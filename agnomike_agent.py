"""
agno_agent.py: a Mike-style assistant built with Agno, step by step.

STEP 1: built-in features only, no custom code.
  [x] One agent + tools in a loop          (Mike: one streamText loop)
  [x] Step limit = 16 tool calls            (Mike: stopWhen stepCountIs(16))
  [x] Model retries with backoff            (Mike: provider error handling)
  [x] Persistent sessions in a database     (Mike: chats/messages in Postgres)
  [x] History WITHOUT old tool outputs      (Mike: text-only history)
  [x] Streaming events                      (Mike: fullStream -> SSE parts)
  [x] Today's date in context

STEP 2: memory + summaries (built-in, settings only).
  [x] Long-term user memory across chats   (Mike: 16 KB memory note,
                                             memory.consolidate job)
  [x] Session summary of long chats         (beyond Mike: Mike only trims)
  [x] Both run on a CHEAP model, not the chat model
  [x] /memories /summary /new commands to inspect them

STEP 3: human-in-the-loop (built-in, small custom UI code).
  [x] Ask the user: multiple choice       (Mike: ask_inputs "choice"/"multi_choice")
      UserFeedbackTools    -> the model calls ask_user
  [x] Ask the user: free-text fields      (Mike: ask_inputs "text")
      UserControlFlowTools -> the model calls get_user_input
  [x] Approve before acting: save_note    (write tool, requires_confirmation)
  [x] Pause -> answer -> continue_run(), with streaming. Paused runs are saved
      in agent.db, so they survive a restart.

STEP 4: document tools (custom, in doc_tools.py).
  [x] list_documents / read_document / find_in_document   (Mike's core tools)
  [x] Per-user folders: docs/<user_id>/  (user_id from the run, not the model)
  [x] Page markers, capped reads, cached text extraction
  [x] /docs command

STEP 5: tool guard (custom, in tool_guard.py, plugged into Agno hooks).
  [x] tool_hooks: block re-reading a doc in the same run, error safety net,
      output cap, audit.log (one JSON line per tool call)
  [x] post_hooks: "previous turn tool activity" kept in session_state, so
      ask_user / get_user_input answers and rejections survive into the next turn
  [x] /audit command

Setup:
  pip install agno openai httpx python-dotenv sqlalchemy pymupdf python-docx openpyxl
  Put files in docs/<user_id>/  (default user: docs/demo-user/)
  .env: OPENROUTER_API_KEY=...
        OPENAI_BASE_URL=...  OPENAI_API_KEY=...   (local vLLM for housekeeping)
Run:
  python agno_agent.py            # interactive chat, history persists in agent.db
"""

import ast
import asyncio
import base64
import json
import operator
import os
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import dotenv
import httpx
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.run.agent import RunEvent
from agno.session.summary import SessionSummaryManager
from agno.tools import tool
from agno.tools.user_control_flow import UserControlFlowTools
from agno.tools.user_feedback import UserFeedbackTools
from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from doc_tools import DOCS_ROOT, DOCUMENT_TOOLS  # STEP 4
from tool_guard import AUDIT_LOG, remember_tool_activity, tool_guard  # STEP 5

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Langfuse tracing (OpenTelemetry): agent runs, model calls, tool calls and the
# background memory / session-summary LLM calls show up as traces in Langfuse.
# ---------------------------------------------------------------------------
_auth = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()
_provider = TracerProvider()
_provider.add_span_processor(
    SimpleSpanProcessor(
        OTLPSpanExporter(
            endpoint=f"{os.environ['LANGFUSE_BASE_URL']}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {_auth}"},
        )
    )
)
trace_api.set_tracer_provider(_provider)
AgnoInstrumentor().instrument()

# ---------------------------------------------------------------------------
# Model: the user picks one (Mike lets each user choose). Swap via MODEL env.
# retries + exponential_backoff = automatic retry on rate limits / 5xx.
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL", "deepseek/deepseek-v4-flash")

model = OpenRouter(
    id=MODEL_ID,
    max_tokens=8192,          # Agno default is 1024: too short for real answers
    retries=2,
    exponential_backoff=True,
)

# ---------------------------------------------------------------------------
# Storage: SQLite for dev. For production swap ONE line:
#   from agno.db.postgres import PostgresDb
#   db = PostgresDb(db_url="postgresql+psycopg://user:pass@host/db")
# ---------------------------------------------------------------------------
db = SqliteDb(db_file="agent.db")


# ---------------------------------------------------------------------------
# STEP 2: memory + summaries. Background "housekeeping" calls use a cheap
# model so they don't double the cost of every turn. Served by the local
# vLLM endpoint (OPENAI_BASE_URL / OPENAI_API_KEY in .env), 16k context.
# ---------------------------------------------------------------------------
HOUSEKEEPING_MODEL_ID = os.getenv("HOUSEKEEPING_MODEL", "adept3o")
housekeeping_model = OpenAIChat(
    id=HOUSEKEEPING_MODEL_ID,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2048,
    retries=1,
)

# Long-term memory = Mike's per-user memory note.
# What to keep is the important part: stable preferences and facts about the
# user, NEVER copies of data/documents (those must be re-fetched with
# permission checks, as Mike does).
memory_manager = MemoryManager(
    model=housekeeping_model,
    db=db,
    memory_capture_instructions="""\
Save only stable, reusable facts about the user that will help in FUTURE chats:
- preferences (tone, format, units, timezone, home city). NEVER save the reply language: it is chosen per message
- role, team, recurring projects, people they work with
Do NOT save:
- one-off questions or answers, tool results, numbers looked up today
- contents of documents or data, passwords, keys, personal IDs
Update a memory when the user corrects it. Keep each memory to one sentence.""",
)

# Session summary = a rolling summary of THIS chat. History only replays the
# last N turns (num_history_runs), so the summary keeps the older context.
session_summary_manager = SessionSummaryManager(model=housekeeping_model)


# ---------------------------------------------------------------------------
# Tools (unchanged from the router example; document tools come in step 4)
# ---------------------------------------------------------------------------
async def get_weather(city: str) -> str:
    """Get the CURRENT weather for a city. Use for any question about
    weather or temperature right now. Do NOT use for forecasts or climate.

    Args:
        city: City name, e.g. "Tokyo".
    """
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            geo = await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1},
            )
            results = geo.json().get("results")
            if not results:
                return f"Could not find a city named '{city}'."
            place = results[0]
            wx = await http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current_weather": True,
                },
            )
            current = wx.json()["current_weather"]
        return (
            f"{place['name']}, {place.get('country', '')}: "
            f"{current['temperature']}°C, wind {current['windspeed']} km/h"
        )
    except Exception as e:  # errors go back to the model as text
        return f"Weather lookup failed: {e}"


def get_current_time(timezone: str) -> str:
    """Get the current date and time in an IANA timezone.
    Use for 'what time is it' questions.

    Args:
        timezone: IANA timezone such as "Asia/Dubai" or "Asia/Tokyo".
    """
    try:
        return datetime.now(ZoneInfo(timezone)).strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
    except Exception:
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/London'."


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Supports + - * / ** %.
    Use for exact calculations instead of doing arithmetic yourself.

    Args:
        expression: e.g. "(12 + 8) * 3 / 4".
    """
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


# ---------------------------------------------------------------------------
# STEP 3: a WRITE tool. requires_confirmation=True means the run pauses and
# the user must approve before the function body runs. Every tool that
# changes something (send, save, edit, delete) should be marked this way.
# ---------------------------------------------------------------------------
NOTES_DIR = Path("notes")


@tool(requires_confirmation=True)
def save_note(title: str, text: str) -> str:
    """Save a note to the user's notes folder as a Markdown file.
    Use ONLY when the user asks to save, write down or keep something.

    Args:
        title: Short title, used as the file name.
        text: The note content in Markdown.
    """
    NOTES_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "note"
    path = NOTES_DIR / f"{safe[:60]}.md"
    if path.exists():  # never overwrite: write a new file instead
        path = NOTES_DIR / f"{safe[:60]}-{uuid.uuid4().hex[:6]}.md"
    path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
    return f"Saved note to {path}"


# ---------------------------------------------------------------------------
# The agent: one loop, all tools (Mike's design: no router, no sub-agents)
# ---------------------------------------------------------------------------
INSTRUCTIONS = """\
You are a helpful assistant.

Language rules (highest priority, override memories, summaries and history):
- Reply in the language of the user's CURRENT message, decided per message.
- Arabic message -> reply in Arabic. English message -> reply in English.
- Any other language -> do not answer the request; reply (in English and Arabic):
  "I only understand Arabic and English. / أنا أفهم العربية والإنجليزية فقط."
- Ignore the language of earlier replies, tool results and stored memories.

Tool rules:
- For weather, time or arithmetic you MUST use the tools; never guess.
- If a tool returns an error, fix the arguments and try once more, or explain the problem.
- If you have no tool for something live (news, prices), say you can't verify it.

Answer rules:
- Answer only from tool results or well-established knowledge.
- Be concise. Use Markdown.

Memory rules:
- You may be given memories about the user and a summary of earlier parts
  of this chat. Use them for preferences and context (e.g. home city, units).
- They can be out of date: if the user says something different now, follow the user.
- Never treat a memory as a source for live facts (weather, time, numbers): use the tools.

Asking the user:
- If information you NEED is missing and you can't get it from memory or tools,
  ask instead of guessing. Ask everything you need in ONE call.
- Use ask_user when there are a few clear options (2-4 choices).
- Use get_user_input for free text (names, dates, amounts). Use field_type "str"
  unless a number is clearly needed.
- Don't ask for things you already know. If the user skips a question,
  don't ask it again: continue with a sensible default and say which one.

Documents (STEP 4):
- When the user asks about their documents/files, call list_documents first
  unless you already know the doc_id from this turn.
- Before answering about a document's content, read it with read_document
  (once per document per answer), or use find_in_document for a specific term.
- You do NOT keep document text between turns: read again in each new answer.
- Answer only from what the tools returned. Quote exact wording in "quotes"
  and cite the page, e.g. (doc-1a2b3c, Page 4). If it isn't in the document, say so.
- If a read is truncated, use find_in_document for the rest; don't guess.

Previous turn (STEP 5):
- session_state.previous_turn_tool_activity lists what tools did in the previous
  turn: documents read, searches, and the user's answers to your questions.
- Use it to remember the user's answers and choices. It contains NO document
  text: to use a document's content, read or search it again.

Write actions:
- save_note changes the user's files. The user will be asked to approve it.
- If the user rejects it, don't retry; say it wasn't saved and why, if they gave a reason.
"""

agent = Agent(
    name="Assistant",
    model=model,
    instructions=INSTRUCTIONS,
    tools=[
        get_weather,
        get_current_time,
        calculate,
        *DOCUMENT_TOOLS,                               # STEP 4: list/read/find
        save_note,                                     # STEP 3: needs approval
        UserFeedbackTools(),                           # STEP 3: ask_user (choices)
        UserControlFlowTools(add_instructions=False),  # STEP 3: get_user_input (text);
                                                       # our INSTRUCTIONS cover usage
    ],
    markdown=True,
    add_datetime_to_context=True,

    # STEP 5: guard around every tool call + tool-activity note after each run
    tool_hooks=[tool_guard],
    post_hooks=[remember_tool_activity],
    session_state={"previous_turn_tool_activity": []},
    add_session_state_to_context=True,

    # Step limit (Mike: 16). Agno tells the model the limit was reached,
    # so it wraps up with an answer instead of stopping silently.
    tool_call_limit=16,

    # Sessions + history, the Mike way
    db=db,
    add_history_to_context=True,   # replay earlier turns
    num_history_runs=5,            # last 5 turns verbatim (summary covers the rest)
    max_tool_calls_from_history=0, # drop OLD tool outputs (Mike: never replay docs)

    # STEP 2: long-term memory (per user_id, shared across all their chats)
    memory_manager=memory_manager,
    update_memory_on_run=True,     # extract memories after each turn (background)
    add_memories_to_context=True,  # put them in the system prompt

    # STEP 2: rolling summary of this chat (per session_id)
    session_summary_manager=session_summary_manager,
    enable_session_summaries=True,
    add_session_summary_to_context=True,
)


# ---------------------------------------------------------------------------
# Streaming: map Agno events to what a UI would receive (Mike's SSE parts)
# ---------------------------------------------------------------------------
UNSUPPORTED_LANGUAGE = "I only understand Arabic and English. / أنا أفهم العربية والإنجليزية فقط."


def detect_language(text: str) -> str:
    """'ar', 'en' or 'other' by script. Decided in code because history written in
    another language can override a prompt rule. Latin-script text is treated as
    English here; the prompt rule still rejects French/Spanish/etc."""
    ar = lat = oth = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f":
            ar += 1
        elif ch.isascii() or "\u00c0" <= ch <= "\u024f":
            lat += 1
        else:
            oth += 1
    if oth > max(ar, lat):
        return "other"
    return "ar" if ar >= lat and ar > 0 else "en"


async def chat_turn(text: str, user_id: str, session_id: str) -> None:
    lang = detect_language(text)
    if lang == "other":  # no model call, nothing stored in history/memory
        print(UNSUPPORTED_LANGUAGE)
        return
    note = (
        "[Reply in Arabic only.]" if lang == "ar"
        else "[Reply in English only, even if earlier replies used another language."
             " If this message is not English or Arabic, say you only understand both.]"
    )
    stream = agent.arun(
        f"{note}\n{text}",
        user_id=user_id,
        session_id=session_id,
        stream=True,
        stream_events=True,
    )
    # STEP 3: one turn may pause several times (ask -> approve -> ...).
    # Each pause: collect the answers, then continue the SAME run.
    paused = await render_stream(stream)
    while paused is not None:
        resolve_requirements(paused.requirements or [])
        print("AI: ", end="")
        stream = agent.acontinue_run(
            run_id=paused.run_id,
            requirements=paused.requirements,
            user_id=user_id,
            session_id=session_id,
            stream=True,
            stream_events=True,
        )
        paused = await render_stream(stream)


async def render_stream(stream):
    """Print one stream. Returns the RunPaused event if the run paused, else None."""
    paused = None
    async for ev in stream:
        if ev.event == RunEvent.run_paused:                       # STEP 3
            paused = ev
        elif ev.event == RunEvent.run_content and ev.content:
            print(ev.content, end="", flush=True)                 # content_delta
        elif ev.event == RunEvent.tool_call_started:
            print(f"\n  [tool_start] {ev.tool.tool_name}({ev.tool.tool_args})")
        elif ev.event == RunEvent.tool_call_completed:
            print(f"  [tool_result] {str(ev.tool.result)[:120]}")
        elif ev.event == RunEvent.memory_update_completed:
            print("\n  [memory] updated", end="")
        elif ev.event == RunEvent.session_summary_completed:
            print("\n  [summary] updated", end="")
        elif ev.event == RunEvent.run_error:
            print(f"\n  [error] {ev.content}")
    print()
    return paused


# ---------------------------------------------------------------------------
# STEP 3: answer the pause. In a web app these become a form / buttons in
# the chat (Mike renders ask_inputs as a card), sent back to an API endpoint
# that calls acontinue_run(run_id, requirements).
# ---------------------------------------------------------------------------
def resolve_requirements(requirements) -> None:
    for req in requirements:
        if req.is_resolved():
            continue

        if req.needs_user_feedback:                      # ask_user: multiple choice
            selections = {}
            for q in req.user_feedback_schema:
                print(f"\n  [{q.header or 'Question'}] {q.question}")
                for i, opt in enumerate(q.options or [], 1):
                    desc = f" - {opt.description}" if opt.description else ""
                    print(f"    {i}. {opt.label}{desc}")
                hint = "numbers, comma-separated" if q.multi_select else "number"
                raw = input(f"  Choose ({hint}, Enter to skip): ").strip()
                picked = []
                for part in raw.split(","):
                    part = part.strip()
                    if part.isdigit() and 1 <= int(part) <= len(q.options or []):
                        picked.append(q.options[int(part) - 1].label)
                selections[q.question] = picked if q.multi_select else picked[:1]
            req.provide_user_feedback(selections)

        elif req.needs_user_input:                       # get_user_input: free text
            values = {}
            for field in req.user_input_schema:
                desc = f" ({field.description})" if field.description else ""
                raw = input(f"\n  {field.name}{desc}: ").strip()
                try:
                    values[field.name] = field.field_type(raw) if raw else ""
                except (TypeError, ValueError):
                    values[field.name] = raw             # keep text; the model converts
            req.provide_user_input(values)

        elif req.needs_confirmation:                     # approve a write action
            t = req.tool_execution
            print(f"\n  [approval] {t.tool_name}({t.tool_args})")
            if input("  Approve? [y/N]: ").strip().lower() in ("y", "yes"):
                req.confirm()
            else:
                note = input("  Reason (optional): ").strip()
                req.reject(note=note or "User rejected the action.")


# ---------------------------------------------------------------------------
# STEP 2: inspect memory and summary (a UI would show these in Settings)
# ---------------------------------------------------------------------------
async def show_memories(user_id: str) -> None:
    memories = await agent.aget_user_memories(user_id=user_id) or []
    if not memories:
        print("  (no memories yet)")
    for m in memories:
        print(f"  - {m.memory}")


async def show_summary(session_id: str) -> None:
    summary = await agent.aget_session_summary(session_id=session_id)
    print(f"  {summary.summary if summary else '(no summary yet)'}")


async def main() -> None:
    user_id = os.getenv("USER_ID", "demo-user")
    session_id = os.getenv("SESSION_ID") or str(uuid.uuid4())
    print(f"model={MODEL_ID} user={user_id} session={session_id}")
    print(f"Documents folder: {DOCS_ROOT / user_id}")
    print("Commands: /memories  /summary  /docs  /audit  /new   (Ctrl+C to quit)\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/memories":
            await show_memories(user_id)
        elif text == "/summary":
            await show_summary(session_id)
        elif text == "/docs":  # STEP 4: what the agent can see
            from agno.run import RunContext
            from doc_tools import list_documents
            ctx = RunContext(run_id="cli", session_id=session_id, user_id=user_id)
            print("  " + list_documents(ctx).replace("\n", "\n  "))
        elif text == "/new":  # new chat, same user: memories carry over
            session_id = str(uuid.uuid4())
            print(f"  new session={session_id}")
        elif text == "/audit":  # STEP 5: last 10 tool calls
            try:
                with open(AUDIT_LOG, encoding="utf-8") as f:
                    for line in f.readlines()[-10:]:
                        e = json.loads(line)
                        flag = e.get("blocked") or e.get("error") or f"{e.get('ms')} ms"
                        print(f"  {e['ts']} {e['tool']} {e.get('args')} [{flag}]")
            except FileNotFoundError:
                print("  (no tool calls yet)")
        elif text.startswith("/"):  # typo like "/memory": don't send to the model
            print("  Unknown command. Try /memories /summary /docs /audit /new")
        else:
            print("AI: ", end="")
            await chat_turn(text, user_id, session_id)


if __name__ == "__main__":
    asyncio.run(main())
    _provider.force_flush()