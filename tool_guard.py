"""
tool_guard.py: STEP 5, the harness around every tool call (Mike's toolDispatcher).

Two Agno hooks:

1. tool_guard  (Agent(tool_hooks=[tool_guard]))  wraps EVERY tool call:
   - Read de-duplication: a document already read in this run is not sent
     again (Mike's "already returned IN FULL" notice). Saves tokens and stops
     read loops.
   - Error safety net: an exception in a tool becomes a readable error for
     the model instead of crashing the run.
   - Output cap: no tool result can exceed MAX_TOOL_RESULT_CHARS.
   - Audit log: one JSON line per call in audit.log (who, what, args,
     result size, duration, error, blocked).

2. remember_tool_activity  (Agent(post_hooks=[...]))  runs after each run:
   Old tool outputs are dropped from history (max_tool_calls_from_history=0),
   so the next turn would forget WHAT happened: which docs were read, what the
   user answered in ask_user/get_user_input, whether an action was rejected.
   This writes a short list into session_state, which Agno shows in the next
   turn's system prompt (add_session_state_to_context=True).
   Mike's equivalent: "[Tool activity in your previous turn]".
   Only identifiers and user answers are kept, never document content.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from agno.run import RunContext

from doc_tools import _resolve
from spotlight import fence_inline, nonce_for  # STEP 6

AUDIT_LOG = os.getenv("AUDIT_LOG", "audit.log")
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "120000"))
DOCUMENT_TOOLS = {"list_documents", "read_document", "find_in_document", "search_documents"}

# run_id -> set of documents already read in that run (cleared after the run)
_reads_by_run: Dict[str, set] = {}

ALREADY_READ = (
    "This document was already returned IN FULL earlier in this answer. "
    "The text is not repeated to save space; this is not an error or a truncation. "
    "Use the text you already have, or find_in_document for a specific phrase. "
    "Do NOT call read_document for this document again."
)


def _audit(run_context: RunContext, **fields: Any) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_id": getattr(run_context, "user_id", None),
        "session_id": getattr(run_context, "session_id", None),
        "run_id": getattr(run_context, "run_id", None),
        **fields,
    }
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# 1. Tool hook: wraps every tool call (async, because we use agent.arun)
# ---------------------------------------------------------------------------
async def tool_guard(
    function_name: str,
    function_call: Callable,
    arguments: Dict[str, Any],
    run_context: RunContext,
) -> Any:
    started = time.perf_counter()
    safe_args = {k: v for k, v in arguments.items() if k != "run_context"}

    # Read de-duplication (per run)
    if function_name == "read_document":
        doc = _resolve(str(arguments.get("doc_id", "")), run_context)
        key = doc.rel if doc else str(arguments.get("doc_id"))
        reads = _reads_by_run.setdefault(run_context.run_id, set())
        if key in reads:
            _audit(run_context, tool=function_name, args=safe_args, blocked="already_read")
            return ALREADY_READ
        reads.add(key)

    # Run the tool; turn crashes into errors the model can act on
    error = None
    try:
        result = await function_call(**arguments)
    except Exception as e:  # tools should catch their own errors; this is the net
        error = f"{type(e).__name__}: {e}"
        result = f"Tool {function_name} failed: {error}. Check the arguments or try another approach."

    # Output cap
    text = result if isinstance(result, str) else None
    if text is not None and len(text) > MAX_TOOL_RESULT_CHARS:
        result = text[:MAX_TOOL_RESULT_CHARS] + f"\n\n[Tool output cut at {MAX_TOOL_RESULT_CHARS:,} chars.]"

    _audit(
        run_context,
        tool=function_name,
        args={k: (str(v)[:300]) for k, v in safe_args.items()},
        result_chars=len(text) if text is not None else None,
        ms=round((time.perf_counter() - started) * 1000),
        error=error,
    )
    return result


# ---------------------------------------------------------------------------
# 2. Post-hook: summarise this run's tool activity for the NEXT turn
# ---------------------------------------------------------------------------
def _first_line(s: Any, limit: int = 120) -> str:
    line = str(s or "").strip().splitlines()[0] if str(s or "").strip() else ""
    return line[:limit]


def _json_after_colon(s: Any) -> List[dict]:
    try:
        return json.loads(str(s).split(":", 1)[1])
    except Exception:
        return []


def _activity_line(t, n: str) -> str:
    name, args = t.tool_name, t.tool_args or {}

    if getattr(t, "confirmed", None) is False:                      # rejected action
        note = getattr(t, "confirmation_note", None) or "no reason given"
        return f"- user REJECTED {name}({_first_line(json.dumps(args, ensure_ascii=False), 80)}): {fence_inline(note, n)}"
    if t.tool_call_error:
        return f"- {name} failed: {_first_line(t.result)}"

    if name == "list_documents":
        return f"- list_documents -> {_first_line(t.result)}"
    if name == "read_document":                                      # ids only, never content
        if str(t.result).startswith("This document was already returned"):
            return ""
        return f"- read_document -> {_first_line(t.result)}"
    if name == "find_in_document":
        return f"- find_in_document({args.get('doc_id') or 'all'}, \"{args.get('query')}\") -> {_first_line(t.result)}"
    if name == "search_documents":                                   # STEP 8: ids only
        return f"- search_documents(\"{args.get('query')}\") -> {_first_line(t.result, 200)}"
    if name == "ask_user":                                           # user's choices
        qs = getattr(t, "user_feedback_schema", None) or []
        rows = [(q.question, q.selected_options) for q in qs] or [
            (r.get("question"), r.get("selected")) for r in _json_after_colon(t.result)]
        picks = "; ".join(f"{q} -> {', '.join(sel or []) or 'skipped'}" for q, sel in rows)
        return f"- user chose: {picks or 'no answer'}"
    if name == "get_user_input":                                     # user's free text
        fields = getattr(t, "user_input_schema", None) or []
        rows = [(f.name, f.value) for f in fields] or [
            (r.get("name"), r.get("value")) for r in _json_after_colon(t.result)]
        vals = "; ".join(f"{k} = {fence_inline(v, n) if v not in (None, '') else 'skipped'}" for k, v in rows)
        return f"- user answered: {vals or 'no answer'}"
    # small tools (weather, time, calculate, save_note): name + short result
    return f"- {name}({_first_line(json.dumps(args, ensure_ascii=False), 80)}) -> {_first_line(t.result)}"


def remember_tool_activity(run_output, run_context: RunContext) -> None:
    _reads_by_run.pop(run_context.run_id, None)  # free the per-run read set

    n = nonce_for(run_context.session_id)  # STEP 6: user answers are untrusted
    lines = [line for t in (run_output.tools or []) if (line := _activity_line(t, n))]
    if run_context.session_state is None:
        run_context.session_state = {}
    # Replace (not append): only the PREVIOUS turn is kept, like Mike.
    run_context.session_state["previous_turn_tool_activity"] = lines[:30]
