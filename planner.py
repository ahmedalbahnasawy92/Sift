"""
planner.py: pick the route for a message BEFORE the agent runs.

  chat       small chat agent on sift-local (greetings, weather, time, maths)
  documents  the full document agent (search, read, workflows, grounding)
  overview   list the user's files (no LLM)

Order (adapted from Mujeeb's planner.py):
  1. Rules, free:   greeting/thanks -> chat, "list my files" -> overview,
                    document words -> documents, weather/time/maths -> chat,
                    follow-up after a documents turn -> documents.
  2. adept3o JSON call (sift-local via the LiteLLM proxy) only if no rule fired,
     with a hard PLANNER_TIMEOUT (default 1.5 s). It also rewrites the query
     into Arabic and English search hints for search_documents.
  3. Anything else (timeout, bad JSON, error, unsure) -> documents.

The two possible mistakes don't cost the same: chat sent to documents costs a
little; documents sent to chat makes the model answer from its own knowledge.
So every doubt goes to documents, and the chat agent has an escape tool
(needs_documents) that re-runs the turn here as route=documents.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PLANNER_TIMEOUT = float(os.getenv("PLANNER_TIMEOUT", "1.5"))
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "sift-local")


@dataclass
class Plan:
    route: str                    # chat | documents | overview
    reason: str                   # why, e.g. "rules:greeting", "llm:...", "fallback:timeout"
    source: str                   # rules | llm | fallback
    queries: Dict[str, List[str]] = field(default_factory=dict)  # {"ar": [...], "en": [...]}
    ms: int = 0

    def metadata(self) -> dict:
        return {"route": self.route, "reason": self.reason, "source": self.source}


# ---------------------------------------------------------------------------
# Rules (anchored where they must be: "what is X in Arabic law" is NOT a greeting)
# ---------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # + tatweel


def _norm(text: str) -> str:
    t = _DIACRITICS.sub("", text.lower())
    t = re.sub("[أإآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s%]", " ", t)          # punctuation, emoji
    return re.sub(r"\s+", " ", t).strip()


_GREETING = re.compile(
    r"^(hi|hello|hey|hiya|good (morning|afternoon|evening)|thanks?( you| a lot)?|thx|ok(ay)?|"
    r"bye|goodbye|see you|how are you|who are you|what can you do|"
    r"مرحبا|اهلا|اهلا وسهلا|هلا|السلام عليكم|وعليكم السلام|صباح الخير|مساء الخير|شكرا|شكرا لك|"
    r"شكرا جزيلا|تمام|حسنا|مع السلامه|كيف حالك|من انت|ماذا تستطيع ان تفعل)( \w+)?$")

_OVERVIEW = re.compile(
    r"^((what|which) (files|documents|docs) (do i have|are (there|available))|"
    r"(list|show)( me)?( all)? (my )?(files|documents|docs)|"
    r"(ما|ماهي|ما هي) (ملفاتي|مستنداتي|وثائقي|الملفات المتاحه|المستندات المتاحه)|"
    r"(اعرض|اظهر|اذكر) (لي )?(ملفاتي|مستنداتي|الملفات|المستندات))$")

_DOC_WORDS = re.compile(
    r"\b(documents?|docs?|files?|pdf|contracts?|lease|tenancy|agreements?|laws?|decrees?|"
    r"decisions?|resolutions?|articles?|clauses?|gazette|page|sections?|polic(y|ies)|reports?|"
    r"according to|summari[sz]e|summary|compare|workflow|"
    r"مستند\w*|ملف\w*|وثيق\w*|وثائق|عقد|عقود|ايجار|قانون|قوانين|مرسوم|قرار\w*|الماده|ماده|بند|"
    r"الجريده|صفح\w*|لائح\w*|سياس\w*|تقرير|وفقا|حسب|لخص|ملخص|قارن|مقارن\w*|اللجنه|لجنه)\b")

_TOOL_WORDS = re.compile(
    r"\b(weather|temperature|forecast|time is it|what time|date today|calculate|"
    r"[\d.,]+ ?[-+*/x%] ?[\d.,]+|percent|"
    r"الطقس|الجو|درجه الحراره|الوقت|الساعه|التاريخ اليوم|احسب|كم يساوي)\b")


def _rules(text: str, last_route: Optional[str]) -> Optional[Plan]:
    t = _norm(text)
    if not t:
        return Plan("chat", "rules:empty", "rules")
    if _OVERVIEW.match(t):
        return Plan("overview", "rules:list_files", "rules")
    if _GREETING.match(t):
        return Plan("chat", "rules:greeting", "rules")
    if _DOC_WORDS.search(t):
        return Plan("documents", "rules:document_words", "rules")
    if _TOOL_WORDS.search(t):
        return Plan("chat", "rules:weather_time_math", "rules")
    if last_route == "documents":
        return Plan("documents", "rules:follow_up_after_documents", "rules")
    return None


# ---------------------------------------------------------------------------
# adept3o JSON call (only when no rule fired)
# ---------------------------------------------------------------------------
_SYSTEM = """You route messages for a document assistant. The user has these files:
{files}

Return ONLY JSON: {{"route": "chat" | "documents", "reason": "<5 words>", "ar": ["..."], "en": ["..."]}}
- "documents": anything that may need the user's files, or facts about laws,
  regulations, contracts, decisions, people, dates or numbers. When unsure: documents.
- "chat": small talk, thanks, questions about you, weather, time, arithmetic,
  or general knowledge clearly unrelated to the files.
- For documents, "ar" and "en": one short search query each (Arabic and English
  words likely to appear in the files). For chat: empty lists."""


def _llm(text: str, history: List[str], files: List[str]) -> Plan:
    from openai import OpenAI
    base = os.getenv("LITELLM_BASE_URL", "").strip()
    if not base:
        return Plan("documents", "fallback:no_proxy", "fallback")
    client = OpenAI(base_url=base, api_key=os.getenv("LITELLM_API_KEY") or "none",
                    timeout=PLANNER_TIMEOUT, max_retries=0)
    convo = "\n".join(history[-4:]) or "(none)"
    try:
        resp = client.chat.completions.create(
            model=PLANNER_MODEL, temperature=0, max_tokens=120,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _SYSTEM.format(files="\n".join(files) or "(none)")},
                      {"role": "user", "content": f"Recent conversation:\n{convo}\n\nMessage: {text}"}],
            extra_body={"metadata": {"component": "planner"}})
        raw = resp.choices[0].message.content or ""
    except Exception as e:  # timeout, proxy down, model down
        return Plan("documents", f"fallback:{type(e).__name__}", "fallback")
    raw = re.sub(r"<\|channel>.*?<channel\|>", "", raw, flags=re.S)  # Gemma thinking tags
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    route = data.get("route")
    if route not in ("chat", "documents"):
        return Plan("documents", "fallback:bad_json", "fallback")
    queries = {k: [q for q in data.get(k, []) if isinstance(q, str) and q.strip()][:2]
               for k in ("ar", "en")} if route == "documents" else {}
    return Plan(route, f"llm:{str(data.get('reason', ''))[:60]}", "llm", queries)


def plan(text: str, history: Optional[List[str]] = None, files: Optional[List[str]] = None,
         last_route: Optional[str] = None) -> Plan:
    """Route one message. history: recent 'role: text' lines; files: document names."""
    t0 = time.time()
    p = _rules(text, last_route) or _llm(text, history or [], files or [])
    p.ms = int((time.time() - t0) * 1000)
    return p
