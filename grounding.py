"""
grounding.py: STEP 7, check the answer against THIS turn's tool results
(Mike's verifyCitations, extended for names, numbers and Arabic).

After every run, a post-hook extracts checkable claims from the answer:

  quote     text in "...", “...”, «...», „...“         must appear in the evidence
  citation  (doc-1a2b3c, Page 4)                      doc must be read/searched this turn
  name      after a title (السيد، الدكتور، Mr., Dr.)   must appear in the evidence
            or an Arabic 3-4 word name ("خالد سالم الحمادي")
  number    2+ digits (45, 50,000, ٢٠٢٦)              must appear in evidence or question

Evidence = results of read_document / find_in_document (and small tools like
calculate) from THIS run only. Memory, summary and earlier turns don't count.

Matching is Arabic-aware: diacritics and tatweel removed, أ/إ/آ->ا, ى->ي,
ة->ه, Arabic-Indic digits -> 0-9, punctuation and extra spaces ignored.

If something can't be verified, agno_agent.py either warns the user or
re-asks the model once to correct itself (GROUNDING_MODE=retry|warn|off).

Limits (it's a safety net, not proof): paraphrases are not checked, and the
Arabic-name pattern is a heuristic. It catches invented names, numbers and
quotes, which are the most harmful hallucinations in document answers.
"""

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GROUNDING_MODE = os.getenv("GROUNDING_MODE", "retry")  # retry | warn | off
DOC_TOOLS = {"read_document", "find_in_document"}
EVIDENCE_TOOLS = DOC_TOOLS | {"calculate", "get_current_time", "get_weather", "list_documents"}

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # + tatweel
_ARABIC_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    **{chr(0x0660 + i): str(i) for i in range(10)},   # ٠-٩
    **{chr(0x06F0 + i): str(i) for i in range(10)},   # ۰-۹
})
_MARKERS = re.compile(r"<<\s*(UNTRUSTED|END)\b[^>]*>>|\[(Page|Sheet) [^\]]*\]|\[Document\]")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = _DIACRITICS.sub("", t).translate(_ARABIC_MAP).lower()
    t = re.sub(r"(?<=\d)[,٬](?=\d{3})", "", t)        # 50,000 -> 50000
    t = re.sub(r"[^\w\s]", " ", t)                     # punctuation -> space
    return " ".join(t.split())


def _found(claim: str, evidence_norm: str) -> bool:
    c = normalize(claim)
    return bool(c) and c in evidence_norm


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------
_QUOTE_RE = re.compile(r'"([^"\n]{8,400})"|“([^”\n]{8,400})”|«([^»\n]{8,400})»|„([^“\n]{8,400})“')
_CITATION_RE = re.compile(r"\b(doc-[0-9a-f]{6})\b")
_PAGE_REF_RE = re.compile(r"\b(?:Page|Sheet|p\.)\s*[0-9٠-٩]+|(?:صفحة|الصفحة|ص)\s*[0-9٠-٩]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])([0-9٠-٩۰-۹]{1,3}(?:[,٬][0-9٠-٩۰-۹]{3})+|[0-9٠-٩۰-۹]{2,})(?![\w])")
_AR = r"[ء-ي٠-٩ـً-ْ]"
_AR_WORD = rf"{_AR}{{2,}}"
_TITLES_AR = r"(?:السيد[ةه]?|الدكتور[ةه]?|د\.|المهندس[ةه]?|الشيخ[ةه]?|الأستاذ[ةه]?|معالي|سعادة)"
_TITLE_NAME_AR = re.compile(rf"{_TITLES_AR}\s*/?\s*((?:{_AR_WORD}\s+){{1,4}}{_AR_WORD})")
_TITLE_NAME_EN = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Eng|Sheikh|H\.E)\.?\s+((?:[A-Z][a-z'-]+\s+){0,3}[A-Z][a-z'-]+)")
# Arabic personal name: 2-3 given/father names (not starting with ال; "عبد الله"
# counts as one), then a family name starting with ال: "خالد سالم الحمادي"
_AR_GIVEN = rf"(?:عبد\s*ال{_AR}{{2,}}|(?!ال){_AR_WORD})"
_AR_NAME = re.compile(rf"(?<!{_AR})((?:{_AR_GIVEN}\s+){{2,3}}ال{_AR}{{2,}})(?!{_AR})")
_NOT_NAME_START = {normalize(w) for w in (
    "في من على الى إلى عن مع هذا هذه ذلك تلك تم قرار مجلس لجنة وزارة رقم المادة بشأن "
    "رئيس عضو نائب مدير هيئة دائرة اعتماد تشكيل تعيين اصدار إصدار نص يتم كان كانت قانون "
    "مرسوم بند فقرة سنة عام شهر يوم حكومة دولة امارة إمارة ورد وردت يوجد لا لم لن ما "
    "اعضاء أعضاء عضوية برئاسة يعين يعيّن تعين السادة"
).split()}


@dataclass
class Claim:
    kind: str     # quote | citation | name | number
    text: str
    ok: bool


@dataclass
class GroundingReport:
    checked: bool = False          # False = not a document answer, nothing to check
    no_evidence: bool = False      # cites/quotes documents but no document tool ran
    claims: List[Claim] = field(default_factory=list)

    @property
    def unverified(self) -> List[Claim]:
        return [c for c in self.claims if not c.ok]

    @property
    def ok(self) -> bool:
        return not self.no_evidence and not self.unverified

    def summary(self) -> str:
        if not self.checked:
            return "not a document answer, skipped"
        if self.no_evidence:
            return "cites documents but none were read this turn"
        n = len(self.claims)
        bad = self.unverified
        if not bad:
            return f"{n} claim(s) checked, all found in the documents"
        items = "; ".join(f"{c.kind}: {c.text[:60]}" for c in bad[:8])
        return f"{len(bad)} of {n} claim(s) NOT found in the documents: {items}"


def _names(answer: str) -> List[str]:
    out = [m.group(1) for m in _TITLE_NAME_AR.finditer(answer)]
    out += [m.group(1) for m in _TITLE_NAME_EN.finditer(answer)]
    for m in _AR_NAME.finditer(answer):
        words = m.group(1).split()
        first = normalize(words[0])
        if first[:1] in ("و", "ف") and len(first) > 3:  # "ورئيس" -> "رئيس"
            first = first[1:]
        if first not in _NOT_NAME_START:
            out.append(m.group(1))
    seen, uniq = set(), []
    for n in out:
        k = normalize(n)
        if k and k not in seen:
            seen.add(k)
            uniq.append(n.strip())
    return uniq


def check_grounding(answer: str, tools: list, user_text: str = "") -> GroundingReport:
    """tools = run_output.tools (ToolExecution list) of THIS run."""
    report = GroundingReport()
    answer = answer or ""
    doc_calls = [t for t in tools or [] if t.tool_name in DOC_TOOLS]
    cites = set(_CITATION_RE.findall(answer))

    if not doc_calls:
        if cites:  # talks about documents without reading any
            report.checked, report.no_evidence = True, True
        return report
    report.checked = True

    evidence_raw = "\n".join(
        _MARKERS.sub(" ", str(t.result)) for t in tools if t.tool_name in EVIDENCE_TOOLS and t.result)
    evidence = normalize(evidence_raw)
    allowed_numbers = evidence + " " + normalize(user_text)
    used_docs = {str((t.tool_args or {}).get("doc_id", "")) for t in doc_calls}
    used_docs |= set(_CITATION_RE.findall(evidence_raw))  # handles shown in results

    # 1. quotes (parts split on ... are checked separately)
    for m in _QUOTE_RE.finditer(answer):
        q = next(g for g in m.groups() if g)
        parts = [p for p in re.split(r"\.\.\.|…", q) if len(normalize(p)) >= 4]
        report.claims.append(Claim("quote", q, all(_found(p, evidence) for p in parts)))

    # 2. citations: the cited doc must have been read/searched this turn
    for doc_id in sorted(cites):
        report.claims.append(Claim("citation", doc_id, doc_id in used_docs))

    # 3. names (also accept the name without a leading و/ف or extra first word)
    for name in _names(answer):
        words = name.split()
        variants = [name, re.sub(r"^[وف]", "", name)]
        if len(words) >= 4:
            variants.append(" ".join(words[1:]))
        report.claims.append(Claim("name", name, any(_found(v, evidence) for v in variants)))

    # 4. numbers (2+ digits), ignoring those inside citations like doc-3e47c5
    answer_wo_cites = _PAGE_REF_RE.sub(" ", _CITATION_RE.sub(" ", answer))
    seen = set()
    for m in _NUMBER_RE.finditer(answer_wo_cites):
        num = normalize(m.group(1))
        if num in seen:
            continue
        seen.add(num)
        report.claims.append(Claim("number", m.group(1), bool(re.search(rf"(?<!\d){num}(?!\d)", allowed_numbers))))

    return report


def correction_prompt(report: GroundingReport) -> str:
    """Message sent to the model for ONE automatic correction attempt."""
    if report.no_evidence:
        problem = "You cited documents without reading them in this turn."
    else:
        items = "\n".join(f"- {c.kind}: {c.text}" for c in report.unverified[:15])
        problem = f"These items in your answer were NOT found in the tool results of this turn:\n{items}"
    return (
        "[Automatic grounding check, not written by the user]\n"
        f"{problem}\n"
        "Read or search the relevant document again now, then give a corrected answer. "
        "Keep only names, numbers and quotes that appear in the tool results, copied exactly. "
        "If something is not in the document, say it was not found. Do not apologise at length."
    )


# ---------------------------------------------------------------------------
# Agno post-hook: run the check after every run and keep the report
# ---------------------------------------------------------------------------
_reports: Dict[str, GroundingReport] = {}   # run_id -> report (read by the CLI/API)


def grounding_hook(run_output, run_context) -> None:
    if GROUNDING_MODE == "off":
        return
    user_text = ""
    if getattr(run_output, "input", None) is not None:
        user_text = str(getattr(run_output.input, "input_content", "") or "")
    report = check_grounding(str(run_output.content or ""), run_output.tools or [], user_text)
    _reports[run_output.run_id] = report
    try:  # also record it in the audit log
        from tool_guard import _audit
        _audit(run_context, tool="grounding_check", ok=report.ok,
               result=report.summary()[:500])
    except Exception:
        pass


def pop_report(run_id: Optional[str]) -> Optional[GroundingReport]:
    return _reports.pop(run_id, None) if run_id else None
