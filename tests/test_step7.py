"""
test_step7.py: offline tests for grounding.py (no LLM, no API key).

Run:  python tests/test_step7.py      (or: pytest tests/test_step7.py)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from grounding import check_grounding, correction_prompt, normalize  # noqa: E402

N = "abcd1234"
GAZETTE = (
    f"Document doc-3e47c5: <<UNTRUSTED {N}>>2026_790.pdf<<END {N}>> (12 section(s))\n\n"
    f'<<UNTRUSTED {N} source="doc-3e47c5">>\n[Page 4]\n'
    "قرار رقم (45) لسنة 2026 بشأن تعيين رئيس لجنة التظلمات المركزية\n"
    "يُعيّن السيد/ خلفان أحمد حارب رئيساً للجنة التظلمات المركزية.\n"
    "[Page 5]\nقرار رقم (46) بتشكيل لجنة البت في التظلمات برئاسة "
    "الدكتورة/ فاطمة علي الكعبي وعضوية سالم راشد المزروعي.\n"
    "The Termination Fee is AED 50,000 and payment is due in 30 days.\n"
    f"<<END {N}>>"
)


@dataclass
class T:  # shaped like agno ToolExecution
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    result: Any = None


READ = [T("read_document", {"doc_id": "doc-3e47c5"}, GAZETTE)]


def test_1_your_case_invented_names_are_caught():
    answer = ("أعضاء اللجنة: خالد سالم الحمادي، محمد خماس المانعي، عبد الله خميس الحبشي. "
              "ورئيس لجنة التظلمات المركزية هو السيد/ خلفان أحمد حارب (doc-3e47c5, Page 4).")
    r = check_grounding(answer, READ)
    bad = {c.text for c in r.unverified}
    assert bad == {"خالد سالم الحمادي", "محمد خماس المانعي", "عبد الله خميس الحبشي"}, bad
    print("1. invented Arabic names flagged, real name passes      OK")
    print("   ->", r.summary())


def test_2_real_names_pass_with_spelling_variants():
    # diacritics, different alef/taa marbuta forms, title with slash
    answer = "برئاسة الدكتوره/ فاطمه علي الكعبي وعضوية سالِم راشد المزروعي."
    r = check_grounding(answer, READ)
    assert r.ok, r.summary()
    print("2. real names pass despite ه/ة, diacritics, titles       OK")


def test_3_quotes():
    good = 'The contract says "The Termination Fee is AED 50,000" (doc-3e47c5, Page 5).'
    bad = 'The contract says "The Termination Fee is waived after one year".'
    assert check_grounding(good, READ).ok
    r = check_grounding(bad, READ)
    assert [c.kind for c in r.unverified] == ["quote"], r.summary()
    print("3. exact quote passes, invented quote flagged            OK")


def test_4_quote_with_ellipsis():
    answer = '«يُعيّن السيد/ خلفان أحمد حارب ... للجنة التظلمات المركزية»'
    assert check_grounding(answer, READ).ok
    print("4. quote with ... checked part by part                   OK")


def test_5_numbers_including_arabic_digits():
    ok = check_grounding("القرار رقم (٤٦) والرسوم 50,000 درهم خلال 30 يوماً.", READ)
    assert ok.ok, ok.summary()
    bad = check_grounding("The fee is AED 75,000.", READ)
    assert [c.text for c in bad.unverified] == ["75,000"], bad.summary()
    print("5. numbers checked (٤٦ = 46, 50,000), invented one flagged OK")


def test_6_page_numbers_are_not_claims():
    r = check_grounding("See (doc-3e47c5, Page 12) and الصفحة 15.", READ)
    assert r.ok, r.summary()
    print("6. page numbers in citations are not flagged             OK")


def test_7_citation_to_unread_document():
    r = check_grounding("As stated in doc-ffffff, the fee is 50,000.", READ)
    assert [c.text for c in r.unverified] == ["doc-ffffff"], r.summary()
    print("7. citing a document NOT read this turn is flagged       OK")


def test_8_cites_documents_without_reading():
    r = check_grounding("According to doc-3e47c5 the fee is AED 50,000.", tools=[])
    assert r.no_evidence and not r.ok
    assert "without reading" in correction_prompt(r)
    print("8. document answer with no read this turn is flagged     OK")


def test_9_non_document_answer_is_skipped():
    r = check_grounding("Dubai is 31°C today.", [T("get_weather", {"city": "Dubai"}, "Dubai: 31°C")])
    assert not r.checked and r.ok
    print("9. weather/chit-chat answers are not checked             OK")


def test_10_numbers_from_user_question_are_allowed():
    r = check_grounding("Decision 99 is not in the document.", READ, user_text="what does decision 99 say?")
    assert r.ok, r.summary()
    print("10. numbers the user asked about are allowed             OK")


def test_11_normalize():
    assert normalize("إِلَى الْمَدِينَةِ") == normalize("الى المدينه")
    assert normalize("٢٠٢٦") == "2026" and normalize("50,000") == "50000"
    print("11. Arabic normalisation (diacritics, alef, digits)      OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items(), key=lambda kv: int(kv[0].split("_")[1]) if kv[0].startswith("test_") else 0):
        if name.startswith("test_"):
            fn()
    print("\nAll step-7 tests passed.")
