"""
test_step8.py: offline tests for search_index.py + the step-8 doc_tools changes.
No LLM or API key: embeddings come from a small fake embedder.

Run (local SQLite store):   python tests/test_step8.py
Run against Postgres:       VECTOR_DB_URL=postgresql://user@host:5432/db python tests/test_step8.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

TMP = Path(tempfile.mkdtemp(prefix="step8_"))
os.environ["DOCS_ROOT"] = str(TMP / "docs")
os.environ["DOC_CACHE_DIR"] = str(TMP / "cache")
os.environ["SPOTLIGHT_SECRET_FILE"] = str(TMP / "secret")
os.environ["SUMMARY_MODEL"] = "none"       # summaries fall back to the first line
os.environ["EMBEDDING_MODEL"] = "none"     # replaced by the fake embedder below
os.environ.setdefault("EMBEDDING_DIM", "8")

U1, U2 = TMP / "docs" / "u1", TMP / "docs" / "u2"
U1.mkdir(parents=True)
U2.mkdir(parents=True)

import pymupdf  # noqa: E402

gazette = pymupdf.open()
p1 = gazette.new_page()
p1.insert_text((72, 72), "Official Gazette issue 790. Table of contents.")
p2 = gazette.new_page()
p2.insert_text((72, 72), "Decision 46 forms the grievance committee. Members: Salem Rashid.")
gazette.save(U1 / "gazette_790.pdf")
(U1 / "قرار_لجنة.txt").write_text(
    "قرار رقم (46) بتشكيل لجنة البت في التظلمات برئاسة الدكتورة فاطمة علي الكعبي "
    "وعضوية سالم راشد المزروعي.\nتختص اللجنة بالنظر في التظلمات المقدمة من الموظفين.")
(U1 / "vendor.txt").write_text("The Termination Fee is AED 50,000. Payment is due in 30 days.")
(U1 / "policy.txt").write_text("Remote work is allowed two days per week for all staff.")
(U2 / "secret.txt").write_text("u2 private: grievance committee salary table")

from agno.run import RunContext  # noqa: E402

import doc_tools as d  # noqa: E402
import search_index as si  # noqa: E402

# ---------------------------------------------------------------------------
# Fake embedder: words of the same "concept" share one dimension, so a query
# can match a passage by MEANING without sharing any word with it.
# ---------------------------------------------------------------------------
CONCEPTS = [
    {"remote", "home", "telework", "wfh"},
    {"fee", "penalty", "charge", "cost"},
    {"committee", "لجنة", "panel", "board"},
    {"grievance", "تظلم", "complaint", "appeal"},
    {"members", "عضوية", "membership", "member"},
    {"staff", "employees", "موظف", "workers"},
    {"payment", "due", "pay"},
    {"week", "days", "weekly"},
]


def fake_embed(texts):
    out = []
    for t in texts:
        toks = set(si.keyword_tokens(t)) | set(t.lower().split())
        out.append([1.0 if (toks & c) else 0.0 for c in CONCEPTS])
    return out


si._embedder.enabled = True
si._embedder.embed = fake_embed

CTX1 = RunContext(run_id="r", session_id="s", user_id="u1")
CTX2 = RunContext(run_id="r", session_id="s", user_id="u2")


def h(name):
    return next(x.doc_id for x in d._list_docs(CTX1) if x.path.name == name)


def test_1_chunking_keeps_pages():
    chunks = si.chunk_pages([("Page 1", "a " * 1500), ("Page 2", "short")])
    assert {c[0] for c in chunks} == {"Page 1", "Page 2"}
    assert all(len(c[1]) <= si.CHUNK_CHARS for c in chunks)
    assert len([c for c in chunks if c[0] == "Page 1"]) >= 3
    print("1. chunks never cross pages, size-limited, overlapping    OK")


def test_2_arabic_keyword_stemming():
    a, b, c = (si.keyword_tokens(x) for x in ("التظلمات", "والتظلم", "تظلُّم"))
    assert a == b == c == ["تظلم"], (a, b, c)
    print("2. التظلمات / والتظلم / تظلُّم -> same keyword            OK")


def test_3_index_and_summaries():
    stats = si.ensure_indexed("u1")
    assert stats["indexed"] == 4, stats
    again = si.ensure_indexed("u1")
    assert again["indexed"] == 0 and again["unchanged"] == 4, again
    block = d.documents_block("u1", "s")
    assert "Termination Fee is AED 50,000" in block, "summary (first line) in the list"
    print("3. 4 files indexed once; unchanged files skipped; summaries in list OK")


def test_4_search_all_docs_finds_right_doc_and_page():
    out = si.search_documents("Decision 46 grievance committee members", CTX1)
    first = out.split("\n")[2]  # header, fence-open, first hit
    assert h("gazette_790.pdf") in out and "[Page 2]" in out, out
    print("4. search across all docs -> right file and page          OK")
    print("   " + first[:110])


def test_5_arabic_query_variants():
    out = si.search_documents("اعضاء لجنه التظلم", CTX1)  # no hamza, ه for ة, singular
    assert h("قرار_لجنة.txt") in out, out
    print("5. Arabic query with spelling/word-form variants finds doc OK")


def test_6_semantic_match_without_shared_words():
    out = si.search_documents("telework", CTX1, top_k=1)  # no doc contains "telework"
    assert h("policy.txt") in out, out
    print("6. match by meaning ('telework' -> remote work policy)    OK")


def test_7_doc_filter_and_permissions():
    only = si.search_documents("committee", CTX1, doc_ids=h("vendor.txt"))
    assert h("gazette_790.pdf") not in only
    other = si.search_documents("grievance committee", CTX2)
    assert "u2 private" in other and h("gazette_790.pdf") not in other
    mine = si.search_documents("salary table", CTX1)
    assert "u2 private" not in mine
    print("7. doc_ids filter works; users never see each other's docs OK")


def test_8_reindex_on_change_and_delete():
    (U1 / "vendor.txt").write_text("The Termination Fee is AED 75,000 from 2027.")
    (U1 / "policy.txt").unlink()
    stats = si.ensure_indexed("u1")
    assert stats["indexed"] == 1 and stats["removed"] == 1, stats
    out = si.search_documents("termination fee", CTX1)
    assert "75,000" in out and "50,000" not in out
    print("8. changed file re-indexed, deleted file removed          OK")


def test_9_find_all_docs_arabic_tolerant():
    out = d.find_in_document("all", "الدكتوره فاطمه", CTX1)  # ه instead of ة
    assert "الدكتورة فاطمة" in out, out          # original wording returned
    assert h("قرار_لجنة.txt") in out
    print("9. find_in_document('all') + ة/ه tolerant, original text   OK")


def test_10_results_are_fenced():
    n = si.nonce_for("s")
    out = si.search_documents("termination", CTX1)
    assert f"<<UNTRUSTED {n} source=\"search_documents\">>" in out and f"<<END {n}>>" in out
    print("10. search results fenced as untrusted data               OK")


def test_11_keywords_only_mode():
    si._embedder.enabled = False
    out = si.search_documents("termination fee", CTX1)
    si._embedder.enabled = True
    assert "keywords only" in out and h("vendor.txt") in out
    print("11. without embeddings, keyword search still works       OK")


if __name__ == "__main__":
    store = "Postgres + pgvector" if si.VECTOR_DB_URL else "local SQLite"
    print(f"store: {store}\n")
    if si.VECTOR_DB_URL:  # clean test users from a shared database
        for uid in ("u1", "u2"):
            for rel in si.get_store().docs(uid):
                si.get_store().remove_doc(uid, rel)
    for name, fn in sorted(globals().items(), key=lambda kv: int(kv[0].split("_")[1]) if kv[0].startswith("test_") else 0):
        if name.startswith("test_"):
            fn()
    print("\nAll step-8 tests passed.")
