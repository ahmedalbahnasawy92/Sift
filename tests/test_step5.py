"""
test_step5.py: offline tests for tool_guard.py (no LLM, no API key, ~1 second).

Run:  python test_step5.py        (or: pytest test_step5.py)

Each test calls the hooks directly, the same way Agno calls them, with fake
tools, so you see exactly what the guard does.
"""

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# Isolated folders/files for the test run (set BEFORE importing the modules)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

TMP = Path(tempfile.mkdtemp(prefix="step5_"))
os.environ["DOCS_ROOT"] = str(TMP / "docs")
os.environ["DOC_CACHE_DIR"] = str(TMP / "cache")
os.environ["AUDIT_LOG"] = str(TMP / "audit.log")
os.environ["MAX_TOOL_RESULT_CHARS"] = "1000"

(TMP / "docs" / "u1").mkdir(parents=True)
(TMP / "docs" / "u1" / "policy.txt").write_text("Remote work is allowed 2 days a week.")

from agno.run import RunContext  # noqa: E402

import tool_guard as g  # noqa: E402


def ctx(run_id="run-1", state=None) -> RunContext:
    return RunContext(run_id=run_id, session_id="s1", user_id="u1", session_state=state)


def audit_lines() -> List[dict]:
    p = Path(os.environ["AUDIT_LOG"])
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def call(name, fn, args, run_context):
    async def function_call(**kwargs):  # what Agno passes as next_func
        return fn(**kwargs)
    return asyncio.run(g.tool_guard(name, function_call, args, run_context))


# A fake tool execution, shaped like agno.models.response.ToolExecution
@dataclass
class FakeTool:
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    result: Any = None
    tool_call_error: bool = False
    confirmed: Optional[bool] = None
    confirmation_note: Optional[str] = None
    user_feedback_schema: Any = None
    user_input_schema: Any = None


@dataclass
class FakeRunOutput:
    tools: list


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_1_read_once_per_run():
    reads = []
    fake_read = lambda doc_id, run_context: reads.append(doc_id) or "Document doc-x: policy.txt\n..."
    c = ctx("run-A")
    first = call("read_document", fake_read, {"doc_id": "policy.txt", "run_context": c}, c)
    second = call("read_document", fake_read, {"doc_id": "policy.txt", "run_context": c}, c)
    assert first.startswith("Document"), first
    assert second == g.ALREADY_READ, second
    assert len(reads) == 1, "the tool body must run only once"
    assert audit_lines()[-1]["blocked"] == "already_read"
    print("1. second read in the same run is blocked              OK")


def test_2_new_run_can_read_again():
    c_new = ctx("run-B")
    out = call("read_document", lambda doc_id, run_context: "Document fresh", {"doc_id": "policy.txt", "run_context": c_new}, c_new)
    assert out == "Document fresh"
    print("2. a NEW run (next turn) can read the same doc again   OK")


def test_3_handle_and_filename_are_the_same_doc():
    from doc_tools import _resolve
    c = ctx("run-C")
    handle = _resolve("policy.txt", c).doc_id               # e.g. doc-1a2b3c
    fake = lambda doc_id, run_context: "Document"
    call("read_document", fake, {"doc_id": handle, "run_context": c}, c)
    out = call("read_document", fake, {"doc_id": "policy.txt", "run_context": c}, c)
    assert out == g.ALREADY_READ
    print(f"3. {handle} and 'policy.txt' count as one document    OK")


def test_4_crash_becomes_error_message():
    def broken(**_):
        raise ValueError("database down")
    c = ctx("run-D")
    out = call("get_weather", broken, {"city": "Dubai"}, c)
    assert "failed" in out and "database down" in out, out
    assert audit_lines()[-1]["error"] == "ValueError: database down"
    print("4. a crashing tool returns an error text (no crash)    OK")


def test_5_output_cap():
    c = ctx("run-E")
    out = call("list_documents", lambda **_: "x" * 5000, {}, c)
    assert len(out) < 1100 and "cut at 1,000" in out
    print("5. huge tool output is cut at MAX_TOOL_RESULT_CHARS    OK")


def test_6_audit_fields():
    e = audit_lines()[-1]
    for key in ("ts", "user_id", "session_id", "run_id", "tool", "args", "result_chars", "ms"):
        assert key in e, key
    assert "run_context" not in e["args"], "run_context must not be logged"
    print("6. audit.log has who/what/size/time, no run_context    OK")


def test_7_activity_note_for_next_turn():
    from agno.tools.function import UserFeedbackQuestion, UserInputField
    run_output = FakeRunOutput(tools=[
        FakeTool("read_document", {"doc_id": "doc-1"}, "Document doc-1: policy.txt (1 section(s))\n\nSECRET BODY TEXT"),
        FakeTool("read_document", {"doc_id": "doc-1"}, g.ALREADY_READ),
        FakeTool("find_in_document", {"doc_id": "doc-1", "query": "remote"}, '1 match(es) for "remote"\n- [Document] ...'),
        FakeTool("ask_user", user_feedback_schema=[
            UserFeedbackQuestion(question="Which format?", selected_options=["Paragraph"])]),
        FakeTool("get_user_input", user_input_schema=[
            UserInputField(name="topic", field_type=str, value="budget review")]),
        FakeTool("save_note", {"title": "Fees"}, confirmed=False, confirmation_note="wrong title"),
    ])
    c = ctx("run-F", state={})
    g.remember_tool_activity(run_output, c)
    lines = c.session_state["previous_turn_tool_activity"]
    text = "\n".join(lines)
    print("7. activity note saved for the next turn:")
    print("   " + text.replace("\n", "\n   "))
    assert "user chose: Which format? -> Paragraph" in text
    assert "user answered: topic = " in text and "budget review" in text
    assert "user REJECTED save_note" in text and "wrong title" in text  # (fenced since step 6)
    assert "SECRET BODY TEXT" not in text, "document content must NOT be kept"
    assert sum("read_document" in l for l in lines) == 1, "blocked re-read is not listed"
    print("   answers + rejection kept, document text dropped      OK")


def test_8_activity_is_replaced_not_appended():
    c = ctx("run-G", state={"previous_turn_tool_activity": ["- old line"]})
    g.remember_tool_activity(FakeRunOutput(tools=[FakeTool("calculate", {"expression": "2+2"}, "4")]), c)
    assert c.session_state["previous_turn_tool_activity"] == ['- calculate({"expression": "2+2"}) -> 4']
    print("8. only the PREVIOUS turn is kept (replaced each turn) OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\nAll step-5 tests passed. (temp files in {TMP})")
