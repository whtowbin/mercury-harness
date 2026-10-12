"""Settings.check_after_edit: after a turn that edits and does not check, the harness runs the
step's tests once and hands the result back with the edits (DESIGN-LOG §90 design C, §99).

A scripted model drives the real tool loop; the edits land in real files and pytest really
runs, since the result text and the trace event are what is under test.
"""

from __future__ import annotations

import json

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.runner import AFTER_EDIT_NOTE, GateChain, _note_ran, run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


def call(name: str, **args) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Scripted:
    """Replies with one turn of tool calls per chat() call, recording what it was sent."""

    def __init__(self, turns: list[list[dict]]):
        self.turns = list(turns)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return Reply(content="", tool_calls=self.turns.pop(0), usage=Usage(10, 5, 0.0, 0.1))


FINISH = call("finish_step", summary="done", verified="true")


def _run(tmp_path, turns, flag=True):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    client = Scripted(turns)
    run_tool_loop(
        client=client,
        model="fake",
        phase="execute:fix-add",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=trace,
        max_iters=len(turns),
        check_after_edit=flag,
    )
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    return client, events


FIX = call("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b")


def test_an_edit_turn_gets_the_test_result_without_another_call(tmp_path):
    client, events = _run(tmp_path, [[FIX], [FINISH]])
    seen = client.sent[1][-1]["content"]  # the edit's tool result, as the next call saw it
    assert AFTER_EDIT_NOTE.strip() in seen and "PASS" in seen
    assert [e["kind"] for e in events].count("after_edit_check") == 1
    runs = [e for e in events if e["kind"] == "run_tests"]
    assert len(runs) == 1 and runs[0]["source"] == "auto"


def test_several_edits_in_one_turn_get_one_run(tmp_path):
    _, events = _run(tmp_path, [[FIX, call("write_file", path="notes.md", content="x")], [FINISH]])
    assert len([e for e in events if e["kind"] == "run_tests"]) == 1


def test_a_turn_that_already_checks_its_edit_is_left_alone(tmp_path):
    _, events = _run(tmp_path, [[FIX, call("run_tests")], [FINISH]])
    runs = [e for e in events if e["kind"] == "run_tests"]
    assert len(runs) == 1 and "source" not in runs[0]
    assert not [e for e in events if e["kind"] == "after_edit_check"]


def test_a_failed_edit_changes_nothing_so_nothing_is_run(tmp_path):
    bad = call("edit_file", path="calc.py", old_string="not in the file", new_string="x")
    _, events = _run(tmp_path, [[bad], [FINISH]])
    assert not [e for e in events if e["kind"] in ("after_edit_check", "run_tests")]


def test_a_failing_result_is_handed_back_too(tmp_path):
    wrong = call("edit_file", path="calc.py", old_string="return a - b", new_string="return a * b")
    client, _ = _run(tmp_path, [[wrong], [FINISH]])
    assert "FAIL" in client.sent[1][-1]["content"]


def test_off_by_default_nothing_changes(tmp_path):
    client, events = _run(tmp_path, [[FIX], [FINISH]], flag=False)
    assert AFTER_EDIT_NOTE.strip() not in client.sent[1][-1]["content"]
    assert not [e for e in events if e["kind"] in ("after_edit_check", "run_tests")]


def test_only_gates_that_track_a_run_hear_about_it():
    class Tracks:
        def __init__(self):
            self.runs = 0

        def ran(self):
            self.runs += 1

    class TddLike:  # no ran(): the model's own test before its first edit stays required
        pass

    tracks = Tracks()
    _note_ran(GateChain(TddLike(), tracks))
    _note_ran(tracks)
    _note_ran(None)
    assert tracks.runs == 2


def test_no_collected_tests_hands_nothing_back(tmp_path):
    """Before a step has written its tests the harness's run finds none; that is noise
    (§99.1 smoke). The check still happens and is traced."""
    client, events = _run_without_tests(tmp_path)
    assert AFTER_EDIT_NOTE.strip() not in client.sent[1][-1]["content"]
    assert [e["kind"] for e in events].count("after_edit_check") == 1


def _run_without_tests(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    client = Scripted([[FIX], [FINISH]])
    run_tool_loop(
        client=client,
        model="fake",
        phase="execute:fix-add",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=trace,
        max_iters=2,
        check_after_edit=True,
    )
    return client, [json.loads(line) for line in trace.path.read_text().splitlines()]


def _run_typed(tmp_path, new_string: str, baseline=frozenset()):
    (tmp_path / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    (tmp_path / "use.py").write_text("from calc import add\n\nTOTAL = add(1, 2)\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    ctx.type_baseline = baseline
    edit = call("edit_file", path="use.py", old_string="add(1, 2)", new_string=new_string)
    client = Scripted([[edit], [FINISH]])
    run_tool_loop(
        client=client,
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=trace,
        max_iters=2,
        check_after_edit=True,
        types_after_edit=True,
    )
    return client.sent[1][-1]["content"]


def test_a_type_error_the_edit_introduced_comes_back_with_it(tmp_path):
    seen = _run_typed(tmp_path, 'add(1, "2")')
    assert "ty: 1 type error(s) this run introduced" in seen and "use.py" in seen


def test_a_clean_edit_adds_nothing_about_types(tmp_path):
    assert "ty:" not in _run_typed(tmp_path, "add(1, 3)")


def test_a_new_project_with_no_baseline_still_gets_its_type_errors(tmp_path):
    """No Python at the start means no baseline (None); every error is then new, as in the
    final check. Semver, where new type errors are common, starts this way (§105 smoke)."""
    seen = _run_typed(tmp_path, 'add(1, "2")', baseline=None)
    assert "ty: 1 type error(s) this run introduced" in seen
