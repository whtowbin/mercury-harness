"""Regression tests for the code review of 2026-09-30 (DESIGN-LOG §91).
One test per fixed finding, numbered as there."""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.llm import BudgetExceeded, Reply, Usage
from agent.numericcheck import NumericCheckGate
from agent.phases import schedule
from agent.phases.execute import workspace_fingerprint
from agent.runner import _touched_fingerprint, run_tool_loop
from agent.steptail import _Call, errors_section
from agent.tools import ToolContext, checks, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


def call(name, **args):
    return {"id": "c", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(text, *calls):
    return Reply(content=text, tool_calls=list(calls), usage=Usage(10, 5, 0.0, 0.1))


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, model, messages, tools=None, tool_choice=None, **kwargs):
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "solution.py").write_text("x = 1\n")
    (tmp_path / "test_solution.py").write_text(
        "from solution import x\n\ndef test_x():\n    assert x == 2\n"
    )
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _loop(ctx, replies, **kwargs):
    run_tool_loop(
        client=Scripted(replies),
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=4,
        **kwargs,
    )
    ctx.trace._fh.flush()
    return [json.loads(line)["kind"] for line in ctx.trace.path.read_text().splitlines()]


FINISH = reply("done", call("finish_step", summary="done", verified="false"))


# -- 1. tdd_tests and a step gate are chained, not one replacing the other -------


def test_tdd_tests_no_longer_drop_the_step_gate(ctx):
    gate = NumericCheckGate()
    gate.observe(call("write_file", path="solution.py", content="def f(): ..."))
    kinds = _loop(ctx, [FINISH], tdd_tests=["test_x"], gate=gate)
    assert "numeric_check_held" in kinds


def test_the_red_first_gate_still_holds_with_a_step_gate_beside_it(ctx):
    edit = reply(
        "edit", call("edit_file", path="solution.py", old_string="x = 1", new_string="x = 2")
    )
    kinds = _loop(ctx, [edit, FINISH], tdd_tests=["test_x"], gate=NumericCheckGate())
    assert "tdd_gate_held" in kinds
    assert (ctx.workspace.root / "solution.py").read_text() == "x = 1\n"


# -- 5. a call that failed is not observed by the gate ---------------------------


def test_a_failed_edit_is_not_an_unchecked_edit(ctx):
    missing = reply(
        "edit", call("edit_file", path="solution.py", old_string="nope", new_string="y")
    )
    kinds = _loop(ctx, [missing, FINISH], gate=NumericCheckGate())
    assert "numeric_check_held" not in kinds


def test_a_landed_edit_still_is(ctx):
    edit = reply(
        "edit", call("edit_file", path="solution.py", old_string="x = 1", new_string="x = 2")
    )
    kinds = _loop(ctx, [edit, FINISH], gate=NumericCheckGate())
    assert "numeric_check_held" in kinds


# -- 6. a harness-run check clears the numeric gate -------------------------------


def test_ran_clears_the_mark_like_a_run_tool_does():
    gate = NumericCheckGate()
    gate.observe(call("write_file", path="solution.py", content="def f(): ..."))
    gate.ran()
    assert gate.hold(call("finish_step", summary="done")) is None


# -- 2. the tail's streak stops at a passing run ---------------------------------


def test_one_failure_after_passes_is_not_a_streak():
    calls = [
        _Call("run_tests", {}, "PASS"),
        _Call("run_tests", {}, "PASS"),
        _Call("run_tests", {}, "FAIL — 2 failed\n\noutput with no FAILED lines"),
    ]
    assert "runs in a row" not in errors_section(calls)


def test_repeated_identical_failures_are_still_a_streak():
    failing = "FAIL — 1 failed\nFAILED test_a.py::test_x - AssertionError"
    calls = [_Call("run_tests", {}, failing)] * 3
    assert "failed 3 runs in a row" in errors_section(calls)


# -- 3. tests_importing prunes ignored directories --------------------------------


def test_tests_importing_skips_ignored_directories(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text("from pkg.mod import x\n")
    for ignored in (".venv", "node_modules"):
        (tmp_path / ignored / "sub").mkdir(parents=True)
        (tmp_path / ignored / "sub" / "test_mod.py").write_text("from pkg.mod import x\n")
    assert checks.tests_importing(tmp_path, ["pkg/mod.py"]) == ["tests/test_mod.py"]


# -- 7. a KeyboardInterrupt in a parallel batch is not lost -----------------------


class _Step:
    def __init__(self, id: str):
        self.id = id
        self.title = id
        self.written: list[str] = []
        self.summary = ""
        self.verified = False
        self.verify_passed = None
        self.verify = ""
        self.attempts = 1


def _concurrent(tmp_path, outcomes):
    steps = [_Step(f"s{i}") for i in range(len(outcomes))]

    def execute_step(client, settings, plan, step, index, ctx, trace):
        exc = outcomes[index]
        if exc is not None:
            raise exc

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    return schedule._run_concurrent(
        steps,
        0,
        len(steps),
        None,
        None,
        None,
        {s.id: i for i, s in enumerate(steps)},
        ctx,
        ctx.trace,
        execute_step,
    )


def test_an_interrupt_behind_an_ordinary_error_is_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, "_checkpoint", lambda *a, **k: None)
    with pytest.raises(KeyboardInterrupt):
        _concurrent(tmp_path, [RuntimeError("boom"), KeyboardInterrupt()])


def test_a_budget_stop_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, "_checkpoint", lambda *a, **k: None)
    with pytest.raises(BudgetExceeded):
        _concurrent(tmp_path, [RuntimeError("boom"), BudgetExceeded("cap")])


# -- 10. one fingerprint recipe ---------------------------------------------------


def test_the_two_fingerprint_helpers_agree(ctx):
    ctx.mark_touched("solution.py")
    ctx.mark_touched("missing.py")
    assert workspace_fingerprint(ctx) == _touched_fingerprint(ctx)
    (ctx.workspace.root / "solution.py").write_text("x = 3\n")
    assert workspace_fingerprint(ctx) == _touched_fingerprint(ctx)
