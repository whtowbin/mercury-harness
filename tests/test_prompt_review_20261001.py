"""The prompt and nudge review of 2026-10-01 (DESIGN-LOG §108).

A scripted model drives the real tool loop; edits land in real files, and pytest and ty really
run, because the text handed back to the model is what is under test.
"""

from __future__ import annotations

import dataclasses
import json

from agent.approval import Approver
from agent.config import Settings
from agent.interact import Interactor
from agent.llm import Reply, Usage
from agent.runner import (
    AFTER_EDIT_FINDING_NOTE,
    AFTER_EDIT_NOTE,
    AFTER_EDIT_TYPES_NOTE,
    _stall_nudge,
    run_tool_loop,
)
from agent.tools import ToolContext, execution_registry, planning_registry
from agent.trace import Trace
from agent.workspace import Workspace


def call(name: str, **args) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Scripted:
    """One reply per chat() call: a list of tool calls, or a string of text with none."""

    def __init__(self, turns: list):
        self.turns = list(turns)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            return Reply(content=turn, tool_calls=[], usage=Usage(10, 5, 0.0, 0.1))
        return Reply(content="", tool_calls=turn, usage=Usage(10, 5, 0.0, 0.1))


FINISH = call("finish_step", summary="done", verified="true", expected_result="it passes")


def _ctx(tmp_path) -> ToolContext:
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _loop(ctx, turns, registry=None, phase="execute:s", **kwargs):
    client = Scripted(turns)
    run_tool_loop(
        client=client,
        model="fake",
        phase=phase,
        system="s",
        user="u",
        registry=registry or execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=len(turns),
        **kwargs,
    )
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    return client, events


# --- Finding 4: new type errors survive a step that has no tests yet --------------------------


def _typed_without_tests(tmp_path, new_string: str):
    (tmp_path / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    (tmp_path / "use.py").write_text("from calc import add\n\nTOTAL = add(1, 2)\n")
    ctx = _ctx(tmp_path)
    ctx.type_baseline = frozenset()
    edit = call("edit_file", path="use.py", old_string="add(1, 2)", new_string=new_string)
    return _loop(ctx, [[edit], [FINISH]], check_after_edit=True, types_after_edit=True)


def test_a_step_with_no_tests_still_hears_about_the_type_error_it_introduced(tmp_path):
    """Before §108 the "no tests were collected" early return dropped the ty text with it:
    11 of 95 such checks carried type errors the model never saw."""
    client, events = _typed_without_tests(tmp_path, 'add(1, "2")')
    seen = client.sent[1][-1]["content"]
    assert AFTER_EDIT_TYPES_NOTE.strip() in seen
    assert "ty: 1 type error(s) this run introduced" in seen and "use.py" in seen
    assert "no tests were collected" not in seen  # the test run itself is still noise here
    check = next(e for e in events if e["kind"] == "after_edit_check")
    assert check["no_tests"] is True and check["delivered"] is True


def test_a_clean_step_with_no_tests_still_hands_nothing_back(tmp_path):
    client, events = _typed_without_tests(tmp_path, "add(1, 3)")
    assert "[harness] After this turn's edits" not in client.sent[1][-1]["content"]
    check = next(e for e in events if e["kind"] == "after_edit_check")
    assert check["delivered"] is False


def test_a_fix_loop_check_is_named_as_the_findings_own(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    ctx = _ctx(tmp_path)
    fix = call("edit_file", path="calc.py", old_string="a - b", new_string="a + b")
    client, _ = _loop(
        ctx,
        [[fix], [FINISH]],
        phase="fix:1",
        check_after_edit=True,
        after_edit_check=lambda c: "PASS — the finding's check",
    )
    seen = client.sent[1][-1]["content"]
    assert AFTER_EDIT_FINDING_NOTE.strip() in seen and AFTER_EDIT_NOTE.strip() not in seen


# --- Finding 2: a planner that writes its plan as text is told to submit it -------------------


def test_the_stall_nudge_names_the_loops_terminal_tool():
    assert "call submit_plan with it now" in _stall_nudge(planning_registry())["content"]
    assert "call finish_step with it now" in _stall_nudge(execution_registry())["content"]


def test_a_text_reply_in_the_loop_gets_the_named_nudge(tmp_path):
    client, _ = _loop(_ctx(tmp_path), ["Here is what I would do: edit calc.py.", [FINISH]])
    nudge = client.sent[1][-1]
    assert nudge["role"] == "user" and "finish_step" in nudge["content"]


def test_a_second_unanswerable_question_is_told_none_will_come():
    interactor = Interactor(enabled=False)
    first = interactor.ask("Should I proceed?")
    second = interactor.ask("Proceeding to create files.")
    assert "Proceed on your own best judgment" in first
    assert "none will come" in second and "Do not ask again" in second


def test_no_tty_repeats_get_the_same_answer(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    interactor = Interactor()
    interactor.ask("one?")
    assert "none will come" in interactor.ask("two?")


def test_an_answered_question_does_not_count_as_unanswerable():
    interactor = Interactor(ask=lambda q, o: "yes")
    assert interactor.ask("one?") == "yes" and interactor.ask("two?") == "yes"
    assert interactor.unanswered == 0


# --- Finding 5: no batching nudge for an edit that follows the harness's check ----------------


def _two_single_edits(tmp_path, check: bool):
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a\n"
    )
    (tmp_path / "test_calc.py").write_text(
        "from calc import add, sub\n\ndef test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_sub():\n    assert sub(3, 2) == 1\n"
    )
    first = call("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b")
    second = call("edit_file", path="calc.py", old_string="return a\n", new_string="return a - b\n")
    return _loop(_ctx(tmp_path), [[first], [second], [FINISH]], check_after_edit=check)


def test_an_edit_after_a_failing_check_gets_no_batching_nudge(tmp_path):
    """41 of 71 edit batching nudges fired right after the harness's check said FAIL: the
    edit-test-then-fix order the executor is told to follow."""
    client, events = _two_single_edits(tmp_path, check=True)
    assert "FAIL" in client.sent[1][-1]["content"]  # the first edit left test_sub failing
    assert not [e for e in events if e["kind"] == "batching_nudge"]


def test_two_single_edits_with_no_check_between_still_get_the_nudge(tmp_path):
    _, events = _two_single_edits(tmp_path, check=False)
    assert [e["tool"] for e in events if e["kind"] == "batching_nudge"] == ["edit_file"]


# --- Finding 3: the step message says the harness runs the verify ----------------------------


def test_the_verify_line_says_who_runs_it():
    from agent.phases.execute import _verify_line

    assert _verify_line("uvx --quiet ruff check a.py", by_harness=False) == (
        "Verify with: uvx --quiet ruff check a.py\n"
    )
    by_harness = _verify_line("uvx --quiet ruff check a.py", by_harness=True)
    assert by_harness.startswith("After finish_step the harness runs `uvx --quiet ruff check")
    assert _verify_line("", by_harness=True) == ""


# --- Finding 1: the reviewer arm --------------------------------------------------------------


def _settings(**flags) -> Settings:
    return dataclasses.replace(Settings(workspace=".", goal="g"), **flags)


def test_the_reviewer_kit_drops_run_tests_only_with_the_switch():
    from agent.phases.review import _review_kit

    assert "run_tests" in _review_kit(_settings(review_presupplied_checks=False)).names()
    on = _review_kit(_settings(review_presupplied_checks=True)).names()
    assert "run_tests" not in on and "check_code" in on and "submit_findings" in on


def _review(tmp_path, monkeypatch, settings, source: str):
    from agent.phases import review as review_module
    from agent.phases.plan import Plan

    (tmp_path / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    (tmp_path / "use.py").write_text(source)
    ctx = _ctx(tmp_path)
    ctx.type_baseline = frozenset()
    ctx.touched.append("use.py")
    sent: dict = {}
    monkeypatch.setattr(review_module, "_suite_result", lambda c, t: "suite output\n\n")
    monkeypatch.setattr(
        review_module,
        "run_tool_loop",
        lambda **kw: (sent.update(kw), ({"verdict": "pass", "findings": []}, []))[1],
    )
    review_module.review(object(), settings, Plan(goal="g", draft=[], steps=[]), ctx, ctx.trace)
    return sent


def test_the_reviewer_is_handed_the_type_check_and_not_asked_to_run_one(tmp_path, monkeypatch):
    sent = _review(
        tmp_path,
        monkeypatch,
        _settings(review_presupplied_checks=True),
        'from calc import add\n\nTOTAL = add(1, "2")\n',
    )
    assert "ty: 1 type error(s) this run introduced" in sent["user"]
    assert sent["user"].endswith("read the code, then call submit_findings.")
    assert "run_tests" not in sent["registry"].names()
    assert "run_tests runs pytest" not in sent["system"]


def test_a_clean_run_tells_the_reviewer_so(tmp_path, monkeypatch):
    sent = _review(
        tmp_path,
        monkeypatch,
        _settings(review_presupplied_checks=True),
        "from calc import add\n\nTOTAL = add(1, 2)\n",
    )
    assert "ty: no type errors this run introduced." in sent["user"]


def test_with_the_switch_off_the_review_is_as_before(tmp_path, monkeypatch):
    from agent.prompts import REVIEWER_SYSTEM

    sent = _review(
        tmp_path,
        monkeypatch,
        _settings(review_presupplied_checks=False),
        'from calc import add\n\nTOTAL = add(1, "2")\n',
    )
    assert sent["system"] == REVIEWER_SYSTEM
    assert "type checker has already been run" not in sent["user"]
    assert sent["user"].endswith(
        "use check_code(types=true) to confirm what you suspect, then call submit_findings."
    )
    assert "run_tests" in sent["registry"].names()


def test_the_switch_is_recorded_in_run_start():
    from agent.loop import _recorded_settings

    recorded = _recorded_settings(_settings(review_presupplied_checks=True))
    assert recorded["review_presupplied_checks"] is True


def test_the_reviewer_checks_are_the_default_with_a_control():
    """On since §108.2, measured with a Mercury reviewer: GLM-reviewed configs keep it off."""
    from bench.configs import CONFIGS

    assert Settings(workspace=".", goal="g").review_presupplied_checks is True
    assert CONFIGS["mercury-only"].review_presupplied_checks is True
    assert CONFIGS["mercury-no-review-checks"].review_presupplied_checks is False
    for name in ("glm-only", "glm-plan-mercury", "glm-plan-mercury-low"):
        assert CONFIGS[name].review_presupplied_checks is False, name


def test_the_s108_label_is_mercury_only_under_another_name():
    """Stage 2 compares mercury-only (the commit before) with mercury-s108 (this commit), so the
    label must not differ in anything but its name (and being retired, so it is never swept)."""
    from bench.configs import CONFIGS

    a, b = CONFIGS["mercury-only"], CONFIGS["mercury-s108"]
    differ = {f.name for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)}
    assert differ == {"name", "description", "retired"}


def test_compare_reports_the_reviewer_arms_metrics():
    """§108's reviewer rule is on review calls a run, guarded by findings a run."""
    from bench.report import render_compare

    rows = [
        {"config": "a", "task": "t", "accuracy": 1.0, "diag_review_calls": 4, "findings": 1},
        {"config": "b", "task": "t", "accuracy": 1.0, "diag_review_calls": 1, "findings": 1},
    ] * 3
    text = render_compare(rows, "a", "b")
    assert "review calls" in text and "review findings" in text
