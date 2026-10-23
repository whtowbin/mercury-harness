"""A step that claims `verified: true` must say what it expected to see.

Modeled on a mitigation Cognition (Devin) describe for self-report
unreliability: state a concrete expectation before the result is visible to
compare against, rather than grading a bare boolean after the fact. This
project's `_verify_step` already trusts a real verify command's exit code
over the model's own report; `expected_result` is the complementary half for
the case a bare `verified: true` still reaches unchecked -- a step with no
verify command at all -- and, for a step that does have one, a coarse
pass/fail cross-check logged for the trace rather than used to gate anything.
"""

from __future__ import annotations

from agent.phases.execute import _apply_result, _expected_pass, _verify_step
from agent.phases.plan import Step


class _Ctx:
    def __init__(self, root, touched=()):
        self.touched = list(touched)
        self.workspace = type("W", (), {"root": root})()


class _Trace:
    def __init__(self):
        self.events = []
        self.said = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))

    def say(self, message):
        self.said.append(message)


class _Settings:
    verify_steps = True


def _step(**kw) -> Step:
    return Step(id="s", title="t", intent="i", acceptance="a", **kw)


# -- _apply_result: verified=true requires an expectation --------------------


def test_verified_true_with_no_expectation_is_downgraded_and_retried():
    step = _step()
    problem = _apply_result(step, {"summary": "did it", "verified": True})
    assert step.verified is False
    assert "without saying what you expected" in problem


def test_verified_true_with_an_expectation_is_accepted():
    step = _step()
    problem = _apply_result(
        step, {"summary": "did it", "verified": True, "expected_result": "4 passed"}
    )
    assert step.verified is True
    assert problem == ""
    assert step.expected_result == "4 passed"


def test_verified_false_needs_no_expectation():
    step = _step()
    problem = _apply_result(step, {"summary": "partial", "verified": False})
    assert step.verified is False
    assert problem == ""


def test_a_salvaged_incomplete_result_is_unaffected_by_this_check():
    """An `incomplete` result is already reported as unfinished by a
    different path; it must not also be blamed for a missing expectation."""
    step = _step()
    problem = _apply_result(
        step, {"summary": "ran out of time", "verified": True, "incomplete": True}
    )
    assert "ran out of time" in problem
    assert "without saying what you expected" not in problem


# -- _expected_pass: the coarse pass/fail read of free text -------------------


def test_expected_pass_reads_empty_text_as_unscored():
    assert _expected_pass("") is None


def test_expected_pass_reads_an_ordinary_claim_as_a_pass():
    assert _expected_pass("4 passed, prints total 42.0") is True


def test_expected_pass_reads_a_failure_word_as_a_failure():
    assert _expected_pass("still fails on the empty-list case") is False


# -- _verify_step: the mismatch is logged, not used to gate ------------------


def test_a_mismatch_is_logged_when_expectation_and_outcome_disagree(tmp_path, monkeypatch):
    from agent.phases import execute

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (True, "4 passed"))
    ctx = _Ctx(tmp_path)
    step = _step(verify="uv run pytest -q", expected_result="this will still fail")
    problem = _verify_step(ctx, (trace := _Trace()), step, _Settings())
    assert problem == ""
    kind, fields = trace.events[-1]
    assert kind == "step_verify"
    assert fields["expectation_mismatch"] is True


def test_no_mismatch_flagged_when_they_agree(tmp_path, monkeypatch):
    from agent.phases import execute

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (True, "4 passed"))
    ctx = _Ctx(tmp_path)
    step = _step(verify="uv run pytest -q", expected_result="4 passed")
    _verify_step(ctx, (trace := _Trace()), step, _Settings())
    _kind, fields = trace.events[-1]
    assert fields["expectation_mismatch"] is False


def test_an_unstated_expectation_is_never_flagged_as_a_mismatch(tmp_path, monkeypatch):
    from agent.phases import execute

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (False, "AssertionError"))
    ctx = _Ctx(tmp_path)
    step = _step(verify="uv run pytest -q", expected_result="")
    _verify_step(ctx, (trace := _Trace()), step, _Settings())
    _kind, fields = trace.events[-1]
    assert fields["expectation_mismatch"] is False


def test_a_failed_verify_that_contradicts_the_expectation_says_so(tmp_path, monkeypatch):
    from agent.phases import execute

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (False, "AssertionError"))
    ctx = _Ctx(tmp_path)
    step = _step(verify="uv run pytest -q", expected_result="4 passed")
    problem = _verify_step(ctx, _Trace(), step, _Settings())
    assert "verify command failed" in problem
    assert "you expected" in problem


# -- §131.9: Settings.expected_result_gate off: record, do not retry ----------


def test_with_the_gate_off_a_missing_expectation_is_accepted():
    step = _step()
    problem = _apply_result(step, {"summary": "did it", "verified": True}, gate_expected=False)
    assert step.verified is True
    assert problem == ""


def _run_step(tmp_path, monkeypatch, gate: bool, verify: str):
    """execute_step with only the model call scripted: a real workspace, trace,
    approver and Settings, and the step's verify command actually run."""
    import json

    from agent.approval import Approver
    from agent.config import Settings
    from agent.phases import execute
    from agent.phases.plan import Plan
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    attempts = []

    def scripted(*args, **kwargs):
        attempts.append(kwargs.get("attempt_no"))
        return {"summary": "did it", "verified": True}, []

    monkeypatch.setattr(execute, "_attempt", scripted)
    ws = tmp_path / "ws"
    ws.mkdir()
    trace = Trace(str(tmp_path / "traces"), run_id="t", echo=False)
    ctx = ToolContext(workspace=Workspace(ws), approver=Approver(auto_approve=True), trace=trace)
    settings = Settings(workspace=str(ws), expected_result_gate=gate, type_check_steps=False)
    step = _step(verify=verify)
    plan = Plan(goal="g", draft=[], steps=[step])
    execute.execute_step(None, settings, plan, step, 0, ctx, trace)
    kinds = [json.loads(line)["kind"] for line in trace.path.read_text().splitlines()]
    return step, attempts, kinds


def test_gate_off_runs_the_verify_once_and_records_the_gap(tmp_path, monkeypatch):
    step, attempts, kinds = _run_step(tmp_path, monkeypatch, gate=False, verify="true")
    assert attempts == [1]
    assert "expected_result_missing" in kinds
    assert "step_retry" not in kinds
    assert step.verify_passed is True


def test_gate_off_still_fails_a_step_whose_verify_fails(tmp_path, monkeypatch):
    step, attempts, kinds = _run_step(tmp_path, monkeypatch, gate=False, verify="false")
    assert len(attempts) == 2 and "step_retry" in kinds
    assert step.verify_passed is False


def test_gate_on_retries_without_running_the_verify(tmp_path, monkeypatch):
    """Today's default, pinned so the A/B compares what it says it does."""
    step, attempts, kinds = _run_step(tmp_path, monkeypatch, gate=True, verify="true")
    assert len(attempts) == 2 and "step_retry" in kinds
    assert step.verify_passed is not True
