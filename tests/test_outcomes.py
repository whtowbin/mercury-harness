"""Whether a run is honest about having failed its own checks.

Before these, a run in which every step failed its verify command produced
exactly the same summary — and the same green "done." — as a clean one.
"""

from __future__ import annotations

from agent.loop import _review_rounds, format_totals, model_tok_per_s, step_outcomes
from agent.phases.finalcheck import FinalCheck
from agent.phases.plan import Plan, Step


def step(id_="s1", verify_passed=None, verified=False, summary="did it") -> Step:
    s = Step(id=id_, title="t", intent="i", acceptance="a")
    s.verify_passed = verify_passed
    s.verified = verified
    s.summary = summary
    return s


def plan_with(*steps: Step) -> Plan:
    return Plan(goal="g", draft=[], steps=list(steps))


# -- step outcomes --------------------------------------------------------


def test_all_steps_passing_reports_all_passed():
    outcomes = step_outcomes(plan_with(step(verify_passed=True), step("s2", verify_passed=True)))
    assert outcomes["all_passed"]
    assert outcomes["failed"] == []
    assert outcomes["total"] == 2


def test_a_failed_verify_is_reported_and_fails_the_run():
    outcomes = step_outcomes(
        plan_with(step("s1", verify_passed=True), step("s2", verify_passed=False))
    )
    assert outcomes["failed"] == ["s2"]
    assert not outcomes["all_passed"]


def test_an_abandoned_step_is_reported_and_fails_the_run():
    outcomes = step_outcomes(plan_with(step("s1", summary="abandoned: too many errors")))
    assert outcomes["abandoned"] == ["s1"]
    assert not outcomes["all_passed"]


def test_a_step_with_no_verify_command_counts_as_unverified_not_failed():
    """Unverified is not the same as failed: plenty of steps have nothing
    meaningful to run. It is surfaced, but it does not fail the run."""
    outcomes = step_outcomes(plan_with(step("s1", verify_passed=None, verified=False)))
    assert outcomes["unverified"] == ["s1"]
    assert outcomes["all_passed"]


def test_a_step_the_executor_verified_itself_is_not_unverified():
    outcomes = step_outcomes(plan_with(step("s1", verified=True)))
    assert outcomes["unverified"] == []


def test_no_plan_is_vacuously_passing():
    outcomes = step_outcomes(None)
    assert outcomes["all_passed"]
    assert outcomes["total"] == 0


# -- review rounds accumulate rather than overwrite -----------------------


class _Recorder:
    """Stands in for review()/apply_fixes() so rounds can be scripted."""

    def __init__(self, rounds: list[list[dict]]):
        self.rounds = rounds
        self.fix_calls: list[list[dict]] = []
        self.reviews = 0

    def review(self, client, settings, plan, ctx, trace, **_kwargs):
        findings = self.rounds[self.reviews] if self.reviews < len(self.rounds) else []
        self.reviews += 1
        return [dict(f) for f in findings]

    def apply_fixes(self, client, settings, findings, ctx, trace, **_kwargs):
        self.fix_calls.append(findings)


def _run_rounds(monkeypatch, tmp_path, rounds):
    """Drive _review_rounds with scripted review results.

    Real Settings/ToolContext/Trace rather than stubs: the stubs these
    replaced did not typecheck against the signature they stood in for.
    """
    import agent.loop as loop_module
    from agent.approval import Approver
    from agent.config import Settings
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(workspace=str(workspace), goal="g")
    settings.budget.max_review_rounds = 2

    recorder = _Recorder(rounds)
    monkeypatch.setattr(loop_module, "review", recorder.review)
    monkeypatch.setattr(loop_module, "apply_fixes", recorder.apply_fixes)
    findings = _review_rounds(None, settings, plan_with(step()), ctx, ctx.trace)
    return findings, recorder


def test_findings_from_an_earlier_round_survive_a_clean_later_round(monkeypatch, tmp_path):
    """The bug this replaced: round two coming back clean replaced round one's
    findings with [], so the summary reported that nothing had been wrong."""
    blocker = {"severity": "blocker", "issue": "boom"}
    findings, recorder = _run_rounds(monkeypatch, tmp_path, [[blocker], []])
    assert len(findings) == 1
    assert findings[0]["issue"] == "boom"
    assert recorder.reviews == 2


def test_each_finding_records_the_round_it_came_from(monkeypatch, tmp_path):
    findings, _ = _run_rounds(
        monkeypatch,
        tmp_path,
        [[{"severity": "blocker", "issue": "a"}], [{"severity": "minor", "issue": "b"}]],
    )
    assert [f["round"] for f in findings] == [1, 2]


def test_a_clean_first_round_stops_immediately(monkeypatch, tmp_path):
    findings, recorder = _run_rounds(monkeypatch, tmp_path, [[], [{"severity": "blocker"}]])
    assert findings == []
    assert recorder.reviews == 1
    assert recorder.fix_calls == []


def test_minor_findings_do_not_trigger_a_fix_round(monkeypatch, tmp_path):
    _, recorder = _run_rounds(monkeypatch, tmp_path, [[{"severity": "minor", "issue": "nit"}]])
    assert recorder.fix_calls == []


def test_blocking_findings_trigger_a_fix_round(monkeypatch, tmp_path):
    _, recorder = _run_rounds(
        monkeypatch, tmp_path, [[{"severity": "blocker", "issue": "boom"}], []]
    )
    assert len(recorder.fix_calls) == 1


# -- one shared totals formatter ------------------------------------------


def test_tok_per_s_is_zero_rather_than_dividing_by_zero():
    assert model_tok_per_s({"completion_tokens": 10, "latency_s": 0}) == 0.0


def test_tok_per_s_divides_completion_tokens_by_latency():
    assert model_tok_per_s({"completion_tokens": 100, "latency_s": 4.0}) == 25.0


def test_format_totals_includes_the_headline_numbers():
    text = format_totals(
        {
            "calls": 3,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.1234,
            "latency_s": 2.0,
            "by_model": {},
        }
    )
    assert "calls 3" in text
    assert "$0.1234" in text


def test_format_totals_lists_each_model():
    text = format_totals(
        {
            "calls": 1,
            "by_model": {
                "m/one": {"calls": 1, "completion_tokens": 100, "latency_s": 2.0, "cost_usd": 0.5}
            },
        }
    )
    assert "m/one" in text
    assert "50 tok/s" in text


# -- the overall ok flag --------------------------------------------------


def test_a_failing_final_check_is_not_ok():
    assert not FinalCheck(tests_ran=True, tests_passed=False).ok


def test_a_clean_final_check_is_ok():
    assert FinalCheck(tests_ran=True, tests_passed=True).ok


# -- the final gate iterates, but not pointlessly -------------------------


class _Gate:
    """Scripts a sequence of FinalCheck results for _final_gate."""

    def __init__(self, checks, effect=None):
        self.checks = list(checks)
        self.fix_rounds = 0
        self.checks_run = 0
        self.effect = effect

    def run_final_check(
        self,
        ctx,
        settings,
        plan,
        trace,
        baseline=None,
        unchanged_since_execute=False,
        type_baseline=None,
    ):
        self.checks_run += 1
        return self.checks.pop(0) if self.checks else self.checks[-1]

    def apply_fixes(self, client, settings, findings, ctx, trace, **_kwargs):
        self.fix_rounds += 1
        if self.effect:
            self.effect(ctx, self.fix_rounds)


def failing(new_failures):
    from agent.phases.finalcheck import FinalCheck

    return FinalCheck(
        tests_ran=True,
        tests_passed=False,
        baseline_known=True,
        new_failures=list(new_failures),
        tests_output="boom",
    )


def passing():
    from agent.phases.finalcheck import FinalCheck

    return FinalCheck(tests_ran=True, tests_passed=True, baseline_known=True)


def _run_gate(monkeypatch, tmp_path, checks, effect=None, **settings_kwargs):
    import agent.loop as loop_module
    from agent.approval import Approver
    from agent.config import Settings
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(workspace=str(workspace), goal="g", final_check=True, **settings_kwargs)

    gate = _Gate(checks, effect)
    monkeypatch.setattr(loop_module, "run_final_check", gate.run_final_check)
    monkeypatch.setattr(loop_module, "apply_fixes", gate.apply_fixes)
    findings = []
    result = loop_module._final_gate(None, settings, plan_with(step()), ctx, ctx.trace, findings)
    return result, gate, findings


def test_a_clean_first_check_runs_no_fix_rounds(monkeypatch, tmp_path):
    result, gate, _ = _run_gate(monkeypatch, tmp_path, [passing()])
    assert result.ok
    assert gate.fix_rounds == 0


def test_a_failure_fixed_on_the_first_round_stops_there(monkeypatch, tmp_path):
    result, gate, _ = _run_gate(monkeypatch, tmp_path, [failing(["t::a"]), passing()])
    assert result.ok
    assert gate.fix_rounds == 1


def test_the_gate_keeps_going_while_it_is_making_progress(monkeypatch, tmp_path):
    """Two failures, one fixed per round: it must not stop after the first."""
    result, gate, _ = _run_gate(
        monkeypatch, tmp_path, [failing(["t::a", "t::b"]), failing(["t::b"]), passing()]
    )
    assert result.ok
    assert gate.fix_rounds == 2


def test_the_gate_stops_early_when_a_round_changes_nothing(monkeypatch, tmp_path):
    """A fix that fixed nothing will not fix anything on an identical retry."""
    result, gate, _ = _run_gate(
        monkeypatch,
        tmp_path,
        [failing(["t::a"]), failing(["t::a"]), failing(["t::a"]), failing(["t::a"])],
    )
    assert not result.ok
    assert gate.fix_rounds == 1, "should not have tried the same thing again"


def _edits_a_file(ctx, round_no):
    """A fix round that really changes a touched file."""
    (ctx.workspace.root / "a.py").write_text(f"x = {round_no}\n")
    if "a.py" not in ctx.touched:
        ctx.touched.append("a.py")


def _distinct_failures(n):
    """Failures whose signature differs every check, so the same-signature stop never fires."""
    return [failing([f"t::{i}"]) for i in range(n)]


def test_an_idle_fix_round_ends_the_gate_when_the_flag_is_on(monkeypatch, tmp_path):
    from agent.loop import MAX_FIX_ROUNDS

    _, gate, _ = _run_gate(
        monkeypatch,
        tmp_path,
        _distinct_failures(MAX_FIX_ROUNDS + 2),
        stop_after_idle_fix_round=True,
    )
    assert gate.fix_rounds == 1, "round 1 changed no file, so round 2 has nothing new to act on"


def test_the_idle_stop_is_recorded_in_the_trace(monkeypatch, tmp_path):
    _run_gate(monkeypatch, tmp_path, _distinct_failures(5), stop_after_idle_fix_round=True)
    trace = next((tmp_path / "t").glob("run-*.jsonl")).read_text()
    assert '"final_fix_idle_stop"' in trace


def test_a_round_that_edits_a_file_is_followed_by_another(monkeypatch, tmp_path):
    from agent.loop import MAX_FIX_ROUNDS

    _, gate, _ = _run_gate(
        monkeypatch,
        tmp_path,
        _distinct_failures(MAX_FIX_ROUNDS + 2),
        effect=_edits_a_file,
        stop_after_idle_fix_round=True,
    )
    assert gate.fix_rounds == MAX_FIX_ROUNDS


def test_only_the_idle_round_stops_it(monkeypatch, tmp_path):
    """Round 1 edits, round 2 does not: the gate stops after round 2, not before."""
    from agent.loop import MAX_FIX_ROUNDS

    def edit_only_round_one(ctx, round_no):
        if round_no == 1:
            _edits_a_file(ctx, round_no)

    _, gate, _ = _run_gate(
        monkeypatch,
        tmp_path,
        _distinct_failures(MAX_FIX_ROUNDS + 2),
        effect=edit_only_round_one,
        stop_after_idle_fix_round=True,
    )
    assert gate.fix_rounds == 2


def test_an_idle_round_changes_nothing_while_the_flag_is_off(monkeypatch, tmp_path):
    from agent.loop import MAX_FIX_ROUNDS

    _, gate, _ = _run_gate(monkeypatch, tmp_path, _distinct_failures(MAX_FIX_ROUNDS + 2))
    assert gate.fix_rounds == MAX_FIX_ROUNDS


def test_a_passing_check_after_an_idle_round_still_passes(monkeypatch, tmp_path):
    result, gate, _ = _run_gate(
        monkeypatch, tmp_path, [failing(["t::a"]), passing()], stop_after_idle_fix_round=True
    )
    assert result.ok
    assert gate.fix_rounds == 1


def test_the_gate_is_bounded_even_while_it_seems_to_progress(monkeypatch, tmp_path):
    from agent.loop import MAX_FIX_ROUNDS

    checks = [failing([f"t::{i}"]) for i in range(MAX_FIX_ROUNDS + 4)]
    _, gate, _ = _run_gate(monkeypatch, tmp_path, checks)
    assert gate.fix_rounds == MAX_FIX_ROUNDS


def test_gate_findings_are_added_to_the_runs_findings(monkeypatch, tmp_path):
    _, _, findings = _run_gate(monkeypatch, tmp_path, [failing(["t::a"]), passing()])
    assert findings
    assert findings[0]["severity"] == "blocker"
    assert str(findings[0]["round"]).startswith("final-check")


def test_the_gate_is_skipped_entirely_when_turned_off(monkeypatch, tmp_path):
    import agent.loop as loop_module
    from agent.approval import Approver
    from agent.config import Settings
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(workspace=str(workspace), goal="g", final_check=False)
    assert loop_module._final_gate(None, settings, plan_with(step()), ctx, ctx.trace, []) is None


def _run_gate_with_ctx(monkeypatch, tmp_path, checks):
    """Like _run_gate, but also returns ctx -- for assertions that need the
    trace file or workspace, not just the gate's return value."""
    import agent.loop as loop_module
    from agent.approval import Approver
    from agent.config import Settings
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(workspace=str(workspace), goal="g", final_check=True)

    gate = _Gate(checks)
    monkeypatch.setattr(loop_module, "run_final_check", gate.run_final_check)
    monkeypatch.setattr(loop_module, "apply_fixes", gate.apply_fixes)
    findings = []
    result = loop_module._final_gate(None, settings, plan_with(step()), ctx, ctx.trace, findings)
    return result, gate, findings, ctx


def _trace_events(ctx):
    import json

    ctx.trace._fh.flush()
    return [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]


def test_a_regressing_round_is_flagged_in_the_trace(monkeypatch, tmp_path):
    """A round whose own fix made the failure count worse must be recorded
    somewhere the harness can see it -- previously the loop just moved on to
    the next round on a changed (not identical) signature, silently. This is
    the traced lives-system bug: 2 failing, one fix round, 5 failing, ran out
    of iterations right after -- never told its own edit was the problem."""
    result, _gate, _findings, ctx = _run_gate_with_ctx(
        monkeypatch, tmp_path, [failing(["t::a"]), failing(["t::a", "t::b", "t::c"]), passing()]
    )
    assert result.ok

    regressed = [e for e in _trace_events(ctx) if e.get("kind") == "final_fix_regressed"]
    assert len(regressed) == 1
    assert regressed[0]["before"] == 1
    assert regressed[0]["after"] == 3


def test_the_round_after_a_regression_is_told_about_it(monkeypatch, tmp_path):
    """Detecting the regression is only useful if the *next* fix round's
    finding actually carries the warning -- apply_fixes reads finding["issue"]
    into the model's prompt, so that is where it has to land."""
    _, _, findings, _ = _run_gate_with_ctx(
        monkeypatch, tmp_path, [failing(["t::a"]), failing(["t::a", "t::b", "t::c"]), passing()]
    )
    round_2 = [f for f in findings if str(f.get("round")) == "final-check-2"]
    assert round_2
    assert "previous fix round made this worse" in round_2[0]["issue"]


def test_a_round_that_does_not_regress_is_not_flagged(monkeypatch, tmp_path):
    """The common, healthy case (steady progress) must not be misread as a
    regression -- only an actual increase in the failure count counts."""
    _, _, findings, ctx = _run_gate_with_ctx(
        monkeypatch, tmp_path, [failing(["t::a", "t::b"]), failing(["t::b"]), passing()]
    )
    assert not [e for e in _trace_events(ctx) if e.get("kind") == "final_fix_regressed"]
    round_2 = [f for f in findings if str(f.get("round")) == "final-check-2"]
    assert round_2
    assert "previous fix round made this worse" not in round_2[0]["issue"]


def test_each_gate_fix_round_escalates_reasoning(monkeypatch, tmp_path):
    """A fix that failed at one reasoning level has shown that level is not
    enough; an identical retry is how a bounded loop learns nothing."""
    import agent.loop as loop_module
    from agent.approval import Approver
    from agent.config import Settings
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(
        workspace=str(workspace), goal="g", final_check=True, executor_reasoning="low"
    )

    levels: list[str | None] = []
    checks = [failing(["t::a"]), failing(["t::b"]), failing(["t::c"]), failing(["t::d"])]
    monkeypatch.setattr(
        loop_module,
        "run_final_check",
        lambda *a, **k: checks.pop(0) if checks else failing(["t::z"]),
    )
    monkeypatch.setattr(
        loop_module, "apply_fixes", lambda *a, **k: levels.append(k.get("reasoning"))
    )
    loop_module._final_gate(None, settings, plan_with(step()), ctx, ctx.trace, [])

    assert levels == ["low", "medium", "high"], levels
