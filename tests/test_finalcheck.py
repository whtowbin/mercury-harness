"""The deterministic gate: phase 4.

These tests are the reason the phase exists — every one of them describes a
run that every earlier phase would have called successful.
"""

from __future__ import annotations

import dataclasses

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.phases.execute import workspace_fingerprint
from agent.phases.finalcheck import FinalCheck, collect_baseline, run_final_check
from agent.phases.plan import Plan, Step
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(workspace=".", goal="g"), **overrides)


def plan_with(*steps: Step) -> Plan:
    return Plan(goal="g", draft=[], steps=list(steps))


def step(id_="s1", verify="", files=None) -> Step:
    return Step(id=id_, title="t", intent="i", acceptance="a", files=files or [], verify=verify)


# -- verify commands, re-run against the finished state ------------------


def test_a_verify_command_that_passes_is_reported_clean(ctx):
    check = run_final_check(ctx, settings(), plan_with(step(verify="true")), ctx.trace)
    assert check.verify_failures == []
    assert check.ok


def test_a_verify_command_that_fails_is_caught(ctx):
    check = run_final_check(ctx, settings(), plan_with(step(verify="false")), ctx.trace)
    assert len(check.verify_failures) == 1
    assert check.verify_failures[0]["step"] == "s1"
    assert not check.ok


# -- goal_verify: the goal's own claim, checked alongside every step's -----


def test_no_goal_verify_by_default_changes_nothing(ctx):
    check = run_final_check(ctx, settings(), plan_with(step(verify="false")), ctx.trace)
    assert len(check.verify_failures) == 1  # just the step's own failure
    assert check.verify_failures[0]["step"] == "s1"


def test_a_passing_goal_verify_is_clean(ctx):
    check = run_final_check(
        ctx, settings(goal_verify="true"), plan_with(step(verify="true")), ctx.trace
    )
    assert check.verify_failures == []
    assert check.ok


def test_a_failing_goal_verify_is_a_blocker_alongside_a_passing_step(ctx):
    """The exact gap this closes: a step's own verify can pass while the
    goal's actual claim is still false -- lint-only verification, or a test
    that covers the feature's shape but not the specific scenario described."""
    check = run_final_check(
        ctx, settings(goal_verify="false"), plan_with(step(verify="true")), ctx.trace
    )
    assert not check.ok
    assert len(check.verify_failures) == 1
    assert check.verify_failures[0]["step"] == "goal"
    assert check.verify_failures[0]["command"] == "false"


def test_goal_verify_failure_is_shaped_like_any_other_verify_failure(ctx):
    """No new finding type: the same rendering, fix path, and escalation as
    a step verify failure, per the proposal's own design constraint."""
    check = run_final_check(ctx, settings(goal_verify="false"), None, ctx.trace)
    findings = check.findings()
    assert len(findings) == 1
    assert findings[0]["severity"] == "blocker"
    assert "goal" in findings[0]["issue"]


def test_goal_verify_runs_even_with_no_plan(ctx):
    """Independent of step verifies entirely -- a goal_verify command must
    still be checked on a bare run, plan or none."""
    check = run_final_check(ctx, settings(goal_verify="true"), None, ctx.trace)
    assert check.verify_failures == []


def test_a_step_broken_by_a_later_step_is_caught_here(ctx):
    """The case the per-step verify structurally cannot see: step one's check
    passed when step one ran, and a later step then broke it."""
    (ctx.workspace.root / "mod.py").write_text("VALUE = 2\n")
    first = step("first", verify='python3 -c "import mod; assert mod.VALUE == 1"')
    # ...and something later changed VALUE, exactly as a second step would.
    check = run_final_check(ctx, settings(), plan_with(first), ctx.trace)
    assert [f["step"] for f in check.verify_failures] == ["first"]


# -- skipping a redundant re-run when nothing has changed ------------------


def test_a_step_already_verified_true_is_not_rerun_when_unchanged(ctx, tmp_path):
    """267 of 296 traced step verifies were `uv run pytest`, run once during
    execute and then unconditionally again here. If nothing has touched the
    workspace since execute finished, the second run can only reproduce the
    first's result — so it should not happen at all."""
    marker = tmp_path / "ran.marker"
    s = step("first", verify=f"python3 -c \"open('{marker}', 'a').write('x')\"")
    s.verify_passed = True  # as execute.py's own _verify_step would have set it
    s.verify_touched = tuple(ctx.touched)
    s.verify_fingerprint = workspace_fingerprint(ctx, s.verify_touched)
    check = run_final_check(ctx, settings(), plan_with(s), ctx.trace, unchanged_since_execute=True)
    assert not marker.exists(), "verify command ran again despite being unchanged"
    assert check.verify_failures == []
    assert s.verify_passed is True


def test_a_later_step_editing_an_earlier_step_s_dependency_forces_a_recheck(ctx, tmp_path):
    """`unchanged_since_execute` alone says only "nothing has changed since
    execute finished" — both fingerprints it compares are taken after every
    step has already run. A second step editing a file the first step's
    already-passed verify depended on happens *during* execute, so it is
    invisible to that comparison even though the first step's result is now
    stale. The step's own fingerprint, taken at the moment it passed, is
    what has to catch this instead."""
    marker = tmp_path / "ran.marker"
    shared = ctx.workspace.root / "shared.py"
    shared.write_text("VALUE = 1\n")
    ctx.mark_touched("shared.py")
    s = step("first", verify=f"python3 -c \"open('{marker}', 'a').write('x')\"")
    s.verify_passed = True
    s.verify_touched = tuple(ctx.touched)
    s.verify_fingerprint = workspace_fingerprint(ctx, s.verify_touched)
    # ...and a later step, still within the same execute phase, edits the
    # file the first step's verify depended on.
    shared.write_text("VALUE = 2\n")
    run_final_check(ctx, settings(), plan_with(s), ctx.trace, unchanged_since_execute=True)
    assert marker.exists(), "a dependency edited after this step passed must force a re-check"


def test_a_never_verified_step_still_runs_even_when_unchanged(ctx, tmp_path):
    """unchanged_since_execute only excuses a step whose result is already
    known. A step whose verify_passed is still None — verify_steps was off
    during execute, say — has no known result to trust, so it must run."""
    marker = tmp_path / "ran.marker"
    s = step("first", verify=f"python3 -c \"open('{marker}', 'a').write('x')\"")
    assert s.verify_passed is None
    run_final_check(ctx, settings(), plan_with(s), ctx.trace, unchanged_since_execute=True)
    assert marker.exists(), "a step with no known result must still be verified"


def test_a_step_that_failed_during_execute_still_runs_even_when_unchanged(ctx, tmp_path):
    """A step whose own verify failed during execute is not excused just
    because nothing has changed since — its result is a known failure, not
    a known pass, and the gate is still the thing that turns it into a
    finding for the fix round."""
    marker = tmp_path / "ran.marker"
    s = step("first", verify=f"python3 -c \"open('{marker}', 'a').write('x')\"")
    s.verify_passed = False
    run_final_check(ctx, settings(), plan_with(s), ctx.trace, unchanged_since_execute=True)
    assert marker.exists(), "a known-failing step must still be re-checked"


# -- workspace_fingerprint(), the signal the skip above depends on --------


def test_fingerprint_is_stable_when_nothing_changes(ctx):
    (ctx.workspace.root / "a.py").write_text("x = 1\n")
    ctx.mark_touched("a.py")
    assert workspace_fingerprint(ctx) == workspace_fingerprint(ctx)


def test_fingerprint_changes_when_a_touched_file_s_content_changes(ctx):
    path = ctx.workspace.root / "a.py"
    path.write_text("x = 1\n")
    ctx.mark_touched("a.py")
    before = workspace_fingerprint(ctx)
    path.write_text("x = 2\n")
    assert workspace_fingerprint(ctx) != before


def test_fingerprint_changes_when_a_new_file_is_touched(ctx):
    (ctx.workspace.root / "a.py").write_text("x = 1\n")
    ctx.mark_touched("a.py")
    before = workspace_fingerprint(ctx)
    (ctx.workspace.root / "b.py").write_text("y = 2\n")
    ctx.mark_touched("b.py")
    assert workspace_fingerprint(ctx) != before


def test_fingerprint_of_an_untouched_workspace_is_a_fixed_value(ctx):
    """No files touched means nothing to hash -- deterministic, not an error."""
    assert workspace_fingerprint(ctx) == workspace_fingerprint(ctx)


def test_steps_without_a_verify_command_are_skipped(ctx):
    check = run_final_check(ctx, settings(), plan_with(step(verify="")), ctx.trace)
    assert check.verify_failures == []


def test_verify_steps_disabled_skips_the_commands(ctx):
    check = run_final_check(
        ctx, settings(verify_steps=False), plan_with(step(verify="false")), ctx.trace
    )
    assert check.verify_failures == []


# -- the test suite -------------------------------------------------------


def test_a_failing_test_suite_is_caught(ctx):
    (ctx.workspace.root / "test_thing.py").write_text("def test_x():\n    assert False\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.tests_ran
    assert not check.tests_passed
    assert not check.ok


def test_a_passing_test_suite_is_clean(ctx):
    (ctx.workspace.root / "test_thing.py").write_text("def test_x():\n    assert True\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.tests_ran
    assert check.tests_passed
    assert check.ok


def test_a_workspace_with_no_tests_does_not_run_the_suite(ctx):
    (ctx.workspace.root / "mod.py").write_text("x = 1\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert not check.tests_ran
    assert check.ok


def test_a_test_file_with_no_collectable_tests_is_not_green(ctx):
    (ctx.workspace.root / "test_helpers.py").write_text("def helper():\n    return True\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.tests_ran
    assert not check.tests_passed
    assert "(no tests collected)" in check.new_failures
    assert not check.ok


def test_a_deleted_test_suite_is_caught_even_with_no_other_change(ctx):
    """`untested_changes` already catches a step that deletes every test
    file *and* edits some other Python file in the same run. This is the
    residual case it cannot see: a run whose only action is deleting its
    own failing test, touching nothing else -- which, judged purely from
    the current filesystem, is indistinguishable from a project that never
    had a test suite at all."""
    (ctx.workspace.root / "test_thing.py").write_text("def test_x():\n    assert False\n")
    baseline = collect_baseline(ctx, ctx.trace)
    assert baseline is not None  # sanity: tests existed and were collectable
    (ctx.workspace.root / "test_thing.py").unlink()
    ctx.mark_touched("test_thing.py")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline=baseline)
    assert check.deleted_test_suite
    assert not check.tests_ran  # nothing left to run
    assert not check.ok


def test_a_workspace_that_never_had_tests_is_not_reported_as_deleted(ctx):
    """No baseline was ever collected (there was nothing to collect it
    from), so this must not be confused with the case above."""
    (ctx.workspace.root / "mod.py").write_text("x = 1\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline=None)
    assert not check.deleted_test_suite
    assert check.ok


def test_test_files_inside_ignored_directories_do_not_count(ctx):
    """A vendored .venv full of somebody else's tests is not this run's suite."""
    vendored = ctx.workspace.root / ".venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "test_vendored.py").write_text("def test_x():\n    assert False\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert not check.tests_ran


# -- syntax ---------------------------------------------------------------


def test_a_file_that_does_not_parse_is_caught(ctx):
    broken = ctx.workspace.root / "broken.py"
    broken.write_text("def f(:\n")
    ctx.mark_touched("broken.py")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.syntax_errors
    assert not check.ok


# -- turning failures into findings the fixer can take --------------------


def test_failures_become_blocker_findings():
    check = FinalCheck(
        verify_failures=[
            {"step": "s1", "command": "pytest", "output": "boom", "file": "a.py", "files": ["a.py"]}
        ]
    )
    findings = check.findings()
    assert len(findings) == 1
    assert findings[0]["severity"] == "blocker"
    assert findings[0]["file"] == "a.py"
    assert "boom" in findings[0]["failure"]


def test_a_failing_suite_becomes_one_finding():
    check = FinalCheck(
        tests_ran=True, tests_passed=False, tests_summary="1 failed", tests_output="assert False"
    )
    findings = check.findings()
    assert len(findings) == 1
    assert "test suite fails" in findings[0]["issue"]


def test_a_clean_check_produces_no_findings():
    assert FinalCheck().findings() == []


def test_pre_existing_type_errors_alone_never_block():
    """ty findings the workspace already had are real but are not defects
    this run introduced, so they are reported, not gated — same treatment
    as a pre-existing test failure. type_errors holds everything ty found;
    new_type_errors (empty here) is what actually gates."""
    from agent.validate import Diagnostic

    check = FinalCheck(
        type_errors=[Diagnostic(file="a.py", line=1, column=1, code="x", message="m")]
    )
    assert check.ok


def test_a_type_error_this_run_introduced_blocks():
    """Unlike pre-existing debt, a type error the run itself introduced is
    exactly as much this run's fault as a test it broke."""
    from agent.validate import Diagnostic

    d = Diagnostic(file="a.py", line=1, column=1, code="x", message="m")
    check = FinalCheck(type_errors=[d], new_type_errors=[d])
    assert not check.ok


def test_a_new_type_error_becomes_a_finding():
    from agent.validate import Diagnostic

    d = Diagnostic(file="a.py", line=1, column=1, code="x", message="m")
    check = FinalCheck(type_errors=[d], new_type_errors=[d])
    findings = check.findings()
    assert len(findings) == 1
    assert findings[0]["severity"] == "blocker"
    assert findings[0]["file"] == "a.py"


# -- environmental failures: inconclusive, not green, and not fixable -----


def test_an_environmental_failure_is_not_ok(ctx):
    """A step whose verify command could not run at all — a broken
    interpreter, not a code defect — must not be reported as a passing run.
    `envfail` correctly keeps it out of `verify_failures` (no model can fix
    a broken virtualenv), but it still has to cost the run its green."""
    s = step("s1", verify="echo 'dyld[1]: Library not loaded: @rpath/x' 1>&2; exit 1")
    check = run_final_check(ctx, settings(require_tests=False), plan_with(s), ctx.trace)
    assert check.env_failures
    assert check.verify_failures == [], "an environmental failure is not a code defect"
    assert not check.ok


def test_an_environmental_failure_alone_produces_no_findings(ctx):
    """env_failures must never reach the fix loop: no model call can repair
    a broken toolchain, and handing it over as a finding was what burned
    fix rounds chasing a dead virtualenv before this was fixed."""
    s = step("s1", verify="echo 'dyld[1]: Library not loaded: @rpath/x' 1>&2; exit 1")
    check = run_final_check(ctx, settings(require_tests=False), plan_with(s), ctx.trace)
    assert check.env_failures
    assert check.findings() == []


def test_an_environmental_failure_alone_is_inconclusive(ctx):
    """The distinction `ok` alone cannot make: nothing here says the work is
    wrong, only that one check could not run. A caller that wants to score
    "broken toolchain" differently from "broken work" reads `inconclusive`."""
    s = step("s1", verify="echo 'dyld[1]: Library not loaded: @rpath/x' 1>&2; exit 1")
    check = run_final_check(ctx, settings(require_tests=False), plan_with(s), ctx.trace)
    assert not check.ok
    assert check.inconclusive


def test_a_real_defect_alongside_an_environmental_failure_is_not_just_inconclusive(ctx):
    """If a real defect is found by a check that *did* run, that must not be
    laundered into "merely inconclusive" by an unrelated environmental
    failure elsewhere in the same run."""
    broken = step("s1", verify="false")
    unrunnable = step("s2", verify="echo 'dyld[1]: Library not loaded: @rpath/x' 1>&2; exit 1")
    check = run_final_check(
        ctx, settings(require_tests=False), plan_with(broken, unrunnable), ctx.trace
    )
    assert check.env_failures and check.verify_failures
    assert not check.ok
    assert not check.inconclusive, "a real defect exists; this is not merely inconclusive"


# -- collection errors are not test failures ------------------------------


def test_a_collection_error_is_reported_as_such_not_as_a_failure(ctx):
    """pytest exit 2 means nothing ran. Reporting it like a failing assertion
    sent one real run on two attempts of packaging archaeology."""
    from agent.tools.checks import run_tests

    (ctx.workspace.root / "test_broken.py").write_text(
        "import a_module_that_does_not_exist\n\ndef test_x():\n    assert True\n"
    )
    output = run_tests(ctx, path=".")
    assert "COLLECTION ERROR" in output
    assert "import" in output.lower()


def test_a_real_assertion_failure_is_still_reported_as_a_failure(ctx):
    from agent.tools.checks import run_tests

    (ctx.workspace.root / "test_real.py").write_text("def test_x():\n    assert False\n")
    output = run_tests(ctx, path=".")
    assert output.startswith("FAIL")
    assert "COLLECTION ERROR" not in output


def test_a_passing_suite_is_still_reported_as_passing(ctx):
    from agent.tools.checks import run_tests

    (ctx.workspace.root / "test_ok.py").write_text("def test_x():\n    assert True\n")
    assert run_tests(ctx, path=".").startswith("PASS")


# -- the baseline, end to end --------------------------------------------


def test_a_pre_existing_failure_is_excused_when_it_is_out_of_scope(ctx):
    (ctx.workspace.root / "test_old.py").write_text(
        "def test_already_broken():\n    assert False\n"
    )
    baseline = collect_baseline(ctx, ctx.trace)
    assert baseline is not None
    assert any("already_broken" in name for name in baseline.failures)

    check = run_final_check(
        ctx, settings(fix_existing_failures=False), plan_with(), ctx.trace, baseline
    )
    assert check.new_failures == []
    assert check.ok, "with --no-fix-existing-failures it is somebody else's problem"
    assert check.findings() == [], "and nothing should be queued to fix"


def test_a_pre_existing_failure_is_the_work_by_default(ctx):
    """The default the other way round was the bug: a run told to fix failing
    tests planned around them, then reported PASS because they predated it."""
    (ctx.workspace.root / "test_old.py").write_text(
        "def test_already_broken():\n    assert False\n"
    )
    baseline = collect_baseline(ctx, ctx.trace)
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline)
    assert not check.ok, "a red suite is not a passing run by default"
    assert any("already failing" in f["issue"] for f in check.findings())


def test_pre_existing_failures_reach_the_fix_loop_as_blockers(ctx):
    (ctx.workspace.root / "test_old.py").write_text(
        "def test_already_broken():\n    assert False\n"
    )
    baseline = collect_baseline(ctx, ctx.trace)
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline)
    findings = check.findings()
    assert findings and all(f["severity"] == "blocker" for f in findings)
    # The fixer must be told to judge the test as well as the code.
    assert "which is wrong" in findings[0]["fix"]


def test_a_suite_that_cannot_collect_is_never_a_pass(ctx):
    """The reported bug in its purest form: pytest exit 2 (collection error)
    before and after, nothing ever ran, and the gate said ok."""
    (ctx.workspace.root / "test_broken.py").write_text(
        "import a_module_that_does_not_exist\n\ndef test_x():\n    assert True\n"
    )
    baseline = collect_baseline(ctx, ctx.trace)
    assert baseline is not None and baseline.collection_broken
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline)
    assert not check.ok, "a suite that does not collect cannot be passing"


def test_a_failure_the_run_introduced_is_caught_against_a_clean_baseline(ctx):
    (ctx.workspace.root / "test_thing.py").write_text("def test_x():\n    assert True\n")
    baseline = collect_baseline(ctx, ctx.trace)
    assert baseline is not None and baseline.failures == []

    # ...the run then breaks it.
    (ctx.workspace.root / "test_thing.py").write_text("def test_x():\n    assert False\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline)
    assert any("test_x" in name for name in check.new_failures)
    assert not check.ok


def test_no_baseline_is_collected_when_there_are_no_tests(ctx):
    assert collect_baseline(ctx, ctx.trace) is None


# -- the cross-session baseline cache --------------------------------------


def test_a_second_collection_on_an_unchanged_workspace_reuses_the_cache(ctx, monkeypatch):
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    first = collect_baseline(ctx, ctx.trace, settings())
    assert first is not None and first.failures == []

    # Nothing touches the workspace between the two calls -- if the second
    # one ran the suite for real rather than reusing the cache, this makes
    # that visible instead of the test happening to agree with a real rerun.
    from agent.phases import finalcheck

    def fail_if_called(*a, **kw):
        raise AssertionError("the suite must not run again on an unchanged workspace")

    monkeypatch.setattr(finalcheck, "_run_suite", fail_if_called)

    second = collect_baseline(ctx, ctx.trace, settings())
    assert second is not None and second.failures == []


def test_a_changed_workspace_does_not_reuse_a_stale_cache(ctx):
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    first = collect_baseline(ctx, ctx.trace, settings())
    assert first is not None and first.failures == []

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert False\n")
    second = collect_baseline(ctx, ctx.trace, settings())
    assert second is not None and any("test_x" in name for name in second.failures)


def test_memory_and_the_cache_file_itself_do_not_defeat_the_cache(ctx, monkeypatch):
    """The cache's own write, and memory.md's ordinary growth between runs,
    must not be mistaken for a workspace change -- otherwise this would miss
    on every single run, on every real project that uses memory, which
    defeats the point entirely."""
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    collect_baseline(ctx, ctx.trace, settings())

    harness_dir = ctx.workspace.root / ".harness"
    harness_dir.mkdir(exist_ok=True)
    (harness_dir / "memory.md").write_text("## an earlier run\nstuff happened\n")

    from agent.phases import finalcheck

    def fail_if_called(*a, **kw):
        raise AssertionError("a memory.md write must not invalidate the cache")

    monkeypatch.setattr(finalcheck, "_run_suite", fail_if_called)

    second = collect_baseline(ctx, ctx.trace, settings())
    assert second is not None and second.failures == []


def test_disabling_the_setting_always_collects_fresh(ctx):
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    collect_baseline(ctx, ctx.trace, settings(cache_baseline_across_runs=False))

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert False\n")
    # Same content the cache would have matched against if it were consulted --
    # the only way to tell "skipped the suite" from "ran it and got lucky"
    # apart is to disable caching and confirm the fresh failure is still seen.
    second = collect_baseline(ctx, ctx.trace, settings(cache_baseline_across_runs=False))
    assert second is not None and any("test_x" in name for name in second.failures)


def test_a_repaired_pre_existing_failure_stops_being_reported(ctx):
    """It was reported from the baseline regardless of current state, so a run
    that fixed every failing test was still told they were failing — and the
    fix loop got a finding about tests that now pass."""
    test_file = ctx.workspace.root / "test_thing.py"
    test_file.write_text("def test_x():\n    assert False\n")
    baseline = collect_baseline(ctx, ctx.trace)
    assert baseline is not None and baseline.failures

    test_file.write_text("def test_x():\n    assert True\n")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline)
    assert check.pre_existing_failures == [], "it passes now; stop calling it failing"
    assert check.fixed_pre_existing, "and say that it was repaired"
    assert check.ok
    assert check.findings() == []


def test_a_still_failing_pre_existing_test_is_still_reported(ctx):
    (ctx.workspace.root / "test_thing.py").write_text("def test_x():\n    assert False\n")
    baseline = collect_baseline(ctx, ctx.trace)
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace, baseline)
    assert check.pre_existing_failures
    assert not check.ok


# -- behaviour written with no tests at all -------------------------------


def test_code_written_into_a_workspace_with_no_tests_fails_the_run(ctx):
    """Six consecutive real runs reported PASS over a game with broken
    scoring and a preferences pane that did nothing, because with no tests
    the gate had nothing deterministic left and `ruff check` stood in for
    "it works"."""
    (ctx.workspace.root / "game.py").write_text("def score():\n    return 0\n")
    ctx.mark_touched("game.py")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.untested_changes == ["game.py"]
    assert not check.ok


def test_the_untested_finding_asks_for_real_behaviour_tests(ctx):
    (ctx.workspace.root / "game.py").write_text("x = 1\n")
    ctx.mark_touched("game.py")
    finding = run_final_check(ctx, settings(), plan_with(), ctx.trace).findings()[0]
    assert finding["severity"] == "blocker"
    assert "no tests" in finding["issue"]
    assert "lint" in finding["failure"] or "ruff" in finding["failure"]


def test_a_workspace_that_already_has_tests_is_left_alone(ctx):
    """The suite itself is the signal there; this check is about its absence."""
    (ctx.workspace.root / "game.py").write_text("x = 1\n")
    (ctx.workspace.root / "test_game.py").write_text("def test_x():\n    assert True\n")
    ctx.mark_touched("game.py")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.untested_changes == []
    assert check.ok


def test_writing_only_a_test_file_does_not_trip_it(ctx):
    (ctx.workspace.root / "test_game.py").write_text("def test_x():\n    assert True\n")
    ctx.mark_touched("test_game.py")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.untested_changes == []


def test_changing_no_python_does_not_trip_it(ctx):
    (ctx.workspace.root / "README.md").write_text("hello\n")
    ctx.mark_touched("README.md")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.untested_changes == []
    assert check.ok


def test_require_tests_can_be_turned_off(ctx):
    (ctx.workspace.root / "game.py").write_text("x = 1\n")
    ctx.mark_touched("game.py")
    check = run_final_check(ctx, settings(require_tests=False), plan_with(), ctx.trace)
    assert check.untested_changes == []
    assert check.ok


def test_a_run_that_changed_nothing_does_not_trip_it(ctx):
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.untested_changes == []
    assert check.ok


def test_the_final_check_updates_a_step_that_was_repaired(ctx):
    """The finished state is the authority: a step that failed mid-run and
    was fixed must not stay marked FAILED, or the run reports "PASS" and
    "FAILED verification" together and exits non-zero on good work."""
    step_obj = step("s1", verify="true")
    step_obj.verify_passed = False  # as execution left it
    run_final_check(ctx, settings(require_tests=False), plan_with(step_obj), ctx.trace)
    assert step_obj.verify_passed is True


def test_the_final_check_also_marks_a_step_that_has_since_broken(ctx):
    step_obj = step("s1", verify="false")
    step_obj.verify_passed = True
    run_final_check(ctx, settings(require_tests=False), plan_with(step_obj), ctx.trace)
    assert step_obj.verify_passed is False


# -- coverage of the changed lines ---------------------------------------


def test_a_change_the_tests_actually_exercise_passes(ctx):
    """The other side of the coverage check: a real test of real code must
    not be penalised, or the gate becomes noise people turn off."""
    (ctx.workspace.root / "mod.py").write_text(
        "def classify(x):\n    if x > 10:\n        return 'big'\n    return 'small'\n"
    )
    (ctx.workspace.root / "test_mod.py").write_text(
        "from mod import classify\n\n"
        "def test_big():\n    assert classify(20) == 'big'\n\n"
        "def test_small():\n    assert classify(1) == 'small'\n"
    )
    ctx.mark_touched("mod.py")
    ctx.mark_touched("test_mod.py")
    check = run_final_check(ctx, settings(), plan_with(), ctx.trace)
    assert check.tests_passed
    if check.coverage and check.coverage.measured:
        assert check.coverage.unexecuted == [], check.coverage.as_dict()
        assert check.ok


def test_coverage_can_be_turned_off(ctx):
    (ctx.workspace.root / "mod.py").write_text("def f():\n    return 1\n")
    (ctx.workspace.root / "test_mod.py").write_text("def test_x():\n    assert True\n")
    ctx.mark_touched("mod.py")
    check = run_final_check(ctx, settings(check_coverage=False), plan_with(), ctx.trace)
    assert check.coverage is None
    assert check.ok


# -- a verify the unattended approver refuses (DESIGN-LOG §93) ------------


@pytest.fixture
def unattended(tmp_path):
    workspace = tmp_path / "uw"
    workspace.mkdir()
    return ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(workspace_root=workspace, unattended=True),
        trace=Trace(str(tmp_path / "ut"), echo=False),
    )


def test_a_refused_verify_could_not_run_and_is_no_blocker(unattended):
    """GAIA scout g2: a refused `cat answer.txt | grep -q ...` became a verify failure, and the
    fix round it started overwrote a correct answer with "PASS"."""
    (unattended.workspace.root / "answer.txt").write_text("red, blue\n")
    check = run_final_check(
        unattended,
        settings(),
        plan_with(step(verify="cat answer.txt | grep -q 'red'", files=["answer.txt"])),
        unattended.trace,
    )
    assert check.verify_failures == []
    assert [f["step"] for f in check.env_failures] == ["s1"]
    assert check.inconclusive and not check.ok


def test_a_file_test_verify_runs_unattended(unattended):
    (unattended.workspace.root / "answer.txt").write_text("41\n")
    passing = run_final_check(
        unattended, settings(), plan_with(step(verify="test -s answer.txt")), unattended.trace
    )
    failing = run_final_check(
        unattended, settings(), plan_with(step(verify="[ -s missing.txt ]")), unattended.trace
    )
    assert passing.ok and passing.env_failures == []
    assert [f["step"] for f in failing.verify_failures] == ["s1"]


def test_a_refused_goal_verify_could_not_run_either(unattended):
    check = run_final_check(
        unattended, settings(goal_verify="curl -s example.com | grep x"), None, unattended.trace
    )
    assert check.verify_failures == []
    assert [f["step"] for f in check.env_failures] == ["goal"]


def test_the_goal_verify_sees_what_the_test_suite_leaves(ctx):
    """A GAIA run's own test removed answer.txt in its cleanup; the goal check had already
    passed on it, and the run ended with no answer (DESIGN-LOG §93.3)."""
    root = ctx.workspace.root
    (root / "answer.txt").write_text("pigs\n")
    (root / "test_answer.py").write_text(
        "import os\n\ndef test_answer():\n    assert open('answer.txt').read() == 'pigs\\n'\n"
        "    os.remove('answer.txt')\n"
    )
    ctx.touched.append("answer.txt")
    check = run_final_check(ctx, settings(goal_verify="test -s answer.txt"), None, ctx.trace)
    assert [f["step"] for f in check.verify_failures] == ["goal"]
