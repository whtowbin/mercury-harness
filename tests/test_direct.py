"""`--direct` (DESIGN-LOG §129): the run skips the planner and executes one step for the goal.

The executor and the planner are replaced by recorders, because what is checked is the control
flow: which of the two the run calls and what plan the executor receives. No model call."""

from __future__ import annotations

import dataclasses

import agent.loop as loop_module
from agent.config import Settings
from agent.phases.plan import Plan, direct_plan


def settings_for(tmp_path, **overrides):
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    base = Settings(
        workspace=str(tmp_path / "ws"),
        goal="fix the thing",
        trace_dir=str(tmp_path / "traces"),
        suggest_next_step=False,
        quick_answers=False,
        final_check=False,
        clarify=False,
    )
    return dataclasses.replace(base, **overrides)


def run_with_recorders(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    planned, executed = [], []
    planner_plan = Plan(goal="fix the thing", draft=[], steps=[])

    def fake_planner(*args, **kwargs):
        planned.append(1)
        return planner_plan

    monkeypatch.setattr(loop_module, "_plan_until_accepted", fake_planner)
    monkeypatch.setattr(loop_module, "execute_plan", lambda c, s, plan, *a: executed.append(plan))
    loop_module.run(settings_for(tmp_path, **overrides), stop_after="execute")
    return planned, executed


def test_direct_runs_the_executor_on_one_step_and_never_plans(tmp_path, monkeypatch):
    planned, executed = run_with_recorders(tmp_path, monkeypatch, direct=True)
    assert planned == []
    (plan,) = executed
    assert [s.intent for s in plan.steps] == ["fix the thing"]
    assert plan.goal == "fix the thing"


def test_without_direct_the_planner_is_still_asked(tmp_path, monkeypatch):
    planned, executed = run_with_recorders(tmp_path, monkeypatch)
    assert planned == [1]
    assert executed[0].steps == []


def test_a_direct_step_declares_no_files_so_it_may_touch_any():
    (step,) = direct_plan("do it").steps
    assert (step.files, step.writes, step.verify) == ([], [], "")


def test_a_direct_step_carries_where_the_goals_names_are_defined_and_loads_those_files(tmp_path):
    from agent.issue_symbols import issue_definitions

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "enums.py").write_text("class ChoicesMeta:\n    pass\n")
    (tmp_path / "pkg" / "other.py").write_text("def unrelated():\n    pass\n")
    found = issue_definitions(tmp_path, "ChoicesMeta returns the wrong value")
    (step,) = direct_plan("ChoicesMeta returns the wrong value", found).steps
    assert step.files == ["pkg/enums.py"]
    assert "ChoicesMeta (class): pkg/enums.py:1" in step.intent
    assert step.intent.startswith("ChoicesMeta returns the wrong value")


def test_direct_with_no_matching_names_is_the_plain_one_step_plan():
    (plain,) = direct_plan("do it", []).steps
    assert (plain.intent, plain.files) == ("do it", [])


def test_the_loop_gives_the_direct_step_the_lookup_only_when_the_flag_is_on(tmp_path, monkeypatch):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "enums.py").write_text("class ChoicesMeta:\n    pass\n")
    kwargs = {"goal": "ChoicesMeta is wrong"}
    _, with_flag = run_with_recorders(
        tmp_path, monkeypatch, direct=True, plan_issue_symbols=True, **kwargs
    )
    assert with_flag[0].steps[0].files == ["enums.py"]
    _, without = run_with_recorders(tmp_path, monkeypatch, direct=True, **kwargs)
    assert without[0].steps[0].files == []
