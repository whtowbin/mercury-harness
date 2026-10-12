"""The executor's reasoning arms: effort configs, draft handoff, `why`.

Every arm is off by default, and the default executor prompt must stay
byte-for-byte what every earlier sweep ran, or those sweeps stop being a
baseline (DESIGN-LOG §53).
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.llm import Reply, Usage
from agent.phases.execute import (
    WHY_TOOLS,
    Step,
    _draft_handoff,
    _executor_system,
    _step_prompt,
    _with_why,
)
from agent.phases.plan import Plan
from agent.prompts import (
    DRAFT_HANDOFF_CHARS,
    EXECUTOR_HYPOTHESIS,
    EXECUTOR_ORDER_LEGACY,
    EXECUTOR_ORDER_TDD,
    EXECUTOR_SYSTEM,
)
from agent.runner import run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace
from bench import configs

# -- the default executor prompt is unchanged ------------------------------------


def settings(**overrides) -> SimpleNamespace:
    base = {"executor_order": "tdd_first", "executor_hypothesis": False}
    return SimpleNamespace(**{**base, **overrides})


def test_the_default_executor_prompt_is_what_the_baseline_ran():
    expected = EXECUTOR_SYSTEM.replace(EXECUTOR_ORDER_LEGACY, EXECUTOR_ORDER_TDD)
    assert _executor_system(settings()) == expected
    # A settings object from before the flag existed reads as off.
    assert _executor_system(SimpleNamespace(executor_order="tdd_first")) == expected


def test_the_hypothesis_arm_appends_its_block_and_changes_nothing_else():
    on = _executor_system(settings(executor_hypothesis=True))
    assert on == _executor_system(settings()) + EXECUTOR_HYPOTHESIS


def test_every_arm_flag_defaults_off():
    fields = {f.name: f.default for f in dataclasses.fields(Settings)}
    assert fields["executor_sees_draft"] is False
    assert fields["executor_hypothesis"] is False
    assert fields["plan_task_kind"] is False
    assert fields["preload_neighbours"] is False
    base = configs.get("mercury-only")
    assert not (
        base.executor_sees_draft
        or base.executor_hypothesis
        or base.plan_task_kind
        or base.preload_neighbours
    )


def test_the_reasoning_arms_differ_from_mercury_only_in_one_field_each():
    base = dataclasses.asdict(configs.get("mercury-only"))
    ignore = {"name", "description", "group"}
    expected = {
        "mercury-exec-none": {"executor_reasoning": "none"},
        "mercury-exec-low": {"executor_reasoning": "low"},
        "mercury-exec-medium": {"executor_reasoning": "medium"},
        "mercury-draft-handoff": {"executor_sees_draft": True},
        "mercury-hypothesis": {"executor_hypothesis": True},
        "mercury-task-kind": {"plan_task_kind": True},
        "mercury-neighbours": {"preload_neighbours": True},
    }
    for name, diff in expected.items():
        arm = dataclasses.asdict(configs.get(name))
        changed = {k: v for k, v in arm.items() if k not in ignore and base[k] != v}
        assert changed == diff, name


# -- the `why` argument ------------------------------------------------------------


def test_why_is_offered_on_exactly_the_failure_acting_tools():
    wrapped = _with_why(execution_registry())
    for tool in wrapped.tools():
        has_why = "why" in tool.parameters.get("properties", {})
        assert has_why == (tool.name in WHY_TOOLS), tool.name
        assert "why" not in tool.parameters.get("required", [])


def test_wrapping_leaves_the_original_registry_untouched():
    original = execution_registry()
    before = json.dumps(original.schemas(), sort_keys=True)
    _with_why(original)
    assert json.dumps(original.schemas(), sort_keys=True) == before


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "pricing.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def tool_call(name: str, **args) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, model, messages, **kwargs):
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def test_an_edit_carrying_why_is_applied_and_the_why_is_traced(ctx):
    edit = tool_call(
        "edit_file",
        path="pricing.py",
        old_string="x = 1",
        new_string="x = 2",
        why="test_x: 1 != 2; set x to 2",
    )
    finish = tool_call("finish_step", summary="done", verified="false")
    replies = [
        Reply(content="", tool_calls=[edit], usage=Usage(10, 5, 0.0, 0.1)),
        Reply(content="", tool_calls=[finish], usage=Usage(10, 5, 0.0, 0.1)),
    ]
    run_tool_loop(
        client=Scripted(replies),
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=_with_why(execution_registry()),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=4,
    )
    assert (ctx.workspace.root / "pricing.py").read_text() == "x = 2\n"
    ctx.trace._fh.flush()
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    calls = [e for e in events if e["kind"] == "tool_call" and e["tool"] == "edit_file"]
    assert calls[0]["args"]["why"] == "test_x: 1 != 2; set x to 2"
    assert not [e for e in events if e["kind"] == "tool_call_failed"]


# -- the draft handoff ----------------------------------------------------------


def plan(draft: list[str]) -> Plan:
    step = Step(id="s1", title="t", intent="i", acceptance="a")
    return Plan(goal="g", draft=draft, steps=[step])


def test_the_handoff_lists_the_notes():
    text = _draft_handoff(plan(["understood: add evict", "risk: evict after insert", "  "]))
    assert "- understood: add evict\n- risk: evict after insert\n" in text
    assert text.startswith("Planner's notes")


def test_an_empty_draft_hands_off_nothing():
    assert _draft_handoff(plan([])) == ""


def test_a_long_draft_is_capped():
    text = _draft_handoff(plan(["word " * 1000]))
    assert len(text) < DRAFT_HANDOFF_CHARS + 200
    assert "…" in text


def test_the_step_prompt_is_unchanged_without_a_draft(ctx):
    p = plan(["understood: x"])
    without = _step_prompt(ctx, p, p.steps[0], 0, 4_000)
    assert "Planner's notes" not in without
    with_draft = _step_prompt(ctx, p, p.steps[0], 0, 4_000, draft=_draft_handoff(p))
    assert with_draft.replace(_draft_handoff(p), "") == without
