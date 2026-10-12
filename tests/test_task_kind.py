"""`task_kind` in submit_plan: the triage signal without a triage call."""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.tools import ToolContext, planning_registry
from agent.tools.control import SUBMIT_PLAN, TASK_KINDS, submit_plan, with_task_kind
from agent.trace import Trace
from agent.workspace import Workspace
from bench import diagnostics


def ev(kind: str, **fields) -> dict:
    return {"kind": kind, **fields}


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "pricing.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def test_the_default_submit_plan_schema_has_no_task_kind():
    tool = planning_registry().get("submit_plan")
    assert tool is not None
    assert "task_kind" not in tool.parameters["properties"]


def test_with_task_kind_adds_an_optional_enum_and_leaves_the_original_alone():
    before = json.dumps(SUBMIT_PLAN.parameters, sort_keys=True)
    tool = with_task_kind(SUBMIT_PLAN)
    assert tool.parameters["properties"]["task_kind"]["enum"] == list(TASK_KINDS)
    assert "task_kind" not in tool.parameters["required"]
    assert json.dumps(SUBMIT_PLAN.parameters, sort_keys=True) == before


def test_submit_plan_records_task_kind_only_when_given(ctx):
    submit_plan(ctx, draft=["d"], steps=[])
    assert "task_kind" not in ctx.result
    submit_plan(ctx, draft=["d"], steps=[], task_kind="bugfix")
    assert ctx.result["task_kind"] == "bugfix"


def test_the_executed_plans_task_kind_is_read_from_the_last_plan_event():
    events = [
        ev("plan", task_kind="feature"),
        ev("plan", task_kind="bugfix"),  # a plan audit resubmitted it
    ]
    assert diagnostics.task_kind(events) == "bugfix"
    assert diagnostics.task_kind([ev("plan")]) == ""
