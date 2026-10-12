"""Phase 1b: the optional GLM-audits-Mercury's-plan check.

The model call is scripted (CountingClient, same pattern as
test_cost_contract.py) since these tests are not about what a real model
says — but the tool loop, the "did the plan actually change" comparison, and
finalize_plan's rewriting/preflight/reconnaissance-dropping all run for real,
since that is exactly the logic this module is responsible for getting right.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.llm import Reply, Usage
from agent.phases.plan import Plan, Step
from agent.phases.plan_audit import audit_plan
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


class CountingClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    ) -> Reply:
        self.calls += 1
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def call(name, **args):
    return {"id": f"c{len(args)}", "function": {"name": name, "arguments": json.dumps(args)}}


def _settings(**overrides):
    overrides.setdefault("plan_auditor_model", "auditor/model")
    return Settings(workspace=".", goal="add a widget", **overrides)


def _plan():
    return Plan(
        goal="add a widget",
        draft=["understood: add a widget"],
        steps=[
            Step(
                id="add-widget",
                title="Add widget",
                intent="Add Widget class",
                acceptance="Widget exists",
                files=["widget.py"],
                signatures=["Widget.__init__(self)"],
            )
        ],
    )


def test_disabled_by_default_makes_no_call(ctx):
    settings = _settings(plan_auditor_model=None)
    plan = _plan()
    client = CountingClient([])
    result = audit_plan(client, settings, plan, ctx, ctx.trace)
    assert result is plan
    assert client.calls == 0


def test_auditor_resubmits_identical_plan_keeps_original(ctx):
    plan = _plan()
    steps_dict = [s.__dict__ for s in plan.steps]
    reply = Reply(
        content="",
        tool_calls=[call("submit_plan", draft=["aligned"], steps=steps_dict)],
        usage=Usage(10, 5, 0.0, 0.1),
    )
    client = CountingClient([reply])
    settings = _settings()
    result = audit_plan(client, settings, plan, ctx, ctx.trace)
    assert result is plan
    assert client.calls == 1


def test_auditor_patches_a_missing_requirement(ctx):
    plan = _plan()
    patched_steps = [s.__dict__ for s in plan.steps] + [
        {
            "id": "add-tests",
            "title": "Add tests",
            "intent": "Add tests for Widget",
            "acceptance": "tests pass",
            "files": ["test_widget.py"],
            "writes": [],
            "signatures": [],
            "tests": ["test_widget_exists: Widget can be constructed"],
            "verify": "",
            "depends_on": [],
        },
    ]
    reply = Reply(
        content="",
        tool_calls=[
            call("submit_plan", draft=["missing: no tests were planned"], steps=patched_steps)
        ],
        usage=Usage(10, 5, 0.0, 0.1),
    )
    client = CountingClient([reply])
    settings = _settings()
    result = audit_plan(client, settings, plan, ctx, ctx.trace)
    assert result is not plan
    assert [s.id for s in result.steps] == ["add-widget", "add-tests"]
    assert result.draft == ["missing: no tests were planned"]


def test_auditor_no_usable_response_keeps_original_plan(ctx):
    plan = _plan()
    reply = Reply(content="I'm not sure", tool_calls=[], usage=Usage(10, 5, 0.0, 0.1))
    client = CountingClient([reply] * 3)
    settings = _settings()
    result = audit_plan(client, settings, plan, ctx, ctx.trace)
    assert result is plan


def test_audit_result_traced(ctx):
    plan = _plan()
    steps_dict = [s.__dict__ for s in plan.steps]
    reply = Reply(
        content="",
        tool_calls=[call("submit_plan", draft=[], steps=steps_dict)],
        usage=Usage(10, 5, 0.0, 0.1),
    )
    client = CountingClient([reply])
    settings = _settings()
    audit_plan(client, settings, plan, ctx, ctx.trace)
    lines = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    events = [e for e in lines if e.get("kind") == "plan_audit_result"]
    assert len(events) == 1
    assert events[0]["changed"] is False
