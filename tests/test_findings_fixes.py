"""Three ways to get convention decisions into the planner's findings (DESIGN-LOG §87).

After numpy and scipy left the ResearchGate, plans with findings fell from 11-13 of
15 to about 1 in 5. (A) the request ends the planner's first message, (B) `findings`
is required, (C) a one-time hold when findings has no "decision: ..." item.
"""

from __future__ import annotations

import json

from agent.approval import Approver
from agent.config import Settings
from agent.llm import Reply, Usage
from agent.phases.plan import make_plan
from agent.research import FINDINGS_LAST, DecisionsGate, with_findings, with_required_findings
from agent.tools import ToolContext
from agent.tools.control import SUBMIT_PLAN
from agent.trace import Trace
from agent.workspace import Workspace

STEP = {"id": "s", "title": "t", "intent": "i", "acceptance": "a", "files": ["solution.py"]}


def submit(findings: list[str] | None = None) -> dict:
    args: dict = {"draft": ["d"], "steps": [STEP]}
    if findings is not None:
        args["findings"] = findings
    return {
        "id": "p",
        "type": "function",
        "function": {"name": "submit_plan", "arguments": json.dumps(args)},
    }


# -- (B) required -------------------------------------------------------------------


def test_required_findings_only_when_the_field_exists():
    assert "findings" in with_required_findings(with_findings(SUBMIT_PLAN)).parameters["required"]
    assert "findings" not in (with_required_findings(SUBMIT_PLAN).parameters.get("required") or [])
    assert "findings" not in (with_findings(SUBMIT_PLAN).parameters.get("required") or [])


# -- (C) the hold ----------------------------------------------------------------------


def test_a_plan_without_a_decision_is_held_once_then_let_through():
    gate = DecisionsGate()
    held = gate.hold(submit(["k = 1.38e-23 J/K (web)"]))
    assert held is not None and "decision: ... because ..." in held
    assert gate.hold(submit(["k = 1.38e-23"])) is None  # once only


def test_a_plan_with_a_decision_is_not_held():
    assert DecisionsGate().hold(submit(["Decision: radians, because the header says pi/4"])) is None
    assert DecisionsGate().hold({"function": {"name": "read_file", "arguments": "{}"}}) is None


# -- (A) and (C) through the real planner loop ------------------------------------------


class Recording:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return (
            self.replies.pop(0) if self.replies else Reply(content="", usage=Usage(1, 1, 0.0, 0.1))
        )


def planned(tmp_path, replies, **flags):
    settings = Settings(
        workspace=str(tmp_path),
        goal="Write solution.py using numpy",
        research_first=True,
        plan_probe="off",
        callsite_check=False,
        # Unattended, as SciCode runs (sci.py passes --no-clarify): with a person present the
        # planner's first message also carries chatflow's closing (§111).
        clarify=False,
        **flags,
    )
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    client = Recording(replies)
    plan = make_plan(client, settings, ctx, ctx.trace)
    return client, plan


def reply(call: dict) -> Reply:
    return Reply(content="", tool_calls=[call], usage=Usage(10, 5, 0.0, 0.1))


def test_findings_last_ends_the_first_planner_message(tmp_path):
    client, _ = planned(
        tmp_path, [reply(submit(["decision: x because y"]))], plan_findings_last=True
    )
    first_user = next(m for m in client.sent[0] if m["role"] == "user")["content"]
    assert first_user.endswith(FINDINGS_LAST)


def test_without_the_flag_the_first_message_is_unchanged(tmp_path):
    client, _ = planned(tmp_path, [reply(submit(["decision: x because y"]))])
    first_user = next(m for m in client.sent[0] if m["role"] == "user")["content"]
    assert FINDINGS_LAST not in first_user
    assert first_user.endswith("Explore only as much as you need, then submit the plan.")


def test_the_hold_sends_the_plan_back_and_the_resubmitted_decisions_are_kept(tmp_path):
    client, plan = planned(
        tmp_path,
        [reply(submit([])), reply(submit(["decision: one sweep per step, because the docstring"]))],
        plan_findings_hold=True,
    )
    assert len(client.sent) == 2
    assert any("no convention decisions" in str(m.get("content")) for m in client.sent[1])
    assert plan.findings[0].startswith("decision: one sweep")
