"""Phase 1b (optional): a second, cheaper model checks the plan against the goal.

Not a second planner. This exists for a planner that is fast to execute against
but terse to read from (Mercury: DESIGN-LOG's plan-comparison finding — GLM's
plans name the actual data structure/algorithm and enumerate edge cases in
roughly twice as many declared tests per plan, Mercury's are shorter and vary
more between repeats of the identical goal) paired with a model that reads more
carefully but is slower to hand-execute against. Rather than pay for a second
full plan (workspace exploration, symbol reading, the whole chain-of-draft), the
auditor sees only the goal and the plan already produced, and its only job is
alignment: does this plan, if executed exactly as written, build everything the
goal asked for. Most of the time nothing is wrong, and the check is one cheap
call that resubmits the plan unchanged — see PLAN_AUDITOR_SYSTEM.
"""

from __future__ import annotations

import json

from ..llm import ChatClient
from ..prompts import PLAN_AUDITOR_SYSTEM
from ..runner import run_tool_loop
from ..tools import ToolContext
from ..tools.base import Registry
from ..tools.control import SUBMIT_PLAN
from ..trace import Trace
from .plan import Plan, Step, finalize_plan

_AUDIT_REGISTRY = Registry([SUBMIT_PLAN])


def _plan_json(plan: Plan) -> str:
    return json.dumps(
        {"draft": plan.draft, "steps": [s.__dict__ for s in plan.steps]}, indent=2, default=str
    )


def audit_plan(client: ChatClient, settings, plan: Plan, ctx: ToolContext, trace: Trace) -> Plan:
    """Check `plan` against `settings.goal` with `settings.plan_auditor_model`.

    Returns the original plan unchanged whenever the auditor is disabled
    (`plan_auditor_model` is None), returns nothing usable, or resubmits the
    same steps. Returns a freshly finalized plan (same rewriting, reconnaissance-
    dropping and preflight checks a fresh plan gets — see `finalize_plan`'s
    docstring) only when the auditor actually changed something.
    """
    if not settings.plan_auditor_model:
        return plan

    trace.say("\n=== phase 1b: plan audit ===", detail=True)
    user = (
        f"Goal:\n{settings.goal}\n\n"
        f"Plan already produced:\n{_plan_json(plan)}\n\n"
        "Check it against the goal, then call submit_plan."
    )
    result, _ = run_tool_loop(
        client=client,
        model=settings.plan_auditor_model,
        phase="plan_audit",
        system=PLAN_AUDITOR_SYSTEM,
        user=user,
        registry=_AUDIT_REGISTRY,
        ctx=ctx,
        trace=trace,
        max_iters=2,
        reasoning_effort=settings.plan_audit_reasoning,
        temperature=0.1,
        max_usd=settings.budget.max_usd,
    )
    if not result or not result.get("steps"):
        trace.say("  plan auditor returned nothing usable; keeping the original plan")
        trace.event("plan_audit_result", changed=False, reason="no usable response")
        return plan

    # Compared only on the fields `submit_plan`'s schema actually declares
    # (via Step.from_dict, the same projection a fresh plan goes through),
    # not on Step's full __dict__: that also carries execution-state fields
    # (verify_passed, attempts, verify_touched as a tuple, ...) no auditor
    # call ever sees or sends back, which made every faithful "unchanged"
    # resubmission compare unequal and defeated this check entirely.
    before = [Step.from_dict(s.__dict__, i).__dict__ for i, s in enumerate(plan.steps)]
    after = [
        Step.from_dict(s, i).__dict__ for i, s in enumerate(result["steps"]) if isinstance(s, dict)
    ]
    if after == before:
        trace.say("  plan audit: no changes")
        trace.event("plan_audit_result", changed=False, notes=result.get("draft", []))
        return plan

    trace.say(f"  plan audit: {len(after)} step(s) resubmitted")
    trace.event(
        "plan_audit_result",
        changed=True,
        notes=result.get("draft", []),
        steps_before=len(before),
        steps_after=len(after),
    )
    # The auditor patches steps; the classification is still the planner's.
    if plan.task_kind and not result.get("task_kind"):
        result = {**result, "task_kind": plan.task_kind}
    # So is the research: the auditor has no tools and looked nothing up.
    if plan.findings:
        result = {**result, "findings": plan.findings}
    return finalize_plan(result, settings, ctx, trace)
