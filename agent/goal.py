"""Keep re-running a goal until it is actually done, not just until one pass
through plan/execute/review/verify finished without raising.

`agent.loop.run()` answers "did this attempt finish cleanly" — a plan
executed, review found nothing blocking, the final gate passed. None of that
is the same question as "does the goal actually say what the person meant,
completely, yet" — a goal can ask for three things and a clean run can
deliver one, and `run()` has no way to notice. `run_until_done()` wraps `run()`
in a loop, judged after each attempt by a cheap model call (`judge_goal`) that
reads the run's own summary and decides: done, worth another narrower
attempt, or blocked in a way another attempt cannot fix.

Every iteration also self-audits its own just-finished trace (reusing
agent/audit.py's model-in-the-loop pass, not a new one) and logs anything it
finds that looks like a HARNESS defect — not a defect in the project being
built — to <workspace>/.harness/feedback.md via agent/feedback.py. That is
the harness noticing its own mistakes as it goes, distinct from judging
whether the goal itself is met.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

from . import tracetools
from .audit import DEFAULT_AUDIT_MODEL, audit_trace
from .config import Settings
from .feedback import FeedbackLog
from .llm import ChatClient
from .loop import run as run_once
from .providers import build_client
from .trace import Trace
from .workspace import Workspace

# Attempts run_until_done will make before giving up and reporting whatever
# the last attempt reached. Not independently measured (this is the shell's
# first real use) — 3 mirrors MAX_FIX_ROUNDS' shape (a small, cheap ceiling
# with headroom, abandoned earlier the moment a round changes nothing) rather
# than a value derived from its own data. See TUNABLES.md.
DEFAULT_MAX_GOAL_ITERATIONS = 3

# The goal judge is a narrow read of a run's own summary, not open-ended
# discovery -- same basis as agent/audit.py's DEFAULT_REASONING_EFFORT and
# plan_audit_reasoning, both cheap alignment checks over text already in
# hand. Not independently measured for this specific call yet.
GOAL_JUDGE_REASONING_EFFORT = "low"

GOAL_JUDGE_SYSTEM = """You are deciding whether a coding agent's attempt actually finished what \
was asked, or whether another attempt is needed. You are reading the harness's own report of \
what happened -- not the agent's opinion of its own work.

Reply with a JSON object: {"status": "done"|"continue"|"blocked", "reason": "one sentence", \
"next_goal": "..."}.

- "done": the original goal is genuinely satisfied, not just that the attempt finished without \
an error. next_goal: "".
- "continue": real progress happened but the goal is not fully met yet, and a narrower, more \
specific instruction for another attempt is worth trying. next_goal: a concrete instruction for \
the next attempt naming what is still missing -- not a repeat of the whole original goal.
- "blocked": another attempt would not help -- something only a person can resolve (a missing \
credential, a genuinely ambiguous requirement, the same failure repeating with no new \
information). next_goal: ""; say why in reason.

Be conservative about "done": an attempt whose own final check failed, or whose summary lists \
unresolved steps, is not done. Return ONLY the JSON object, no prose before or after it.
"""

_JSON_OBJECT_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class GoalVerdict:
    status: str  # "done" | "continue" | "blocked"
    reason: str
    next_goal: str = ""


def _parse_verdict(text: str) -> GoalVerdict | None:
    candidate = text.strip()
    fence = _JSON_OBJECT_RE.search(candidate)
    if fence:
        candidate = fence.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("status") not in ("done", "continue", "blocked"):
        return None
    return GoalVerdict(
        status=parsed["status"],
        reason=str(parsed.get("reason", ""))[:500],
        next_goal=str(parsed.get("next_goal", ""))[:2000],
    )


def judge_goal(client: ChatClient, model: str, original_goal: str, summary: dict) -> GoalVerdict:
    """One cheap call judging whether `summary` (a `run()` result dict) means
    `original_goal` is actually done. Never raises: a judge that cannot
    answer is treated as "blocked" rather than looping on a guess, or
    silently claiming "done" on the strength of nothing -- a wrong stop
    costs one wasted iteration; a wrong continue could spend every remaining
    one on a goal that was already finished.
    """
    report = summary.get("session_summary") or json.dumps(
        {
            "ok": summary.get("ok"),
            "error": summary.get("error"),
            "final_check": summary.get("final_check"),
            "suggested_next_step": summary.get("suggested_next_step"),
        },
        default=str,
    )
    user = f"Original goal:\n{original_goal}\n\nWhat this attempt reported:\n{report[:6000]}"
    try:
        reply = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": GOAL_JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            reasoning_effort=GOAL_JUDGE_REASONING_EFFORT,
        )
        verdict = _parse_verdict(reply.content)
    except Exception as exc:
        return GoalVerdict(
            status="blocked",
            reason=f"the goal judge failed to answer ({type(exc).__name__}: "
            f"{exc}); stopping rather than looping on a guess",
        )
    if verdict is None:
        return GoalVerdict(
            status="blocked",
            reason="the goal judge's reply could not be parsed as a verdict; "
            "stopping rather than looping on a guess",
        )
    return verdict


def _self_audit(
    client: ChatClient,
    trace_path: str,
    workspace_root,
    goal: str,
    iteration: int,
    feedback: FeedbackLog,
    trace: Trace,
) -> None:
    """Audit this iteration's own just-finished trace for harness-level
    defects (not project bugs) and log any into .harness/feedback.md.
    Reuses agent/audit.py's existing model-in-the-loop pass rather than a
    second one -- same model, same prompt, same cost profile as
    `harness trace audit` / the `audit_previous_runs` tool. A failure here
    (a bad trace file, a client error) is logged and skipped, never raised:
    this is bookkeeping about the harness, not part of the goal loop's own
    success criteria.
    """
    try:
        events = tracetools.load_trace(trace_path)
        result = audit_trace(
            events, client, model=DEFAULT_AUDIT_MODEL, workspace_root=workspace_root
        )
    except Exception as exc:
        trace.event(
            "goal_loop_self_audit_failed", iteration=iteration, error=f"{type(exc).__name__}: {exc}"
        )
        return
    harness_findings = [f for f in result["findings"] if f.get("category") == "harness_bug"]
    written = feedback.record(goal, iteration, trace_path, harness_findings)
    trace.event(
        "goal_loop_self_audit",
        iteration=iteration,
        cost_usd=result["cost_usd"],
        findings=len(result["findings"]),
        harness_findings=written,
    )


def run_until_done(
    settings: Settings,
    max_goal_iterations: int = DEFAULT_MAX_GOAL_ITERATIONS,
    plan_path: str | None = None,
    stop_after: str = "review",
    on_event=None,
    on_ask=None,
    on_approve=None,
    cancel=None,
    on_call_start=None,
    on_call_end=None,
    on_plan=None,
) -> dict:
    """Re-run settings.goal (refined by the judge between attempts) until
    judge_goal reports "done", reports "blocked", or max_goal_iterations is
    reached -- whichever comes first. Returns the last attempt's `run()`
    summary, with `goal_loop` added: {"iterations", "status", "reason"}.

    plan_path, when given, is only honored on the first attempt -- a refined
    goal from a later iteration needs its own fresh plan, not a plan written
    for the original one.
    """
    workspace_root = Workspace(settings.workspace).root
    judge_client = build_client(settings, stream=False)
    feedback = FeedbackLog(workspace_root)
    trace = Trace(settings.trace_dir, echo=not settings.quiet, on_event=on_event)

    original_goal = settings.goal
    current_goal = settings.goal
    summary: dict = {}
    iterations: list[dict] = []
    verdict = GoalVerdict(status="continue", reason="not yet attempted")

    for iteration in range(1, max(1, max_goal_iterations) + 1):
        trace.say(f"\n=== goal loop: attempt {iteration}/{max_goal_iterations} ===")
        trace.event("goal_loop_attempt", iteration=iteration, goal=current_goal)
        iter_settings = (
            settings
            if current_goal == settings.goal
            else (dataclasses.replace(settings, goal=current_goal))
        )
        summary = run_once(
            iter_settings,
            plan_path=plan_path if iteration == 1 else None,
            stop_after=stop_after,
            on_event=on_event,
            on_ask=on_ask,
            on_approve=on_approve,
            cancel=cancel,
            on_call_start=on_call_start,
            on_call_end=on_call_end,
            on_plan=on_plan,
        )

        trace_path = summary.get("trace_path")
        if trace_path:
            _self_audit(
                judge_client, trace_path, workspace_root, current_goal, iteration, feedback, trace
            )

        verdict = judge_goal(judge_client, settings.reviewer_model, original_goal, summary)
        trace.event(
            "goal_loop_verdict", iteration=iteration, status=verdict.status, reason=verdict.reason
        )
        iterations.append(
            {
                "iteration": iteration,
                "goal": current_goal,
                "ok": summary.get("ok"),
                "status": verdict.status,
                "reason": verdict.reason,
            }
        )

        if verdict.status == "done":
            trace.say(f"\ngoal loop: done after {iteration} attempt(s) -- {verdict.reason}")
            break
        if verdict.status == "blocked":
            trace.say(f"\ngoal loop: blocked after {iteration} attempt(s) -- {verdict.reason}")
            break
        if iteration >= max_goal_iterations:
            verdict = GoalVerdict(
                status="blocked",
                reason=f"reached max_goal_iterations ({max_goal_iterations}) "
                f"without the judge reporting done",
            )
            iterations[-1]["status"] = verdict.status
            iterations[-1]["reason"] = verdict.reason
            trace.say(f"\ngoal loop: {verdict.reason}")
            break
        if not verdict.next_goal.strip():
            # "continue" with nothing concrete to try differently is the same
            # as being stuck -- looping on the unchanged original goal would
            # very likely just repeat this attempt.
            verdict = GoalVerdict(
                status="blocked",
                reason="the judge said to continue but gave no concrete "
                "next step; stopping rather than repeating this "
                "attempt unchanged",
            )
            iterations[-1]["status"] = verdict.status
            iterations[-1]["reason"] = verdict.reason
            trace.say(f"\ngoal loop: {verdict.reason}")
            break
        current_goal = verdict.next_goal

    summary["goal_loop"] = {
        "iterations": iterations,
        "status": verdict.status,
        "reason": verdict.reason,
    }
    return summary
