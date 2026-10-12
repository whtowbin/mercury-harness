"""The orchestrator: clarify -> plan -> execute -> review -> fix -> final check."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

from . import (
    chatflow,
    faillog,
    issue_symbols,
    relevance,
    scaffold,
    testcheck,
    toolchain,
    userconfig,
)
from . import skills as skills_module
from . import summary as summary_module
from . import trace as trace_rates
from .approval import Approver
from .cancel import Cancelled, CancelToken
from .config import Settings
from .guards import ResourceGuard
from .interact import Interactor, command_asker
from .llm import BudgetExceeded, ChatClient, supports_caching
from .memory import MemoryStore
from .phases.clarify import gather_clarifications, is_assent
from .phases.execute import escalate, execute_plan, workspace_fingerprint
from .phases.finalcheck import (
    BaselineTests,
    FinalCheck,
    collect_baseline,
    collect_type_baseline,
    run_final_check,
)
from .phases.plan import Plan, Step, direct_plan, make_plan
from .phases.plan_audit import audit_plan
from .phases.review import apply_fixes, review
from .prompts import COMPACT_MEMORY_SYSTEM, NEXT_STEP_SYSTEM, QUICK_ANSWER_SYSTEM
from .providers import build_client
from .runner import over_spend_cap
from .tools import ToolContext
from .tools.web import search_backend
from .trace import Trace
from .workspace import NestedWorkspaceError, Workspace

# What on_plan may return. "" runs the plan as-is, "abort" stops the run, and
# anything else is treated as feedback to re-plan with.
ABORT = "abort"
# How many times a caller may send the plan back for another attempt before
# the harness stops offering. Generous, but not unbounded: each re-plan is a
# full planning call.
MAX_REPLANS = 3
# Fix/re-check cycles the final gate may run. Bounded, and abandoned earlier
# than this the moment a round changes nothing — a fix that fixed nothing will
# not fix anything on the next identical attempt either, and the spend cap
# should not be the only thing that stops it.
MAX_FIX_ROUNDS = 3
# Memory is condensed once the file grows past this multiple of the per-run
# recall budget, so it cannot grow without bound across a long-lived project.
MEMORY_COMPACT_FACTOR = 6
MEMORY_KEEP_RECENT = 8


def _check_not_nested(workspace: Workspace, allow_nested: bool) -> Path | None:
    """The enclosing project, if any. Raises unless `allow_nested` accepts it."""
    enclosing = workspace.enclosing_project()
    if enclosing and not allow_nested:
        raise NestedWorkspaceError(
            f"the workspace {workspace.root} is inside the Python project at {enclosing}.\n"
            "Build tools search upward, so commands the agent runs (`uv pip install`, "
            "`uv sync`) can reach that project's environment and modify it. The path "
            "jail does not constrain subprocesses.\n"
            "Use a workspace outside any project — `mkdir -p ~/agent-sandbox` — or pass "
            "--allow-nested if you accept the risk."
        )
    return enclosing


def _build_run(
    settings: Settings,
    on_event,
    on_ask,
    on_approve,
    cancel: CancelToken | None,
    on_call_start,
    on_call_end,
    on_ask_many=None,
) -> tuple[Workspace, Path | None, Trace, ChatClient, ToolContext]:
    """Everything a run needs before any phase can start: the workspace (and
    its nested-project check), the trace, the model client, and the shared
    tool context. Pulled out of run() as pure construction — same objects,
    same order, same side effects (the skills_loaded/toolchain trace events
    still fire from here), just not read all at once."""
    workspace = Workspace(settings.workspace)
    enclosing = _check_not_nested(workspace, settings.allow_nested)
    trace = Trace(settings.trace_dir, echo=not settings.quiet, on_event=on_event)
    client = build_client(
        settings,
        stream=settings.stream,
        cache_prompts=settings.cache_prompts,
        max_call_s=settings.max_call_s,
        strict_tools=getattr(settings, "strict_tool_schemas", True),
        provider_sort=getattr(settings, "provider_sort", ""),
        on_call_start=on_call_start,
        on_call_end=on_call_end,
        # A dropped connection and a slow provider both look like a retry
        # from outside; only one of them is fixed by checking the network.
        on_reconnect=lambda attempt, error: trace.event(
            "llm_reconnected", attempt=attempt, error=error[:200]
        ),
        # Every failed attempt, not just the ones a later success reports.
        on_attempt_failed=lambda attempt, seconds, error: trace.event(
            "llm_attempt_failed", attempt=attempt, seconds=round(seconds, 1), error=error
        ),
    )
    guard = ResourceGuard(
        step_subprocess_budget_s=settings.limits.step_shell_budget_s,
        max_workspace_mb=settings.limits.max_workspace_mb,
        max_file_write_mb=settings.limits.max_file_write_mb,
        max_step_errors=settings.limits.max_step_errors,
    )
    loaded_skills = (
        skills_module.load_merged(userconfig.user_skills_dir(), settings.skills_dir)
        if settings.skills_enabled
        else []
    )
    loaded_skills = skills_module.for_web_research(
        loaded_skills, getattr(settings, "web_research", False)
    )
    answer_command = getattr(settings, "answer_command", "")
    if answer_command and on_ask is None:
        on_ask = command_asker(answer_command)
    # on_ask_many: a form that shows clarify's questions together (the TUI's; DESIGN-LOG §101).
    interactor = Interactor(ask=on_ask, enabled=settings.ask_enabled, ask_many=on_ask_many)
    ctx = ToolContext(
        workspace=workspace,
        approver=Approver(
            settings.auto_approve,
            ask=on_approve,
            auto_approve_python=settings.auto_approve_python,
            notify=trace.say,
            workspace_root=workspace.root,
            # A real run, not a test fixture reaching for
            # auto_approve as a shortcut -- see Approver's own
            # docstring for what turning this on changes.
            unattended=settings.auto_approve,
        ),
        trace=trace,
        interactor=interactor,
        guard=guard,
        cancel=cancel or CancelToken(),
        autolint=settings.autolint,
        read_assist=getattr(settings, "read_assist", True),
        web_research=getattr(settings, "web_research", False),
        narrate=settings.narrate,
        replay_checks=settings.replay_identical_checks,
        skills=loaded_skills,
        skill_token_budget=settings.skill_token_budget,
        skill_triggers=settings.skill_triggers,
        audit_client=client,
        relevance_judge=relevance.build_judge(
            settings,
            client,
            on_call=lambda model, reply: trace.llm_call(
                "relevance", model, reply.usage, reply.finish_reason, reply=reply
            ),
        ),
        test_command=getattr(settings, "test_command", ""),
        test_label_root=getattr(settings, "test_label_root", ""),
        web_deny=tuple(getattr(settings, "web_deny", ())),
    )
    if settings.memory:
        ctx.memory = MemoryStore(workspace.root)
        ctx.memory.start_session()
    if loaded_skills:
        trace.event("skills_loaded", names=[s.name for s in loaded_skills])
        trace.say("skills: " + ", ".join(s.name for s in loaded_skills), detail=True)
    return workspace, enclosing, trace, client, ctx


# Settings never written to a trace: credentials, by name.
_SECRET_FIELD = re.compile(r"api_key|secret|password|token$|_key$", re.IGNORECASE)


def _recorded_settings(settings: Settings) -> dict:
    """Every behaviour setting of this run, so a trace says which features were on.

    A platformer-app run could only be dated against commits to tell whether combined fixes
    and the 5-question clarify were active (DESIGN-LOG §103). Scalars and tuples only;
    credentials are left out by name.
    """
    out: dict = {}
    for field in dataclasses.fields(settings):
        if field.name in ("goal", "budget") or _SECRET_FIELD.search(field.name):
            continue
        value = getattr(settings, field.name)
        if isinstance(value, (bool, int, float, str, type(None))):
            out[field.name] = value
        elif isinstance(value, tuple):
            out[field.name] = list(value)
    return out


@functools.lru_cache(maxsize=1)
def _harness_commit() -> dict:
    """The harness's own commit, and whether its code had uncommitted changes. Once a
    process: a TUI session runs the code it loaded at start, whatever the checkout says now."""
    package = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(package), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(package), "status", "--porcelain", "--untracked-files=no", "--", "."],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if head.returncode != 0:
        return {}
    return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def _narrate_run_start(
    trace: Trace, workspace: Workspace, enclosing: Path | None, settings: Settings
) -> None:
    """The banner lines printed once, right before the pipeline starts."""
    tools_present = toolchain.available(workspace.root)
    trace.event("toolchain", **tools_present)
    if enclosing:
        trace.say(
            f"WARNING: workspace is nested inside the project at {enclosing}; "
            "agent subprocesses can reach it"
        )
    trace.event(
        "run_start",
        goal=settings.goal,
        workspace=str(workspace.root),
        nested_inside=str(enclosing) if enclosing else "",
        planner=settings.planner_model,
        executor=settings.executor_model,
        reviewer=settings.reviewer_model,
        mercury_provider=getattr(settings, "mercury_provider", "openrouter"),
        mercury_temperature=getattr(settings, "mercury_temperature", None),
        budget=asdict(settings.budget),
        settings=_recorded_settings(settings),
        harness=_harness_commit(),
        # Which search service web_search used: GAIA scores moved 22 to 15 of 40 with no harness
        # change when the local SearXNG's engines degraded (DESIGN-LOG §112.2, §115).
        search_backend=search_backend(),
    )
    trace.say(f"goal: {settings.goal}", detail=True)
    trace.say(f"workspace: {workspace.root}", detail=True)
    trace.say(
        f"planner {settings.planner_model} | executor {settings.executor_model} "
        f"| reviewer {settings.reviewer_model}",
        detail=True,
    )
    missing = [name for name, ok in tools_present.items() if not ok]
    trace.say(
        "toolchain: "
        + ", ".join(sorted(n for n, ok in tools_present.items() if ok))
        + (f"  (unavailable: {', '.join(missing)})" if missing else ""),
        detail=True,
    )
    _note_inert_cache_flag(trace, settings)


def _note_inert_cache_flag(trace: Trace, settings: Settings) -> None:
    """Say when `cache_prompts` cannot do anything for this run's models.

    The flag sends an explicit cache breakpoint, and only Anthropic models
    accept one, so on the all-Mercury defaults both settings behave
    identically. Someone passing `--no-cache` to cut cost is entitled to
    know it changed nothing, rather than concluding caching does not help.
    Recorded as an event and a detail line: worth having in the trace,
    not worth a line of the conversation.
    """
    if not settings.cache_prompts:
        return
    models = {settings.planner_model, settings.executor_model, settings.reviewer_model}
    if any(supports_caching(m) for m in models):
        return
    trace.event("cache_control_inert", models=sorted(models))
    trace.say(
        "cache_prompts is on, but no model in this run accepts an explicit "
        "cache breakpoint; provider-side caching still applies",
        detail=True,
    )


def _obtain_plan(client, settings, ctx, trace, on_plan, baselines, plan_path) -> Plan | None:
    """The plan to execute: a saved one, one on stdin, or one planned (and, if
    configured, audited) now. None when the person declined the planned one."""
    if plan_path == "-":
        return _load_plan_from_stdin(settings)
    if plan_path:
        return _load_plan(plan_path, settings)
    if settings.direct:
        trace.say("\n=== direct: no planner; one step for the whole goal ===", detail=True)
        found = (
            issue_symbols.issue_definitions(ctx.workspace.root, settings.goal)
            if settings.plan_issue_symbols
            else None
        )
        return direct_plan(settings.goal, found)
    plan = _plan_until_accepted(client, settings, ctx, trace, on_plan, baselines)
    if plan is None:
        trace.say("\nplan was not accepted; nothing was executed")
        return None
    if settings.plan_auditor_model:
        plan = audit_plan(client, settings, plan, ctx, trace)
    return plan


def _try_quick_answer_first(client, settings, ctx, trace, workspace) -> str:
    """The lookup _plan_until_accepted makes before planning, reused so a bare
    "yes" accepting an earlier suggestion isn't classified blind — see
    try_quick_answer's docstring. Reads ctx.memory directly rather than
    recall_memory() so this internal check doesn't narrate "recalled earlier
    work" before the run has decided whether it is doing any."""
    memory_hint = (
        ctx.memory.relevant(
            settings.goal,
            budget_tokens=settings.memory_token_budget,
            judge=ctx.relevance_judge,
        )
        if settings.memory and ctx.memory is not None
        else ""
    )
    return try_quick_answer(
        client,
        settings,
        trace,
        workspace_map=ctx.workspace.tree(),
        memory_context=memory_hint,
    )


def run(
    settings: Settings,
    plan_path: str | None = None,
    stop_after: str = "review",
    on_event=None,
    on_ask=None,
    on_approve=None,
    cancel: CancelToken | None = None,
    on_call_start=None,
    on_call_end=None,
    on_plan=None,
    on_ask_many=None,
) -> dict:
    workspace, enclosing, trace, client, ctx = _build_run(
        settings, on_event, on_ask, on_approve, cancel, on_call_start, on_call_end, on_ask_many
    )
    _narrate_run_start(trace, workspace, enclosing, settings)

    findings: list[dict] = []
    plan: Plan | None = None
    error = ""
    quick_answer = ""
    check: FinalCheck | None = None
    test_audit: dict | None = None
    interrupted: KeyboardInterrupt | None = None
    baselines: Baselines | None = None
    # Everything _finish needs that does not change as the run progresses; each
    # exit point supplies only what it knows.
    finish = partial(
        _finish,
        trace=trace,
        ctx=ctx,
        goal=settings.goal,
        failure_log_dir=settings.failure_log_dir,
        show_stats=settings.show_stats,
        client=client,
        settings=settings,
    )
    try:
        if settings.quick_answers and not plan_path:
            quick_answer = _try_quick_answer_first(client, settings, ctx, trace, workspace)
        if quick_answer:
            trace.say(f"\n{quick_answer}")
            trace.event("quick_answer", text=quick_answer)
            return finish(plan=plan, findings=findings, quick_answer=quick_answer)

        # Started here, needed by the planner, and waited for in between —
        # see _start_baselines. It runs the whole suite, which on a real
        # project is the single longest thing a run does before it has said
        # anything: 49.3 seconds in one traced run, against 0.9s for the
        # clarifying questions that followed it. Those questions, and the
        # time the person spends answering them, are dead CPU otherwise.
        if (
            getattr(settings, "scaffold_new_projects", False)
            and not plan_path
            and scaffold.is_new_project(workspace.root)
            and scaffold.wants_python(settings.goal)
        ):
            scaffold.scaffold(workspace.root, trace)
        baselines = _start_baselines(ctx, trace, settings)

        plan = _obtain_plan(client, settings, ctx, trace, on_plan, baselines, plan_path)
        if plan is None:
            return finish(plan=None, findings=findings, error="stopped: the plan was not accepted")
        if stop_after == "plan":
            return finish(plan=plan, findings=findings, stopped_after="plan")

        execute_plan(client, settings, plan, ctx, trace)
        if stop_after == "execute":
            return finish(plan=plan, findings=findings)

        # Captured right here, not inside the gate: this is the one point
        # that means "execute just finished, nothing has touched the
        # workspace since." Comparing against it below, after review has had
        # its chance to edit things, is what lets the gate skip re-running a
        # step's verify command when the answer can only be what it already
        # was.
        post_execute_fingerprint = workspace_fingerprint(ctx)

        if settings.review:
            findings = _review_rounds(client, settings, plan, ctx, trace, baselines.tests)
        else:
            trace.say("\nreview is off for this run; going straight to the final check")
        unchanged_since_execute = workspace_fingerprint(ctx) == post_execute_fingerprint
        check = _final_gate(
            client,
            settings,
            plan,
            ctx,
            trace,
            findings,
            baselines.tests,
            unchanged_since_execute,
            baselines.types,
        )
        if getattr(settings, "test_audit", True) and check is not None and check.tests_passed:
            test_audit = _audit_tests(ctx, trace)
            if getattr(settings, "fix_weak_tests", False) and test_audit.get("findings"):
                check, test_audit = _repair_weak_tests(
                    client, settings, plan, ctx, trace, findings, baselines, check, test_audit
                )
    except BudgetExceeded as exc:
        trace.say(f"\nSTOPPED: {exc}")
        trace.event("budget_exceeded", message=str(exc))
        error = str(exc)
    except Cancelled as exc:
        trace.say(f"\nSTOPPED: {exc}")
        trace.event("cancelled", message=str(exc))
        error = f"stopped: {exc}"
    except KeyboardInterrupt as exc:
        # Still recorded and logged like any other stop, rather than
        # discarding the run's progress — but re-raised once that's done, so
        # a caller relying on KeyboardInterrupt reaching it (the CLI's exit
        # code 130) keeps seeing it.
        trace.say("\ninterrupted by user")
        trace.event("interrupted")
        error = "stopped: interrupted by user (Ctrl-C)"
        interrupted = exc
    except Exception as exc:
        # A failed run still spent money and may have produced work. Returning the
        # summary keeps both visible; raising would discard them.
        trace.say(f"\nFAILED: {type(exc).__name__}: {exc}")
        trace.event("run_failed", error=f"{type(exc).__name__}: {exc}")
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if baselines is not None:
            baselines.close()

    # Only on a genuinely completed run: --stop-after plan/execute return
    # earlier, above, and skip this — there is nothing to weigh in on yet.
    suggestion = ""
    over_cap = over_spend_cap(trace, settings.budget.max_usd)
    if not error and plan is not None and settings.suggest_next_step and not over_cap:
        suggestion = suggest_next_step(client, settings, plan, findings, ctx, trace)

    summary = finish(
        plan=plan,
        findings=findings,
        error=error,
        suggestion=suggestion,
        check=check,
        test_audit=test_audit,
    )
    if interrupted is not None:
        raise interrupted
    return summary


# How long Baselines.close() waits for a stopped baseline's thread: the suite's process
# group gets SIGTERM, then SIGKILL after a second (toolchain._stop_group).
BASELINE_CLOSE_WAIT_S = 10.0


class Baselines:
    """The pre-run baselines, collected on a worker thread.

    Both are read-only measurements of the workspace as it was before the
    agent touched anything, so nothing depends on when they run — only on
    their being finished before the planner needs them. Running the suite is
    the longest thing a run does before it speaks (49.3s in one traced run),
    and the clarifying questions that follow take 0.9s to produce and then
    wait on a person. Overlapping the two costs nothing and hides most of it.

    `result()` blocks until they are done, so every consumer sees exactly
    what it saw when this was sequential. A failure on the thread is
    re-raised there rather than swallowed: a baseline that silently became
    None would make the gate treat every pre-existing failure as this run's
    fault.
    """

    def __init__(self, future, trace: Trace, cancel: threading.Event | None = None):
        self._future = future
        self._trace = trace
        self._cancel = cancel or threading.Event()
        self._done: tuple[BaselineTests | None, frozenset[str] | None] | None = None

    def close(self) -> None:
        """Stop a baseline nobody will read, and wait for its thread.

        Called from run()'s `finally`. A run that ended before `result()` (a plan
        rejected or not accepted, a stop after planning, Cancelled, an exception)
        used to leave the whole suite running: `harness run` returned only when it
        finished, and in the TUI the next goal started a second suite beside it
        (DESIGN-LOG §91, finding 4). Setting the event kills the suite's process
        group; a run that already read the result loses nothing.
        """
        if self._future is None or self._done is not None:
            return
        self._cancel.set()
        # A stopped baseline's outcome, or its failure, is discarded.
        with contextlib.suppress(Exception):
            self._future.result(timeout=BASELINE_CLOSE_WAIT_S)

    def result(self) -> tuple[BaselineTests | None, frozenset[str] | None]:
        if self._done is None:
            started = time.time()
            self._done = self._future.result() if self._future else (None, None)
            waited = time.time() - started
            self._trace.event("baseline_wait", seconds=round(waited, 2))
            if waited > 1.0:
                self._trace.say(f"   (waited {waited:.0f}s for the pre-run test baseline)")
        return self._done

    @property
    def tests(self) -> BaselineTests | None:
        return self.result()[0]

    @property
    def types(self) -> frozenset[str] | None:
        return self.result()[1]


def _start_baselines(ctx: ToolContext, trace: Trace, settings: Settings) -> Baselines:
    """Kick the baselines off on a thread and return a handle to them."""
    if not settings.final_check:
        return Baselines(None, trace)

    cancel = threading.Event()

    def collect():
        tests = collect_baseline(ctx, trace, settings, cancel=cancel)
        if cancel.is_set():
            return None, None
        types = collect_type_baseline(ctx, trace)
        # Steps read this to tell a type error they introduced from one the
        # project already had. Set on the thread, read after `result()`.
        ctx.type_baseline = types
        return tests, types

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="baseline")
    future = pool.submit(collect)
    pool.shutdown(wait=False)
    return Baselines(future, trace, cancel)


def _failing_tests_context(baseline: BaselineTests | None, settings: Settings) -> str:
    """What is already red, as text for the planner.

    Without this the planner is asked to fix failing tests while being told
    nothing about which tests fail — it then spends its whole budget running
    the suite itself, or asks the human to run it, which is what a real trace
    shows it doing seven times in a row.
    """
    command = getattr(settings, "test_command", "")
    if command:
        return (
            f"\n\nThis project's tests run with its own command, `{command}`, and "
            "run_tests uses it: name what to run in its `path` (a test module label such "
            "as `pagination.tests`, or a test file). There is no pre-run baseline, since "
            "the whole suite is too large to run first.\n"
        )
    if baseline is None or not (baseline.failures or baseline.collection_broken):
        return ""
    if baseline.unrunnable:
        hint = (
            f" {baseline.runner_hint[0].upper()}{baseline.runner_hint[1:]}."
            if baseline.runner_hint
            else ""
        )
        return (
            f"\n\npytest could not load {len(baseline.failures)} of this project's test files "
            "(they fail at import or configuration, before any test runs), so the project "
            f"runs its tests some other way and which of them pass is unknown.{hint} Run "
            "tests the project's way (for example through run_bash), and do not spend the "
            "run making pytest collect the suite unless the goal asks for that.\n"
        )
    if baseline.collection_broken:
        header = (
            "\n\nThe test suite does not currently collect — pytest fails before "
            "running anything, which is usually an import error or a missing "
            "dependency. Nothing else can be verified until that is fixed."
        )
    else:
        header = ""
    listed = "\n".join(f"  - {name}" for name in baseline.failures[:20])
    scope = (
        "These are part of the work: fix them unless the goal says otherwise."
        if settings.fix_existing_failures
        else "These were already failing and are out of scope unless the goal names them."
    )
    why = (
        f"What each reports (deepest frame in the code, then its error):\n{baseline.digest}\n"
        if baseline.digest and getattr(settings, "baseline_digest", True)
        else ""
    )
    return (
        f"{header}\n\nTests already failing before this run started "
        f"({len(baseline.failures)}):\n{listed}\n{why}{scope}\n"
        "For each one, read the test and the code it exercises and decide which is "
        "wrong before changing either. Never delete a test or weaken an assertion "
        "to make it pass.\n"
    )


def _plan_until_accepted(
    client,
    settings: Settings,
    ctx: ToolContext,
    trace: Trace,
    on_plan,
    baselines: Baselines | None = None,
) -> Plan | None:
    """Plan, then give the caller a chance to accept, reject, or redirect it.

    A plan is the cheapest possible place to catch a misunderstanding: fixing
    one here costs a single planning call, while discovering the same mistake
    after execution costs the whole run. `on_plan` returning feedback rather
    than acceptance re-plans with that feedback in the prompt, which is what
    makes this worth more than a yes/no gate.

    Returns None when the caller aborted. With no `on_plan` (a library or
    unattended caller) the first plan is used, exactly as before.
    """
    # What earlier runs on this workspace did, and left broken. Loaded before
    # clarification so the triage call can see it too: a follow-up goal like
    # "why did that fail?" is answerable only from this, and without it the
    # clarifier asks a question nobody can answer.
    remembered = recall_memory(settings, ctx, trace)
    # Joined here, before clarify, because clarify's prompt gets the failing tests
    # too. (This comment used to say the join came after clarify's questions; it
    # never did.) Measured in DESIGN-LOG §95: the wait is a median 0.02 s over
    # 7,643 runs and at most 1 s in app runs; of the 407 runs that waited over 5 s
    # (unattended sweeps), 15 asked clarify anything. Moving the join after clarify
    # would save almost nothing and cost clarify that context.
    failing = _failing_tests_context(baselines.tests if baselines else None, settings)
    context = (
        remembered
        + failing
        + gather_clarifications(
            client,
            settings,
            ctx,
            trace,
            remembered=remembered + failing,
            suggestion=_pending_suggestion(settings, ctx),
        )
    )
    plan = make_plan(client, settings, ctx, trace, extra_context=context)
    chatflow.show_answers(plan.draft, trace)
    # With nobody reviewing the whole plan, a person who is present still confirms what it
    # assumes (DESIGN-LOG §111); a reviewer of the whole plan sees the assumptions in it.
    if on_plan is None and chatflow.enabled(settings):
        on_plan = chatflow.assumptions_gate(ctx.interactor, trace)
    if on_plan is None:
        return plan

    for attempt in range(MAX_REPLANS):
        decision = (on_plan(plan) or "").strip()
        if not decision:
            return plan
        if decision.lower() == ABORT:
            trace.event("plan_rejected", attempt=attempt + 1)
            return None
        trace.event("plan_feedback", attempt=attempt + 1, feedback=decision[:500])
        trace.say(f"\nre-planning with your feedback: {decision}")
        context += (
            f"\n\nA previous plan was rejected with this feedback:\n{decision}\n"
            "Address it directly in the new plan."
        )
        plan = make_plan(client, settings, ctx, trace, extra_context=context)
        chatflow.show_answers(plan.draft, trace)
    # Out of re-plans: the caller has seen three attempts, so use the last one
    # rather than discarding the work silently.
    return plan


def _review_rounds(
    client,
    settings: Settings,
    plan: Plan,
    ctx: ToolContext,
    trace: Trace,
    baseline: BaselineTests | None = None,
) -> list[dict]:
    """Review, fix, re-review. Returns every finding from every round.

    Accumulating rather than overwriting matters for the record: a run whose
    first round found three blockers and whose second came back clean used to
    report no findings at all, because the second round's empty list replaced
    the first's. What was wrong and then repaired is exactly what someone
    reading the summary afterwards wants to know.
    """
    all_findings: list[dict] = []
    for round_no in range(1, settings.budget.max_review_rounds + 1):
        round_findings = review(
            client,
            settings,
            plan,
            ctx,
            trace,
            already_failing=baseline.failures if baseline else None,
        )
        for finding in round_findings:
            finding.setdefault("round", round_no)
        all_findings.extend(round_findings)
        blocking = [f for f in round_findings if f.get("severity") in ("blocker", "major")]
        if not blocking:
            trace.say(f"\nreview round {round_no}: clean")
            break
        trace.say(f"\nreview round {round_no}: {len(blocking)} blocking finding(s)")
        apply_fixes(client, settings, round_findings, ctx, trace)
    return all_findings


def _final_gate(
    client,
    settings: Settings,
    plan: Plan,
    ctx: ToolContext,
    trace: Trace,
    findings: list[dict],
    baseline: BaselineTests | None = None,
    unchanged_since_execute: bool = False,
    type_baseline: frozenset[str] | None = None,
) -> FinalCheck | None:
    """Check, fix, re-check — up to MAX_FIX_ROUNDS times, stopping early.

    This is also what closes the review loop's last gap: the final round of
    fixes used to be applied and never looked at again, so the edits most
    likely to be wrong — a model's repairs to a model's findings — were the
    only ones nothing checked.

    It iterates because one fix round is frequently not enough: a real run
    diagnosed two failing tests correctly, ran out of iterations before it
    could act, and the gate reported failure and stopped rather than trying
    again with what it had learned. It stops early when a round leaves the
    failure set byte-for-byte unchanged, because a fix that fixed nothing
    will not fix anything on an identical retry — better to report honestly
    than to spend two more rounds proving it.

    `unchanged_since_execute` only ever applies to this first check. Every
    later one in the loop below follows an `apply_fixes` call that just
    edited the workspace, so it is never passed there — the default of
    False is exactly right for a state this function knows has changed.
    """
    if not settings.final_check:
        return None
    check = run_final_check(
        ctx, settings, plan, trace, baseline, unchanged_since_execute, type_baseline
    )
    if check.ok:
        return check

    last_signature = check.signature()
    last_failure_count = len(check.failing_tests)
    regressed = False
    for round_no in range(1, MAX_FIX_ROUNDS + 1):
        gate_findings = check.findings()
        if not gate_findings:
            break
        for finding in gate_findings:
            finding.setdefault("round", f"final-check-{round_no}")
        # A round whose own edit made the failure count worse is not told so
        # anywhere else: the loop just moves on to the next round on a
        # different (not identical) signature, so the "stop on no progress"
        # check below never fires either. A traced fix round went from 2
        # failing to 5 this way and ran out of iterations immediately after,
        # never having been told the edit it just made was the problem.
        if regressed and gate_findings:
            gate_findings[0]["issue"] = (
                "[harness] the previous fix round made this worse — "
                f"{last_failure_count} failure(s) before it, now more. Consider whether "
                f"to undo that edit before trying a different approach. "
                + gate_findings[0]["issue"]
            )
            regressed = False
        findings.extend(gate_findings)
        trace.say(
            f"\nfinal check found {len(gate_findings)} failure(s); "
            f"fix round {round_no} of {MAX_FIX_ROUNDS}"
        )
        # Turn reasoning up each round. A fix that failed at this level has
        # shown the level is not enough, and an identical retry is how a
        # bounded loop spends its remaining rounds learning nothing. Same
        # number of calls, different setting — the step-retry path has done
        # this since the beginning; the gate's rounds did not.
        reasoning = settings.executor_reasoning
        for _ in range(round_no - 1):
            reasoning = escalate(reasoning)
        trace.event(
            "final_fix_round", round=round_no, failures=len(gate_findings), reasoning=reasoning
        )
        before_fixes = workspace_fingerprint(ctx)
        apply_fixes(client, settings, gate_findings, ctx, trace, reasoning=reasoning)
        idle_round = workspace_fingerprint(ctx) == before_fixes

        check = run_final_check(ctx, settings, plan, trace, baseline, type_baseline=type_baseline)
        if check.ok:
            trace.say(f"final check passes after {round_no} fix round(s)")
            return check

        # A round that changed no file leaves the next one nothing new to act on: on 2-5 Oct
        # runs 0 of 59 fix rounds after such a round passed (DESIGN-LOG §124.2).
        if idle_round and settings.stop_after_idle_fix_round:
            trace.say("that fix round changed no file; stopping rather than running another")
            trace.event("final_fix_idle_stop", round=round_no)
            break

        current_failure_count = len(check.failing_tests)
        if current_failure_count > last_failure_count:
            trace.say(
                f"   ! that fix round made things worse: {last_failure_count} "
                f"failure(s) before, {current_failure_count} after"
            )
            trace.event(
                "final_fix_regressed",
                round=round_no,
                before=last_failure_count,
                after=current_failure_count,
            )
            regressed = True
        last_failure_count = current_failure_count

        signature = check.signature()
        if signature == last_signature:
            trace.say(
                "the same failures survived that round; stopping rather than "
                "repeating an attempt that changed nothing"
            )
            trace.event("final_fix_no_progress", round=round_no)
            break
        last_signature = signature

    trace.say("final check still failing — see the summary")
    return check


def _pending_suggestion(settings: Settings, ctx: ToolContext) -> str:
    """What the previous run on this workspace offered to do next, if anything.

    Only used to resolve a goal that is a bare assent ("yes"), and only after
    the person confirms it — see clarify.resolve_assent.
    """
    if not settings.memory or ctx.memory is None:
        return ""
    return ctx.memory.pending_suggestion()


def recall_memory(settings: Settings, ctx: ToolContext, trace: Trace) -> str:
    """The parts of this workspace's memory worth putting in the prompt."""
    if not settings.memory or ctx.memory is None:
        return ""
    block = ctx.memory.relevant(
        settings.goal, budget_tokens=settings.memory_token_budget, judge=ctx.relevance_judge
    )
    if block:
        trace.event("memory_recalled", chars=len(block))
        trace.say(f"recalled earlier work on this workspace ({ctx.memory.memory_path})")
    return block


def remember_run(client, settings: Settings, ctx: ToolContext, trace: Trace, summary: dict) -> None:
    """Record how this run went, and condense the file if it has grown large.

    Never raises: failing to write a note must not turn a finished run into a
    failed one, and a caller has no useful response to "the memory file could
    not be updated" anyway.
    """
    if not settings.memory or ctx.memory is None:
        return
    try:
        ctx.memory.record_run(settings.goal, summary)
        budget = settings.memory_token_budget * MEMORY_COMPACT_FACTOR
        if ctx.memory.oversized(budget) and not over_spend_cap(trace, settings.budget.max_usd):
            _compact_memory(client, settings, ctx, trace)
    except Exception as exc:
        trace.event("memory_write_failed", error=f"{type(exc).__name__}: {exc}")


def _compact_memory(client, settings: Settings, ctx: ToolContext, trace: Trace) -> None:
    """Condense everything older than the recent sections into one summary.

    One cheap call, the same shape as every other summarisation in the
    harness. A failure here leaves the file as it was — too long, but intact
    and readable — which is strictly better than a half-written one.
    """
    if ctx.memory is None:
        return
    sections = ctx.memory.sections(include_scratch=False)
    older = sections[:-MEMORY_KEEP_RECENT]
    if not older:
        return
    body = "\n\n".join(s.render() for s in older)
    try:
        reply = client.chat(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": COMPACT_MEMORY_SYSTEM},
                {"role": "user", "content": body[:40_000]},
            ],
            temperature=0.2,
            max_tokens=700,
            reasoning_effort="low",
        )
        trace.llm_call(
            "compact_memory", settings.planner_model, reply.usage, reply.finish_reason, reply=reply
        )
        condensed = reply.content.strip()
    except Exception as exc:
        trace.event("memory_compact_failed", error=f"{type(exc).__name__}: {exc}")
        return
    if not condensed:
        return
    ctx.memory.compact(MEMORY_KEEP_RECENT, condensed)
    trace.event("memory_compacted", condensed_sections=len(older))
    trace.say(f"condensed {len(older)} older memory entries")


# Suggestions that are the harness's own job, not a question for a person.
# The prompt forbids these in as many words; ten of seventeen traced runs took
# one anyway, so it is checked rather than only asked for. Matching is on the
# opening clause, because "run the tests" as the *subject* of the sentence is
# the chore, while "should pipes scroll faster once the score passes ten" is a
# real question that might mention tests later on.
_CHORE_VERBS = r"(?:re-?)?(?:run|execute|verify|check|confirm|ensure|validate|make sure)"
_CHORE_OPENERS = re.compile(
    r"^(?:"
    # "Should I run the tests?", "Do you want me to verify...?", "Shall we check...?"
    r"(?:should|shall|can|could|may)\s+(?:i|we)\s+"
    r"|(?:do|would)\s+you\s+(?:want|like)\s+(?:me|us)\s+to\s+"
    r"|(?:you\s+(?:should|could|might|may)\s+)?(?:now\s+|next,?\s+|then\s+|finally,?\s+)?"
    r")" + _CHORE_VERBS + r"\b",
    re.IGNORECASE,
)
_NOTHING = re.compile(r"^\s*(nothing obvious|none|n/?a)\b[.!]?\s*$", re.IGNORECASE)
# The model's "there is nothing to suggest", written out as prose.
_COMPLETE = re.compile(
    r"\b(?:task|goal|work)\b[^.?!]{0,60}\b(?:is|are|was|appears|seems|has been)\b"
    r"[^.?!]{0,30}\b(?:complete|completed|done|finished|achieved|fulfilled)\b"
    r"|\bno (?:further|additional|obvious) (?:action|steps?|work|changes?)\b",
    re.IGNORECASE,
)


def is_chore(suggestion: str) -> bool:
    """Whether a suggestion should be kept to the harness rather than offered.

    The rule is shape, because shape is what separates the two cases reliably.
    A *question* is a decision that belongs to the person: which of two designs,
    whether a feature is wanted, what a spec left open. A *statement* is an
    instruction — "Update Game.reset() to reset cloud_spawn_counter", "Fix the
    syntax error in tests/test_game.py" — and an instruction is work the run
    should have finished, not a choice to hand over. Offering it as the next
    goal is how it stays unfinished: across seventeen traced runs, ten goals
    were the previous run's suggestion, and the feature the person actually
    wanted went unbuilt for four of them.

    Withheld on top of that: the "this is finished" answers, which are prose
    for "nothing obvious" and are not suggestions at all, and questions that
    ask permission to do the harness's own job ("Should I run the tests?").
    """
    text = suggestion.strip().strip("*-• ").lstrip("1234567890. ")
    if not text or _NOTHING.match(text) or _COMPLETE.search(text):
        return True
    if "?" not in text:
        return True
    return bool(_CHORE_OPENERS.match(text))


def suggest_next_step(
    client, settings: Settings, plan: Plan, findings: list[dict], ctx: ToolContext, trace: Trace
) -> str:
    """One cheap, tool-free completion proposing a concrete next step.

    Never raises: a failure here must not turn a successful run into a
    failed one. The caller treats "" exactly like "the model had nothing
    to add" — there is no separate error path a caller needs to handle.
    """
    try:
        diff = ctx.baseline.run_diff(ctx.workspace.root, context=1)
        step_lines = (
            "\n".join(f"- [{s.id}] {s.summary}" for s in plan.steps if s.summary)
            or "(no step summaries recorded)"
        )
        finding_lines = "\n".join(f"- {f.get('issue', '')}" for f in findings) or "(none)"
        user = (
            f"Goal:\n{plan.goal}\n\nSteps completed:\n{step_lines}\n\n"
            f"Review findings:\n{finding_lines}\n\nDiff (may be truncated):\n{diff[:6000]}"
        )
        reply = client.chat(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": NEXT_STEP_SYSTEM},
                {"role": "user", "content": user},
            ],
            # Deliberately not settings.planner_reasoning: that defaults to
            # "high", and paying top-of-ladder reasoning for a two-sentence
            # suggestion nobody is blocked on is the most expensive way to
            # buy the least.
            #
            # 200 was not a budget, it was a bug: every one of 43 traced calls
            # came back at the cap, so the suggestion the TUI offers as the
            # next goal — pre-filled into the prompt box, ready to press Enter
            # on — was cut off mid-sentence every single time. The prompt still
            # asks for 1-3 sentences; this is the room to finish them, and the
            # model is not paid by the token it does not use.
            temperature=0.3,
            max_tokens=600,
            reasoning_effort="low",
        )
        trace.llm_call(
            "suggest", settings.planner_model, reply.usage, reply.finish_reason, reply=reply
        )
        suggestion = reply.content.strip()
        if is_chore(suggestion):
            trace.event("suggestion_withheld", text=suggestion[:300])
            return ""
        return suggestion
    except Exception as exc:
        trace.event("suggest_next_step_failed", error=f"{type(exc).__name__}: {exc}")
        return ""


# Phrases that mean "here is what I would do, may I do it" rather than an
# actual answer. QUICK_ANSWER_SYSTEM explicitly forbids this ("Do not
# describe a plan or what you would do; just answer the question") and tells
# the model to prefer NEEDS_PLAN when unsure — but two real runs against a
# live workspace ("are there obvious problems preventing the fixes from
# working?", "audit the traces... to determine why this keeps happening")
# got exactly this instead: a paragraph of hypotheses ending in "Should I
# proceed?", which the caller then treated as a finished answer and ended the
# run having read nothing and changed nothing. Nobody was there to say yes,
# so the run just stopped -- the harness's only signal that something needed
# investigating was a paragraph nobody could act on.
_HEDGE_PHRASES = (
    "should i proceed",
    "do you want me to",
    "would you like me to",
    "shall i proceed",
    "want me to proceed",
    "should i go ahead",
    "want me to go ahead",
    "let me know if you",
    "should i continue",
)


def _reads_as_a_hedge(text: str) -> bool:
    """True when a quick-answer reply is an offer to act, not an answer.

    Deliberately narrow -- keyed on specific permission-seeking phrasing, not
    "ends with a question mark" (a real answer can legitimately end by asking
    a genuine follow-up). A false negative here just means the old behaviour
    (the hedge is shown as the answer); a false positive would route a truly
    answerable question into a full plan for no reason, which is the
    asymmetry this stays narrow for.
    """
    lowered = text.lower()
    return any(phrase in lowered for phrase in _HEDGE_PHRASES)


# A goal that opens with one of these is an instruction to do work, whatever the model
# makes of it: SciCode goals opening "Write solution.py, a single Python module..." were
# answered with the code as a chat reply, and the runs ended with no file written (2 of 15;
# DESIGN-LOG §76). Checked before the model is asked, so these goals cost no quick-answer call.
_WORK_VERBS = re.compile(
    r"^\s*(?:please\s+)?(?:write|implement|add|create|build|fix|refactor|update|make|"
    r"rename|remove|delete|change|convert|port|migrate|replace)\b",
    re.IGNORECASE,
)


def is_plainly_work(goal: str) -> bool:
    return bool(_WORK_VERBS.match(goal or ""))


# A goal with no question mark that opens with none of these is a statement, and a
# statement about the workspace's own behaviour is a report to act on. Vague SWE-bench
# reports ("When I delete a record ... Django sends a separate delete") were answered as
# questions, wrongly ("This is actually already supported in Django!"), in 4 of 20 runs;
# in the real traces the only question-mark-free quick answers were a pasted traceback
# and the goal "q" (DESIGN-LOG §78). "when" is deliberately absent: it opens reports.
_ASK_OPENERS = re.compile(
    r"^\s*(?:please\s+)?(?:what|how|why|where|which|who|is|are|was|does|do|did|can|could|"
    r"should|would|will|explain|describe|summari[sz]e|list|show|tell|compare|walk)\b",
    re.IGNORECASE,
)


def reads_as_a_question(goal: str) -> bool:
    return "?" in (goal or "") or bool(_ASK_OPENERS.match(goal or ""))


def try_quick_answer(
    client, settings: Settings, trace: Trace, workspace_map: str = "", memory_context: str = ""
) -> str:
    """One cheap, tool-free completion tried before planning: for a goal
    that's really just a question, answer it directly and let the caller
    skip the whole plan/execute/review pipeline. Returns "" for anything
    the model itself flags as needing real work (NEEDS_PLAN) — the normal
    signal to fall through to the full pipeline — and also on any failure
    here, same as suggest_next_step: this must never be the reason a run
    that would otherwise have proceeded normally doesn't.

    `memory_context` matters for exactly the goal shape this call used to get
    wrong: a bare "yes" (or "sure"/"continue"/"do it") accepting the previous
    run's suggested next step. A traced run's goal was literally "yes", this
    call saw nothing else, and correctly had no way to tell a question from a
    request — it answered "How can I help you with this workspace?" and the
    run exited having changed nothing, silently discarding the suggestion
    "yes" was accepting. `_plan_until_accepted` already recalls memory before
    planning; the caller here reuses the very same lookup so this call sees
    it too, before it ever decides whether to skip planning at all.
    """
    # The workspace map goes in because without it every question *about* the
    # workspace was NEEDS_PLAN by definition, and the pipeline answered it the
    # only way it can: with a step summary. Asked which files were in the
    # folder, a traced run planned, executed, called list_dir, and
    # reported "Listed all files in the workspace root and tests/ subdirectory"
    # — the claim, without the list. A question about what is there is
    # answerable from the map, directly, for one cheap call.
    if is_plainly_work(settings.goal):
        trace.event("quick_answer_skipped", reason="goal opens with a work verb")
        return ""
    # A bare "yes" keeps its own route: it is resolved against memory here and in clarify.
    if not reads_as_a_question(settings.goal) and not is_assent(settings.goal):
        trace.event("quick_answer_skipped", reason="goal is a statement, not a question")
        return ""
    question = settings.goal
    if memory_context:
        question += f"\n\nRecalled context from earlier work on this workspace:\n{memory_context}"
    if workspace_map:
        question += f"\n\nWorkspace map:\n{workspace_map}"
    try:
        reply = client.chat(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": QUICK_ANSWER_SYSTEM},
                {"role": "user", "content": question},
            ],
            # "low", not settings.planner_reasoning: this call runs before
            # every interactive run, and deciding "question or work?" is not
            # a reasoning-heavy judgement. Paying the planner's default
            # "high" here taxed every run for a classification.
            temperature=0.2,
            max_tokens=600,
            reasoning_effort="low",
        )
        trace.llm_call(
            "quick_answer_check",
            settings.planner_model,
            reply.usage,
            reply.finish_reason,
            reply=reply,
        )
        text = reply.content.strip()
        if not text or text.startswith("NEEDS_PLAN"):
            return ""
        # An answer that ran into the token cap is not a quick answer. A
        # SWE-bench issue ("Migration crashes deleting an index_together...")
        # came back as 595 tokens of workarounds, cut off mid-code, and the run
        # ended there having changed nothing (DESIGN-LOG §69).
        if reply.finish_reason == "length":
            trace.event("quick_answer_truncated", text=text[:500])
            return ""
        if _reads_as_a_hedge(text):
            trace.event("quick_answer_hedged", text=text[:500])
            return ""
        return text
    except Exception as exc:
        trace.event("quick_answer_failed", error=f"{type(exc).__name__}: {exc}")
        return ""


def step_outcomes(plan: Plan | None) -> dict:
    """How the plan's steps actually turned out.

    Every ingredient of this was already recorded on the Step objects, but
    nothing aggregated it, so a run where three of four steps failed their own
    verify command produced exactly the same summary — and the same green
    "done." — as one where all four passed. Callers branch on `all_passed`.
    """
    if plan is None:
        return {"total": 0, "failed": [], "unverified": [], "abandoned": [], "all_passed": True}
    failed = [s.id for s in plan.steps if s.verify_passed is False]
    abandoned = [s.id for s in plan.steps if s.summary.startswith("abandoned:")]
    unverified = [
        s.id
        for s in plan.steps
        if s.verify_passed is None and not s.verified and s.id not in abandoned
    ]
    return {
        "total": len(plan.steps),
        "failed": failed,
        "unverified": unverified,
        "abandoned": abandoned,
        "all_passed": not failed and not abandoned,
    }


def _audit_tests(ctx: ToolContext, trace: Trace) -> dict:
    """Ask whether the finished suite can see this run's change (agent/testcheck.py).

    After the gate rather than inside it: the gate re-runs across fix rounds,
    and this is a report about the end state that should be computed once, on
    a suite already known to be green.
    """
    result = testcheck.audit(ctx.workspace.root, ctx.baseline, list(ctx.touched))
    payload = result.as_dict()
    trace.event("test_audit", **payload)
    if result.ran:
        trace.say(
            f"\ntest audit ({result.seconds:.0f}s): "
            + (f"{len(result.findings)} finding(s)" if result.findings else "clean")
        )
        for finding in result.findings:
            trace.say(f"   - {finding}")
    return payload


def _source_diff(ctx: ToolContext, max_lines: int = 80) -> str:
    """The run's diff for non-test Python only, bounded, for the test-repair round."""
    changed = [p for p in ctx.baseline.snapshots if p.endswith(".py") and not testcheck._is_test(p)]
    if not changed:
        return ""
    return ctx.baseline.run_diff(ctx.workspace.root, paths=changed, context=2, max_lines=max_lines)


def _repair_weak_tests(
    client,
    settings: Settings,
    plan: Plan,
    ctx: ToolContext,
    trace: Trace,
    findings: list[dict],
    baselines: Baselines,
    check: FinalCheck,
    audit_payload: dict,
) -> tuple[FinalCheck, dict]:
    """One fix round aimed at tests that cannot see the change, then re-check.

    The audit says the suite is green but would stay green if the code were
    wrong. That is a finding like any other, so it goes through the same fix
    machinery the gate uses; what differs is that there is exactly one round,
    because the audit is report-only evidence and not a gate that should be
    allowed to spend the run's budget chasing an equivalent mutant.

    The gate runs again afterwards, since a round that edits tests can break
    them, and the audit runs again so the summary reports the state the person
    is left with, not the state before the repair. What the first audit found
    is kept on the result.
    """
    fixable = testcheck.TestAudit(
        added_tests=audit_payload.get("added_tests") or [],
        added_tests_pass_on_old_code=audit_payload.get("added_tests_pass_on_old_code"),
        mutants_tried=audit_payload.get("mutants_tried", 0),
        survivors=audit_payload.get("survivors") or [],
        test_edits=audit_payload.get("test_edits") or {},
        killed_before=audit_payload.get("killed_before"),
        killed_after=audit_payload.get("killed_after"),
    ).fix_findings(_source_diff(ctx))
    if not fixable:
        return check, audit_payload
    for finding in fixable:
        finding.setdefault("round", "test-audit")
    findings.extend(fixable)
    trace.say(f"\ntest audit found {len(fixable)} weak spot(s); one fix round")
    trace.event("test_fix_round", findings=len(fixable))
    apply_fixes(client, settings, fixable, ctx, trace, reasoning=settings.executor_reasoning)
    repaired = _final_gate(
        client, settings, plan, ctx, trace, findings, baselines.tests, False, baselines.types
    )
    if repaired is None or not repaired.tests_passed:
        trace.event("test_fix_regressed")
        trace.say("   ! the test-repair round left the suite failing")
        return repaired or check, {**audit_payload, "repaired": False}
    after = _audit_tests(ctx, trace)
    after["repaired"] = True
    after["findings_before"] = audit_payload.get("findings") or []
    return repaired, after


def _finish(
    trace: Trace,
    plan: Plan | None,
    findings: list[dict],
    ctx: ToolContext,
    error: str = "",
    suggestion: str = "",
    goal: str = "",
    failure_log_dir: str | None = None,
    quick_answer: str = "",
    show_stats: bool = True,
    check: FinalCheck | None = None,
    client=None,
    settings: Settings | None = None,
    test_audit: dict | None = None,
    stopped_after: str = "",
) -> dict:
    # ctx is required, not optional: run() constructs it unconditionally before
    # any code path that can reach _finish, and no caller has ever omitted it.
    diff = ctx.baseline.run_diff(ctx.workspace.root)
    trace.event("run_diff", files=list(ctx.baseline.snapshots), diff=diff)
    trace.say("\nchanges this run:\n" + ctx.baseline.stat(ctx.workspace.root))
    if ctx.dependencies:
        trace.say("dependencies added: " + ", ".join(ctx.dependencies))

    outcomes = step_outcomes(plan)
    _narrate_outcomes(trace, outcomes, check)
    _narrate_cache_and_totals(trace, ctx.cache, show_stats)
    if suggestion:
        trace.say(f"\nworth deciding next: {suggestion}")
    trace.say(f"trace: {trace.path}", detail=True)

    # "ok" is the question a caller actually has — did this produce working
    # work — which neither `error` (harness failures only) nor the findings
    # list answers on its own.
    ok = not error and outcomes["all_passed"] and (check is None or check.ok)
    summary = {
        "ok": ok,
        "plan": {
            "goal": plan.goal,
            "draft": plan.draft,
            "findings": plan.findings,
            "steps": [s.__dict__ for s in plan.steps],
        }
        if plan
        else None,
        "findings": findings,
        "steps": outcomes,
        "final_check": check.as_dict() if check else None,
        "test_audit": test_audit,
        "test_changes": list(ctx.test_change_state["log"]),
        "totals": trace.summary(),
        "trace_path": str(trace.path),
        "changed_files": list(ctx.baseline.snapshots),
        "dependencies_added": list(ctx.dependencies),
        "questions_asked": ctx.interactor.asked,
        "suggested_next_step": suggestion,
        "quick_answer": quick_answer,
        "error": error,
        # --stop-after plan: the steps were planned, not run (DESIGN-LOG §92.3).
        "stopped_after": stopped_after,
    }
    # Written from the finished summary, so it can never disagree with it.
    # The frontends render this; the trace keeps a copy for anyone reading
    # the run back later.
    summary["session_summary"] = summary_module.render(goal, summary)
    trace.event("session_summary", text=summary["session_summary"])
    # A run that raised is not the only kind of failed run: one that
    # completes cleanly but fails its own verify/finalcheck gates sets `ok`
    # False with `error` left empty, and the CLI exits non-zero for it the
    # same as for a raised error (see cli.py: `not summary.get("ok", True)`)
    # -- so it needs a failure log for the same reason, not just the crash
    # case. `error` itself still means what it always has (a harness
    # exception), so a synthesized description stands in only when there
    # wasn't one to begin with.
    log_reason = error or (
        ""
        if ok
        else "run completed, but failed its own verify/finalcheck gates "
        "-- see the transcript below for what did not pass"
    )
    log_path = faillog.save(failure_log_dir, goal, log_reason, trace.path) if log_reason else None
    summary["failure_log_path"] = str(log_path) if log_path else ""
    if log_path:
        trace.say(f"failed run saved: {log_path}")
    # Written last, so it records the finished summary rather than a guess at
    # one — this is what the *next* run reads to know what happened here.
    try:
        if settings is not None:
            remember_run(client, settings, ctx, trace, summary)
    finally:
        # In a finally because these are the run's two real resources and
        # `remember_run` makes a model call, which can raise. The trace file
        # handle is held open for the whole run (it is appended to as events
        # happen), and the client holds a connection pool that nothing else
        # releases -- across a TUI session that is one set of idle sockets
        # per prompt, never reused, which is the shape of client a host's
        # connection limits exist to refuse. Both closes are idempotent.
        trace.close()
        if client is not None and hasattr(client, "close"):
            client.close()
    return summary


def _narrate_cache_and_totals(trace: Trace, cache, show_stats: bool) -> None:
    looked_up = cache.hits + cache.misses
    if looked_up:
        trace.event("cache_stats", hits=cache.hits, misses=cache.misses)
    if not show_stats:
        return
    if looked_up:
        trace.say(f"read cache: {cache.hits}/{looked_up} hits")
    trace.say("\n" + format_totals(trace.summary()))


def _narrate_outcomes(trace: Trace, outcomes: dict, check: FinalCheck | None) -> None:
    """Say plainly whether the work actually passed its own checks."""
    if outcomes["total"]:
        parts = [f"{outcomes['total']} step(s)"]
        if outcomes["failed"]:
            parts.append(
                f"{len(outcomes['failed'])} FAILED verification ({', '.join(outcomes['failed'])})"
            )
        if outcomes["abandoned"]:
            parts.append(
                f"{len(outcomes['abandoned'])} abandoned ({', '.join(outcomes['abandoned'])})"
            )
        if outcomes["unverified"]:
            parts.append(f"{len(outcomes['unverified'])} unverified")
        if len(parts) == 1:
            parts.append("all verified")
        trace.say("steps: " + ", ".join(parts))
    # Said whether or not the run passed: an unrunnable check is not a failing
    # check, but it is also not a passing one, and the human is the only one
    # who can do anything about it.
    if check is not None and check.env_failures:
        trace.say(
            f"could not verify {len(check.env_failures)} command(s) — the "
            f"toolchain, not the code: "
            + "; ".join(str(f.get("note", "")) for f in check.env_failures[:3])
        )
    if check is not None and not check.ok:
        trace.say("final check: FAILED — this work does not pass its own checks")


def format_totals(totals: dict) -> str:
    """The end-of-run cost/throughput block, as text.

    One implementation, because there were three: this prose block, the CLI's
    Rich table, and the TUI's status line each formatted the same numbers
    their own way, and the CLI's had already drifted into recomputing tok/s by
    hand instead of using the figure Totals already knows how to derive.
    """
    by_model = totals.get("by_model") or {}
    lines = ["=== totals ==="]
    lines.append(
        f"calls {totals.get('calls', 0)} | "
        f"in {totals.get('prompt_tokens', 0):,} tok | "
        f"out {totals.get('completion_tokens', 0):,} tok | "
        f"${totals.get('cost_usd', 0.0):.4f} | "
        f"{totals.get('latency_s', 0.0):.1f}s of model time"
    )
    for model, slot in by_model.items():
        line = (
            f"  {model}: {slot.get('calls', 0)} calls, "
            f"{slot.get('completion_tokens', 0):,} out tok, "
            f"{model_tok_per_s(slot):.0f} tok/s end-to-end"
        )
        if slot.get("decode_s", 0) > 0:
            line += (
                f", {trace_rates.decode_tok_per_s(slot):.0f} tok/s decoding, "
                f"{trace_rates.mean_ttft(slot):.2f}s to first token"
            )
        if slot.get("cached_tokens"):
            share = 100 * slot["cached_tokens"] / max(1, slot.get("prompt_tokens", 1))
            line += f", {slot['cached_tokens']:,} cached ({share:.0f}%)"
        lines.append(line + f", ${slot.get('cost_usd', 0.0):.4f}")
    return "\n".join(lines)


def model_tok_per_s(slot: dict) -> float:
    """End-to-end token rate for one model's slot. Kept as a name here because
    the CLI's stats table imports it from this module."""
    return trace_rates.tok_per_s(slot)


def _load_plan(path: str, settings: Settings) -> Plan:
    """Re-run a saved plan, so executor models can be compared on identical input."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return _plan_from_dict(data, settings)


def _load_plan_from_stdin(settings: Settings) -> Plan:
    """Load a plan from stdin, accepting JSON-serialized plan from another agent.

    Reads JSON from stdin until EOF. The JSON should contain a plan object
    (see agent/API.md for schema).
    """
    data = json.load(sys.stdin)
    return _plan_from_dict(data, settings)


def _plan_from_dict(data: dict, settings: Settings) -> Plan:
    """Parse a plan from a dict (file or stdin)."""
    body = data.get("plan", data)
    return Plan(
        goal=body.get("goal") or settings.goal,
        draft=body.get("draft", []),
        steps=[Step.from_dict(s, i) for i, s in enumerate(body.get("steps", []))],
    )
