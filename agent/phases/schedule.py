"""Deciding which plan steps can run at the same time.

The executor's wall time is dominated by per-call overhead, so the largest saving
available is not a faster model but overlapping steps that do not need each other.
Two steps may run together only when both conditions hold: neither depends on the
other transitively, and the files they declare are disjoint. A step that declares
no files is treated as touching everything, because nothing else can be proven.

Concurrency is off by default. Sequential execution is the safe reading of a plan,
and a plan whose `depends_on` is wrong will corrupt a run that trusts it.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import checkpoints
from ..guards import GuardTripped
from ..llm import BudgetExceeded
from ..tools import ToolContext
from ..trace import Trace
from .plan import Plan, Step


def _deps_met(step: Step, satisfied: set[str], known: set[str]) -> bool:
    """True when every dependency of `step` is either done or unknown.

    Unknown ids are ignored rather than treated as blocking: a planner that
    names a step that does not exist should not deadlock the run.
    """
    return all(dep in satisfied or dep not in known for dep in step.depends_on)


def build_waves(steps: list[Step], trace: Trace | None = None) -> list[list[Step]]:
    """Layer the steps so every step comes after everything it depends on.

    Unknown ids are ignored rather than treated as blocking: a planner that names
    a step that does not exist should not deadlock the run. A cycle collapses to
    the original order, which is always executable. Both cases are silent by
    design — a plan with a stray or circular `depends_on` should still run — but
    both are also always a planner mistake, so when `trace` is given they are
    reported rather than left for someone to notice the schedule looked off.
    """
    known = {step.id for step in steps}
    if trace is not None:
        _warn_unknown_deps(steps, known, trace)
    pending = list(steps)
    satisfied: set[str] = set()
    waves: list[list[Step]] = []

    while pending:
        ready = [s for s in pending if _deps_met(s, satisfied, known)]
        if not ready:
            # A dependency cycle. Fall back to running the rest in plan order.
            if trace is not None:
                cyclic = [s.id for s in pending]
                trace.event("schedule_cycle", steps=cyclic)
                trace.say(
                    f"   ! dependency cycle among steps {', '.join(cyclic)}; "
                    "running the rest in plan order instead of failing the run"
                )
            return [*waves, *([s] for s in pending)]
        waves.append(ready)
        satisfied.update(s.id for s in ready)
        # Identity, not value-equality: Step is a dataclass with mutable fields
        # (summary, attempts, ...) that default to the same value on every step
        # before execution starts, so `s not in ready` (structural `==`) risks
        # dropping a distinct step that happens to compare equal, rather than
        # the specific object just scheduled.
        scheduled = {id(s) for s in ready}
        pending = [s for s in pending if id(s) not in scheduled]
    return waves


def _warn_unknown_deps(steps: list[Step], known: set[str], trace: Trace) -> None:
    """Flag any `depends_on` naming a step id that is not in this plan at all.

    This is different from a step that was legitimately dropped earlier
    (reconnaissance pruning removes its id from every remaining step's
    `depends_on`, so the name is gone by the time this runs) — a survivor
    here means the planner typed or invented an id that never existed.
    """
    for s in steps:
        missing = [dep for dep in s.depends_on if dep not in known]
        if missing:
            trace.event("schedule_unknown_deps", step=s.id, missing=missing)
            trace.say(
                f"   ! [{s.id}] depends_on names unknown step(s) "
                f"{', '.join(missing)}; ignoring them"
            )


def _conflicts(step: Step, claimed_writes: set[str], claimed_reads: set[str]) -> bool:
    """Whether `step` can join a batch that already writes and reads those files.

    Only writes conflict. Two steps reading the same file do not interfere —
    reading game.py to write a test for it is the common shape of an audit —
    and treating every declared file as a conflict is what kept four steps of
    one traced plan strictly sequential when they shared only their reads.
    A write still conflicts with a sibling's read, because a reader is entitled
    to see the file as it was, not half-rewritten.
    """
    writes = step.written
    return bool(writes & (claimed_writes | claimed_reads)) or bool(step.touches & claimed_writes)


def partition_disjoint(wave: list[Step], limit: int) -> list[list[Step]]:
    """Split one wave into batches whose writes do not overlap each other."""
    batches: list[list[Step]] = []
    current: list[Step] = []
    writes: set[str] = set()
    reads: set[str] = set()

    for step in wave:
        # No declared files at all means no provable isolation, so it runs on
        # its own. This is different from an explicit empty `writes`, which is
        # a step promising to modify nothing and is safe beside anything.
        if not step.files:
            if current:
                batches.append(current)
                current, writes, reads = [], set(), set()
            batches.append([step])
            continue
        if current and _conflicts(step, writes, reads):
            batches.append(current)
            current, writes, reads = [], set(), set()
        current.append(step)
        writes |= step.written
        reads |= step.touches - step.written
        if len(current) >= max(1, limit):
            batches.append(current)
            current, writes, reads = [], set(), set()
    if current:
        batches.append(current)
    return batches


def plan_schedule(plan: Plan, concurrency: int, trace: Trace | None = None) -> list[list[Step]]:
    """The full execution order: a list of batches, each run together."""
    if concurrency <= 1:
        return [[step] for step in plan.steps]
    return [
        batch
        for wave in build_waves(plan.steps, trace)
        for batch in partition_disjoint(wave, concurrency)
    ]


StepRunner = Callable[..., Any]


def run_waves(
    client, settings, plan: Plan, ctx: ToolContext, trace: Trace, execute_step: StepRunner
) -> None:
    """Run the plan, overlapping steps where the plan says it is safe."""
    index_of = {step.id: i for i, step in enumerate(plan.steps)}
    batches = plan_schedule(plan, settings.step_concurrency, trace)
    concurrent_batches = sum(1 for b in batches if len(b) > 1)
    trace.event(
        "schedule",
        batches=len(batches),
        concurrent_batches=concurrent_batches,
        concurrency=settings.step_concurrency,
        order=[[s.id for s in batch] for batch in batches],
    )
    if concurrent_batches:
        trace.say(f"schedule: {len(batches)} batches, {concurrent_batches} run steps in parallel")

    done = 0
    for batch in batches:
        ctx.cancel.check()
        if len(batch) == 1:
            done = _run_single(
                batch[0],
                done,
                len(plan.steps),
                client,
                settings,
                plan,
                index_of,
                ctx,
                trace,
                execute_step,
            )
        else:
            done = _run_concurrent(
                batch,
                done,
                len(plan.steps),
                client,
                settings,
                plan,
                index_of,
                ctx,
                trace,
                execute_step,
            )


def _run_single(
    step: Step,
    done: int,
    total: int,
    client,
    settings,
    plan: Plan,
    index_of: dict[str, int],
    ctx: ToolContext,
    trace: Trace,
    execute_step: StepRunner,
) -> int:
    """One step, sequentially. Returns the updated `done` count."""
    done += 1
    trace.say(f"\n-- step {done}/{total}: [{step.id}] {step.title}")
    _guarded(execute_step, client, settings, plan, step, index_of[step.id], ctx, trace)
    _report(trace, step)
    _checkpoint(ctx, trace, step.id, [step.id])
    return done


def _run_concurrent(
    batch: list[Step],
    done: int,
    total: int,
    client,
    settings,
    plan: Plan,
    index_of: dict[str, int],
    ctx: ToolContext,
    trace: Trace,
    execute_step: StepRunner,
) -> int:
    """A batch of steps the plan says are safe to overlap. Returns the
    updated `done` count. Raises the first error a thread hit, once every
    thread has finished — a spend cap reached in one thread must still stop
    the run, and every step's own trace/report line is still worth having
    even when a sibling step failed."""
    names = ", ".join(s.id for s in batch)
    done += len(batch)
    trace.say(f"\n-- steps {done - len(batch) + 1}-{done}/{total} in parallel: {names}")
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
        futures = {
            pool.submit(
                _guarded,
                execute_step,
                client,
                settings,
                plan,
                step,
                index_of[step.id],
                ctx.for_step(writable=frozenset(step.written)),
                trace,
            ): step
            for step in batch
        }
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
    for step in batch:
        _report(trace, step)
    _checkpoint(ctx, trace, f"batch-{done}", [s.id for s in batch])
    for exc in errors:
        if isinstance(exc, BudgetExceeded):
            raise exc
    # A KeyboardInterrupt (or any other non-Exception) in one worker must not
    # hide behind an ordinary failure in an earlier one, or Ctrl-C during a
    # parallel batch reads as a step error (code review of 2026-09-30, finding 7).
    for exc in errors:
        if not isinstance(exc, Exception):
            raise exc
    if errors:
        raise errors[0]
    return done


def _guarded(
    execute_step: StepRunner,
    client,
    settings,
    plan: Plan,
    step: Step,
    index: int,
    ctx: ToolContext,
    trace: Trace,
) -> None:
    """A step that trips a resource guard fails that step, not the run."""
    try:
        execute_step(client, settings, plan, step, index, ctx, trace)
    except GuardTripped as exc:
        step.summary = f"abandoned: {exc}"
        trace.event("step_abandoned", step=step.id, reason=str(exc))


def _checkpoint(ctx: ToolContext, trace: Trace, label: str, step_ids: list[str]) -> None:
    """A best-effort git snapshot after a step (or batch) finishes.

    Never raises and never blocks the run on failure: see agent/checkpoints.py
    for why a checkpoint is a convenience, not something a run should depend on.
    """
    try:
        sha = checkpoints.create(
            ctx.workspace.root, trace.run_id, label, f"checkpoint after {', '.join(step_ids)}"
        )
    except Exception as exc:
        trace.event("checkpoint_error", steps=step_ids, error=str(exc))
        return
    if sha:
        trace.event(
            "checkpoint",
            steps=step_ids,
            sha=sha,
            ref=f"{checkpoints.REF_PREFIX}/{trace.run_id}/{label}",
        )


def _report(trace: Trace, step: Step) -> None:
    marks = []
    if step.verified:
        marks.append("verified")
    if step.verify_passed is True:
        marks.append("verify passed")
    elif step.verify_passed is False:
        marks.append("VERIFY FAILED")
    if step.attempts > 1:
        marks.append(f"{step.attempts} attempts")
    suffix = f"  [{', '.join(marks)}]" if marks else ""
    trace.say(f"   [{step.id}] {step.summary}{suffix}")
