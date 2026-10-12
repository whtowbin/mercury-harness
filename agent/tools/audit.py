"""Auditing this workspace's own earlier runs, as a callable tool.

A goal that says "the fix didn't work" or "still broken" is asking the agent
to know what a previous attempt actually did -- not guess at it from the
current file state. `recall` (notes.py) answers that from the harness's own
condensed memory.md; this tool goes one level deeper, straight at the raw
trace files, for two things memory.md doesn't carry: the tested deterministic
checks in tracetools.py (undeclared writes, verify commands that never run a
test, plan/goal coverage gaps) and, on request, the same model-in-the-loop
judgment pass `harness trace audit` runs from the CLI -- test-gaming, a
misread goal, a step marked verified that never finished.

The deterministic pass is always free. The judgment pass costs one real model
call per run examined, so it is opt-in (`deep=true`) and its cost is recorded
into *this* run's own trace via `ctx.trace.llm_call`, using the client this
run already built (`ctx.audit_client` -- see loop.py's `_prepare`), not a
second one spun up just for this call.
"""

from __future__ import annotations

from pathlib import Path

from .. import tracetools
from ..audit import (
    DEFAULT_AUDIT_MODEL,
    DEFAULT_REASONING_EFFORT,
    build_audit_prompt,
    parse_findings,
)
from .base import Tool, ToolContext, obj

MAX_PREVIOUS_RUNS = 8
DEFAULT_LIMIT = 3


def previous_trace_files(ctx: ToolContext, limit: int, query: str) -> list[Path]:
    """Earlier `run-*.jsonl` files next to this run's own trace, newest first.

    Reads the directory straight from `ctx.trace.path.parent` rather than
    from `Settings.trace_dir` -- that is a string a caller might not have
    resolved the same way twice, where the live `Trace` object already knows
    exactly where *this* run is writing, and prior runs against the same
    workspace write to the same place.
    """
    trace_dir = ctx.trace.path.parent
    if not trace_dir.is_dir():
        return []
    candidates = sorted(
        (p for p in trace_dir.glob("run-*.jsonl") if p != ctx.trace.path),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if query:
        needle = query.lower()
        goals = {row["trace"]: row.get("goal") or "" for row in tracetools.run_outcomes(trace_dir)}
        candidates = [p for p in candidates if needle in goals.get(p.name, "").lower()]
    capped = max(1, min(int(limit or DEFAULT_LIMIT), MAX_PREVIOUS_RUNS))
    return candidates[:capped]


def _deterministic_findings(events: list[dict], workspace_root: Path) -> list[str]:
    """Every tracetools structural check, each isolated so one failing check
    (an older trace format, a plan event tracetools doesn't recognise) can't
    take the rest down with it -- the same per-check resilience agent/audit.py
    already uses for whole traces."""
    findings: list[str] = []

    def _run(label: str, fn) -> None:
        try:
            result = fn()
        except Exception as exc:
            findings.append(f"{label}: check failed ({exc})")
            return
        if result:
            findings.append(f"{label}: {result}")

    _run(
        "undeclared writes (touched but not declared in any step's `writes`)",
        lambda: tracetools.undeclared_writes(events),
    )
    _run(
        "verify coverage gaps (lint-only or missing verification)",
        lambda: tracetools.verify_coverage_gaps(events),
    )
    _run("declared writes that were never touched", lambda: tracetools.declared_vs_actual(events))
    _run("plan/behaviour mismatches", lambda: tracetools.plan_conformance(events, workspace_root))
    _run(
        "goal requirements missing from the plan", lambda: tracetools.goal_requirement_gaps(events)
    )
    return findings


def _run_header(path: Path, events: list[dict]) -> tuple[str, str]:
    goal = next((e.get("goal") for e in events if e.get("kind") == "run_start"), "") or ""
    finals = [e for e in events if e.get("kind") == "final_check"]
    ok = finals[-1].get("ok") if finals else None
    status = "passed" if ok else "failed" if ok is False else "unknown"
    return goal, status


def _deep_audit_one(ctx: ToolContext, events: list[dict]) -> tuple[list[str], float]:
    """One model call over one run's events. Returns (report lines, cost_usd)."""
    if ctx.audit_client is None:
        return ["llm audit: unavailable (no model client wired into this run)"], 0.0
    try:
        prompt = build_audit_prompt(events, ctx.workspace.root)
        reply = ctx.audit_client.chat(
            model=DEFAULT_AUDIT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
    except Exception as exc:
        return [f"llm audit failed: {exc}"], 0.0

    ctx.trace.llm_call(
        "audit_previous_runs",
        DEFAULT_AUDIT_MODEL,
        reply.usage,
        finish_reason=reply.finish_reason,
        reply=reply,
    )
    found = parse_findings(reply.content)
    if not found:
        return ["llm audit findings: none"], reply.usage.cost_usd
    lines = ["llm audit findings:"]
    for f in found:
        category = f.get("category", "other")
        summary = f.get("summary", "")
        evidence = f.get("evidence", "")
        lines.append(f"  - [{category}] {summary}")
        if evidence:
            lines.append(f"    evidence: {evidence}")
        fix = f.get("proposed_fix", "")
        if fix:
            lines.append(f"    proposed fix: {fix}")
    return lines, reply.usage.cost_usd


def audit_previous_runs(
    ctx: ToolContext, deep: bool = False, limit: int = DEFAULT_LIMIT, query: str = ""
) -> str:
    query = (query or "").strip()
    files = previous_trace_files(ctx, limit, query)
    if not files:
        where = f" matching {query!r}" if query else ""
        return f"no previous runs found for this workspace{where} in {ctx.trace.path.parent}"

    sections: list[str] = []
    total_cost = 0.0
    for path in files:
        try:
            events = tracetools.load_trace(path)
        except Exception as exc:
            sections.append(f"## {path.name}\ncould not read trace: {exc}")
            continue

        goal, status = _run_header(path, events)
        lines = [f"## {path.name}  ({status})"]
        if goal:
            lines.append(f"goal: {goal[:200]}")

        det = _deterministic_findings(events, ctx.workspace.root)
        lines.append("deterministic findings:")
        if det:
            lines.extend(f"  - {d}" for d in det)
        else:
            lines.append("  - none")

        if deep:
            deep_lines, cost = _deep_audit_one(ctx, events)
            lines.extend(deep_lines)
            total_cost += cost

        sections.append("\n".join(lines))

    ctx.trace.event(
        "audit_previous_runs",
        runs=len(files),
        deep=deep,
        query=query,
        cost_usd=round(total_cost, 4),
    )
    report = "\n\n".join(sections)
    if deep and total_cost:
        report += f"\n\n(llm audit cost this call: ${total_cost:.4f})"
    return report


AUDIT_TOOLS = [
    Tool(
        "audit_previous_runs",
        "Look at this workspace's earlier harness runs to see what was tried, what broke, and "
        "why -- before redoing work someone already attempted, or repeating a failure someone "
        "already hit. Always runs free deterministic checks (undeclared file writes, verify "
        "commands that never actually run a test, plan steps that don't cover what the goal "
        "asked for). Set deep=true to also spend one model call per run reading the narration "
        "itself for the judgment calls a deterministic check can't make -- a misread goal, a "
        "test edited instead of the code, a step marked verified that never actually finished. "
        "Worth calling when the goal refers to something from before ('still broken', 'the fix "
        "didn't work', 'same bug'), or when a step keeps failing the same way and you want to "
        "know whether this happened last time too, and if so why.",
        obj(
            {
                "deep": {
                    "type": "boolean",
                    "description": "Also run an LLM judgment audit on each run examined. Costs "
                    "a small amount per run (a few tenths of a cent, GLM at low "
                    "reasoning). Default false: deterministic checks only, free.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many of the most recent previous runs to look at, up "
                    f"to {MAX_PREVIOUS_RUNS}. Default {DEFAULT_LIMIT}.",
                },
                "query": {
                    "type": "string",
                    "description": "Only look at previous runs whose goal mentions this "
                    "(case-insensitive substring match). Leave empty for the "
                    "most recent runs regardless of topic.",
                },
            }
        ),
        audit_previous_runs,
    ),
]
