"""Phase 3: an autoregressive model reviews large chunks, then the dLLM applies fixes."""

from __future__ import annotations

import re
import shlex

from .. import promptadapt
from .. import skills as skills_module
from ..autoformat import format_touched
from ..llm import BudgetExceeded, ChatClient, request_tokens
from ..prompts import FIXER_SYSTEM, REVIEWER_SYSTEM
from ..runner import run_tool_loop
from ..text import numbered
from ..tools import Registry, ToolContext, execution_registry, review_registry, with_edit_blocks
from ..trace import Trace
from .execute import premise_gate, with_justification
from .packing import LoadedFile, batch_to_budget, load_files
from .plan import Plan

# Whole files, not slices — the reviewer's advantage is seeing everything at once.
REVIEW_FILE_CHAR_CAP = 60_000
# _load_batches used to size a chunk against file bodies alone and know
# nothing about everything review() appends afterward -- the diff, the test
# suite output, the step report, the system prompt, the tool schemas. A real
# run (14 changed files, a large pytest failure dump, a real diff against
# baseline) sent a chunk request_tokens itself measured at ~153,216 -- already
# over the review budget _load_batches thought it was packing against -- and
# the provider then refused it for exceeding the model's actual 260,000-token
# limit. Two separate gaps, both handled here: the overhead this margin's
# denominator subtracts out before batching (the traced case alone was
# ~20,000 tokens of goal/step-report/suite/system/tools that batching never
# saw), and the fact that even a "corrected" char/4 estimate still isn't
# trustworthy at face value for dense, numeric, scientific Python -- the
# traced chunk's estimate was already inside a same-sized nominal budget and
# still got refused, meaning real tokenization ran well ahead of the
# estimate. Halving what's left after overhead is the cheap fix that doesn't
# require a real tokenizer; a target for a real tokenizer if this recurs.
REVIEW_BUDGET_SAFETY_MARGIN = 0.5


def _load_batches(ctx: ToolContext, files: list[str], budget_tokens: int) -> list[list[LoadedFile]]:
    """Group changed files into as few large chunks as the budget allows."""
    loaded, _ = load_files(ctx, files, REVIEW_FILE_CHAR_CAP)
    return batch_to_budget(loaded, budget_tokens)


# The whole command-evidence block, in characters. The newest outputs are kept
# when it is over: a number in a document is usually the last thing a run
# measured.
COMMAND_EVIDENCE_CHAR_CAP = 6_000


def _command_evidence(ctx: ToolContext) -> str:
    """The run's run_python / run_bash outputs that contain a number, for the
    reviewer's check that every measured figure in a changed document was
    actually produced by something this run ran."""
    entries = [e for e in ctx.command_log if re.search(r"\d", e["output"])]
    if not entries:
        return (
            "Commands run this run: none printed a number, so no measured figure in a "
            "changed document can be traced to this run.\n\n"
        )
    blocks: list[str] = []
    used = 0
    for entry in reversed(entries):
        block = f"$ {entry['tool']}: {entry['command']}\n{entry['output']}"
        if used + len(block) > COMMAND_EVIDENCE_CHAR_CAP:
            break
        blocks.append(block)
        used += len(block)
    dropped = len(entries) - len(blocks)
    note = f" ({dropped} older outputs omitted)" if dropped else ""
    return (
        f"Output of the commands run this run that printed numbers{note}:\n"
        + "\n---\n".join(reversed(blocks))
        + "\n\n"
    )


def _known_failures_note(already_failing: list[str] | None) -> str:
    """What was already red when the run started, for the reviewer's prompt.

    Cheap grounding: the reviewer is otherwise guessing at the suite's state,
    and a traced run had it return "pass" on work the deterministic gate then
    failed. Knowing which tests were broken going in tells it what to check
    was actually repaired, rather than assuming.
    """
    if not already_failing:
        return ""
    listed = ", ".join(already_failing[:10])
    return (
        f"Tests that were failing before this run started ({len(already_failing)}): "
        f"{listed}\nConfirm for yourself whether they pass now — do not assume "
        "either way.\n\n"
    )


def _suite_result(ctx: ToolContext, trace: Trace) -> str:
    """Run the suite once for the whole review, and say so in every chunk.

    "Run run_tests before you decide anything" was issued per chunk, and the
    reviewer's registry cannot edit anything — so with N chunks it mandated N
    runs of the same suite over a workspace that provably could not have
    changed between them. Traced: 99 of 662 runs made two or more `run_tests`
    calls during review, up to nine, and the cross-chunk repeats were
    invisible to every dedupe the harness has, because each chunk gets a
    fresh `seen_calls` counter.

    Running it here instead is the same grounding move `_known_failures_note`
    already makes: give the reviewer the fact rather than an instruction to
    go and get it. On a failure the output is handed over in full, since
    that is the thing most worth reviewing.
    """
    from ..tools.checks import run_tests

    output = run_tests(ctx)
    trace.say("   … ran the suite once for this review")
    return (
        "The test suite has already been run for you, just now, against "
        "exactly the code shown above. This is its output:\n\n"
        f"{output}\n\n"
        "Nothing has changed since, and you cannot change anything, so "
        "running it again would return this same text — spend your calls on "
        'reading code instead. A verdict of "pass" that contradicts the '
        "output above is a guess, and a deterministic check runs after you "
        "either way.\n\n"
    )


def _types_result(ctx: ToolContext) -> str:
    """ty's report on the type errors this run introduced, run once for the whole review, like
    the suite (Settings.review_presupplied_checks, DESIGN-LOG §108). "" where the final check's
    type gate does not apply, so the reviewer is never told about a check that did not run.

    The reviewer called check_code(types=true) in 66 of 138 recent runs, as the message's last
    line asked, on code that could not change while it looked."""
    from .finalcheck import _has_python, new_type_errors_text, type_gate_applies

    baseline = getattr(ctx, "type_baseline", None)
    if not _has_python(ctx) or not type_gate_applies(ctx.workspace.root, baseline):
        return ""
    found = new_type_errors_text(ctx) or "ty: no type errors this run introduced."
    return (
        "The type checker has already been run too, on the whole workspace, reporting only "
        f"what this run introduced:\n\n{found}\n\n"
    )


# The review message's last line. With the checks supplied, it no longer asks for one.
REVIEW_CLOSING = (
    "Read anything else you need, use check_code(types=true) to confirm what you "
    "suspect, then call submit_findings."
)
REVIEW_CLOSING_SUPPLIED = (
    "The suite and the type check above ran on exactly this code, and nothing can change "
    "while you review: read the code, then call submit_findings."
)
REVIEW_CLOSING_SUITE = (
    "The suite above ran on exactly this code, and nothing can change while you review: read "
    "the code, use check_code(types=true) to confirm what you suspect, then call "
    "submit_findings."
)


def _review_kit(settings) -> Registry:
    """The reviewer's tools. With review_presupplied_checks, without run_tests: it could only
    re-run the suite already in the message, on code it cannot change. 125 of about 420
    recent review turns were run_tests alone (§108)."""
    registry = review_registry()
    if not getattr(settings, "review_presupplied_checks", False):
        return registry
    return Registry([t for t in registry.tools() if t.name != "run_tests"])


def review(
    client: ChatClient,
    settings,
    plan: Plan,
    ctx: ToolContext,
    trace: Trace,
    already_failing: list[str] | None = None,
) -> list[dict]:
    files = list(ctx.touched)
    if not files:
        trace.say("nothing was changed; skipping review")
        return []

    registry = _review_kit(settings)
    supplied = getattr(settings, "review_presupplied_checks", False)
    types = _types_result(ctx) if supplied else ""
    suite = _suite_result(ctx, trace) + types
    system = promptadapt.reviewer(REVIEWER_SYSTEM, settings, types_supplied=bool(types))
    closing = (
        REVIEW_CLOSING
        if not supplied
        else REVIEW_CLOSING_SUPPLIED
        if types
        else REVIEW_CLOSING_SUITE
    )
    step_report = "\n".join(
        f"- [{s.id}] {s.title}\n    done when: {s.acceptance}\n    executor said: {s.summary}"
        + (f"\n    notes: {s.notes}" if s.notes else "")
        for s in plan.steps
    )
    known = _known_failures_note(already_failing)
    evidence = _command_evidence(ctx)
    # Everything review() sends besides the per-chunk diff and file listing --
    # measured once, for real, rather than guessed, so the budget batching
    # packs against reflects what a chunk actually costs on top of its files.
    overhead = request_tokens(
        system,
        f"Goal the work was supposed to achieve:\n{plan.goal}\n\n"
        f"Plan and what the executor reported:\n{step_report}\n\n"
        + (
            f"Dependencies added during the run: {', '.join(ctx.dependencies)}\n\n"
            if ctx.dependencies
            else ""
        )
        + known
        + evidence
        + suite,
        registry.schemas(),
    )
    file_budget = max(
        10_000, int((settings.budget.review_budget() - overhead) * REVIEW_BUDGET_SAFETY_MARGIN)
    )
    batches = _load_batches(ctx, files, file_budget)

    findings: list[dict] = []
    for n, batch in enumerate(batches, 1):
        listing = "\n\n".join(
            f"=== {item.rel} ===\n{item.body}"
            + ("\n... (file truncated for review)" if item.truncated else "")
            for item in batch
        )
        # The diff says what the run changed; the listing says what the code now is.
        # The reviewer needs both, and neither substitutes for the other.
        diff = ctx.baseline.run_diff(ctx.workspace.root, [item.rel for item in batch], context=3)
        user = (
            f"Goal the work was supposed to achieve:\n{plan.goal}\n\n"
            f"Plan and what the executor reported:\n{step_report}\n\n"
            + (
                f"Dependencies added during the run: {', '.join(ctx.dependencies)}\n\n"
                if ctx.dependencies
                else ""
            )
            + f"What changed in this chunk (diff against the pre-run state):\n{diff}\n\n"
            f"Current contents of the changed files ({n} of {len(batches)}): "
            f"{', '.join(item.rel for item in batch)}\n\n"
            f"{listing}\n\n" + known + evidence + suite + closing
        )
        trace.say(
            f"\n=== phase 3: review chunk {n}/{len(batches)} "
            f"({len(batch)} files, "
            f"~{request_tokens(system, user, registry.schemas())} tokens) ===",
            detail=True,
        )
        result, _ = run_tool_loop(
            client=client,
            model=settings.reviewer_model,
            phase=f"review:{n}",
            system=system,
            user=user,
            registry=registry,
            ctx=ctx,
            trace=trace,
            max_iters=10,
            reasoning_effort=settings.planner_reasoning,
            temperature=0.1,
            max_usd=settings.budget.max_usd,
        )
        if not result:
            trace.say("  reviewer returned nothing for this chunk")
            continue
        chunk = [f for f in result.get("findings", []) if isinstance(f, dict)]
        trace.event(
            "review_result",
            chunk=n,
            verdict=result.get("verdict"),
            summary=result.get("summary", ""),
            findings=chunk,
        )
        trace.say(f"  verdict: {result.get('verdict')} ({len(chunk)} finding(s))")
        findings.extend(chunk)
    return findings


def apply_fixes(
    client: ChatClient,
    settings,
    findings: list[dict],
    ctx: ToolContext,
    trace: Trace,
    reasoning: str | None = None,
) -> None:
    """Hand each finding back to the dLLM as a small, scoped edit task.

    `reasoning` overrides the executor's usual level. The final gate turns it
    up on each successive round: a fix that failed at low reasoning has
    already shown that level is not enough, and retrying it identically is
    how a bounded loop wastes its remaining rounds. Escalating costs nothing
    extra per call — it is the same number of round trips at a different
    setting — which is why the step-retry path has always done it and the
    fix rounds, until now, did not.
    """
    # Findings vary as much as steps do — one is about a git history, the next
    # about a package version — so each fix gets the kit its own text asks for.
    actionable = [f for f in findings if f.get("severity") in ("blocker", "major")]
    if not actionable:
        trace.say("no blocker or major findings to fix")
        return

    trace.say(
        f"\n=== phase 3b: fixing {len(actionable)} finding(s) on {settings.executor_model} ===",
        detail=True,
    )
    if getattr(settings, "combined_fixes", False) and len(actionable) >= 2:
        _apply_fixes_together(client, settings, actionable, ctx, trace, reasoning)
        return
    for i, finding in enumerate(actionable, 1):
        # The first finding was just seen failing; a later one may have been fixed by an
        # earlier fix in this same round.
        if i > 1 and _already_resolved(finding, ctx, trace):
            continue
        _apply_one_fix(client, settings, finding, i, len(actionable), ctx, trace, reasoning)


def _apply_fixes_together(
    client: ChatClient,
    settings,
    findings: list[dict],
    ctx: ToolContext,
    trace: Trace,
    reasoning: str | None,
) -> None:
    """Settings.combined_fixes: one fixer loop for every finding of a round.

    One loop per finding re-reads the same files and re-runs the same tests for each:
    calls grow with the findings (7.3 for one, 19.3 for two, 28.4 for three or more over all
    traces), and rounds with 2+ findings are 42% of fix calls (DESIGN-LOG §100). Together,
    the files are loaded once, the edits can go in one turn, and one check covers them.
    The budget is the per-fix one for up to two findings' worth, since the reading is shared.
    """
    total = len(findings)
    wanted = sorted({path for finding in findings for path in _related_files(finding)[2]})
    blocks = []
    for i, finding in enumerate(findings, 1):
        rel, related, _ = _related_files(finding)
        blocks.append(_finding_block(i, total, finding, rel, related))
    user = (
        "\n\n".join(blocks)
        + f"\n\n{_finding_listing(ctx, wanted)}\n\n"
        + f"Fix all {total} findings: make the edits together, check them once, then call "
        "finish_step."
    )
    issues = " ".join(f"{f.get('issue', '')} {f.get('fix', '')}" for f in findings)
    registry = execution_registry(
        f"{' '.join(wanted)} {issues}",
        ctx.language_profile,
        extra_tools=frozenset(settings.force_tools) or None,
    )
    if getattr(settings, "test_change_justification", False):
        registry = with_justification(registry)
    if getattr(settings, "edit_blocks", False):
        registry = with_edit_blocks(registry)
    trace.say(f"-- fixing {total} findings together: {', '.join(wanted)[:80]}")
    available = skills_module.for_project(ctx.skills, ctx.workspace.root)
    chosen = skills_module.select(available, "fix", issues, wanted, ctx.skill_token_budget)
    try:
        result, _ = run_tool_loop(
            client=client,
            model=settings.executor_model,
            phase="fix:all",
            system=promptadapt.fixer(FIXER_SYSTEM, settings)
            + promptadapt.skills(skills_module.render(chosen), settings, fixer=True),
            user=user,
            registry=registry,
            ctx=ctx,
            trace=trace,
            max_iters=settings.budget.max_iters_per_fix * min(total, 2),
            reasoning_effort=reasoning or settings.executor_reasoning,
            temperature=0.1,
            max_usd=settings.budget.max_usd,
            gate=premise_gate(settings, ctx),
            check_after_edit=getattr(settings, "check_after_edit", False),
            after_edit_check=_findings_check(findings),
        )
    except BudgetExceeded:
        trace.say("  ! spend cap reached during fixes")
        raise
    if getattr(settings, "autoformat", True):
        format_touched(ctx, trace, "fix:all")
    trace.event("fix_result", index=0, file=", ".join(wanted), findings=total, result=result)
    if result:
        trace.say(f"   {result.get('summary', '')}")


def _findings_check(findings: list[dict]):
    """The after-edit check for a fix loop: each finding's own `recheck`, so the result the
    fixer reads after an edit is about what it is fixing (DESIGN-LOG §103). None when no
    finding carries one: the loop then runs the step's tests as usual."""
    checks = [f["recheck"] for f in findings if f.get("recheck")]
    if not checks:
        return None

    def run(ctx: ToolContext) -> str:
        return "\n\n".join(_run_recheck(ctx, check) for check in checks)

    return run


def _run_recheck(ctx: ToolContext, recheck: dict) -> str:
    """One finding's check as text: a verify command, the failing tests, or the checker."""
    from ..tools.checks import check_code_readonly
    from .execute import run_verify

    if recheck.get("types") or recheck.get("code"):
        paths = list(recheck.get("types") or recheck.get("code") or [])
        return check_code_readonly(ctx, paths=paths, types=bool(recheck.get("types")))
    command = recheck.get("verify") or ""
    tests = recheck.get("tests") or []
    if not command and tests and not getattr(ctx, "test_command", ""):
        command = "pytest -q " + " ".join(shlex.quote(t) for t in tests)
    if not command:
        return ""
    passed, output = run_verify(ctx, command)
    return f"PASS — {command}" if passed else f"FAIL — {command}\n{output[-1_500:]}"


def _already_resolved(finding: dict, ctx: ToolContext, trace: Trace) -> bool:
    """Whether a final-check finding's own check passes now, so its fixer can be skipped.

    Only findings that carry an exact, cheap check (`recheck`: a step's verify command, or
    the failing test ids) are rechecked; a review finding or a type error always gets its
    fix. In the platformer app, fix 2 spent 9 calls re-running tests for a regression fix 1 had
    already resolved; 6% of fix calls over all traces went to fix rounds that changed
    nothing (DESIGN-LOG §98). A project with its own test command runs targets its own way,
    so its tests are not rechecked here.
    """
    recheck = finding.get("recheck") or {}
    command = recheck.get("verify") or ""
    tests = recheck.get("tests") or []
    if not command and tests and not getattr(ctx, "test_command", ""):
        command = "pytest -q " + " ".join(shlex.quote(t) for t in tests)
    if not command:
        return False
    from .execute import run_verify

    passed, _output = run_verify(ctx, command)
    if passed:
        issue = str(finding.get("issue", ""))
        trace.event("fix_skipped", issue=issue[:300], recheck=command)
        trace.say(f"   ✓ already resolved by an earlier fix, skipped: {issue[:80]}")
    return passed


def _related_files(finding: dict) -> tuple[str, list[str], list[str]]:
    """(the finding's own file, every other file it names, both together).

    The reviewer's whole advantage is seeing across files, so its most
    valuable findings — a signature that disagrees with its call sites — are
    exactly the ones that span more than one. Sending only finding["file"]
    left the fixer unable to see the other half of the defect it was asked to
    repair, so `files` (which submit_findings now accepts) is loaded too.
    """
    rel = str(finding.get("file", ""))
    related = [str(f) for f in (finding.get("files") or []) if str(f) != rel]
    wanted = [rel, *related] if rel else related
    return rel, related, wanted


def _finding_listing(ctx: ToolContext, wanted: list[str]) -> str:
    blocks = [
        f"=== {name} ===\n"
        + (numbered(text) if (text := ctx.try_read(name)) is not None else "(file not found)")
        for name in wanted[:6]
    ]
    return "\n\n".join(blocks) or "(no files could be read)"


def _finding_block(i: int, total: int, finding: dict, rel: str, related: list[str]) -> str:
    """One finding as the fixer reads it, without the files or the closing instruction."""
    return (
        f"Finding {i} of {total} — severity {finding.get('severity')}\n"
        f"File: {rel}"
        + (f" line {finding['line']}" if finding.get("line") else "")
        + "\n"
        + (f"Also involves: {', '.join(related)}\n" if related else "")
        + f"Issue: {finding.get('issue')}\n"
        f"Fails when: {finding.get('failure', '(not stated)')}\n"
        f"Suggested fix: {finding.get('fix')}"
    )


def _finding_prompt(
    i: int, total: int, finding: dict, rel: str, related: list[str], listing: str
) -> str:
    return (
        _finding_block(i, total, finding, rel, related)
        + f"\n\n{listing}\n\n"
        + "Apply this fix only, then call finish_step."
    )


def _apply_one_fix(
    client: ChatClient,
    settings,
    finding: dict,
    i: int,
    total: int,
    ctx: ToolContext,
    trace: Trace,
    reasoning: str | None = None,
) -> None:
    rel, related, wanted = _related_files(finding)
    listing = _finding_listing(ctx, wanted)
    user = _finding_prompt(i, total, finding, rel, related, listing)
    registry = execution_registry(
        f"{rel} {finding.get('issue', '')} {finding.get('fix', '')}",
        ctx.language_profile,
        extra_tools=frozenset(settings.force_tools) or None,
    )
    if getattr(settings, "test_change_justification", False):
        registry = with_justification(registry)
    if getattr(settings, "edit_blocks", False):
        registry = with_edit_blocks(registry)
    trace.say(f"-- fix {i}/{total}: {rel} — {finding.get('issue', '')[:80]}")
    try:
        available = skills_module.for_project(ctx.skills, ctx.workspace.root)
        chosen = skills_module.select(
            available, "fix", f"{rel} {finding.get('issue', '')}", wanted, ctx.skill_token_budget
        )
        result, _ = run_tool_loop(
            client=client,
            model=settings.executor_model,
            phase=f"fix:{i}",
            system=promptadapt.fixer(FIXER_SYSTEM, settings)
            + promptadapt.skills(skills_module.render(chosen), settings, fixer=True),
            user=user,
            registry=registry,
            ctx=ctx,
            trace=trace,
            max_iters=settings.budget.max_iters_per_fix,
            reasoning_effort=reasoning or settings.executor_reasoning,
            temperature=0.1,
            max_usd=settings.budget.max_usd,
            gate=premise_gate(settings, ctx),
            check_after_edit=getattr(settings, "check_after_edit", False),
            after_edit_check=_findings_check([finding]),
        )
    except BudgetExceeded:
        trace.say("  ! spend cap reached during fixes")
        raise
    if getattr(settings, "autoformat", True):
        format_touched(ctx, trace, f"fix:{i}")
    trace.event("fix_result", index=i, file=rel, result=result)
    if result:
        trace.say(f"   {result.get('summary', '')}")
