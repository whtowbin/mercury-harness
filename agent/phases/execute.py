"""Phase 2: the dLLM executes plan steps, each in a small context.

Three things happen here beyond running the loop once per step. A step's files are
preloaded rather than fetched, because a round trip costs about a second of queue
and prefill and a preloaded file costs nothing. A step's `verify` command is run
after it finishes, and a failure is handed back. And a step that fails is retried
once with the executor's reasoning turned up, which is cheaper than paying for high
reasoning on every step that was going to succeed anyway.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
import subprocess
import time
from collections import Counter
from collections.abc import Iterable
from functools import partial

from .. import envfail, promptadapt, toolchain, validate, verifycmd
from .. import skills as skills_module
from ..approval import Denied
from ..autoformat import format_touched
from ..editcalls import CHANGE_TOOLS, EDIT_TOOLS
from ..evidence import EvidenceGate
from ..llm import BudgetExceeded, ChatClient, approx_tokens, request_tokens
from ..numericcheck import EXECUTOR_NUMERIC_CHECK, NumericCheckGate
from ..projectconfig import uses_a_formatter
from ..prompts import (
    DRAFT_HANDOFF_CHARS,
    EXECUTOR_BATCH_LAST,
    EXECUTOR_BATCH_RULE,
    EXECUTOR_HYPOTHESIS,
    EXECUTOR_ORDER_LEGACY,
    EXECUTOR_ORDER_TDD,
    EXECUTOR_SYSTEM,
)
from ..research import findings_handoff
from ..runner import GateChain, PremiseGate, run_tool_loop
from ..steptail import TailItems, step_tail
from ..text import clip, numbered
from ..tools import (
    INTROSPECT_TOOLS,
    Registry,
    Tool,
    ToolContext,
    execution_registry,
    execution_tool_names,
    with_edit_blocks,
    workspace_manifest,
)
from ..tools.base import content_fingerprint
from ..trace import Trace
from . import neighbours as neighbours_module
from .packing import LoadedFile, fit_to_budget, load_files
from .plan import Plan, Step

# Per-file slice preloaded into a step. Anything bigger, the model reads on
# demand — which is the expensive path, so the cap is sized to hold a real
# source file rather than a small one.
#
# At 16,000 this became a cliff. A workspace that started at 950 lines grew to
# 1,732 (game.py 20KB, its tests 37KB), both files crossed the cap, and the
# model went from having them in the prompt to paging through them: 88 of 125
# read_file calls in a traced session carried an offset, most of them literally
# `offset=401` — read_file's default limit — and 45 of the 63 in execute were
# of files the step had *declared*, so the preload was meant to cover them.
#
# The trade is measured, not assumed. TTFT by prompt size across 944 traced
# calls: 1.22s at 10-15k tokens, 1.41s at 15-20k, 1.64s at 20-30k. So +15,000
# tokens of prefill costs about 0.4s, while one avoided round trip saves ~1.3s.
# And the file's content reaches the context either way — paging only decides
# whether it arrives in one prefill or over several round trips.
PRELOAD_CHAR_CAP = 48_000
# With `preload_neighbours`: how many importing files, and how much of each.
# Smaller than a declared file's slice because these are context, not the
# code being changed; a test file's top usually carries its fixtures and the
# first cases, which is what shows the expected behaviour.
MAX_NEIGHBOURS = 4
NEIGHBOUR_CHAR_CAP = 12_000
VERIFY_TIMEOUT_S = 120
# Ordered weakest to strongest; a failed step steps one rung up.
REASONING_LADDER = ["none", "low", "medium", "high"]


def escalate(level: str | None) -> str:
    """The next reasoning level up, or the top of the ladder."""
    if level not in REASONING_LADDER:
        return "medium"
    index = REASONING_LADDER.index(level)
    return REASONING_LADDER[min(index + 1, len(REASONING_LADDER) - 1)]


def _preload(
    ctx: ToolContext, files: list[str], budget_tokens: int, neighbours: bool = False
) -> str:
    """Inline the step's named files, smallest first, until the budget is spent.

    With `neighbours`, the files that import them (tests first, then callers)
    fill what the step's own files leave of the budget, as read-only context.
    """
    loaded, missing = load_files(ctx, files, PRELOAD_CHAR_CAP)
    included, skipped = fit_to_budget(loaded, budget_tokens, smallest_first=True)

    blocks: list[str] = []
    for item in included:
        tail = (
            "\n... (truncated; use read_file with an offset for the rest)" if item.truncated else ""
        )
        blocks.append(f"--- {item.rel} ---\n{item.body}{tail}")
    for item in skipped:
        blocks.append(f"--- {item.rel} (not preloaded; read_file it if you need it) ---")
    if missing:
        blocks.append("Files named by this step that do not exist yet: " + ", ".join(missing))
    _warn_if_short(ctx, included, skipped, budget_tokens)
    if neighbours:
        spent = sum(item.tokens for item in included)
        blocks.extend(_neighbour_blocks(ctx, files, budget_tokens - spent))
    return "\n\n".join(blocks) if blocks else "(no files preloaded)"


def _neighbour_blocks(ctx: ToolContext, files: list[str], budget_tokens: int) -> list[str]:
    """Preloaded blocks for the files importing `files`, within `budget_tokens`."""
    if budget_tokens <= 0:
        return []
    found = neighbours_module.importers(ctx.workspace.root, files)[:MAX_NEIGHBOURS]
    if not found:
        # Recorded all the same: how often the arm has nothing to add (31 of
        # 40 runs in §53.2, mostly because the plan already named the test) is
        # its reach, and a sweep cannot read that from events that never fire.
        ctx.trace.event("neighbours_preloaded", found=[], included=[], tokens=0)
        return []
    loaded, _ = load_files(ctx, found, NEIGHBOUR_CHAR_CAP)
    # Tests before callers, as `importers` ordered them: the test pins the
    # behaviour, which is the more useful of the two when only one fits.
    included, _ = fit_to_budget(loaded, budget_tokens, smallest_first=False)
    ctx.trace.event(
        "neighbours_preloaded",
        found=found,
        included=[item.rel for item in included],
        tokens=sum(item.tokens for item in included),
    )
    return [
        f"--- {item.rel} (imports this step's code; context only, not this step's to edit) ---"
        f"\n{item.body}" + ("\n... (truncated)" if item.truncated else "")
        for item in included
    ]


def _warn_if_short(
    ctx: ToolContext, included: list[LoadedFile], skipped: list[LoadedFile], budget_tokens: int
) -> None:
    """Say out loud when a step's own files did not fit.

    This is the cliff that cost a session. A workspace grew past the preload
    budget, its files stopped arriving in the prompt, and the model paged them
    back in one `read_file(offset=401)` at a time — 88 of 125 reads in one
    session — for a ~1.3s round trip each. Nothing in the trace said so; the
    runs simply got slower, and it read as a regression in something else.
    A budget quietly failing to hold its input should be an event, not a mood.
    """
    # Two different shortfalls with two different remedies, so they are
    # reported separately. Truncation is PRELOAD_CHAR_CAP: one file too big for
    # its own slice, which no budget flag can change. Skipping is the token
    # budget: the files together outgrew the step, which --step-budget and
    # MODEL_CONTEXT_TOKENS do fix. An earlier version conflated them and told
    # people to raise two knobs that could not affect the case they had hit.
    truncated = [item for item in included if item.truncated]
    if not (truncated or skipped):
        return
    ctx.trace.event(
        "preload_short",
        budget=budget_tokens,
        char_cap=PRELOAD_CHAR_CAP,
        truncated=[i.rel for i in truncated],
        skipped=[i.rel for i in skipped],
    )
    if truncated:
        ctx.trace.say(
            f"   ! {', '.join(i.rel for i in truncated)} is larger than the "
            f"{PRELOAD_CHAR_CAP:,}-character per-file preload cap and was cut short; "
            "the rest will be read back in at a round trip apiece "
            "(raise PRELOAD_CHAR_CAP)."
        )
    if skipped:
        # Real sizes, not the already-capped ones: a file cut to the char cap
        # reports the capped figure, which understates what was actually needed.
        wanted = sum(_real_tokens(ctx, item.rel) for item in [*included, *skipped])
        ctx.trace.say(
            f"   ! this step's files need about {wanted:,} tokens and the preload "
            f"budget is {budget_tokens:,}; {', '.join(i.rel for i in skipped)} was "
            "left out and will be read back in at a round trip apiece "
            "(raise --step-budget or MODEL_CONTEXT_TOKENS)."
        )


def _real_tokens(ctx: ToolContext, rel: str) -> int:
    """A file's full size in tokens, ignoring any preload cap applied to it."""
    text = ctx.try_read(rel)
    return approx_tokens(numbered(text)) if text else 0


def _step_prompt(
    ctx: ToolContext,
    plan: Plan,
    step: Step,
    index: int,
    budget_tokens: int,
    feedback: str = "",
    draft: str = "",
    neighbours: bool = False,
    verify_by_harness: bool = False,
) -> str:
    done = [s for s in plan.steps if s.summary and s.id != step.id]
    history = (
        "\n".join(
            f"- [{s.id}] {s.summary}" + (f" (note: {s.notes})" if s.notes else "") for s in done
        )
        or "(this is the first step)"
    )
    # Earlier steps may have installed things this step can now import.
    deps = "\nDependencies added so far: " + ", ".join(ctx.dependencies) if ctx.dependencies else ""
    # A one-line change summary is cheap and stops the executor redoing earlier work.
    changed = ctx.baseline.stat(ctx.workspace.root) if ctx.baseline.snapshots else ""
    changed = f"\n\nFiles changed so far this run:\n{changed}" if changed else ""
    # Reserve most of the budget for preloaded content. Every file that arrives
    # in the prompt is a round trip the executor does not have to spend, and
    # the rest of this prompt is a few hundred tokens of plan and history.
    preload = _preload(ctx, step.files, max(1_000, budget_tokens * 2 // 3), neighbours=neighbours)
    retry = f"\n\n*** THIS STEP ALREADY FAILED ONCE ***\n{feedback}\n" if feedback else ""
    return (
        f"Overall goal:\n{plan.goal}\n\n"
        f"Full plan (for orientation only — do NOT work ahead):\n{plan.outline()}\n\n"
        f"{draft}"
        f"Already completed:\n{history}{deps}{changed}\n\n"
        f"=== YOUR STEP {index + 1} of {len(plan.steps)}: [{step.id}] {step.title} ===\n"
        f"Intent: {step.intent}\n"
        f"Done when: {step.acceptance}\n"
        + (
            "Shapes to write (from the planner, which read the code):\n"
            + "".join(f"  - {sig}\n" for sig in step.signatures)
            if step.signatures
            else ""
        )
        + (
            "Tests this step should produce:\n" + "".join(f"  - {t}\n" for t in step.tests)
            if step.tests
            else ""
        )
        + _verify_line(step.verify, verify_by_harness)
        + f"Files named by this step: {', '.join(step.files) or '(none named)'}\n\n"
        f"Current contents:\n{preload}"
        f"{retry}\n\n"
        "Do this step now, then call finish_step."
    )


def _verify_line(verify: str, by_harness: bool) -> str:
    """The step's verify command, as the step message names it.

    "Verify with: <cmd>" read as an instruction: a code step whose verify was synthesized as
    `ruff check` ran it through run_bash (36 calls in 44 runs) and finished "verified, ruff
    passed". With the after-edit check on, the harness has already run the step's tests, and
    it runs the verify itself after finish_step, so the line says that (DESIGN-LOG §108).
    """
    if not verify:
        return ""
    if by_harness:
        return f"After finish_step the harness runs `{verify}` itself; you do not need to.\n"
    return f"Verify with: {verify}\n"


def _step_text(step: Step) -> str:
    """Everything the step says about itself, for tool and skill selection."""
    return " ".join([step.title, step.intent, step.acceptance, step.verify, " ".join(step.files)])


def _skills_for(ctx: ToolContext, phase: str, step: Step) -> str:
    """Reference material relevant to this step, appended to the system prompt.

    It goes in the prompt rather than behind a tool because prefill is cheap and
    round trips are not. Prompt caching means the stable part is paid for once.
    """
    if not ctx.skills:
        return ""
    if ctx.skill_triggers == "step":
        # Only what the step itself says and touches (DESIGN-LOG §117).
        text = " ".join([step.title, step.intent, step.acceptance, " ".join(step.files)])
        names: list[str] = []
    else:
        text = _step_text(step)
        names = (
            [p.name for p in ctx.workspace.root.iterdir()] if ctx.workspace.root.is_dir() else []
        )
    available = skills_module.for_project(ctx.skills, ctx.workspace.root)
    chosen = skills_module.select(
        available, phase, text, names + step.files, ctx.skill_token_budget
    )
    if chosen:
        ctx.trace.event(
            "skills_injected", step=step.id, phase=phase, names=[s.name for s in chosen]
        )
    # No catalogue of the other skills: it points at read_skill, which the executor and fixer
    # kits no longer carry (DESIGN-LOG §113).
    return skills_module.render(chosen)


# A verify command that is only `ruff check <files>`, however it is invoked. Anything
# chained, piped or redirected, or a `--fix`, runs as written.
_LINT_VERIFY = re.compile(
    r"^\s*(?:uvx\s+(?:--quiet\s+)?|uv\s+run\s+(?:--quiet\s+)?|python3?\s+-m\s+)?"
    r"ruff\s+check\b(?P<rest>[^|;&<>`$()]*)$"
)
_RUFF_VALUE_OPTIONS = frozenset(
    {
        "--select",
        "--ignore",
        "--extend-select",
        "--extend-ignore",
        "--line-length",
        "--target-version",
        "--config",
        "--output-format",
    }
)
_RUFF_REWRITES = frozenset({"--fix", "--unsafe-fixes", "--fix-only", "--add-noqa", "--watch"})


def lint_verify_parts(command: str) -> tuple[list[str], list[str]] | None:
    """(options, paths) when `command` is only a Ruff check of named files, else None."""
    match = _LINT_VERIFY.match(command or "")
    if not match:
        return None
    try:
        tokens = shlex.split(match.group("rest"))
    except ValueError:
        return None
    options: list[str] = []
    paths: list[str] = []
    expect_value = False
    for token in tokens:
        if expect_value:
            options.append(token)
            expect_value = False
        elif token.split("=", 1)[0] in _RUFF_REWRITES:
            return None
        elif token.startswith("-"):
            options.append(token)
            expect_value = token in _RUFF_VALUE_OPTIONS
        else:
            paths.append(token)
    paths = [p for p in paths if p not in (".", "./")]
    return (options, paths) if paths else None


def _new_lint_findings(ctx: ToolContext, options: list[str], paths: list[str]) -> list:
    """Ruff findings in `paths` that the files did not have before this run.

    Compared by rule and message, not line, since an edit moves lines. A file the
    run has not touched cannot have new findings. The same rule as the type
    baseline: debt a file arrived with is not this step's to pay.
    """
    from collections import Counter

    root = ctx.workspace.root
    new = []
    for path in paths:
        after = validate.ruff_check(root, [path], options=options)
        if path not in ctx.baseline.snapshots:
            continue
        original = ctx.baseline.snapshots[path]
        before = (
            validate.ruff_check_text(root, path, original, options) if original is not None else []
        )
        remaining = Counter((d.code, d.message) for d in before)
        for d in after:
            key = (d.code, d.message)
            if remaining[key]:
                remaining[key] -= 1
            else:
                new.append(d)
    return new


def _lint_verify(ctx: ToolContext, command: str) -> tuple[bool, str] | None:
    """A `ruff check <files>` verify judged on what this run introduced.

    Four django plans verified a step with `uvx ruff check <source file>`; the file
    arrived with dozens of findings under Ruff's defaults (UP031 on every %-format),
    so the step could only pass by restyling the whole file, and the rewrites broke
    passing tests (14 in one run). The one resolved task verified with its tests
    alone (DESIGN-LOG §74).
    """
    parts = lint_verify_parts(command)
    if parts is None:
        return None
    options, paths = parts
    new = _new_lint_findings(ctx, options, paths)
    serious = [d for d in new if validate.is_serious(d)]
    if serious:
        return False, "ruff: findings this run introduced:\n" + validate.render(serious)[:3_000]
    if new:
        # Style only (I001, UP031, ...): reported, not blocking. Since 25 Sept 41 runs failed
        # their final verify on lint-only findings and spent up to 970 fix calls restyling;
        # one django run rewrote %-formatting the file already used everywhere (§110).
        return True, (
            "ruff: style findings this run introduced (not blocking):\n"
            + validate.render(new)[:3_000]
        )
    return (
        True,
        "ruff: no findings this run introduced (anything already in these files is not counted)",
    )


# The prefix of run_verify's output when the approver refused the command. The final check
# reads it as "could not run", not "failed" (DESIGN-LOG §93).
VERIFY_NOT_APPROVED = "verification command was not approved: "
# The prefix of run_verify's output when the command ran no test at all (verifycmd): read as
# "could not verify", not as a defect a fix round could repair (DESIGN-LOG §110).
VERIFY_SELECTS_NOTHING = "the verify command selects no test: "


def unrunnable_note(output: str) -> str:
    """The note for a verify that could not check anything, from run_verify's output."""
    if output.startswith(VERIFY_NOT_APPROVED):
        return "needs approval, and this run is unattended"
    if output.startswith(VERIFY_SELECTS_NOTHING):
        return output.splitlines()[0]
    return ""


def run_verify(ctx: ToolContext, command: str) -> tuple[bool, str]:
    """Run a step's verify command. Returns (passed, output).

    During a concurrent batch (`ctx.writable` set), a verify command is just
    as capable of writing outside this step's declared files as run_bash is
    — `before_write`'s jail only guards edit_file/write_file, not a `pytest`
    invocation that happens to regenerate a sibling step's fixture or a
    `ruff --fix` that rewrites a file another step in the same batch is mid-
    edit on. Two steps' verify commands racing each other over the same file
    is exactly the hazard the scheduler's disjointness check exists to rule
    out, so any write outside the declared set is treated as a failure
    rather than silently accepted.
    """
    try:
        command = ctx.approver.check_verify(command)
    except Denied as exc:
        return False, f"{VERIFY_NOT_APPROVED}{exc}"
    exhausted = ctx.guard.check_subprocess_budget()
    if exhausted:
        return False, exhausted
    linted = _lint_verify(ctx, command)
    if linted is not None:
        return linted
    asked = command
    command = verifycmd.normalize(
        command, getattr(ctx, "test_command", ""), getattr(ctx, "test_label_root", "")
    )
    ran_as = f"(ran as: {command})\n" if command != asked else ""
    effective = int(min(VERIFY_TIMEOUT_S, max(5, ctx.guard.remaining_seconds())))
    jailed = ctx.writable is not None
    before = workspace_manifest(ctx.workspace.root) if jailed else None
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.workspace.root),
            capture_output=True,
            text=True,
            timeout=effective,
            env=toolchain.restricted_env(ctx.workspace.root),
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        ctx.guard.record_subprocess(elapsed)
        return False, f"verify command timed out after {effective}s"
    except OSError as exc:
        ctx.guard.record_subprocess(time.time() - started)
        return False, f"could not run the verify command: {exc}"
    ctx.guard.record_subprocess(time.time() - started)
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if before is not None:
        violation = _verify_jail_violation(ctx, before)
        if violation:
            return False, violation
    if proc.returncode == 0:
        return True, ran_as + output[-3_000:]
    nothing = verifycmd.selects_nothing(command, output)
    if nothing:
        return False, f"{VERIFY_SELECTS_NOTHING}{nothing}\n{ran_as}{output[-3_000:]}"
    return False, ran_as + output[-3_000:]


def _verify_jail_violation(ctx: ToolContext, before: dict[str, tuple[int, int]]) -> str:
    """None if a verify command run under a write jail stayed inside it,
    else a message naming what it touched outside its declared files."""
    after = workspace_manifest(ctx.workspace.root)
    changed = {p for p, stat in after.items() if before.get(p) != stat}
    changed |= before.keys() - after.keys()
    outside = sorted(p for p in changed if p not in (ctx.writable or frozenset()))
    if not outside:
        return ""
    return (
        "verify command modified files outside this step's declared scope while "
        f"running alongside another step: {outside}. Declare every file the verify "
        "command can change, or run it as a step that does not overlap with others."
    )


def workspace_fingerprint(ctx: ToolContext, paths: Iterable[str] | None = None) -> str:
    """A cheap content fingerprint of a set of files.

    Exists to answer one question cheaply: has anything changed since an
    earlier point in the run? Two fingerprints taken at different times being
    equal means the given path set and every byte of it are identical.

    `paths` defaults to every file this run has touched so far, which is what
    tells whether *anything* has changed since an earlier point — used to
    skip re-running a step's verify command when it already passed during
    execute and nothing has touched the workspace since: 296 traced step
    verifies were being run a second time against the finished state even
    when review found nothing to fix, which is most runs.

    Callers that need to know whether one *step's* dependencies changed,
    rather than the whole workspace's, pass that step's own frozen
    `verify_touched` snapshot instead — the touched set at large keeps
    growing as later steps run, so comparing against it would compare a
    step's old footprint to a larger, unrelated one.
    """
    return content_fingerprint(ctx, paths)


def _tdd_tests(settings, step: Step, attempt_no: int) -> list[str] | None:
    """The tests the red-first gate should hold a step's source edit for, or None
    when the gate does not apply: it is off, this is a retry (which follows a
    failure, so a test has by definition already run), or the plan named no test
    for the step to write."""
    if not getattr(settings, "tdd_gate", False) or attempt_no != 1 or not step.tests:
        return None
    return list(step.tests)


def _executor_system(settings, ctx: ToolContext | None = None) -> str:
    """The executor prompt, with the ordering guidance for the style in use.

    Swapped rather than appended, for the same reason as the planner's draft
    style: two paragraphs about the order to work in is the contradiction the
    `tdd_first` style exists to remove. `tdd_first` is the default since
    DESIGN-LOG §51; "legacy" is byte-for-byte the prompt every earlier sweep used.
    """
    system = EXECUTOR_SYSTEM
    if getattr(settings, "executor_order", "tdd_first") == "tdd_first":
        system = system.replace(EXECUTOR_ORDER_LEGACY, EXECUTOR_ORDER_TDD)
    if getattr(settings, "executor_hypothesis", False):
        system += EXECUTOR_HYPOTHESIS
    if getattr(settings, "numeric_check", False):
        system += EXECUTOR_NUMERIC_CHECK
    if getattr(settings, "executor_batch_last", False):
        system = system.replace(EXECUTOR_BATCH_RULE, "") + EXECUTOR_BATCH_LAST
    formatted = ctx is not None and uses_a_formatter(ctx.workspace.root)
    return promptadapt.executor(system, settings, formatted)


# The calls a model makes to act on a failure. `why` is offered on exactly
# these, so the note sits in the call that acts on the hypothesis it states.
WHY_TOOLS = EDIT_TOOLS - {"write_file"} | {"run_tests", "run_python"}
_WHY_SCHEMA = {
    "type": "string",
    "description": "After a failure: the failure you read and what this call changes "
    "about it, in 15 words at most. Leave empty otherwise.",
}


def _with_why(registry: Registry) -> Registry:
    """The same tools, with an optional `why` argument on WHY_TOOLS.

    The argument is logged (the tool_call event records every argument) and
    then dropped before the tool runs, so no tool needs to know about it.
    """

    def wrap(tool: Tool) -> Tool:
        if tool.name not in WHY_TOOLS:
            return tool
        inner = tool.fn

        def fn(ctx, why: str = "", **kwargs):
            return inner(ctx, **kwargs)

        params = dict(tool.parameters)
        params["properties"] = {**params.get("properties", {}), "why": _WHY_SCHEMA}
        return dataclasses.replace(tool, parameters=params, fn=fn)

    return Registry([wrap(t) for t in registry.tools()])


# The edits that can change a pre-existing test, and so may carry a reason.
JUSTIFICATION_TOOLS = CHANGE_TOOLS
_JUSTIFICATION_SCHEMA = {
    "type": "string",
    "description": "Only when this edit removes, weakens or rewrites a test that existed "
    "before this run: why. Quote the goal's own words asking for the behaviour to change, "
    "or say specifically what is wrong with the test. Leave empty otherwise.",
}


def with_justification(registry: Registry) -> Registry:
    """The same tools, with an optional `justification` on JUSTIFICATION_TOOLS.

    The gate reads it from the call before the tool runs (agent/testchange.py); the
    tool itself never sees it, as with `why` above.
    """

    def wrap(tool: Tool) -> Tool:
        if tool.name not in JUSTIFICATION_TOOLS:
            return tool
        inner = tool.fn

        def fn(ctx, justification: str = "", **kwargs):
            return inner(ctx, **kwargs)

        params = dict(tool.parameters)
        params["properties"] = {
            **params.get("properties", {}),
            "justification": _JUSTIFICATION_SCHEMA,
        }
        return dataclasses.replace(tool, parameters=params, fn=fn)

    return Registry([wrap(t) for t in registry.tools()])


def premise_gate(settings, ctx) -> PremiseGate | None:
    """The premise gate for this run's settings, or None when it is off."""
    if not getattr(settings, "premise_gate", False):
        return None
    return PremiseGate(
        ctx,
        goal=str(getattr(settings, "goal", "") or ""),
        justify=bool(getattr(settings, "test_change_justification", False)),
    )


def _step_gate(settings, ctx) -> PremiseGate | GateChain | None:
    """The gates on one step attempt: the premise gate, the numeric check and the bug-fix
    evidence check (DESIGN-LOG §120), each when on."""
    premise = premise_gate(settings, ctx)
    numeric = NumericCheckGate() if getattr(settings, "numeric_check", False) else None
    evidence = EvidenceGate(ctx) if getattr(settings, "bugfix_evidence", False) else None
    if numeric is None and evidence is None:
        return premise
    return GateChain(premise, numeric, evidence)


def _draft_handoff(plan: Plan) -> str:
    """The planner's draft notes, for an executor step's prompt."""
    notes = [str(note).strip() for note in plan.draft if str(note).strip()]
    if not notes:
        return ""
    text = "\n".join(f"- {note}" for note in notes)
    if len(text) > DRAFT_HANDOFF_CHARS:
        text = text[:DRAFT_HANDOFF_CHARS].rstrip() + " …"
    return f"Planner's notes (its reading of the request and the risks it named):\n{text}\n\n"


def _attempt(
    client: ChatClient,
    settings,
    plan: Plan,
    step: Step,
    index: int,
    ctx: ToolContext,
    trace: Trace,
    reasoning: str | None,
    feedback: str,
    attempt_no: int,
    seen_calls: Counter[str] | None = None,
) -> tuple[dict | None, list[dict]]:
    budget = settings.budget
    draft = _draft_handoff(plan) if getattr(settings, "executor_sees_draft", False) else ""
    # Research the planner did reaches every step whole, whether or not the draft does.
    draft = findings_handoff(plan.findings) + draft
    user = _step_prompt(
        ctx,
        plan,
        step,
        index,
        budget.step_budget(),
        feedback,
        draft,
        neighbours=getattr(settings, "preload_neighbours", False),
        verify_by_harness=getattr(settings, "check_after_edit", False),
    )
    system = _executor_system(settings, ctx) + promptadapt.skills(
        _skills_for(ctx, "execute", step), settings
    )
    registry = execution_registry(
        _step_text(step), ctx.language_profile, extra_tools=frozenset(settings.force_tools) or None
    )
    if getattr(settings, "research_first", False):
        # A step can check an API itself, offline and read-only. Research-first runs
        # only, so the default kit's schema (and its cached prefix) is unchanged.
        registry = Registry([*registry.tools(), *INTROSPECT_TOOLS])
    if getattr(settings, "executor_hypothesis", False):
        registry = _with_why(registry)
    if getattr(settings, "test_change_justification", False):
        registry = with_justification(registry)
    if getattr(settings, "edit_blocks", False):
        registry = with_edit_blocks(registry)
    # The whole request, not just the user half: see llm.request_tokens.
    trace.event(
        "step_attempt",
        step=step.id,
        attempt=attempt_no,
        reasoning=reasoning,
        tools=len(registry.names()),
        # Which trigger-gated tools this kit carried, so offered-vs-called is
        # a join on the trace rather than a recomputation from step text.
        optional=sorted(set(registry.names()) - execution_tool_names("")),
        prompt_tokens_est=request_tokens(system, user, registry.schemas()),
    )
    result, messages = run_tool_loop(
        client=client,
        model=settings.executor_model,
        phase=f"execute:{step.id}",
        system=system,
        user=user,
        registry=registry,
        ctx=ctx,
        trace=trace,
        max_iters=budget.max_iters_per_step,
        reasoning_effort=reasoning,
        temperature=0.1,
        max_usd=budget.max_usd,
        prompt_token_budget=budget.step_budget() * 3,
        check_after_edit=getattr(settings, "check_after_edit", False),
        types_after_edit=getattr(settings, "types_after_edit", False),
        seen_calls=seen_calls,
        salvage=True,
        tdd_tests=_tdd_tests(settings, step, attempt_no),
        gate=_step_gate(settings, ctx),
        tail=_tail(settings, step, plan, ctx),
    )
    if getattr(settings, "autoformat", True):
        format_touched(ctx, trace, f"execute:{step.id}")
    return result, messages


def _tail(settings, step: Step, plan: Plan | None = None, ctx: ToolContext | None = None):
    """The per-turn tail for this step (Settings.executor_tail_reminder), with the
    optional sections Settings.executor_tail_items names, or None."""
    if not getattr(settings, "executor_tail_reminder", False):
        return None
    wanted = set(getattr(settings, "executor_tail_items", ()) or ())
    items = TailItems(
        **{name: name in wanted for name in ("errors", "ledger", "api", "inspect", "key")}
    )
    facts = tuple(plan.findings) if plan is not None else ()
    file_text = _step_file_text(ctx, step) if (ctx is not None and items.inspect) else ""
    tail = partial(step_tail, step, items=items, facts=facts, file_text=file_text)
    return lambda *args: promptadapt.tail(tail(*args), settings)


def _step_file_text(ctx: ToolContext, step: Step) -> str:
    """The step's files as they stand when it starts, for the imports its new code uses."""
    parts: list[str] = []
    for name in step.files:
        path = ctx.workspace.root / name
        if path.is_file() and path.suffix == ".py":
            parts.append(path.read_text(errors="replace")[:20_000])
    return "\n".join(parts)


def _already_tried(messages: list[dict], limit: int = 14) -> str:
    """A compact list of the tool calls an attempt already made.

    A retry used to re-send the identical prompt with reasoning turned up,
    which meant a step that had spent nineteen calls reading the same files
    and running the same failing command simply did it all again. Telling the
    retry what has been tried is what makes it a second attempt rather than a
    louder copy of the first.
    """
    lines: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name", "?")
            args = str(function.get("arguments") or "")
            lines.append(f"- {name} {clip(args, 90)}")
    if not lines:
        return ""
    shown = lines[-limit:]
    return "\nWhat the previous attempt already did (do not simply repeat it):\n" + "\n".join(shown)


def _apply_result(step: Step, result: dict | None) -> str:
    """Update `step` from one attempt's result. Returns "" when the attempt
    looks like it succeeded (still subject to its own verify command),
    otherwise the problem text to feed the next retry."""
    if not result:
        step.summary = "executor never called finish_step; step may be incomplete"
        return "You never called finish_step, so the step was left unfinished."
    step.summary = str(result.get("summary", ""))
    step.verified = bool(result.get("verified"))
    step.notes = str(result.get("notes", ""))
    step.expected_result = str(result.get("expected_result", "")).strip()
    if not result.get("incomplete"):
        if step.verified and not step.expected_result:
            step.verified = False
            return (
                "You reported verified=true without saying what you expected your "
                "check to show. State that in expected_result — concretely, before "
                "looking back at output you have already seen — then call "
                "finish_step again."
            )
        return ""
    # A salvaged result is the model's last reasoning, not a finished step:
    # keep what it worked out, but do not let it pass as done.
    return (
        "The step ran out of iterations before calling finish_step. "
        f"Its own last analysis was:\n{step.summary}"
    )


def _unwritten_declared_files(ctx: ToolContext, step: Step) -> list[str]:
    """Files `step.writes` promised and this run never actually touched.

    `ctx.touched` accumulates for the whole run, not just this step — see the
    comment on `step.verify_touched` below — so membership in it does not by
    itself prove *this* step is the one that wrote a given path. But absence
    proves the opposite unconditionally: a path in `step.writes` that is not
    in `ctx.touched` by the time this step finishes was not written by this
    step, by any earlier step, or by anything else all run — there has been
    no chance for it to be. That is a real defect whatever the step's
    `verify` command says, because `writes` is not context the planner
    thought worth mentioning, it is the step's own declared job.

    This is what would have caught a traced step named "Verify README
    summary" that called `finish_step` on its very first tool call, claiming
    the README's summary sections were already complete, having never opened
    the file. Deliberately scoped to `step.writes` rather than `step.files`:
    `files` also holds paths a step only reads for context, and requiring
    those to change too would break every ordinary review-only step.
    """
    if not step.writes:
        return []
    return [f for f in step.writes if f not in ctx.touched]


def _undeclared_shadow_files(
    ctx: ToolContext, step: Step, touched_before: frozenset[str]
) -> list[str]:
    """New files this step created that it never declared in `step.writes`,
    whose name already exists somewhere else in the workspace.

    The counterpart to `_unwritten_declared_files`, found from the opposite
    real failure: a step declared `writes=[coin.py, sprites.py]`, and while
    chasing an import that did not resolve the way it expected mid-step,
    created six files instead of two — including a second `sprites.py` inside
    a `tests/` directory it invented on the spot, rather than fixing the
    import or inspecting the existing root-level file. That fork went
    undetected for every run afterward: fixes to the real `sprites.py` never
    touched the shadow copy, so the same "pygame font not initialized" defect
    kept reappearing across unrelated later runs, each one patching whichever
    copy the currently-failing test happened to import from. `pytest
    --collect-only` on that project now silently runs two divergent copies of
    nearly the same suite, 147 tests instead of 42, and nothing ever flagged
    the split.

    Deliberately narrow, unlike a blanket "flag every undeclared write": a
    step creating one small new helper file it did not think to declare is
    normal and not a problem on its own — `_unwritten_declared_files` above
    already covers the direction that actually matters (declared work not
    done). A *newly created* file whose basename already exists elsewhere in
    the tree is a different, much rarer, and far more diagnosable shape:
    there is essentially never a legitimate reason for a project to gain a
    second `sprites.py`. `__init__.py` is excluded outright — many files of
    that exact name, one per package, is completely ordinary.
    """
    new_this_step = [p for p in ctx.touched if p not in touched_before]
    if not new_this_step:
        return []
    declared = set(step.writes or [])
    manifest = workspace_manifest(ctx.workspace.root)
    by_basename: dict[str, list[str]] = {}
    for rel in manifest:
        by_basename.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)

    flagged = []
    for path in new_this_step:
        if path in declared or path.rsplit("/", 1)[-1] == "__init__.py":
            continue
        if path not in ctx.baseline.snapshots or ctx.baseline.snapshots[path] is not None:
            continue  # existed before this run touched it -- an edit, not a new file
        others = [p for p in by_basename.get(path.rsplit("/", 1)[-1], []) if p != path]
        if others:
            flagged.append(f"{path} (shadows existing {others[0]})")
    return flagged


def _type_check_step(ctx: ToolContext, trace: Trace, step: Step, settings) -> str:
    """Type-check what this step changed, against the rest of the project.

    A step that changes a function's shape breaks its callers immediately,
    and until now nothing said so until review — one whole phase later, after
    every other step had been written on top of it. The measurement is
    lopsided: across every traced run the reviewer called
    `check_code(types=true)` 486 times, and the executor 16, while calling
    ruff-only 80 times. The executor was checking style five times as often
    as it checked whether its own edit still fitted the code around it.

    `ty` is the right instrument and was already installed: it reads the
    whole project to resolve imports, so it sees a call site in a file this
    step never opened. That is precisely the "does my change still connect"
    question, answered statically, with no new dependency and no runtime
    schema — which is why this is a type check rather than a validation
    layer.

    Only diagnostics this run introduced count. `ctx.type_baseline` is the
    set that existed before the run started, for the same reason the test
    baseline exists: ty on a partly-typed project reports pre-existing debt,
    and failing a step for inherited debt would make the signal useless.
    """
    if not getattr(settings, "type_check_steps", True):
        return ""
    changed = [p for p in step.writes or step.files if p.endswith(".py")]
    changed = [p for p in changed if p in ctx.touched]
    if not changed:
        return ""
    baseline = getattr(ctx, "type_baseline", None)
    if baseline is None:
        return ""
    from ..phases.finalcheck import _diag_key, type_gate_applies

    # The same rule as the final check's (TYPE_DEBT_UNGATED): in a project that
    # does not type-check, "new" errors shift with every edit, and a django step
    # failed its verification on three of them (DESIGN-LOG §72).
    if not type_gate_applies(ctx.workspace.root, baseline):
        return ""
    found = validate.ty_check(ctx.workspace.root, changed)
    new = [d for d in found if _diag_key(d) not in baseline]
    if not new:
        return ""
    trace.event(
        "step_type_errors",
        step=step.id,
        count=len(new),
        files=changed,
        errors=[_diag_key(d) for d in new[:10]],
    )
    trace.say(f"   ✗ [{step.id}] {len(new)} new type error(s) in what this step changed")
    return (
        "This step introduced type errors that did not exist before the run. "
        "They are how a changed signature shows up at its call sites, so a "
        "caller elsewhere may now be broken:\n" + validate.render(new)[:2_000]
    )


def _already_proven_by_tests(ctx: ToolContext, command: str, settings) -> bool:
    """Whether the model already ran this exact check, and nothing has changed.

    A step's `verify` is usually the same pytest invocation the executor just
    ran on itself: across 960 traced step verifies, 667 followed a *passing*
    `run_tests` with no file-changing tool call in between, and 403 of those
    had run the whole suite. The second run is the same subprocess over
    byte-identical files, and on a real project that is twenty seconds each
    time.

    Three things must hold, and each rules out a way the two could differ.
    The recorded run must have passed with no `-k` narrowing, or it proves
    nothing about the tests it skipped. The workspace must be unchanged since
    — by stat manifest over every file, not just the touched ones, so a
    `run_python` write or a virtualenv repair invalidates it. And the verify
    must be a plain pytest command whose targets the recorded run covered:
    `_subsumed_by_suite` rejects shell chains and every flag that changes
    which tests run, so `pytest x && ruff check .` is never waved through on
    the strength of a pytest run.

    What this deliberately does not do is trust the model's report. The
    evidence is a command the harness itself ran and whose exit code it read.
    """
    from .finalcheck import _subsumed_by_suite

    if not getattr(settings, "skip_verified_by_tests", True):
        return False
    recorded = getattr(ctx, "tests_green", None)
    if not recorded or not _subsumed_by_suite(command):
        return False
    state, targets = recorded
    from ..tools.base import workspace_state

    if state != workspace_state(ctx.workspace.root):
        return False
    if "." in targets:
        return True
    # `.split("::", 1)[0]` before the `.py` check, not after: a node id like
    # `tests/test_b.py::test_x` does not itself end in `.py` (it ends in the
    # test name), so checking the whole argument first silently dropped
    # every node-id-suffixed file from `wanted` -- meaning a command mixing
    # a bare file with a node id, e.g. `pytest test_a.py test_b.py::test_x`,
    # could be waved through as "already proven" by a prior run that only
    # ever covered test_a.py, with test_b.py::test_x never actually checked.
    wanted = set()
    for arg in shlex.split(command)[1:]:
        if arg.startswith("-"):
            continue
        file_part = arg.split("::", 1)[0]
        if file_part.endswith(".py"):
            wanted.add(file_part)
    return bool(wanted) and wanted <= set(targets)


_EXPECTED_FAILURE_WORDS = (
    "fail",
    "error",
    "won't pass",
    "wont pass",
    "does not pass",
    "doesn't pass",
    "broken",
    "still missing",
    "non-zero",
    "nonzero",
    "exits 1",
    "exit 1",
    "red",
)


def _expected_pass(expected_result: str) -> bool | None:
    """Best-effort read of whether a step's stated `expected_result` was a
    pass or a failure. None when the field is empty -- a step with no verify
    command, or an executor that skipped it despite the prompt asking for it,
    has nothing to score.

    This is deliberately coarse: a keyword scan, not a semantic comparison
    against the verify command's actual output. Comparing free text against
    free text reliably is its own project, and a wrong mismatch flag would
    train distrust of a signal that is otherwise cheap and genuinely useful.
    So it is used for one thing only -- surfacing, in the trace and in the
    retry text, whenever the model's own prediction and the harness's
    independently-run result disagree on the one axis that is easy to score
    -- and never to gate a step on its own.
    """
    text = expected_result.lower().strip()
    if not text:
        return None
    return not any(word in text for word in _EXPECTED_FAILURE_WORDS)


def _verify_step(
    ctx: ToolContext,
    trace: Trace,
    step: Step,
    settings,
    touched_before: frozenset[str] = frozenset(),
) -> str:
    """Run the step's verify command, if it has one. Returns "" on a pass, on
    nothing to run, or on a failure that is not the step's fault; otherwise
    the problem text to feed the retry.

    The third case is the one worth spelling out. A verify command can fail
    because the code is wrong or because the virtualenv it runs in is broken,
    and telling the model the second is the first is how a correct step gets
    retried into a fourteen-iteration scope explosion (see envfail). So an
    environmental failure is repaired and re-run once, and if it is still
    environmental the step is left unverified rather than blamed.

    A missing declared write is checked independently of all that, and
    reported even when there is no `verify` command to fail: an empty
    `verify` was, before this check existed, silent permission to skip the
    step's own declared job entirely (see `_unwritten_declared_files`). A
    shadow file the step never declared — a duplicate module under a new
    name or directory — is checked the same way, for the same reason (see
    `_undeclared_shadow_files`).
    """
    unwritten = _unwritten_declared_files(ctx, step)
    shadows = _undeclared_shadow_files(ctx, step, touched_before)
    problems = []
    if unwritten:
        trace.event("step_writes_missing", step=step.id, files=unwritten)
        noun = "it" if len(unwritten) == 1 else "them"
        problems.append(
            f"This step declared it would write {', '.join(unwritten)}, but the "
            f"workspace shows no change to {noun}. Write {noun} before finishing."
        )
    if shadows:
        trace.event("step_shadow_files", step=step.id, files=shadows)
        problems.append(
            "This step created new file(s) that were not declared in `writes` and "
            f"duplicate an existing file elsewhere in the project: {'; '.join(shadows)}. "
            "This is almost always a mistake — check why the existing file wasn't used "
            "or edited instead, and delete the duplicate rather than keeping both."
        )
    writes_problem = "\n\n".join(problems)
    if not (step.verify and settings.verify_steps):
        return writes_problem
    if _already_proven_by_tests(ctx, step.verify, settings):
        step.verify_passed = True
        step.verify_touched = tuple(ctx.touched)
        step.verify_fingerprint = workspace_fingerprint(ctx, step.verify_touched)
        expected = _expected_pass(step.expected_result)
        trace.event(
            "step_verify",
            step=step.id,
            command=step.verify,
            passed=True,
            skipped_already_run=True,
            expected_result=step.expected_result,
            expectation_mismatch=(expected is not None and expected is not True),
        )
        trace.say(f"   ✓ [{step.id}] {step.verify} (already run by this step)")
        type_problem = _type_check_step(ctx, trace, step, settings)
        return "\n\n".join(p for p in (writes_problem, type_problem) if p)
    passed, output = run_verify(ctx, step.verify)
    env_note = "" if passed else unrunnable_note(output)
    if not passed and not env_note:
        passed, output, env_note = envfail.repair_and_retry(
            output, ctx.workspace.root, lambda: run_verify(ctx, step.verify)
        )
    if env_note:
        step.verify_passed = None
        step.verify_error = env_note
        trace.event(
            "step_verify",
            step=step.id,
            command=step.verify,
            passed=False,
            environmental=True,
            note=env_note,
            output=output[-600:],
        )
        trace.say(f"   ~ [{step.id}] could not verify — {env_note}")
        return writes_problem
    step.verify_passed = passed
    if passed:
        # Frozen now, not read later: `ctx.touched` keeps growing as later
        # steps run, so a fingerprint taken against it at final-check time
        # would compare this step's old footprint to a bigger, unrelated
        # one and never match. Freezing the paths here is what lets a later
        # step editing one of them be caught instead of silently trusted.
        step.verify_touched = tuple(ctx.touched)
        step.verify_fingerprint = workspace_fingerprint(ctx, step.verify_touched)
    else:
        step.verify_fingerprint = ""
        step.verify_touched = ()
    expected = _expected_pass(step.expected_result)
    mismatch = expected is not None and expected is not passed
    trace.event(
        "step_verify",
        step=step.id,
        command=step.verify,
        passed=passed,
        output=output[-600:],
        expected_result=step.expected_result,
        expectation_mismatch=mismatch,
    )
    verify_problem = (
        "" if passed else f"The step's verify command failed.\n$ {step.verify}\n{output}"
    )
    if verify_problem and mismatch:
        verify_problem += f'\n\nNote: you expected "{step.expected_result}".'
    # Only when the command itself passed: a step whose tests are red has a
    # more actionable problem than its type errors, and reporting both sends
    # the retry after two things at once.
    type_problem = "" if verify_problem else _type_check_step(ctx, trace, step, settings)
    parts = [p for p in (writes_problem, verify_problem, type_problem) if p]
    return "\n\n".join(parts)


def _rescued_by_verify(
    ctx: ToolContext,
    trace: Trace,
    step: Step,
    settings,
    result: dict | None,
    touched_before: frozenset[str] = frozenset(),
) -> bool:
    """Whether a step that never reported success can be shown to have succeeded.

    Running out of tool-call iterations is not evidence that the work is
    wrong; it is evidence that the model stopped *narrating*. One traced run
    made this vivid: `add-integration-test` wrote its test, watched it pass,
    then spent its last three iterations tidying import lines and hit the
    cap. The harness declared the step failed — and ninety seconds later the
    final gate ran that same step's verify command, `uv run pytest
    test_eval_and_compare.py::TestMain -v`, which passed on the first try.
    The evidence needed to clear the step existed the whole time and was not
    consulted, because a non-empty `problem` short-circuited the check.

    That ordering had the priority backwards. A verify command is
    *independent* of the model's self-report, which is exactly why it is
    worth the most in the case where the self-report is missing or is a
    salvaged fragment of reasoning. So it is consulted here too.

    It clears the step only on `verify_passed is True` — an actual command
    that actually ran and actually passed. A step with no verify command,
    or one whose command could not run (a broken virtualenv, which
    `_verify_step` reports as unverified rather than failed), is *not*
    rescued: treating "nothing contradicted it" as "it worked" is precisely
    the hole `_unwritten_declared_files` was built to close, and this must
    not reopen it from the other side.
    """
    salvaged = bool(result and result.get("incomplete"))
    # Only the ran-out-of-iterations case. `_apply_result` also reports a
    # problem when the executor produced no terminal call and nothing
    # salvageable at all, and that is a different situation entirely: there
    # is no evidence the model did anything. A step whose verify command was
    # already green before the run started — "add a case to test_x.py", where
    # test_x.py exists and passes — would otherwise be rescued on the
    # strength of a test it never touched, marked verified, and returned
    # before the escalate-and-retry that would have done the work.
    if not salvaged:
        return False
    if not (settings.verify_steps and step.verify.strip()):
        return False
    if _verify_step(ctx, trace, step, settings, touched_before) or step.verify_passed is not True:
        return False
    step.verified = True
    step.notes = (step.notes + " | " if step.notes else "") + (
        "ran out of iterations, but its verify command passed against the finished workspace"
    )
    # The salvaged summary is the model's last reasoning, not a report of
    # what it built (see runner._salvage). Leaving it as the step's summary
    # is how a mid-thought narration ends up presented as a deliverable and
    # written into the workspace's memory file as the record of this step.
    step.summary = (
        f"Completed; confirmed by its verify command "
        f"({step.verify}). The executor ran out of tool-call "
        f"iterations before reporting, so this is the command's "
        f"result rather than its own summary."
    )
    trace.event("step_rescued_by_verify", step=step.id, command=step.verify, attempts=step.attempts)
    trace.say(f"   ✓ [{step.id}] ran out of iterations, but its verify command passed")
    return True


def execute_step(
    client: ChatClient, settings, plan: Plan, step: Step, index: int, ctx: ToolContext, trace: Trace
) -> None:
    """Run one step, verify it, and retry once at higher reasoning if it failed."""
    reasoning = settings.executor_reasoning
    feedback = ""
    # Shared across this step's attempts, not per loop: three identical calls
    # in attempt one plus three more in attempt two is a model stuck on one
    # command, and counting them separately let it stay stuck.
    seen_calls: Counter[str] = Counter()
    # `ctx.touched` accumulates for the whole run, not just this step (see
    # `_unwritten_declared_files`), so isolating what *this* step created
    # needs a snapshot taken before its first attempt — used by
    # `_undeclared_shadow_files` to catch a step that fabricates a duplicate
    # module instead of using or fixing the existing one.
    touched_before = frozenset(ctx.touched)

    for attempt in range(max(1, settings.max_step_attempts)):
        started = time.time()
        result, messages = _attempt(
            client,
            settings,
            plan,
            step,
            index,
            ctx,
            trace,
            reasoning,
            feedback,
            attempt_no=attempt + 1,
            seen_calls=seen_calls,
        )
        step.attempts = attempt + 1
        step.seconds = time.time() - started

        problem = _apply_result(step, result)
        # The verify command is the only independent evidence the step
        # worked, so it runs whenever the result itself raised no problem —
        # and, per _rescued_by_verify below, also in the one case where the
        # result is missing precisely because the model stopped reporting.
        if not problem:
            problem = _verify_step(ctx, trace, step, settings, touched_before)
        elif _rescued_by_verify(ctx, trace, step, settings, result, touched_before):
            return
        if not problem:
            return

        step.notes = (step.notes + " | " if step.notes else "") + "attempt failed"
        last_attempt = attempt + 1 >= max(1, settings.max_step_attempts)
        if last_attempt or not settings.escalate_on_failure:
            trace.event("step_failed", step=step.id, attempts=step.attempts, problem=problem[:600])
            trace.say(f"   ! step did not verify: {problem.splitlines()[0][:100]}")
            return

        feedback = problem + _already_tried(messages)
        raised = escalate(reasoning)
        # `executor_reasoning` defaults to "high", the top of the ladder, so on
        # a default run escalate() returns what it was given and the retry is
        # at identical settings. Saying "retrying at reasoning=high" there
        # claimed an escalation that never happened. The retry is still worth
        # making — `feedback` hands the next attempt the failure and what was
        # already tried, which the first attempt did not have — so what is
        # narrated is that, and the reasoning bump only when it is real.
        escalated = raised != reasoning
        reasoning = raised
        trace.say(
            f"   retrying at reasoning={reasoning}"
            if escalated
            else "   retrying, with the failure fed back"
        )
        trace.event("step_retry", step=step.id, reasoning=reasoning, escalated=escalated)


def execute_plan(client: ChatClient, settings, plan: Plan, ctx: ToolContext, trace: Trace) -> None:
    from .schedule import run_waves

    trace.say(
        f"\n=== phase 2: execute {len(plan.steps)} steps on {settings.executor_model} ===",
        detail=True,
    )
    try:
        run_waves(client, settings, plan, ctx, trace, execute_step)
    except BudgetExceeded:
        trace.say("  ! spend cap reached, stopping execution")
        raise
