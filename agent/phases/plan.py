"""Phase 1: an autoregressive model drafts the plan with chain-of-draft prompting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from .. import (
    chatflow,
    dataformats,
    envfail,
    issue_symbols,
    langtools,
    outline,
    promptadapt,
    symptom,
    verifycmd,
)
from .. import skills as skills_module
from ..callsites import CallSiteGate
from ..langtools import LanguageProfile
from ..llm import ChatClient
from ..prompts import (
    DRAFT_HANDOFF_PROSE,
    DRAFT_HANDOFF_TERSE,
    DRAFT_PROSE,
    DRAFT_PROSE_REWORDINGS,
    DRAFT_TERSE,
    PLAN_PROBE,
    PLAN_REPRO,
    PLANNER_SYSTEM,
    SYMPTOM_FIRST_PLANNING,
)
from ..research import (
    FINDINGS_LAST,
    DecisionsGate,
    complete_findings,
    looked_up_signatures,
    with_findings,
    with_required_findings,
)
from ..runner import GateChain, ProbeGate, ReproGate, run_tool_loop
from ..steptail import planner_tail, with_key_fact
from ..tools import Registry, ToolContext, planning_registry
from ..tools.checks import run_in_scratch
from ..tools.control import with_reproduce, with_task_kind
from ..trace import Trace


@dataclass
class Step:
    id: str
    title: str
    intent: str
    acceptance: str
    # What the planner worked out while reading the code, carried forward so
    # the executor does not rediscover it: the shapes it should write, and the
    # tests that would prove the step done.
    signatures: list[str] = field(default_factory=list)
    # Settings.plan_key_facts: the one fact this step must not get wrong, repeated
    # in the executor's tail (DESIGN-LOG §84).
    key_fact: str = ""
    tests: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    # The subset of `files` this step may modify. Empty means "not declared",
    # which is read as "all of them" — the conservative reading, and what every
    # plan written before this field existed meant.
    writes: list[str] = field(default_factory=list)
    verify: str = ""
    # Ids of steps that must finish first. Empty means it depends on nothing.
    depends_on: list[str] = field(default_factory=list)
    # Filled in by the execute phase.
    summary: str = ""
    verified: bool = False
    # What the executor said, when it called finish_step, that its verify
    # command (or whatever it ran in lieu of one) would show — written down
    # before the harness's own check ran or the model looked back at output
    # it had already seen. Compared against the real outcome in
    # `_verify_step` rather than trusted; the point is not to grade the
    # prediction but to force it, since a self-report made before the
    # result is visible cannot be rationalized to fit it afterward the way
    # a bare `verified: true` can.
    expected_result: str = ""
    # None when the step declared no verify command, or it was not run, or
    # it could not be run because the toolchain itself was broken.
    verify_passed: bool | None = None
    # Set when the verify command failed for a reason that is not the step's
    # doing — a broken virtualenv, a missing tool. Kept separate from a
    # failure so it is reported to the human without ever being fed back to
    # the model as "your step failed".
    verify_error: str = ""
    notes: str = ""
    attempts: int = 0
    seconds: float = 0.0
    # A content fingerprint of `verify_touched`, taken at the moment this
    # step's own verify command last passed. Lets the final-check gate tell
    # "nothing this step depended on has changed since it passed" apart from
    # "nothing has changed since execute finished" — a later step editing a
    # file this one already depended on is the former without being the
    # latter, and used to slip past the gate unnoticed. Empty when the step
    # has never had a passing verify fingerprinted.
    verify_fingerprint: str = ""
    # The workspace's touched-file set at the moment `verify_fingerprint` was
    # taken, frozen so a later step's own touches don't change what this
    # step's fingerprint is re-checked against.
    verify_touched: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict, index: int) -> Step:
        return cls(
            id=str(data.get("id") or f"step-{index + 1}"),
            title=str(data.get("title") or data.get("id") or f"step {index + 1}"),
            intent=str(data.get("intent") or ""),
            acceptance=str(data.get("acceptance") or ""),
            signatures=[str(x) for x in (data.get("signatures") or [])],
            key_fact=str(data.get("key_fact") or ""),
            tests=[str(x) for x in (data.get("tests") or [])],
            files=[str(f) for f in (data.get("files") or [])],
            writes=[str(f) for f in (data.get("writes") or [])],
            verify=str(data.get("verify") or ""),
            depends_on=[str(d) for d in (data.get("depends_on") or [])],
        )

    @property
    def written(self) -> set[str]:
        """Files this step may modify. An undeclared `writes` means all of them."""
        return set(self.writes) if self.writes else set(self.files)

    @property
    def touches(self) -> set[str]:
        return set(self.files) | set(self.writes)


@dataclass
class Plan:
    goal: str
    draft: list[str]
    steps: list[Step]
    # The planner's own classification, when `plan_task_kind` asked for one.
    # Recorded only: nothing routes on it until its agreement with the known
    # task categories has been measured.
    task_kind: str = ""
    # Facts the planner looked up that the steps rely on, with sources
    # (Settings.research_first; agent/research.py). Every executor step sees them.
    findings: list[str] = field(default_factory=list)

    def outline(self) -> str:
        return "\n".join(f"{i + 1}. [{s.id}] {s.title}" for i, s in enumerate(self.steps))


# A verify command that only proves a tool exists. The planner prompt forbids
# these, and a traced run produced `uv run pytest --version` anyway, on a step
# called "add pytest" — so it is checked rather than merely asked for. Such a
# command passes no matter how wrong the code is, which makes the step's own
# correctness signal, and the final gate's re-run of it, worthless.
_TOOL_PROBE = re.compile(
    r"(?:^|\s)(?:--version|-V|--help|-h)(?:\s|$)"
    r"|^\s*(?:which|type|command\s+-v)\s"
    r"|(?:^|\s)(?:uv|pip|pip3)\s+show(?:\s|$)"
    r"|(?:^|\s)uv\s+pip\s+list(?:\s|$)"
)


def is_tool_probe(command: str) -> bool:
    """True when the command checks that a tool is installed, not that the work is right."""
    return bool(_TOOL_PROBE.search(command.strip()))


# `uv run <tool>` only works when <tool> is a dependency of the workspace.
# A traced run made the harness itself author `uv run ruff check game.py` in a
# project whose pyproject declares only pygame and pytest, so the command could
# never run; the step reported a broken environment, envfail dutifully rebuilt
# the virtualenv, and of course that changed nothing, because nothing was
# broken. toolchain.resolve already knows how to find a tool that the project
# does not declare — it falls back to `uvx` — so verify commands go through it
# rather than assuming.
_UV_RUN = re.compile(r"^\s*uv\s+run\s+(?:--\S+\s+)*(?P<tool>[\w.-]+)(?P<rest>\s.*)?$", re.DOTALL)
# Only tools that do not need the project's own dependencies to do their job.
# pytest is deliberately absent: see runnable_verify.
_REWRITABLE = ("ruff", "ty")
# `uv run python -m pytest` and `uv run pytest` look interchangeable and are
# not: a traced bench sweep had six steps write the `python -m` form, and
# every one failed with "No module named pytest" against the identical
# workspace where bare `uv run pytest` succeeds. `uv run <name>` resolves
# `<name>` on PATH -- and a workspace's inherited PATH can carry a working
# pytest even with no virtualenv of its own -- but `uv run python ...`
# resolves the interpreter through uv's own toolchain discovery instead,
# which reaches for a freshly-provisioned, package-free managed Python and
# never looks at PATH at all. Both forms run the same test files in the same
# directory when they run; only one of them reliably does.
_PYTEST_VIA_PYTHON = re.compile(
    r"^(?P<pre>\s*uv\s+run\s+(?:--\S+\s+)*)python3?\s+-m\s+pytest(?P<rest>\s.*)?$", re.DOTALL
)


# A step whose acceptance is that it finished looking. These produce no code,
# cost a full executor loop each, and discard what they learn at the step
# boundary — and they are usually re-fetching something the harness already
# has: the test baseline is collected before planning, the planner has read the
# workspace, and each step's own files are preloaded into the executor prompt.
# 13 of 62 traced planned steps were this shape, costing 67 model calls and
# 1.1M prompt tokens.
_LOOKING = re.compile(
    r"\b(?:read|captured|understood|inspected|examined|identified|reviewed|"
    r"analysed|analyzed|documented|noted|gathered|located)\b",
    re.IGNORECASE,
)
# Anything that could actually be checked: a comparison, a state, an outcome.
# Its presence is what separates "the tests pass" from "the tests were run".
_CHECKABLE = re.compile(
    r"\b(?:pass(?:es|ed|ing)?|fail(?:s|ed|ing)?|return(?:s|ed)?|equals?|match(?:es|ed)?|"
    r"raises?|exits?|draws?|renders?|increments?|activates?|contains?|shows?|"
    r"persists?|resets?|appears?|updates?|creates?|removes?|prevents?|rejects?|"
    # Past participles that claim work was done, as opposed to looked at:
    # "identified and tested" wrote six real tests, "test output captured" did not.
    r"tested|verified|covered|added|fixed|written|implemented)\b"
    r"|[=<>]|\bno longer\b|\bat y\b|\d+\s*,\s*\d+",
    re.IGNORECASE,
)


def is_reconnaissance(step: Step) -> bool:
    """A step whose acceptance is that it finished looking.

    Judged on acceptance, not intent: "Read the handler and fix the off-by-one"
    is real work that happens to start with "Read", while an acceptance of
    "files read successfully" admits nothing changes. And only when the
    acceptance contains nothing checkable — "All tests pass or failures are
    documented" is a weak acceptance but "all tests pass" is real, so it stays.
    """
    acceptance = step.acceptance.strip()
    if not acceptance:
        return False
    return bool(_LOOKING.search(acceptance)) and not _CHECKABLE.search(acceptance)


def runnable_verify(command: str) -> str:
    """Rewrite a `uv run` verify command that the workspace cannot run.

    Two unrelated failure modes share this function because both are "the
    command as written cannot reliably start," never "the code is wrong."

    `uv run X` only finds X if the workspace declares it. A traced run had the
    harness itself author `uv run ruff check game.py` against a project whose
    pyproject lists pygame and pytest and nothing else, so the command could
    never start: the step reported a broken environment, envfail rebuilt the
    virtualenv on the strength of that, and naturally nothing changed, because
    nothing was broken. `uvx` fetches the tool into its own environment and is
    exactly the tool for this. Scoped to ruff and ty on purpose. A linter and
    a type checker do not need the project's dependencies — they read its
    files and its pyproject config — so running them from a separate
    environment is correct. pytest is the opposite: it must import the code
    under test and everything that code imports, so `uv run pytest` is
    rewritten to nothing, ever. Swapping which interpreter and which packages
    the tests see, to work around a command that probably runs fine, is not a
    trade worth making.

    `uv run python -m pytest` is the second case, and it is not a dependency
    problem: see _PYTEST_VIA_PYTHON. Rewriting it to `uv run pytest` changes
    nothing about which packages the tests see — both forms run in the same
    directory against the same PATH — it only avoids the interpreter-lookup
    codepath that `uv run python` takes and `uv run pytest` does not.
    """
    # `pytest -k "*"` is not the syntax for running the whole suite: pytest
    # parses `*` as an expression and exits before collecting anything.
    # Omitting `-k` has the intended meaning and is safe for a generated
    # verification command.
    command = re.sub(r"\s+", " ", _PYTEST_ALL_SELECTOR.sub(" ", command)).strip()
    python_form = _PYTEST_VIA_PYTHON.match(command)
    if python_form:
        rest = (python_form.group("rest") or "").strip()
        return f"{python_form.group('pre')}pytest {rest}".strip()
    match = _UV_RUN.match(command)
    if not match or match.group("tool") not in _REWRITABLE:
        return command
    rest = (match.group("rest") or "").strip()
    return f"uvx --quiet {match.group('tool')} {rest}".strip()


# Matches the leading `... pytest` / `... ruff` / `... ty` of an already-
# rewritten verify command, whatever its prefix — `uv run pytest`, `uvx
# --quiet ruff`, a bare `ty` — so a preflight probe can be built from exactly
# the same invocation shape the real verify command will use.
_VERIFY_TOOL = re.compile(r"^.*?\b(pytest|ruff|ty)\b")
_PYTEST_ALL_SELECTOR = re.compile(r"(?P<prefix>(?:^|\s)-k\s+)(?P<quote>[\"']?)\*(?P=quote)(?=\s|$)")


def _preflight_verify(ctx: ToolContext, trace: Trace, command: str) -> str:
    """Whether this verify command's tool can run at all, checked once with
    `--version` rather than discovered mid-step — after a step has already
    spent its whole execution budget on work nothing could then verify.

    Six traced steps wrote `uv run python -m pytest`, which failed with "No
    module named pytest" every time; runnable_verify() above now rewrites
    that specific shape, but nothing caught the general case, where the tool
    genuinely is not runnable in this workspace for some other reason.
    envfail already tells a broken environment apart from broken code once a
    step fails; this asks the same question earlier and once, using the same
    repair path, rather than once per step that happens to share the tool —
    a workspace with five pytest-verified steps and a broken venv used to pay
    for `uv sync` five times over, once inside each step's own failure.

    Returns "" when the tool is fine (or was just repaired), or the reason
    it is not, meant for the caller to act on rather than to feed back to a
    model as a defect in its own work.
    """
    from .execute import run_verify

    match = _VERIFY_TOOL.match(command.strip())
    if not match:
        return ""
    probe = f"{match.group(0)} --version"
    passed, output = run_verify(ctx, probe)
    if passed:
        return ""
    passed, output, env_note = envfail.repair_and_retry(
        output, ctx.workspace.root, lambda: run_verify(ctx, probe)
    )
    if passed:
        trace.say(f"   toolchain repaired: {probe.split()[0]} now runs")
        return ""
    return env_note or f"`{probe}` did not run cleanly"


def _synthesize_verify(step: Step, profile: LanguageProfile | None = None) -> str:
    """A default `verify` for a step that declared none.

    The retry-and-escalate path in execute.py only ever engages for a step
    that has a verify command, so a planner that leaves the field blank
    silently opts that step out of the harness's only per-step correctness
    signal. Rather than requiring a command the planner may have no good
    answer for, fall back to the cheapest thing that is still real.

    Deliberately scoped to the step's own files, never the whole suite: a
    plan whose tests arrive in step 3 would otherwise fail step 1 against
    tests that do not exist yet, burning a retry at higher reasoning on a
    step that was correct. Breakage *across* steps is what phase 4's final
    check exists to catch, and it runs when the whole plan is done.

    `step.writes` is preferred over `step.files` as the target list. A traced
    "document findings" step declared `writes: ["README.md"]` but also listed
    a `.py` file in `files` (context it read, not something it changed); the
    old version of this function looked only at `files`, so it lint-checked
    that Python file and never examined README.md at all — the step's actual
    deliverable was unverified by construction. `writes` is a step author's
    explicit claim of what it will change, so it is the more specific answer
    whenever it is given; `files` remains the fallback for a step that never
    declared `writes` (read as "all of them", same as everywhere else this
    field is used).

    `profile` — the workspace's detected `LanguageProfile` (see langtools.py),
    `None` when it's Python or nothing recognised — only matters here when
    `targets` holds no `.py` file: a step touching `.test.ts`/`.js` files in a
    detected Node workspace gets that profile's own test/lint command instead
    of falling straight to the data-format/prose checks below.
    """
    targets = step.writes or step.files
    python_files = [f for f in targets if f.endswith(".py")]
    non_python = [f for f in targets if not f.endswith(".py")]
    # The data-format/prose check for whatever this step also touched that
    # isn't Python -- composed in below rather than returned on its own,
    # since a step that ships a code change alongside a config or doc file
    # (common: a feature plus its README mention) previously lost that half
    # of its own declared `writes` entirely the moment any .py file was
    # present, with nothing else in the harness checking it either.
    extra = _synthesize_text_verify(non_python) if python_files and non_python else ""
    if python_files:
        tests = [
            f
            for f in python_files
            if f.rsplit("/", 1)[-1].startswith("test_") or f.endswith("_test.py")
        ]
        if tests:
            command = "uv run pytest -q " + " ".join(tests)
        else:
            # Ruff's default rule set includes pyflakes, so this catches
            # undefined names and bad imports, not only formatting opinions.
            command = "uv run ruff check " + " ".join(python_files)
        return command + (" && " + extra if extra else "")
    if profile is not None and profile is not langtools.PYTHON:
        own_files = [f for f in targets if langtools.is_test_file(profile, f)]
        if own_files:
            return profile.test_cmd.format(files=" ".join(own_files))
    return _synthesize_text_verify(targets)


_WHOLE_SUITE = re.compile(
    r"^(uv run |uvx --quiet )?(python3? -m )?pytest"
    r"(\s+-[\w-]+)*\s*$"
)


def _narrow_whole_suite_verify(step: Step, profile: LanguageProfile | None, trace: Trace) -> None:
    """Scope a step's whole-suite verify down to the step's own files.

    `_synthesize_verify` explains at length why a step's verify must not be
    the whole suite — a plan whose tests arrive in step 3 fails step 1
    against tests that do not exist yet, burning a retry at higher reasoning
    on a step that was correct — and then only applies that reasoning to
    steps whose verify was left blank. PLANNER_SYSTEM meanwhile offers
    `uv run pytest -q` as its worked example, twice, so the planner writes
    it: 429 of 960 traced step verifies were exactly that, 47% of them, each
    running the entire suite to check one step.

    Breakage *across* steps is phase 4's job, and phase 4 runs the whole
    suite when the plan is done. Nothing is lost by scoping here; what is
    gained is that the per-step signal is about the step.

    Left alone when the step declares nothing to narrow to, when the step's
    files yield no test command, or when the command is anything other than
    a bare pytest invocation — a verify with its own `-k`, path, or
    non-pytest tool already says what it means.
    """
    command = step.verify.strip()
    if not _WHOLE_SUITE.match(command):
        return
    scoped = _synthesize_verify(step, profile)
    if not scoped or "pytest" not in scoped or scoped.strip() == command:
        return
    trace.event("verify_scoped", step=step.id, was=command, now=scoped)
    trace.say(f"   [{step.id}] verify scoped to this step's own files")
    step.verify = scoped


def _synthesize_text_verify(targets: list[str]) -> str:
    """A syntax check for a step's non-Python deliverables, where one exists.

    Delegates to `agent.dataformats` (structured data: JSON/YAML/TOML/XML/…,
    `yq`-first) and its prose check (a narrow, real Markdown parse via
    `pandoc`) — see that module's docstring for what each does and does not
    catch, and why.

    Markdown, plain text and CSV having no syntax to validate against beyond
    what `dataformats.build_prose_validator` covers is deliberate, not a gap:
    `_verify_step` in execute.py separately checks that a step actually
    touched every file it declared under `writes`, which is a weaker claim
    than "the content is right" but a real one, and is the check that would
    have caught the traced step that called `finish_step` claiming a
    README's summary was already correct without ever opening it.
    """
    if not targets:
        return ""
    checks = [
        c
        for c in (dataformats.build_validator(targets), dataformats.build_prose_validator(targets))
        if c
    ]
    return " && ".join(checks)


DRAFT_NARRATION_CHARS = 600


def _narrate_understanding(plan: Plan, trace: Trace) -> None:
    """Show the planner's reading of the request on its own line.

    The draft was already narrated, joined into one line among the others, and
    a misreading slid past in it: "fix lamp spawn to align vertically with
    wall gaps" was note three of four, in a scrolling log, and the person did
    not see it until the game showed lamps in the gaps. What the agent thinks
    it was asked deserves its own line, before any code is written.
    """
    restatement = next(
        (note for note in plan.draft if note.strip().lower().startswith("understood")), ""
    )
    rest = [note for note in plan.draft if note != restatement]
    if restatement:
        trace.say("\nunderstood as: " + restatement.split(":", 1)[-1].strip())
    if rest:
        text = " | ".join(rest)
        if len(text) > DRAFT_NARRATION_CHARS:
            # A prose draft can run to paragraphs; the terminal shows the start
            # and the trace keeps all of it.
            text = text[:DRAFT_NARRATION_CHARS].rstrip() + " … (full draft in the trace)"
        trace.say("draft: " + text)


def _drop_reconnaissance(plan: Plan, trace: Trace) -> None:
    """Remove steps whose acceptance admits they change nothing.

    Dropped rather than merged: there is nothing to merge, by the step's own
    account. Kept when they are the whole plan, because an empty plan is worse
    than a wasteful one. Dependencies naming a dropped step are pruned, since
    build_waves treats an unknown id as satisfied anyway and leaving them would
    only make the schedule harder to read.
    """
    recon = [step for step in plan.steps if is_reconnaissance(step)]
    if not recon or len(recon) == len(plan.steps):
        return
    for step in recon:
        trace.event(
            "step_dropped", step=step.id, reason="reconnaissance", acceptance=step.acceptance
        )
        trace.say(
            f"   [{step.id}] dropped: its acceptance is that it finished looking "
            f"({step.acceptance.strip()[:60]!r}), and the executor is preloaded "
            "with the step's files anyway"
        )
    discarded = {id(step) for step in recon}
    plan.steps = [step for step in plan.steps if id(step) not in discarded]
    surviving = {step.id for step in plan.steps}
    for step in plan.steps:
        step.depends_on = [dep for dep in step.depends_on if dep in surviving]


def _planner_system(settings, probe: str | None = None) -> str:
    """The planner prompt, plus the probe and symptom-first blocks when on.

    `probe` overrides `settings.plan_probe` with the mode actually in force
    for this workspace (see `_probe_mode`).

    Behind a setting so a sweep can price it: prompt edits are otherwise the
    one mechanism in this harness that cannot be A/B'd, and the first attempt
    to measure this one by running the old and new prompts in sequence was
    confounded by a 2.3x swing in provider latency between the two sweeps.
    """
    base = PLANNER_SYSTEM
    if getattr(settings, "draft_style", "terse") == "prose":
        # Swapped, not appended: two instructions about how to draft, one saying
        # "brief notes, not paragraphs" and one saying paragraphs, is exactly the
        # contradiction this style exists to remove.
        base = PLANNER_SYSTEM.replace(DRAFT_TERSE, DRAFT_PROSE).replace(
            DRAFT_HANDOFF_TERSE, DRAFT_HANDOFF_PROSE
        )
        for terse, prose in DRAFT_PROSE_REWORDINGS:
            base = base.replace(terse, prose)
    base = promptadapt.planner(base, settings)
    mode = probe or getattr(settings, "plan_probe", "off")
    if mode == "repro":
        form = getattr(settings, "repro_form", "snippet")
        base = base + "\n\n" + symptom.planner_block(PLAN_REPRO, form)
    elif mode != "off":
        base = base + "\n\n" + PLAN_PROBE
    if getattr(settings, "research_first", False):
        from ..research import PLAN_RESEARCH_FIRST

        base = base + "\n\n" + PLAN_RESEARCH_FIRST
    if not getattr(settings, "symptom_first_planning", False):
        return base
    return base + "\n\n" + SYMPTOM_FIRST_PLANNING


def _probe_mode(setting: str, symbols: str, trace: Trace) -> str:
    """`plan_probe` as it applies to this workspace: "off" when there is no code to probe.

    An empty or non-Python workspace has nothing probe_python could reproduce a
    symptom through. Holding the plan there costs a round trip for nothing, and
    merely offering the tool is not free either: on the five greenfield bench
    tasks the planner still probed 1-2 times a run on three of them (§54). The
    symbol map is the test, empty exactly when there is no Python source.
    """
    if setting != "off" and not symbols:
        trace.event("probe_withheld", reason="no Python source in the workspace")
        return "off"
    return setting


def _plan_gate(probe: str, ctx: ToolContext, trace: Trace) -> ProbeGate | ReproGate | None:
    if probe == "require":
        return ProbeGate()
    if probe == "repro":
        return ReproGate(lambda code: run_in_scratch(ctx, code), trace)
    return None


def _with_callsite_check(gate, settings, ctx: ToolContext, trace: Trace):
    """`gate`, chained after a CallSiteGate when Settings.callsite_check is on, and
    after a ResearchGate when Settings.research_first is on."""
    gates = [gate]
    if getattr(settings, "research_first", False):
        from ..research import ResearchGate

        gates.append(ResearchGate(settings.goal, ctx.workspace.root))
        if getattr(settings, "plan_findings_hold", False):
            gates.append(DecisionsGate())
    if getattr(settings, "callsite_check", False):
        gates.append(CallSiteGate(settings.goal, ctx.workspace.root, trace))
    live = [g for g in gates if g is not None]
    if len(live) <= 1:
        return live[0] if live else None
    return GateChain(*live)


def _findings_last(settings) -> bool:
    """Settings.plan_findings_last, which only means anything with research_first."""
    return bool(
        getattr(settings, "plan_findings_last", False)
        and getattr(settings, "research_first", False)
    )


def _planning_kit(settings, probe: str) -> Registry:
    """The planner's tools, with submit_plan's optional fields for the settings in use."""
    registry = planning_registry(
        getattr(settings, "draft_style", "terse"), probe=probe in ("offer", "require")
    )
    research = getattr(settings, "research_first", False)
    form = getattr(settings, "repro_form", "snippet")
    # Applied to submit_plan in this order. With task_kind present,
    # runner.ReproGate sends an empty snippet on a self-declared bugfix back once.
    # That is opt-in through plan_task_kind, not part of "repro": it cost +1.9
    # calls a run and moved honest by 0 over 216 runs (DESIGN-LOG §54.3).
    transforms = [
        (getattr(settings, "plan_task_kind", False), with_task_kind),
        (research, with_findings),
        (research and getattr(settings, "plan_findings_required", False), with_required_findings),
        (getattr(settings, "plan_key_facts", False), with_key_fact),
        (probe == "repro", lambda t: symptom.with_form(with_reproduce(t), form)),
    ]
    for enabled, transform in transforms:
        if enabled:
            registry = Registry(
                [transform(t) if t.name == "submit_plan" else t for t in registry.tools()]
            )
    return registry


def _issue_symbols_section(settings, ctx: ToolContext) -> str:
    """Where the goal's identifiers are defined, as a block of the planner's first message."""
    if not getattr(settings, "plan_issue_symbols", False):
        return ""
    found = issue_symbols.issue_symbols(ctx.workspace.root, settings.goal)
    if not found:
        return ""
    return (
        "Where the names in the goal are defined (a search of the workspace, not a claim about "
        f"where to edit):\n{found}\n\n"
    )


def make_plan(
    client: ChatClient, settings, ctx: ToolContext, trace: Trace, extra_context: str = ""
) -> Plan:
    # The tree says which files exist; the symbol map says what is in them.
    # The second is what `files` and `writes` are declared from, and it costs
    # 6-8% of what reading the sources costs.
    symbols = outline.symbol_map(ctx.workspace.root)
    probe = _probe_mode(getattr(settings, "plan_probe", "off"), symbols, trace)
    registry = _planning_kit(settings, probe)
    user = (
        f"Goal:\n{settings.goal}{extra_context}\n\n"
        f"Workspace root: {ctx.workspace.root}\n"
        f"Repo map:\n{ctx.workspace.tree()}\n\n"
        + (
            f"What is defined where (top-level symbols, not bodies):\n{symbols}\n\n"
            if symbols
            else ""
        )
        + _issue_symbols_section(settings, ctx)
        + "Use this to decide which files each step touches. Read a file when you need "
        "to know how something works; you should not need to read one just to find out "
        "what is in it.\n\n"
        "Explore only as much as you need, then submit the plan."
        # Before FINDINGS_LAST, whose place at the very end was what made it work (§87).
        + (chatflow.PLANNER_CLOSING if chatflow.enabled(settings) else "")
        + (FINDINGS_LAST if _findings_last(settings) else "")
    )
    trace.say("\n=== phase 1: plan (chain of draft) ===", detail=True)
    names = [p.name for p in ctx.workspace.root.iterdir()] if ctx.workspace.root.is_dir() else []
    chosen = skills_module.select(ctx.skills, "plan", settings.goal, names, ctx.skill_token_budget)
    result, messages = run_tool_loop(
        client=client,
        model=settings.planner_model,
        phase="plan",
        system=_planner_system(settings, probe) + skills_module.render(chosen, ctx.skills, "plan"),
        user=user,
        registry=registry,
        ctx=ctx,
        trace=trace,
        max_iters=12,
        reasoning_effort=settings.planner_reasoning,
        temperature=0.2,
        max_usd=settings.budget.max_usd,
        gate=_with_callsite_check(gate := _plan_gate(probe, ctx, trace), settings, ctx, trace),
        # A planner that runs out of iterations gets one last turn to submit what
        # it has (runner._last_word) instead of the run dropping to a one-step
        # plan. The salvaged prose it would otherwise leave has no steps, so the
        # fallback below still catches a planner that does not submit.
        salvage=True,
        tail=partial(planner_tail, settings.goal)
        if getattr(settings, "planner_tail", False)
        else None,
        last_word_gates=getattr(settings, "last_word_gates", "off"),
    )
    if isinstance(gate, ReproGate) and gate.passed is not None:
        ctx.repro_passed = gate.passed
    if isinstance(gate, ReproGate) and gate.reproduced:
        ctx.repro_reproduced = gate.reproduced
    if not result or not result.get("steps"):
        # Not every tool-capable model can reliably drive a structured plan call.
        # Degrading to a single step is far better than failing the run: the
        # executor can still do the work, just without decomposition.
        trace.say("planner returned no plan; falling back to a single-step plan")
        trace.event("plan_fallback", model=settings.planner_model)
        return Plan(
            goal=settings.goal,
            draft=["planner produced no plan", "executing goal as one step"],
            steps=[
                Step(
                    id="whole-goal",
                    title="Complete the goal",
                    intent=settings.goal,
                    acceptance="The goal is fully met.",
                )
            ],
        )
    if getattr(settings, "research_first", False):
        _complete_findings(result, messages, trace)
    return finalize_plan(result, settings, ctx, trace)


def _complete_findings(result: dict, messages: list[dict], trace: Trace) -> None:
    """Add the looked-up signatures the steps use that the planner's findings left out."""
    stated = [str(f) for f in (result.get("findings") or [])]
    steps_text = json.dumps(result.get("steps") or [])
    completed = complete_findings(stated, looked_up_signatures(messages), steps_text)
    if len(completed) > len(stated):
        trace.event("findings_completed", added=completed[len(stated) :])
    result["findings"] = completed


def _settle_step_verify(step: Step, profile, settings, trace: Trace) -> None:
    """Bring one step's verify command to the form the executor will run:
    rewritten to something this workspace can execute, never a bare toolchain
    probe, synthesized when absent, and narrowed off the whole suite."""
    if step.verify.strip():
        # In the project's own test form too (django labels for pytest ids, the label root
        # off), so the executor is shown the command that will run (DESIGN-LOG §110).
        fixed = verifycmd.normalize(
            runnable_verify(step.verify),
            getattr(settings, "test_command", ""),
            getattr(settings, "test_label_root", ""),
        )
        if fixed != step.verify:
            trace.event("verify_rewritten", step=step.id, was=step.verify, now=fixed)
            trace.say(f"   [{step.id}] verify rewritten to one this workspace can run")
            step.verify = fixed
    if step.verify.strip() and is_tool_probe(step.verify):
        trace.event(
            "verify_rejected",
            step=step.id,
            command=step.verify,
            reason="checks that a tool is installed, not that the step worked",
        )
        trace.say(f"   [{step.id}] verify only probes the toolchain; replacing it")
        step.verify = ""
    if not step.verify.strip():
        # Synthesized commands go through the same rewrite: they are the
        # ones that named `uv run ruff` in a project without ruff.
        step.verify = runnable_verify(_synthesize_verify(step, profile))
        if step.verify:
            trace.event("verify_synthesized", step=step.id, command=step.verify)
    elif getattr(settings, "scope_step_verify", True):
        _narrow_whole_suite_verify(step, profile, trace)


def _preflight_plan_verifies(plan: Plan, ctx: ToolContext, trace: Trace) -> None:
    # One preflight pass, after every step's verify command has settled into
    # its final, rewritten form. Cached per tool rather than run per step: a
    # plan with five pytest-verified steps checks pytest once, not five
    # times, and a broken environment is repaired (or given up on) once,
    # before any step has spent a call on work nothing could then verify.
    preflight_cache: dict[str, str] = {}
    for step in plan.steps:
        if not step.verify.strip():
            continue
        tool_match = _VERIFY_TOOL.match(step.verify.strip())
        tool_key = tool_match.group(0) if tool_match else step.verify
        if tool_key not in preflight_cache:
            preflight_cache[tool_key] = _preflight_verify(ctx, trace, step.verify)
        issue = preflight_cache[tool_key]
        if issue:
            trace.event("verify_preflight_failed", step=step.id, command=step.verify, reason=issue)
            trace.say(
                f"   [{step.id}] verify command unavailable ({issue}); "
                "retaining it so the final gate cannot report a false pass"
            )
            step.verify_error = issue


def finalize_plan(result: dict, settings, ctx: ToolContext, trace: Trace) -> Plan:
    """Turn a raw `submit_plan`-shaped result into a runnable `Plan`.

    Shared by the planner's own call and the optional plan-audit phase
    (`phases/plan_audit.py`): a plan the auditor patches needs exactly the
    same rewriting, reconnaissance-dropping and preflight checking a freshly
    planned one gets, or its patched steps would reach the executor with
    verify commands nothing has yet confirmed can even run.
    """
    # Non-dict entries are dropped rather than crashing the run: a model that
    # sends `steps` as prose should cost one bad plan, not the whole job.
    raw_steps: list[Any] = [s for s in result["steps"] if isinstance(s, dict)]
    raw_steps = raw_steps[: settings.budget.max_steps]
    if not raw_steps:
        trace.say("planner returned no usable steps; executing the goal as one step")
        trace.event("plan_fallback", model=settings.planner_model, reason="no dict steps")
        return Plan(
            goal=settings.goal,
            draft=[str(d) for d in (result.get("draft") or [])],
            steps=[
                Step(
                    id="whole-goal",
                    title="Complete the goal",
                    intent=settings.goal,
                    acceptance="The goal is fully met.",
                )
            ],
        )
    plan = Plan(
        goal=settings.goal,
        draft=[str(d) for d in result.get("draft", [])],
        steps=[Step.from_dict(s, i) for i, s in enumerate(raw_steps)],
        task_kind=str(result.get("task_kind") or "").strip().lower(),
        findings=[str(f) for f in (result.get("findings") or [])],
    )
    _drop_reconnaissance(plan, trace)
    # Detected once for the whole plan, not once per step: a language
    # profile is a property of the workspace, not of any one step's files.
    profile = langtools.detect_profile(ctx.workspace.root)
    for step in plan.steps:
        _settle_step_verify(step, profile, settings, trace)

    _preflight_plan_verifies(plan, ctx, trace)

    trace.event(
        "plan",
        goal=plan.goal,
        draft=plan.draft,
        steps=[s.__dict__ for s in plan.steps],
        task_kind=plan.task_kind,
        findings=plan.findings,
    )
    _narrate_understanding(plan, trace)
    trace.say(plan.outline())
    return plan


DIRECT_FILES = 3  # the files a direct step is given the contents of, from the issue lookup


def direct_plan(goal: str, definitions: list | None = None) -> Plan:
    """A plan of one step whose intent is the goal, for `--direct`: no planner call, so the
    executor finds the files itself. Acceptance is the project's own tests, which the final
    check runs either way. `definitions` (agent/issue_symbols.py, no model call) say where the
    goal's own names are defined; they go into the intent and the first DIRECT_FILES distinct
    files become the step's files, which the executor is handed the contents of."""
    intent = goal
    files: list[str] = []
    if definitions:
        intent += (
            "\n\nWhere the names in this task are defined (a search, not a plan):\n"
            + "\n".join(f"- {d.render()}" for d in definitions)
        )
        files = list(dict.fromkeys(d.path for d in definitions))[:DIRECT_FILES]
    step = Step(
        id="s1",
        title="Resolve the task",
        intent=intent,
        acceptance="The task is done and the project's existing tests still pass.",
        files=files,
    )
    return Plan(goal=goal, draft=["Resolve the task directly."], steps=[step])
