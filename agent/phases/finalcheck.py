"""Phase 4: the deterministic gate.

Everything before this point is a model's opinion about whether the work is
right. The step-level `verify` commands come closest, but each one runs in
isolation right after its own step, so a later step that breaks an earlier
one's work passes unnoticed — and the review phase, which is the only thing
that looks at the whole surface, is an LLM making a judgement call.

This phase makes no model calls at all. It re-runs every step's verify
command against the *finished* state, runs the project's test suite, and
type-checks the changed files. What it reports is not an opinion: a command
either exited zero or it did not. That makes it both the cheapest signal in
the harness and the only one that can contradict a model that believes it
succeeded.

Failures here are converted into review findings and handed to the ordinary
fix path, so the thing that repairs them is code that already exists.
"""

from __future__ import annotations

import shlex
import threading
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .. import baseline_cache, envfail, failsig, toolchain, validate
from ..projectconfig import configures_type_checker
from ..tools import ToolContext
from ..trace import Trace
from ..workspace import walk_files
from . import coverage as coverage_module
from .plan import Plan

TEST_TIMEOUT_S = 300


@dataclass
class FinalCheck:
    """What the gate found. `ok` is the only thing callers need to branch on."""

    verify_failures: list[dict] = field(default_factory=list)
    # Commands that could not be run at all because the toolchain is broken,
    # kept apart from verify_failures on purpose: these are not defects in the
    # work and must not be handed to model fix rounds. They still make the
    # result non-green: a run that could not verify its work is inconclusive,
    # not successful. See `ok` and `inconclusive` for how those two states
    # (a real defect vs. nothing could be checked) are told apart.
    env_failures: list[dict] = field(default_factory=list)
    tests_ran: bool = False
    tests_passed: bool = True
    tests_summary: str = ""
    tests_output: str = ""
    type_errors: list[validate.Diagnostic] = field(default_factory=list)
    # The subset of type_errors this run actually introduced, as opposed to
    # type debt the workspace already had — see collect_type_baseline. This,
    # not type_errors as a whole, is what `ok` gates on.
    new_type_errors: list[validate.Diagnostic] = field(default_factory=list)
    # Whether a pre-run baseline of type diagnostics was actually collected;
    # without one there is no way to tell a type error this run introduced
    # from one the workspace already had, so (mirroring baseline_known for
    # tests) every diagnostic found counts as new.
    type_baseline_known: bool = False
    syntax_errors: list[validate.Diagnostic] = field(default_factory=list)
    # Test ids failing now that were not failing before the run started, and
    # the ones that were already failing when it started. Only the first kind
    # is this run's problem.
    new_failures: list[str] = field(default_factory=list)
    pre_existing_failures: list[str] = field(default_factory=list)
    # Ones that were red at the start and are green now — worth saying out
    # loud, since on a "fix the failing tests" goal it is the whole result.
    fixed_pre_existing: list[str] = field(default_factory=list)
    # Why the ones still red are red, "name: file:line in func -- error" per
    # line, so the summary can say what stopped the fix rather than only that
    # something did.
    pre_existing_why: str = ""
    # Non-test Python this run wrote or changed, when the workspace has no
    # tests at all. Six consecutive runs on one project reported PASS while
    # the game they produced had broken scoring and a preferences pane that
    # did nothing — because with no tests to run, the only thing standing
    # between "ruff check passed" and "ship it" was nothing.
    untested_changes: list[str] = field(default_factory=list)
    # True when a baseline was collected (so tests existed and were
    # collectable when the run started) and none exist now. `untested_changes`
    # already catches a step that deletes every test file *and* changes some
    # other Python file in the same run; this is the residual case it cannot
    # see -- a run whose only action is deleting its own failing test(s),
    # touching nothing else -- which would otherwise be indistinguishable
    # from a project that never had tests to begin with.
    deleted_test_suite: bool = False
    # What the suite actually executed of what this run changed.
    coverage: coverage_module.CoverageReport | None = None
    # Whether failures that predate the run count against it. False treats
    # them as somebody else's problem (right when the goal is unrelated and
    # the workspace merely happens to have red tests); True treats them as
    # the work (right when the goal is "the tests are failing, fix them",
    # which is most of why anyone points this at a broken repository).
    fix_pre_existing: bool = False

    @property
    def failing_tests(self) -> list[str]:
        """Every test failure this run is answerable for."""
        if not self.baseline_known:
            return self.new_failures
        return sorted(
            {*self.new_failures, *(self.pre_existing_failures if self.fix_pre_existing else [])}
        )

    @property
    def _has_defect(self) -> bool:
        """Whether a real defect was found in the work itself — independent
        of whether some other check could not be run at all. Split out from
        `ok` so `inconclusive` below can ask the same question minus
        `env_failures` without duplicating this logic.

        Which failures count depends on `fix_pre_existing`. Excusing
        pre-existing ones unconditionally made the gate useless for the most
        common reason to reach for this agent at all — a suite that is
        already red — and produced runs that reported PASS over a test suite
        that could not even be collected.
        """
        if self.baseline_known:
            tests_broke = bool(self.failing_tests)
        else:
            tests_broke = self.tests_ran and not self.tests_passed
        unexercised = bool(
            self.coverage and (self.coverage.unexecuted or self.coverage.dead_functions)
        )
        return bool(
            self.verify_failures
            or self.syntax_errors
            or tests_broke
            or self.untested_changes
            or self.deleted_test_suite
            or unexercised
            or self.new_type_errors
        )

    @property
    def ok(self) -> bool:
        """Whether this run's work passes.

        False for either reason a run can fall short: a real defect
        (`_has_defect`), or a check that could not be run at all
        (`env_failures`) — a run that never got to look is not one a caller
        should treat as green either. `inconclusive` below is how a caller
        that wants to tell those two apart can.
        """
        return not (self._has_defect or self.env_failures)

    @property
    def inconclusive(self) -> bool:
        """True when nothing was found wrong with the work itself, but some
        check could not run at all, so nothing was ruled out either — a
        broken virtualenv, not a broken step. False whenever a real defect
        exists too (found by a check that *did* run), since that is the more
        actionable fact and should not be laundered into "merely
        inconclusive" by an unrelated environmental failure alongside it.

        `ok` does not use this: both reasons a run isn't `ok` need a caller's
        attention, and collapsing them into one boolean is deliberate there.
        This exists for a caller — the benchmark harness in particular —
        that wants to score "the work was wrong" differently from "nothing
        could tell either way."
        """
        return bool(self.env_failures) and not self._has_defect

    # Set when a pre-run baseline was actually collected; without one there is
    # no way to tell a new failure from an old one, so any failure counts.
    baseline_known: bool = False

    def signature(self) -> str:
        """A stable identity for this failure set, to spot a fix round that
        changed nothing."""
        parts = sorted(f"verify:{f.get('step')}" for f in self.verify_failures)
        parts += sorted(f"test:{name}" for name in self.failing_tests)
        parts += sorted(f"syntax:{d.file}:{d.line}" for d in self.syntax_errors)
        parts += sorted(f"untested:{name}" for name in self.untested_changes)
        if self.deleted_test_suite:
            parts.append("deleted_test_suite")
        parts += sorted(f"type:{d.file}:{d.line}:{d.code}" for d in self.new_type_errors)
        if self.coverage:
            parts += sorted(f"unexercised:{n}" for n in self.coverage.unexecuted)
            parts += sorted(f"deadfn:{n}" for n in self.coverage.dead_functions)
        if not self.baseline_known and self.tests_ran and not self.tests_passed:
            parts.append(f"suite:{self.tests_summary}")
        return "|".join(parts)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "inconclusive": self.inconclusive,
            "verify_failures": self.verify_failures,
            "env_failures": self.env_failures,
            "tests_ran": self.tests_ran,
            "tests_passed": self.tests_passed,
            "tests_summary": self.tests_summary,
            "new_failures": self.new_failures,
            "pre_existing_failures": self.pre_existing_failures,
            "fixed_pre_existing": self.fixed_pre_existing,
            "pre_existing_why": self.pre_existing_why,
            "failing_tests": self.failing_tests,
            "untested_changes": self.untested_changes,
            "deleted_test_suite": self.deleted_test_suite,
            "coverage": self.coverage.as_dict() if self.coverage else None,
            "fix_pre_existing": self.fix_pre_existing,
            "type_errors": len(self.type_errors),
            "new_type_errors": len(self.new_type_errors),
            "syntax_errors": len(self.syntax_errors),
        }

    def findings(self) -> list[dict]:
        """The failures, shaped like review findings so apply_fixes can take them.

        Severity is always "blocker": unlike a reviewer's judgement, every one
        of these is a command that actually failed, so there is no such thing
        as a stylistic or speculative entry here.
        """
        out: list[dict] = []
        for failure in self.verify_failures:
            out.append(
                {
                    "file": failure.get("file", ""),
                    "files": failure.get("files", []),
                    "severity": "blocker",
                    "issue": (
                        f"the verify command for step [{failure['step']}] fails against the "
                        f"finished code: {failure['command']}"
                    ),
                    "failure": failure["output"][-2_000:],
                    "fix": "Make the verify command pass without weakening what it checks.",
                    "recheck": {"verify": failure["command"]},
                }
            )
        if self.syntax_errors:
            out.append(
                {
                    "file": self.syntax_errors[0].file,
                    "severity": "blocker",
                    "issue": f"{len(self.syntax_errors)} file(s) do not parse",
                    "failure": validate.render(self.syntax_errors)[:2_000],
                    "fix": "Fix the syntax errors.",
                    "recheck": {"code": sorted({e.file for e in self.syntax_errors if e.file})},
                }
            )
        if self.new_failures:
            listed = ", ".join(self.new_failures[:10])
            excused = (
                ""
                if self.fix_pre_existing
                else " Tests that were already failing before this run started are "
                "not your concern: " + (", ".join(self.pre_existing_failures[:10]) or "(none)")
            )
            out.append(
                {
                    "file": _file_of(self.new_failures[0]),
                    "severity": "blocker",
                    "issue": f"tests that passed before this run now fail: {listed}",
                    "failure": self.tests_output[-3_000:],
                    "fix": (
                        "Make these specific tests pass again without deleting or "
                        "weakening them." + excused
                    ),
                    "recheck": {"tests": list(self.new_failures)},
                }
            )
        # Separately from the regression above: the tests that were already
        # red when the run started. Reported as their own finding so the fix
        # loop actually iterates on them, which is the whole point when the
        # goal was "the tests are failing, fix them" — excusing these was
        # making the agent ignore exactly the work it had been asked to do.
        if self.fix_pre_existing and self.pre_existing_failures:
            listed = ", ".join(self.pre_existing_failures[:10])
            out.append(
                {
                    "file": _file_of(self.pre_existing_failures[0]),
                    "severity": "blocker",
                    "issue": f"tests failing (and already failing before this run): {listed}",
                    "failure": self.tests_output[-3_000:],
                    "fix": (
                        "Make these pass. Read the test and the code it exercises and "
                        "decide which is wrong before changing either: if the test "
                        "encodes the intended behaviour, fix the code; if the test "
                        "itself is wrong or out of date, fix the test and say so. Do "
                        "not delete a test or weaken an assertion to make it pass."
                    ),
                    "recheck": {"tests": list(self.pre_existing_failures)},
                }
            )
        if self.new_type_errors:
            out.append(
                {
                    "file": self.new_type_errors[0].file,
                    "files": sorted({d.file for d in self.new_type_errors if d.file}),
                    "severity": "blocker",
                    "issue": (
                        f"{len(self.new_type_errors)} type error(s) this run introduced "
                        "that the workspace did not have before it started"
                    ),
                    "failure": validate.render(self.new_type_errors)[:2_000],
                    "fix": (
                        "Fix these type errors. Pre-existing type debt elsewhere in the "
                        "workspace is not your concern; only what this run newly broke is."
                    ),
                    "recheck": {"types": sorted({d.file for d in self.new_type_errors if d.file})},
                }
            )
        if self.untested_changes:
            listed = ", ".join(self.untested_changes[:10])
            out.append(
                {
                    "file": self.untested_changes[0],
                    "files": list(self.untested_changes[:6]),
                    "severity": "blocker",
                    "issue": (
                        f"this run wrote or changed behaviour with no tests anywhere in "
                        f"the workspace: {listed}"
                    ),
                    "failure": (
                        "Nothing ran the code. `ruff check` passing means it parses "
                        "and is tidy, not that it works — a lint pass is not a "
                        "behaviour check."
                    ),
                    "fix": (
                        "Write tests that exercise what this code is actually supposed to "
                        "do, then run them. Cover the behaviour a person would notice if "
                        "it broke: the thing the goal asked for, what happens at the "
                        "boundaries, and the failure modes you thought about while "
                        "writing it. If a test fails, that is the point — fix the code."
                    ),
                }
            )
        if self.deleted_test_suite:
            out.append(
                {
                    "file": "",
                    "severity": "blocker",
                    "issue": "tests existed when this run started and do not exist now",
                    "failure": (
                        "The suite is gone, so nothing runs to prove any of this "
                        "workspace's behaviour still works. A green gate over no "
                        "tests is not evidence of anything."
                    ),
                    "fix": (
                        "Restore the tests, or replace them with equivalents that "
                        "exercise the same behaviour. Never delete a test to make a run "
                        "look done — a red test is the thing this gate exists to find, "
                        "not a problem to make disappear."
                    ),
                }
            )
        gap = coverage_module.finding_for(self.coverage) if self.coverage else None
        if gap:
            out.append(gap)
        if not self.baseline_known and self.tests_ran and not self.tests_passed:
            out.append(
                {
                    "file": "",
                    "severity": "blocker",
                    "issue": f"the test suite fails: {self.tests_summary}",
                    "failure": self.tests_output[-3_000:],
                    "fix": "Make the failing tests pass without deleting or weakening them.",
                }
            )
        return out


def _file_of(node_id: str) -> str:
    """The file part of a pytest node id like `tests/test_x.py::test_y`."""
    return node_id.split("::", 1)[0] if node_id else ""


def failing_tests(output: str) -> list[str]:
    """Node ids from pytest's short summary (`FAILED x::y` / `ERROR x`)."""
    return sorted(failsig.FAILED_ID_RE.findall(output or ""))


def collect_baseline(
    ctx: ToolContext, trace: Trace, settings=None, cancel: threading.Event | None = None
) -> BaselineTests | None:
    """What was already failing before the agent touched anything.

    Run once, before execution. Without it the final check cannot tell work
    the agent broke from work that arrived broken, and in any real repository
    the second kind is common — which would either fail every run or, worse,
    send fix rounds off editing code the goal never mentioned.
    """
    if not _has_tests(ctx):
        return None
    if getattr(ctx, "test_command", ""):
        # The project's own command runs targets, not the whole suite (django's
        # takes minutes); before the run there are none to run.
        trace.event("test_baseline_skipped", reason="project test command")
        trace.say("   (no pre-run test baseline: the project's own test command runs targets)")
        return None
    root = ctx.workspace.root
    state = baseline_cache.relevant_state(root)
    if getattr(settings, "cache_baseline_across_runs", True):
        cached = baseline_cache.load(root, state)
        # A cache written before failure digests existed has failing tests and
        # no reason for them; rerunning once is cheaper than planning blind.
        if cached is not None and (
            cached.get("digest") is not None
            or not (cached["failures"] or cached["collection_broken"])
        ):
            baseline = _mark_unrunnable(
                BaselineTests(
                    failures=cached["failures"],
                    collection_broken=cached["collection_broken"],
                    digest=cached.get("digest") or "",
                ),
                root,
            )
            trace.event(
                "test_baseline",
                failures=baseline.failures,
                collection_broken=baseline.collection_broken,
                digest=baseline.digest,
                cached=True,
            )
            trace.say("workspace unchanged since the last recorded baseline — reusing it")
            _say_baseline(baseline, settings, trace)
            return baseline
    # Said before the run, not after it. This is the first thing that happens
    # once a goal is entered, it runs the whole suite, and in a traced run it
    # took 49 seconds during which the harness printed nothing at all -- so
    # the very first impression of a run was a frozen terminal. It now also
    # overlaps clarify and the person answering it -- see loop._start_baselines
    # -- but is still worth saying, since a run with clarify off has nothing
    # else happening while this runs.
    trace.say("checking which tests already pass before starting...")
    code, out, err, _ = _run_suite(
        ctx, parallel=getattr(settings, "test_parallel", False), cancel=cancel
    )
    if code == 127 or (cancel is not None and cancel.is_set()):
        # Stopped part-way (the run ended first): nothing measured, and nothing cached.
        return None
    combined = f"{out}\n{err}".strip()
    failures = failing_tests(combined)
    baseline = _mark_unrunnable(
        BaselineTests(
            failures=failures,
            collection_broken=code in (2, 3, 4),
            digest=failsig.format_digest(failsig.failure_digest(combined), total=len(failures)),
        ),
        root,
    )
    if getattr(settings, "cache_baseline_across_runs", True):
        baseline_cache.save(
            root, state, baseline.failures, baseline.collection_broken, baseline.digest
        )
    trace.event(
        "test_baseline",
        failures=baseline.failures,
        collection_broken=baseline.collection_broken,
        digest=baseline.digest,
    )
    _say_baseline(baseline, settings, trace)
    return baseline


def _say_baseline(baseline: BaselineTests, settings, trace: Trace) -> None:
    """Tell the person what the run will do about tests that start red.

    The old line said "they will not be counted against it" whatever the
    setting, including when `fix_existing_failures` (on by default) made them
    exactly the work: it read as if the run had been told to ignore them.
    """
    if baseline.unrunnable:
        hint = f"; {baseline.runner_hint}" if baseline.runner_hint else ""
        trace.say(
            f"note: pytest could not load {len(baseline.failures)} of this project's test "
            f"files, so it probably runs its tests another way{hint}. They are not "
            "treated as failing tests to fix (--test-command runs its own runner)."
        )
    elif baseline.failures:
        if getattr(settings, "fix_existing_failures", True):
            consequence = "this run will try to fix them"
        else:
            consequence = "they will not be counted against it"
        trace.say(
            f"note: {len(baseline.failures)} test(s) were already failing "
            f"before this run; {consequence}"
        )
    elif baseline.collection_broken:
        trace.say("note: the test suite could not be collected before this run started")


@dataclass
class BaselineTests:
    failures: list[str] = field(default_factory=list)
    collection_broken: bool = False
    # Why each is red, as "name: file:line in func -- error" lines. The run
    # that collected the baseline already held the whole traceback and kept
    # only the ids, so the planner was told which tests fail and nothing about
    # what they say.
    digest: str = ""
    # pytest could not load most of the suite: the project runs its tests some
    # other way, so its "failures" are not a measure of anything (§70).
    unrunnable: bool = False
    runner_hint: str = ""


# A baseline counts as unrunnable by pytest when at least this many test files
# fail at collection and they are at least half of what it reports. django's
# suite under pytest: 242 to 533 collection errors (ImproperlyConfigured, or no
# `django` on the path) and nothing else (DESIGN-LOG §69). A few broken imports
# in a working suite stay ordinary pre-existing failures.
UNRUNNABLE_MIN_FILES = 5
UNRUNNABLE_SHARE = 0.5
# Files that suggest how a project does run its tests, checked in this order.
RUNNER_HINTS = ("tests/runtests.py", "runtests.py", "manage.py", "tox.ini", "noxfile.py")


def _mark_unrunnable(baseline: BaselineTests, root) -> BaselineTests:
    """Flag a baseline pytest could mostly not collect, with a hint at the
    project's own runner when one is visible."""
    collection = [f for f in baseline.failures if "::" not in f]
    if len(collection) >= UNRUNNABLE_MIN_FILES and len(collection) >= UNRUNNABLE_SHARE * max(
        1, len(baseline.failures)
    ):
        baseline.unrunnable = True
        found = next((h for h in RUNNER_HINTS if (root / h).is_file()), "")
        baseline.runner_hint = f"`{found}` looks like its test runner" if found else ""
    return baseline


# `-p no:X` is a no-op when X is not installed (verified: pytest does not
# error on an unregistered plugin name), so both are safe to pass
# unconditionally rather than probing the workspace for what it has.
# no:cacheprovider skips the .pytest_cache read/write neither baseline nor
# gate ever benefit from (this run's tests are never "last failed"). no:randomly
# makes the suite's own ordering deterministic, which every fingerprint- and
# replay-based skip in this file already assumes.
_PYTEST_ARGS = [
    "-q",
    "--no-header",
    "--tb=short",
    "-p",
    "no:cacheprovider",
    "-p",
    "no:randomly",
    ".",
]


def _stop(cancel: threading.Event | None) -> dict:
    """`cancel` as a keyword for toolchain.run, only when there is one: the gate's own
    suite runs keep exactly the call they had before the baseline became cancellable."""
    return {"cancel": cancel} if cancel is not None else {}


def _run_suite(
    ctx: ToolContext,
    with_coverage: bool = False,
    parallel: bool = False,
    cancel: threading.Event | None = None,
) -> tuple[int, str, str, bool]:
    """Run the suite. Returns (code, stdout, stderr, coverage_recorded).

    With coverage on, the suite runs *once*, under `coverage run` — the data
    is read afterwards by a separate command that executes no tests, so
    measuring what the tests touched costs one extra fast subprocess rather
    than a second run of the suite. `uv run --with coverage` injects the
    dependency for that invocation only, leaving the workspace's own
    pyproject untouched.

    Falls back to a plain run if the coverage-wrapped one cannot start at
    all (no network to fetch coverage, no uv): a missing measurement must
    never cost the run its actual pass/fail signal.

    `parallel` is mutually exclusive with `with_coverage` on purpose. Plain
    `coverage run -m pytest -n auto` has every xdist worker write the same
    `.coverage` file at once — pytest-cov exists specifically to coordinate
    that, and this does not use it — so the combination is silently
    corrupted data rather than a faster measurement. Coverage wins: it gates
    `ok`, and a wrong answer there is worse than a slow one.
    """
    if getattr(ctx, "test_command", ""):
        return _run_project_suite(ctx)
    if with_coverage:
        code, out, err = toolchain.run(
            "uv",
            [
                "run",
                "--with",
                "coverage",
                "python",
                "-m",
                "coverage",
                "run",
                "--source=.",
                "-m",
                "pytest",
                *_PYTEST_ARGS,
            ],
            ctx.workspace.root,
            timeout=TEST_TIMEOUT_S,
            # coverage.py's default tracer is sys.settrace, which instruments
            # every line; coverage 7.4+ on Python 3.12+ can use sys.monitoring
            # instead, which is built for exactly this and costs far less per
            # line. Measured on a real 76-test suite: 52.5s -> 47.9s, ~9%,
            # on every single gate check that measures coverage -- which is
            # most of them, since check_coverage defaults on. An unrecognised
            # value degrades to a stderr warning and the default tracer, not
            # a failure (confirmed: exit code 0), so a coverage.py old enough
            # to predate this is unaffected rather than broken.
            extra_env={"COVERAGE_CORE": "sysmon"},
            **_stop(cancel),
        )
        if code != 127 and not _coverage_failed_to_start(out, err):
            return code, out, err, True
    elif parallel:
        code, out, err = toolchain.run(
            "uv",
            ["run", "--with", "pytest-xdist", "pytest", "-n", "auto", *_PYTEST_ARGS],
            ctx.workspace.root,
            timeout=TEST_TIMEOUT_S,
            **_stop(cancel),
        )
        if code != 127 and not _xdist_failed_to_start(out, err):
            return code, out, err, False
    code, out, err = toolchain.run(
        "pytest", _PYTEST_ARGS, ctx.workspace.root, timeout=TEST_TIMEOUT_S, **_stop(cancel)
    )
    return code, out, err, False


def _coverage_failed_to_start(out: str, err: str) -> bool:
    """True when the wrapper itself failed, as opposed to tests failing.

    "no module named" is the telling one and covers more than coverage's own
    absence: in a directory with no project for uv to inherit from, `uv run
    --with coverage` builds an environment holding coverage and nothing else,
    so it is *pytest* that goes missing. Either way the measurement is not
    available and the plain run has to provide the real answer — a bonus
    signal must never cost the run the one that matters.
    """
    blob = f"{out}\n{err}".lower()
    return any(
        sign in blob
        for sign in (
            "no module named",
            "failed to resolve",
            "no solution found",
            "error: failed to",
            "could not find",
        )
    )


def _xdist_failed_to_start(out: str, err: str) -> bool:
    """True when `uv run --with pytest-xdist` itself failed to start.

    Same detector as `_coverage_failed_to_start`, under its own name: the
    failure mode is identical (uv could not build the ephemeral environment,
    typically no network) and a wrapper that never ran must fall back to the
    plain suite rather than cost the run its actual pass/fail signal.
    """
    return _coverage_failed_to_start(out, err)


def _is_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def _untested_behaviour(ctx: ToolContext) -> list[str]:
    """Non-test Python this run changed, when the workspace has no tests.

    Deliberately narrow. It does not fire for a project that already has a
    suite (the suite itself is then the signal), nor for a run that only
    touched docs or config. It fires for the case that actually went wrong:
    hundreds of lines of behaviour written across several runs with no test
    ever created, so the gate had nothing to run and said PASS every time.
    """
    changed = [p for p in ctx.touched if p.endswith(".py") and not _is_test_file(p)]
    if not changed or _has_tests(ctx):
        return []
    return sorted(changed)


def _has_tests(ctx: ToolContext) -> bool:
    root = ctx.workspace.root
    for path in walk_files(root):
        name = path.name
        if path.suffix == ".py" and (name.startswith("test_") or name.endswith("_test.py")):
            return True
    return False


def fixes_pre_existing(settings, baseline: BaselineTests | None) -> bool:
    """Whether the gate's fix rounds take on tests that were red before the run:
    the setting, except when pytest could not load the suite at all."""
    unrunnable = baseline is not None and baseline.unrunnable
    return bool(getattr(settings, "fix_existing_failures", True)) and not unrunnable


def _run_project_suite(ctx: ToolContext) -> tuple[int, str, str, bool]:
    """Settings.test_command on the test files this run touched.

    Not the whole suite: a large project's (django's) takes minutes, and the
    hidden or pre-existing tests are the benchmark's business, not this gate's.
    No touched tests means nothing ran (127), which the gate reads as "tests did
    not run" rather than as a failure.
    """
    from ..tools.checks import project_test_targets, run_project_tests

    targets = project_test_targets(ctx)
    if not targets:
        return 127, "", "no test files touched this run to give the project's test command", False
    code, text = run_project_tests(ctx, targets, "", TEST_TIMEOUT_S)
    return code, text, "", False


def _run_suite_repairing(
    ctx: ToolContext,
    result: FinalCheck,
    trace: Trace,
    want_coverage: bool,
    want_parallel: bool = False,
) -> tuple[int, str, bool]:
    """The suite, with one repair attempt if it could not start. Returns
    (exit code, combined output, coverage recorded).

    A suite that never ran is not a suite that failed. Without this the gate
    reported "1 error in 0.03s" as a test failure and handed the fix rounds a
    blocker about code that was never executed — the error was a virtualenv
    missing its libpython. The extra run only ever happens on a suite that
    already produced nothing, so it costs nothing on the normal path.
    """
    root = ctx.workspace.root
    code, out, err, recorded = _run_suite(ctx, with_coverage=want_coverage, parallel=want_parallel)
    combined = f"{out}\n{err}".strip()
    if code == 0 or not envfail.diagnose(combined, root):
        return code, combined, recorded
    ran, note = envfail.repair(root)
    if ran:
        trace.say(f"   ~ test environment was broken; {note}, re-running")
        code, out, err, recorded = _run_suite(
            ctx, with_coverage=want_coverage, parallel=want_parallel
        )
        combined = f"{out}\n{err}".strip()
        if code == 0 or not envfail.diagnose(combined, root):
            return code, combined, recorded
    persisted = envfail.diagnose(combined, root)
    trace.say(f"   ~ could not run the test suite: {persisted}")
    trace.event("suite_environmental", note=str(persisted), repaired=ran)
    result.env_failures.append(
        {
            "step": "(test suite)",
            "command": "pytest",
            "note": f"{persisted} ({note})",
            "output": combined[-2_000:],
        }
    )
    # 127 is this module's existing "the suite did not run" code, which
    # _check_test_suite already reads as tests_ran = False.
    return 127, combined, False


_PYTEST_HEADS = (
    ("uv", "run", "pytest"),
    ("uv", "run", "python", "-m", "pytest"),
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
)
# Flags that make a pytest invocation ask a different question than the
# suite does, so its passing cannot be inferred from the suite's. `-p` loads
# or disables plugins, `--rootdir`/`-c` change which config applies, and
# `--doctest-modules` collects tests the plain suite run never collects.
_NOT_SUBSUMED_FLAGS = (
    "-p",
    "--rootdir",
    "-c",
    "--doctest-modules",
    "--co",
    "--collect-only",
    "-x",
    "--exitfirst",
    "--deselect",
    # Each of these changes which tests run or how, so the
    # suite's own green result says nothing about them:
    # markers the suite's addopts may exclude, warning
    # filters promoted to errors, ini overrides, a
    # different assertion or import mode, a coverage
    # threshold the suite never applied.
    "-m",
    "-W",
    "-o",
    "--assert",
    "--import-mode",
    "--pythonpath",
    "--cov-fail-under",
)
# `run_verify` executes with shell=True, and shlex.split keeps these as
# ordinary tokens — so `pytest x && ruff check .` matched the pytest head and
# was declared settled by a green suite that never ran ruff. Any of these
# means the command is not merely a pytest invocation.
_SHELL_METACHARACTERS = ("&&", "||", ";", "|", ">", "<", "$(", "`", "&")
# Directories pytest's own norecursedirs skips, which an explicit path
# argument would override -- so a verify naming one reaches tests the
# suite's bare `pytest .` never collected.
_NEVER_COLLECTED = frozenset({"build", "dist", "node_modules", "venv", "__pycache__"})


def _subsumed_by_suite(command: str) -> bool:
    """Whether a green whole-suite run already proves this command passes.

    The gate runs each step's own verify command and then, moments later,
    the entire test suite. In practice the first is almost always a subset
    of the second: one traced run's two step commands were `uv run pytest
    test_eval_and_compare.py -v -k permutation_importance` and `uv run
    pytest test_eval_and_compare.py::TestMain -v`, both strictly inside the
    72-test suite that followed them — 23 seconds of the gate, twice over,
    establishing nothing the suite did not.

    "Subsumed" is meant literally, not approximately: a pytest run that
    only *selects* (a path, `::Class`, `-k`, `-v`) asks a question the
    whole-suite run has already answered, because every test it would run,
    the suite ran. Anything that changes plugins, rootdir, config or
    collection is not a subset and is not treated as one, and neither is a
    command that is not pytest at all — a `pandoc` or `ruff` verify has no
    relationship to the suite whatsoever.

    This is only ever *acted* on when the suite comes back green; see
    `_resolve_deferred_verifies` for why that matters.
    """
    if any(meta in command for meta in _SHELL_METACHARACTERS):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    # An inline environment assignment ("FOO=1 pytest") runs under different
    # settings than the suite does, so it is not a subset of it.
    if not argv or "=" in argv[0]:
        return False
    head_len = next((len(head) for head in _PYTEST_HEADS if tuple(argv[: len(head)]) == head), 0)
    if not head_len:
        return False
    # Only the arguments *after* the invocation: `python -m pytest` spells the
    # interpreter's module flag `-m`, which is pytest's marker flag as well,
    # and reading it as the latter rejected every `python -m pytest` command.
    rest = argv[head_len:]
    if any(_carries_flag(arg, flag) for arg in rest for flag in _NOT_SUBSUMED_FLAGS):
        return False
    # Every path it names must be one the suite's own `pytest .` would have
    # collected. `pytest ../sibling/test_x.py` or an absolute path outside
    # the workspace is not inside the suite's collection root at all, and an
    # explicit path argument also overrides pytest's norecursedirs, so
    # `pytest build/test_x.py` reaches tests the suite never saw.
    # A flag's value is not a path: `-k slow` must not be read as one.
    paths = [
        arg
        for prev, arg in zip(["", *rest], rest, strict=False)
        if not arg.startswith("-") and not prev.startswith("-")
    ]
    return all(_inside_collection_root(arg) for arg in paths)


def _carries_flag(arg: str, flag: str) -> bool:
    """Whether `arg` is `flag`, `flag=value`, or a combined short form.

    The combined case is the one that bit: `-xvs` is three short flags in one
    token, so an equality test never saw the `-x`, and `-pno:randomly` hid a
    `-p` that disables a plugin the suite was running with.
    """
    if arg == flag or arg.startswith(flag + "="):
        return True
    single_dash = len(flag) == 2 and flag.startswith("-") and not flag.startswith("--")
    return bool(
        single_dash and arg.startswith("-") and not arg.startswith("--") and flag[1] in arg[1:]
    )


def _inside_collection_root(arg: str) -> bool:
    """Whether a pytest path argument sits under the suite's own root.

    Only the path part: `tests/test_x.py::TestCase::test_y` selects inside a
    file the suite collected, which is exactly the subset case this is for.
    """
    path = PurePosixPath(arg.split("::", 1)[0])
    if path.is_absolute() or ".." in path.parts:
        return False
    return not any(part in _NEVER_COLLECTED or part.startswith(".") for part in path.parts[:-1])


def _resolve_deferred_verifies(
    ctx: ToolContext, settings, deferred: list, result: FinalCheck, trace: Trace
) -> None:
    """Settle the step verifies that `_check_step_verifies` chose not to run.

    A green suite settles them by itself: every test each command selects is
    a test the suite just ran and passed, so running it again is arithmetic,
    not evidence.

    A red suite settles nothing, and this is the case worth being careful
    about. The gate turns a failed step verify into a finding carrying that
    step's id and files, which is what lets a fix round be pointed at the
    right place; a bare list of failing node ids does not carry that. So
    when the suite fails, every deferred command runs after all — the saving
    is given back precisely when the attribution it would have cost is worth
    having.
    """
    from .execute import workspace_fingerprint

    if not deferred:
        return
    if result.tests_ran and result.tests_passed:
        for step in deferred:
            step.verify_passed = True
            step.verify_touched = tuple(ctx.touched)
            step.verify_fingerprint = workspace_fingerprint(ctx, step.verify_touched)
            trace.event(
                "final_verify",
                step=step.id,
                command=step.verify,
                passed=True,
                skipped_subsumed=True,
            )
            trace.say(f"   ✓ [{step.id}] {step.verify} (covered by the suite)")
        return
    trace.say(
        f"   the suite is not green, so {len(deferred)} deferred step "
        "verify command(s) run after all"
    )
    for step in deferred:
        _run_one_step_verify(ctx, step, result, trace)


def _check_step_verifies(
    ctx: ToolContext,
    settings,
    plan: Plan | None,
    result: FinalCheck,
    trace: Trace,
    unchanged_since_execute: bool = False,
) -> list:
    """1. Every step's own verify command, against the finished code rather
    than against the state that existed right after that step ran.

    `unchanged_since_execute` lets a step skip re-running its own command a
    second time. Re-running unconditionally was the point once — see the
    comment below about a step whose failure only a later step revealed —
    but every step whose command already passed during execute, in a run
    where nothing has touched the workspace since, produces the identical
    result the second time by construction: the command is the same, the
    files it reads are byte-for-byte what they were, and nothing about the
    machine it runs on changed between the two calls. 267 of 296 traced step
    verifies in one sweep were `uv run pytest`, run once during execute and
    then unconditionally again here — most of that second run bought nothing
    a review round with no findings could not have told you for free.

    Commands a green suite would settle by itself are not run here at all.
    They are returned instead, for `_resolve_deferred_verifies` to settle
    once the suite's own result is known — deferred rather than dropped,
    because a red suite still needs each one's per-step attribution.
    """
    from .execute import workspace_fingerprint

    if not (plan and settings.verify_steps):
        return []
    suite_will_run = _has_tests(ctx)
    deferred: list = []
    for step in plan.steps:
        if not step.verify.strip():
            continue
        # Two separate questions have to both be "no" to skip re-running: has
        # anything changed anywhere since execute finished (the review phase
        # is the only thing that runs after that point), and has anything
        # this specific step's own passing run depended on changed since
        # *it* passed. The first alone misses a later step, during execute,
        # editing a file an earlier step's already-passed verify depended
        # on — both fingerprints exist by the time execute finishes, so
        # "unchanged since execute" is trivially true even though the
        # earlier step's result is now stale. Requiring the step's own
        # fingerprint to still match is what catches that case.
        if (
            unchanged_since_execute
            and step.verify_passed is True
            and step.verify_fingerprint
            and workspace_fingerprint(ctx, step.verify_touched) == step.verify_fingerprint
        ):
            trace.event(
                "final_verify",
                step=step.id,
                command=step.verify,
                passed=True,
                skipped_unchanged=True,
            )
            trace.say(f"   ✓ [{step.id}] {step.verify} (unchanged since execute)")
            continue
        if (
            suite_will_run
            and getattr(settings, "skip_subsumed_verifies", True)
            and _subsumed_by_suite(step.verify)
        ):
            deferred.append(step)
            continue
        _run_one_step_verify(ctx, step, result, trace)
    return deferred


def _verify_outcome(ctx: ToolContext, command: str) -> tuple[bool, str, str]:
    """(passed, output, env_note) for a verify command against the finished state.

    `env_note` is set when the command could not run at all: a broken environment that
    survived one repair (envfail), a command the approver refused in an unattended run, or a
    command that selects no test at all (verifycmd: a label that does not import, a named test
    that does not exist; 11 of 55 django runs, DESIGN-LOG §110).
    A refusal used to count as a failure: GAIA scout g2 had 50 of 83 final verifies refused
    (`test -s answer.txt`), and one fix round, told the check failed, overwrote a correct
    answer with "PASS" (DESIGN-LOG §93). A fix round cannot change what is approved.

    `run_verify` is imported here, not at module scope: tests patch `execute.run_verify`.
    """
    from .execute import run_verify, unrunnable_note

    passed, output = run_verify(ctx, command)
    if passed:
        return True, output, ""
    note = unrunnable_note(output)
    if note:
        return False, output, note
    return envfail.repair_and_retry(output, ctx.workspace.root, lambda: run_verify(ctx, command))


def _run_one_step_verify(ctx: ToolContext, step, result: FinalCheck, trace: Trace) -> None:
    """Run one step's verify command (`_verify_outcome`) and record what it says."""
    from .execute import workspace_fingerprint

    ctx.cancel.check()
    passed, output, env_note = _verify_outcome(ctx, step.verify)
    if env_note:
        # Unrunnable, not failed. Recording it as a verify failure would
        # make it a blocker finding, and the fix rounds would spend
        # themselves asking the model to repair a virtualenv.
        step.verify_passed = None
        step.verify_error = env_note
        trace.event(
            "final_verify",
            step=step.id,
            command=step.verify,
            passed=False,
            environmental=True,
            note=env_note,
        )
        trace.say(f"   ~ [{step.id}] could not run: {env_note}")
        result.env_failures.append(
            {
                "step": step.id,
                "command": step.verify,
                "note": env_note,
                "output": output[-2_000:],
            }
        )
        return
    # The finished state is the authority, so this overrides whatever the
    # verify said mid-run. Without it a step that failed during execution
    # and was repaired by a fix round stayed marked FAILED forever: one
    # real run ended "final check: PASS" and "4 FAILED verification" in
    # the same breath, and exited non-zero on work that was fine.
    step.verify_passed = passed
    if passed:
        step.verify_touched = tuple(ctx.touched)
        step.verify_fingerprint = workspace_fingerprint(ctx, step.verify_touched)
    else:
        step.verify_fingerprint = ""
        step.verify_touched = ()
    trace.event(
        "final_verify", step=step.id, command=step.verify, passed=passed, output=output[-600:]
    )
    if passed:
        trace.say(f"   ✓ [{step.id}] {step.verify}")
    else:
        trace.say(f"   ✗ [{step.id}] {step.verify}")
        result.verify_failures.append(
            {
                "step": step.id,
                "command": step.verify,
                "output": output,
                "file": step.files[0] if step.files else "",
                "files": list(step.files),
            }
        )


def _check_goal_verify(ctx: ToolContext, settings, result: FinalCheck, trace: Trace) -> None:
    """The goal's own verification command, if one was given (Settings.goal_verify
    / `--verify`), run once against the finished state alongside every step's
    own verify. Failure lands in the same `verify_failures` list a step verify
    failure would — same shape, same fix path, same escalation — with
    `step: "goal"` as its identity so it reads clearly in a report without
    needing a second kind of finding anywhere downstream.
    """
    command = getattr(settings, "goal_verify", None)
    if not command or not command.strip():
        return
    ctx.cancel.check()
    passed, output, env_note = _verify_outcome(ctx, command)
    if env_note:
        trace.event(
            "final_verify",
            step="goal",
            command=command,
            passed=False,
            environmental=True,
            note=env_note,
        )
        trace.say(f"   ~ [goal] could not run: {env_note}")
        result.env_failures.append(
            {
                "step": "goal",
                "command": command,
                "note": env_note,
                "output": output[-2_000:],
            }
        )
        return
    trace.event("final_verify", step="goal", command=command, passed=passed, output=output[-600:])
    if passed:
        trace.say(f"   ✓ [goal] {command}")
    else:
        trace.say(f"   ✗ [goal] {command}")
        result.verify_failures.append(
            {
                "step": "goal",
                "command": command,
                "output": output,
                "file": "",
                "files": [],
            }
        )


def _check_syntax(ctx: ToolContext, changed: list[str], result: FinalCheck, trace: Trace) -> None:
    """2. Syntax, which nothing should ever fail this late but which costs
    milliseconds and makes every later signal trustworthy."""
    for rel in changed:
        path = ctx.resolve_existing(rel)
        if path is None:
            continue
        diagnostic = validate.syntax_check(path)
        if diagnostic:
            result.syntax_errors.append(diagnostic)
    if result.syntax_errors:
        trace.say(f"   ✗ {len(result.syntax_errors)} file(s) do not parse")


def _check_test_suite(
    ctx: ToolContext,
    baseline: BaselineTests | None,
    result: FinalCheck,
    trace: Trace,
    settings=None,
) -> None:
    """3. The test suite. Whole-suite, not -x: the point is to find
    everything that is broken now, not to stop at the first thing."""
    if not _has_tests(ctx):
        # A baseline only exists when tests were present and collectable at
        # the start of the run (see collect_baseline) -- so reaching here
        # with one in hand means tests existed then and don't now. Distinct
        # from a project that never had a suite, which is exactly what
        # "no test files found; skipping" would otherwise make this look
        # identical to.
        if baseline is not None:
            result.deleted_test_suite = True
            trace.say("   ✗ tests existed when this run started and do not now")
        else:
            trace.say("   - no test files found; skipping the suite")
        return

    ctx.cancel.check()
    trace.say("   … running the test suite")
    # Whether to wrap the suite in `coverage run` has to be decided before it
    # runs, because the data only exists if it was collected. So the question
    # is not "did anything else fail" — types have not been checked yet — but
    # "is this round already lost". A round with a failed verify command or a
    # syntax error is going back through a fix round whatever coverage says,
    # and measuring what a doomed round covered is the one case where the
    # wrapper's overhead buys nothing. In traced runs coverage decided the
    # verdict alone in 2 of 12 failing gates and co-occurred with a verify
    # failure in 3 more; it is those 3, and every intermediate fix round,
    # that this skips.
    already_failed = bool(result.verify_failures or result.syntax_errors)
    want_coverage = getattr(settings, "check_coverage", True) and not already_failed
    if already_failed and getattr(settings, "check_coverage", True):
        trace.event(
            "coverage_skipped",
            reason="round already failing",
            verify_failures=len(result.verify_failures),
            syntax_errors=len(result.syntax_errors),
        )
    # Coverage takes priority when both are requested — see _run_suite's
    # docstring on why the combination corrupts rather than helps.
    want_parallel = getattr(settings, "test_parallel", False) and not want_coverage
    code, combined, recorded = _run_suite_repairing(
        ctx, result, trace, want_coverage, want_parallel
    )
    result.tests_ran = code != 127
    result.tests_passed = code == 0
    # A test file that collects zero tests is not a successful verification.
    # Treat pytest's exit code 5 as a failed gate instead of allowing a
    # file containing only helpers or malformed discovery patterns to produce
    # a false green result.
    if code == 5:
        result.tests_ran, result.tests_passed = True, False
    result.tests_output = combined
    result.tests_summary = failsig.pytest_summary(combined, code)

    if baseline is not None:
        result.baseline_known = True
        was_failing = set(baseline.failures)
        now_failing = set(failing_tests(combined))
        # Only the ones *still* failing. Reporting the whole baseline here
        # meant a run that repaired every one of them was still told "1 test
        # already failing before this run", and the fix loop was handed a
        # finding about tests that now pass — which the executor correctly
        # called a false positive before giving up on it.
        result.pre_existing_failures = sorted(was_failing & now_failing)
        result.new_failures = sorted(now_failing - was_failing)
        result.fixed_pre_existing = sorted(was_failing - now_failing)
        if result.pre_existing_failures:
            result.pre_existing_why = failsig.format_digest(
                failsig.failure_digest(combined), total=len(now_failing)
            )
        # Collection breaking during the run is this run's doing even though
        # no individual node id can be named for it.
        if code in (2, 3, 4) and not baseline.collection_broken:
            result.new_failures = sorted({*result.new_failures, "(the suite no longer collects)"})
        if code == 5:
            result.new_failures = sorted({*result.new_failures, "(no tests collected)"})
    elif code == 5:
        result.new_failures = ["(no tests collected)"]

    unparsed = code not in (0, 127) and not failing_tests(combined)
    trace.event(
        "final_tests",
        exit_code=code,
        summary=result.tests_summary,
        new_failures=result.new_failures,
        pre_existing=len(result.pre_existing_failures),
        # See checks.UNPARSED_TAIL_CHARS: a failure that names no test id.
        **({"unparsed_tail": combined[-1_500:]} if unparsed else {}),
    )
    if result.tests_ran:
        trace.say(f"   {_test_mark(result)} tests: {result.tests_summary}{_test_note(result)}")
    # Only worth asking "did the tests run the new code" once they pass:
    # while they are failing, that is the finding to act on, and coverage of
    # a half-executed suite would just be noise on top of it.
    if recorded and result.tests_passed:
        result.coverage = coverage_module.measure(
            ctx,
            trace,
            minimum=getattr(settings, "min_changed_coverage", coverage_module.DEFAULT_MIN_COVERAGE),
        )


def _test_mark(result: FinalCheck) -> str:
    return "✓" if (result.tests_passed or not result.failing_tests) else "✗"


def _test_note(result: FinalCheck) -> str:
    parts = []
    if result.new_failures:
        parts.append(f"{len(result.new_failures)} newly failing")
    if result.pre_existing_failures:
        parts.append(
            f"{len(result.pre_existing_failures)} still failing from before"
            + ("" if result.fix_pre_existing else ", not counted against it")
        )
    if result.fixed_pre_existing:
        parts.append(f"{len(result.fixed_pre_existing)} pre-existing failure(s) fixed")
    return f" — {'; '.join(parts)}" if parts else ""


def _diag_key(d: validate.Diagnostic) -> str:
    """A stable identity for one ty diagnostic, for comparing a pre-run
    baseline against what the finished workspace reports. Deliberately
    excludes the message: a timeout diagnostic's message embeds raw stderr
    that can vary harmlessly between two runs of the same timed-out check,
    and would otherwise look like a "new" error every time."""
    return f"{d.file}:{d.line}:{d.code}"


def collect_type_baseline(ctx: ToolContext, trace: Trace) -> frozenset[str] | None:
    """Type diagnostics that already existed before this run touched
    anything. Mirrors collect_baseline for tests, for the same reason: ty on
    a partly-typed project routinely reports pre-existing debt, and without
    knowing what that debt already was, a type error this run actually
    introduced can never be told apart from one that arrived broken — which
    is exactly why type errors were never gated on at all until now.
    """
    if not _has_python(ctx):
        return None
    diagnostics = validate.ty_check(ctx.workspace.root, None)
    baseline = frozenset(_diag_key(d) for d in diagnostics)
    trace.event("type_baseline", count=len(baseline))
    if baseline:
        trace.say(
            f"note: {len(baseline)} type issue(s) already existed before this run; "
            "they will not be counted against it"
        )
    return baseline


# New type errors gate a run only in a project that type-checks: one that
# configures a checker, or one whose existing type debt is small enough that a
# new error means something. django configures none and arrives with ~15,000;
# there, 1-4 "new" errors that shift with every edit drove all three of a
# run's fix rounds while its tests passed (DESIGN-LOG §71). Across 3,285 bench
# runs the pre-run debt was 0-10, so the bench keeps its gate unchanged.
TYPE_DEBT_UNGATED = 100


def type_gate_applies(root, type_baseline: frozenset[str] | None) -> bool:
    """Whether this run's new type errors should fail the final check."""
    if type_baseline is None or len(type_baseline) <= TYPE_DEBT_UNGATED:
        return True
    return configures_type_checker(root)


def _has_python(ctx: ToolContext) -> bool:
    """Whether ty has anything at all to check. Skips the pre-run baseline
    (and the cost of a ty invocation) for a workspace this harness might
    still legitimately be pointed at that just is not a Python project."""
    root = ctx.workspace.root
    return any(p.suffix == ".py" for p in walk_files(root))


def new_type_errors_text(ctx: ToolContext) -> str:
    """Type errors this run has introduced so far, as text for the after-edit check, or "".

    The final check's own rule (new against the run's starting baseline, gated only where
    type_gate_applies), run right after the edit that caused them: 15% of failed final checks
    failed on new type errors alone (288 of 1,882), each then a fix round (DESIGN-LOG §104).
    "" when the gate does not apply or nothing is new. A baseline of None means the workspace
    had no Python when the run started (a new project): as in `_check_types`, every error is
    then new. Skipping that case left semver, the task where new type errors are common, with
    no report at all (§105 smoke). By execution time the planner has waited for the baseline,
    so None never means "not collected yet".
    """
    baseline = getattr(ctx, "type_baseline", None)
    root = ctx.workspace.root
    if not type_gate_applies(root, baseline):
        return ""
    found = validate.ty_check(root, None)
    new = found if baseline is None else [d for d in found if _diag_key(d) not in baseline]
    if not new:
        return ""
    return f"ty: {len(new)} type error(s) this run introduced:\n" + validate.render(new, limit=8)


def _check_types(
    ctx: ToolContext,
    changed: list[str],
    result: FinalCheck,
    trace: Trace,
    type_baseline: frozenset[str] | None = None,
) -> None:
    """4. Types, across the whole workspace ty can see (it needs the rest of
    the project to resolve imports even when only `changed` was touched).

    Type debt the workspace already had is reported, not gated — ty on a
    partially-typed project produces findings that are real but predate this
    run. A type error this run actually introduced is different: it is
    exactly as much this run's fault as a test it broke, and is now treated
    that way (see FinalCheck.new_type_errors / .ok).
    """
    if not changed:
        return
    result.type_errors = validate.ty_check(ctx.workspace.root, None)
    if type_baseline is None:
        result.new_type_errors = list(result.type_errors)
    else:
        result.type_baseline_known = True
        result.new_type_errors = [
            d for d in result.type_errors if _diag_key(d) not in type_baseline
        ]
    if result.new_type_errors and not type_gate_applies(ctx.workspace.root, type_baseline):
        # Reported, not gated: see TYPE_DEBT_UNGATED.
        trace.event(
            "type_gate_skipped",
            new=len(result.new_type_errors),
            pre_existing=len(type_baseline or ()),
        )
        trace.say(
            f"   - ty: {len(result.new_type_errors)} new issue(s) reported, not gated: the "
            f"project configures no type checker and already had {len(type_baseline or ())}"
        )
        result.new_type_errors = []
    if result.type_errors:
        pre_existing = len(result.type_errors) - len(result.new_type_errors)
        note = f"{len(result.new_type_errors)} new" if result.new_type_errors else "none new"
        if pre_existing:
            note += f", {pre_existing} pre-existing"
        trace.say(f"   - ty: {len(result.type_errors)} issue(s) ({note})")


def run_final_check(
    ctx: ToolContext,
    settings,
    plan: Plan | None,
    trace: Trace,
    baseline: BaselineTests | None = None,
    unchanged_since_execute: bool = False,
    type_baseline: frozenset[str] | None = None,
) -> FinalCheck:
    """Re-verify the finished state. Makes no model calls.

    Four checks, run in this order for exactly the reason they're numbered in
    each helper's docstring: cheapest and most trustworthy first. Each one
    still runs exactly the subprocesses it always did — one run_verify per
    step with a verify command, one pytest invocation, one ty_check call —
    this function just no longer reads them all at once.

    `unchanged_since_execute` is the caller's answer to "did review or a fix
    round touch anything after execute finished?" — computed once, cheaply,
    from workspace_fingerprint(), and passed down rather than recomputed
    here so this function stays about what to check, not how to tell whether
    checking again is necessary.

    `type_baseline` is collect_type_baseline's answer to "what type errors
    already existed before this run," in the same spirit and for the same
    reason `baseline` is for tests.
    """
    # An unrunnable baseline's "failures" are pytest failing to load the suite,
    # not tests to repair: chasing them sent fix rounds after the environment.
    result = FinalCheck(fix_pre_existing=fixes_pre_existing(settings, baseline))
    trace.say("\n=== phase 4: final check (no model calls) ===", detail=True)
    # The guard still holds the last step's spent shell budget, and run_verify checks it:
    # 10 final verifies in 5 runs "failed" with "used its entire 90s shell budget" before
    # running (found by a parallel session, DESIGN-LOG §110).
    ctx.guard.reset_step()

    deferred = _check_step_verifies(ctx, settings, plan, result, trace, unchanged_since_execute)
    changed = [p for p in ctx.touched if p.endswith(".py")]
    _check_syntax(ctx, changed, result, trace)
    _check_test_suite(ctx, baseline, result, trace, settings)
    # After the suite, because a green suite is what settles these without
    # running them — and a red one is what makes running them worth it.
    _resolve_deferred_verifies(ctx, settings, deferred, result, trace)
    # Last of the commands that run, so it sees the state the run ends in: a GAIA run's own
    # test deleted answer.txt in its cleanup, after the goal check had passed on it
    # (DESIGN-LOG §93.3).
    _check_goal_verify(ctx, settings, result, trace)
    _check_types(ctx, changed, result, trace, type_baseline)
    if getattr(settings, "require_tests", True):
        result.untested_changes = _untested_behaviour(ctx)
        if result.untested_changes:
            trace.say(
                f"   ✗ {len(result.untested_changes)} file(s) of behaviour changed "
                "with no tests anywhere in the workspace"
            )

    trace.event("final_check", **result.as_dict())
    trace.say(
        "   final check: "
        + ("PASS" if result.ok else "INCONCLUSIVE" if result.inconclusive else "FAIL")
    )
    return result
