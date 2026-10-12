"""Did the tests actually execute the code this run changed?

"Tests exist and pass" turned out to be a much weaker signal than it sounds.
A real run wrote five passing tests about scoring and preferences while both
reported bugs survived, because the tests reached into the objects and set
state by hand instead of driving the code that had just been written. The
suite was green and the new code had never run.

This answers the narrower, checkable question instead: of the lines this run
added or modified, how many did the test suite execute? It is deliberately
scoped to *changed* lines rather than whole-file coverage — a project at 40%
coverage is a judgement call, but "you wrote forty lines and the tests ran
none of them" is not.

What it cannot see is worth stating plainly: a bug in a line the run did not
touch is invisible here, and a line being executed says only that it ran, not
that anything asserted about the result. This narrows the gap between "green"
and "works"; it does not close it.
"""

from __future__ import annotations

import ast
import difflib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import toolchain
from ..tools import ToolContext
from ..trace import Trace

# Below this many changed executable lines, a low fraction is usually a
# one-line tweak rather than a body of untested work, and saying so is noise.
MIN_LINES_TO_REPORT = 3
# A file is flagged when the tests ran less than this fraction of what the run
# changed in it. Not "zero lines covered", which sounds stricter and is in
# fact unreachable: `def f(...)` executes at import, so a function nobody ever
# calls still shows one covered line and would slip through untouched.
DEFAULT_MIN_COVERAGE = 0.5
COVERAGE_JSON = "coverage-report.json"


@dataclass
class CoverageReport:
    """Per file: how many of its changed executable lines the tests ran."""

    measured: bool = False
    # rel path -> (covered changed lines, total changed executable lines)
    per_file: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Files with a real body of changed code that the suite barely runs.
    unexecuted: list[str] = field(default_factory=list)
    # "game.py::Bird.draw" for each changed function no test ever entered,
    # reported whatever the file's overall ratio says.
    dead_functions: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def covered(self) -> int:
        return sum(c for c, _ in self.per_file.values())

    @property
    def total(self) -> int:
        return sum(t for _, t in self.per_file.values())

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.total if self.total else 100.0

    def as_dict(self) -> dict:
        return {
            "measured": self.measured,
            "covered_changed_lines": self.covered,
            "total_changed_lines": self.total,
            "percent": round(self.percent, 1),
            "unexecuted_files": self.unexecuted,
            "unexecuted_functions": self.dead_functions,
            "note": self.note,
        }


def changed_lines(before: str | None, after: str) -> set[int]:
    """1-indexed line numbers in `after` that are new or modified.

    A file with no baseline (created this run) counts entirely as changed.
    """
    after_lines = after.splitlines()
    if before is None:
        return set(range(1, len(after_lines) + 1))
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after_lines, autojunk=False)
    touched: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            touched.update(range(j1 + 1, j2 + 1))
    return touched


# A changed function with this many executable lines, none of which ran, is
# reported even when the file's overall ratio looks fine. Below it, a one- or
# two-line helper nobody calls is noise.
MIN_FUNCTION_LINES = 3


def functions_at(source: str) -> list[tuple[str, int, int]]:
    """(qualified name, first line, last line) for every def in the module."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                end = getattr(child, "end_lineno", child.lineno) or child.lineno
                # From the first statement, not from `def`: the signature line
                # executes when the module is imported, so counting it would
                # make every function look entered. This is the same trap that
                # made "zero lines covered" an unreachable threshold.
                start = child.body[0].lineno if child.body else child.lineno
                found.append((f"{prefix}{child.name}", start, end))
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return found


def untouched_functions(
    source: str, touched: set[int], executed: set[int], executable: set[int]
) -> list[str]:
    """Changed functions the suite never entered.

    The gate used to ask only "what fraction of this file's changed lines
    ran", and a file answers that with an average. Over ten traced runs on one
    game, `Bird.draw` was edited repeatedly and never executed by a single
    test, while the same runs also changed `Bird.update` and `Bird.tap`, which
    are tested — so the ratio came out at 50-90% and the gate reported nothing
    unexecuted, ten times, while the feature the person had asked for was
    invisible on screen. An average over a file hides a function inside it.
    """
    out: list[str] = []
    for name, first, last in functions_at(source):
        body = {n for n in range(first, last + 1)} & executable
        changed_here = body & touched
        if len(changed_here) < MIN_FUNCTION_LINES:
            continue
        if not (body & executed):
            out.append(name)
    return out


def _run_json(ctx: ToolContext) -> dict | None:
    """Turn the recorded coverage data into JSON we can read. No tests run."""
    # Written outside the workspace: at the workspace root, a project's own
    # file of this name was overwritten and then deleted by the cleanup below.
    scratch = Path(tempfile.mkdtemp(prefix="harness-coverage-"))
    report = scratch / COVERAGE_JSON
    code, _out, _err = toolchain.run(
        "uv",
        [
            "run",
            "--with",
            "coverage",
            "python",
            "-m",
            "coverage",
            "json",
            "-o",
            str(report),
            "--quiet",
        ],
        ctx.workspace.root,
        timeout=120,
    )
    if code != 0:
        report.unlink(missing_ok=True)
        return None
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def measure(
    ctx: ToolContext, trace: Trace, minimum: float = DEFAULT_MIN_COVERAGE
) -> CoverageReport:
    """Compare what the suite executed against what the run changed.

    Assumes the suite has just been run under `coverage run` (see
    finalcheck._run_suite), so this only reads the data it left behind.
    """
    result = CoverageReport()
    data = _run_json(ctx)
    if data is None:
        result.note = "coverage data was not available"
        return result

    files = data.get("files") or {}
    with ctx.lock:
        snapshots = dict(ctx.baseline.snapshots)

    for rel, before in snapshots.items():
        if rel.endswith(".py"):
            _measure_one(ctx, rel, before, files, result, minimum)

    result.measured = bool(result.per_file)
    result.unexecuted.sort()
    result.dead_functions.sort()
    trace.event("coverage", **result.as_dict())
    if result.measured:
        trace.say(
            f"   - coverage of changed lines: {result.covered}/{result.total} "
            f"({result.percent:.0f}%)"
        )
    if result.dead_functions:
        trace.say("   ✗ changed but never run by a test: " + ", ".join(result.dead_functions[:6]))
    return result


def _measure_one(
    ctx: ToolContext,
    rel: str,
    before: str | None,
    files: dict,
    result: CoverageReport,
    minimum: float,
) -> None:
    """Fold one changed file's coverage into `result`."""
    after = ctx.try_read(rel)
    if after is None:
        return  # deleted this run; nothing left to execute
    touched = changed_lines(before, after)
    if not touched:
        return
    info = _entry_for(files, rel)
    executed = set(info.get("executed_lines") or [])
    executable = touched & (executed | set(info.get("missing_lines") or []))
    if not executable:
        return  # comments, docstrings, blank lines
    covered = len(touched & executed)
    result.per_file[rel] = (covered, len(executable))
    if len(executable) >= MIN_LINES_TO_REPORT and covered / len(executable) < minimum:
        result.unexecuted.append(rel)
    result.dead_functions.extend(
        f"{rel}::{name}" for name in untouched_functions(after, touched, executed, executable)
    )


def _entry_for(files: dict, rel: str) -> dict:
    """Coverage keys paths its own way; match on the tail of the path."""
    if rel in files:
        return files[rel] or {}
    for key, info in files.items():
        if key.endswith(rel) or rel.endswith(key.lstrip("./")):
            return info or {}
    return {}


def finding_for(report: CoverageReport) -> dict | None:
    """The blocker for code the tests never ran, or None."""
    if report.dead_functions and not report.unexecuted:
        return _dead_function_finding(report)
    if not report.unexecuted:
        return None
    listed = ", ".join(report.unexecuted[:10])
    return {
        "file": report.unexecuted[0],
        "files": list(report.unexecuted[:6]),
        "severity": "blocker",
        "issue": (
            f"the tests barely execute the code this run changed in {listed} — "
            f"overall they ran {report.covered} of {report.total} changed lines "
            f"({report.percent:.0f}%)"
        ),
        "failure": (
            "The suite passes, but it hardly runs this work. A green suite that "
            "does not reach the new code proves only that the old code still "
            "behaves; it says nothing about what was just written. Note that a "
            "`def` line counts as executed merely by being imported, so even the "
            "covered count here overstates how much was really exercised."
        ),
        "fix": (
            "Add tests that exercise these lines through the entry point a caller "
            "actually uses, rather than setting internal state by hand until an "
            "assertion holds. If a line genuinely cannot be reached from a test, say "
            "why in your summary instead of leaving it silently unexercised."
        ),
    }


def _dead_function_finding(report: CoverageReport) -> dict:
    """A function the run changed and no test ever entered.

    Distinct from the whole-file ratio, which an average can hide. The traced
    case: ten runs editing `Bird.draw` to make a wing appear, six passing tests
    about the wing, and `draw` never executed once — because the tests called a
    helper in the test file that reimplemented the drawing arithmetic and
    compared it to hardcoded points. Every run reported PASS; the wing was
    never on screen.
    """
    listed = ", ".join(report.dead_functions[:6])
    first = report.dead_functions[0].split("::")[0]
    return {
        "file": first,
        "files": sorted({name.split("::")[0] for name in report.dead_functions})[:6],
        "severity": "blocker",
        "issue": (
            f"this run changed {listed} and no test ever called "
            f"{'them' if len(report.dead_functions) > 1 else 'it'}"
        ),
        "failure": (
            "The suite is green and these functions never ran during it. Passing "
            "tests about code that is never executed prove nothing about the code — "
            "the usual way this happens is a test that reimplements the logic it "
            "means to check, or builds the expected value with the same arithmetic "
            "the function uses, so the assertion holds no matter what the function "
            "does."
        ),
        "fix": (
            "Call the function itself from a test and assert on what it produces or "
            "changes. If it draws to a surface or otherwise has no return value, pass it "
            "a stub or a recording double and assert on the calls it made — that still "
            "executes the real code. Do not copy the function's body into the test file "
            "and compare the copy against constants."
        ),
    }
