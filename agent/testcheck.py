"""Whether the tests a run leaves behind can actually see the change it made.

A green suite says the code and the tests agree. It does not say the tests
would notice if the code were wrong, and the two commonest ways this harness
has of being wrong are both invisible to it: a test written after the fix that
asserts what the fix happens to do (so it passes on the code from before the
run too), and a test that recomputes its expectation the way the code does (so
it passes on almost any code). Measured over 204 bench runs on existing-code
tasks, 15% wrote tests that pass against the original bug; weakening a seeded
assertion happened in none, and removing a seeded test in 2.5%. See DESIGN-LOG
§46.

Three checks, none needing a model call, run once at the end of a green run in
a scratch copy of the workspace (the real one is never edited):

- **revert**: restore the source files this run changed to their pre-run
  contents and run the tests the run added. If they all still pass, they do not
  exercise the change. This is bench/integrity.py's `own_tests_miss_the_bug`,
  ported to real runs using the snapshots the run already keeps.
- **mutation**: make small changes (flip a comparison, swap an operator, shift a
  constant) to the functions this run changed, and run the tests. A change no
  test notices is a place the tests cannot see. Survivors are reported, not
  proven meaningful: some mutants are equivalent to the original.
- **ledger**: classify each edit to a test file that existed before the run as
  additive, rewritten or reduced, and for a rewrite or reduction compare how
  many mutants the old version of the file killed against the new one. A test
  edit is judged by whether the file still sees as much, not by whether its
  diff looks suspicious, which is how a legitimate update to a test of
  behaviour the goal changed is told apart from a loosened assertion.

Report-only: nothing here changes `ok`. Mutation on tiny fixtures separated the
designed-weak tasks from the adequate ones cleanly (seeded tests killed 1 of 11,
0 of 8 and 2 of 13 mutants on the three tests-that-cannot-see-it fixtures, 11 of
11 on the adequate one) but a real project has no oracle to say a surviving
mutant matters, so this informs a reviewer rather than gating a run.

Python only, like `agent/outline.py`; docs/multi-language-gaps.md lists what
another language would need.
"""

from __future__ import annotations

import ast
import contextlib
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import langtools, toolchain
from .diffs import Baseline
from .phases.coverage import changed_lines
from .workspace import IGNORED_DIRS

MAX_MUTANTS = 12
AUDIT_BUDGET_S = 45.0
PER_RUN_TIMEOUT_S = 20
MAX_COPY_FILE_BYTES = 2_000_000

_COMPARE_FLIP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
_BINOP_SWAP: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult,
}
_PYTEST_FLAGS = [
    "-q",
    "--no-header",
    "--tb=no",
    "-x",
    "-p",
    "no:cacheprovider",
    "-p",
    "no:randomly",
]


@dataclass
class Mutant:
    file: str
    line: int
    kind: str
    function: str
    source: str

    def describe(self) -> str:
        where = f"{self.file}:{self.line}"
        return f"{where} ({self.kind}{', in ' + self.function if self.function else ''})"


@dataclass
class TestAudit:
    ran: bool = False
    skipped: str = ""
    # Test names the run added, and whether every one still passed against the
    # source as it was before the run. None when the check could not run.
    added_tests: list[str] = field(default_factory=list)
    added_tests_pass_on_old_code: bool | None = None
    mutants_tried: int = 0
    survivors: list[str] = field(default_factory=list)
    # {"tests/test_x.py": "additive" | "rewritten" | "reduced"} for test files
    # that existed before the run.
    test_edits: dict[str, str] = field(default_factory=dict)
    # Mutants killed by the pre-run and the current versions of the edited test
    # files, when a file was rewritten or reduced. None otherwise.
    killed_before: int | None = None
    killed_after: int | None = None
    seconds: float = 0.0

    @property
    def findings(self) -> list[str]:
        out: list[str] = []
        if self.added_tests_pass_on_old_code:
            names = ", ".join(self.added_tests[:4])
            out.append(
                f"the {len(self.added_tests)} test(s) added this run pass against the "
                f"code as it was before the run, so they do not exercise the change "
                f"({names})"
            )
        if self.survivors:
            out.append(
                f"{len(self.survivors)} of {self.mutants_tried} small changes to the "
                f"functions edited this run were not noticed by any test: "
                + "; ".join(self.survivors[:3])
            )
        if (
            self.killed_before is not None
            and self.killed_after is not None
            and self.killed_after < self.killed_before
        ):
            edited = ", ".join(
                sorted(k for k, v in self.test_edits.items() if v in ("rewritten", "reduced"))
            )
            out.append(
                f"the edited test file(s) ({edited}) now catch {self.killed_after} of "
                f"{self.mutants_tried} mutants, down from {self.killed_before}"
            )
        return out

    def fix_findings(self, source_diff: str = "") -> list[dict]:
        """Findings in the shape the fix round takes, for the two signals
        strong enough to act on.

        Survivors alone are not: a mutant no test notices is sometimes
        equivalent to the original, so acting on it would send a model to
        "fix" tests that were right. They ride along as evidence when one of
        the strong signals fires, because "a test that notices these" is the
        most concrete instruction there is for what the weak tests are missing.
        """
        evidence = ""
        if source_diff:
            # A test that fails on the old code has to be written against the
            # difference between the old and new code, and the fix round sees
            # only the finished files. Without the diff the model is told to
            # write a test that fails "before the run" with no view of what
            # the run changed: on the first sweep of this repair it fixed 2 of
            # 7 flagged runs, and failed all 4 on the task where the change was
            # a layout constant it had no way to see had moved.
            evidence += (
                "What this run changed in the source (the behaviour a test must "
                "tell apart):\n" + source_diff + "\n\n"
            )
        if self.survivors:
            evidence += (
                "Small changes to the edited code that no test noticed:\n  - "
                + "\n  - ".join(self.survivors[:6])
            )
        out: list[dict] = []
        if self.added_tests_pass_on_old_code:
            files = sorted({name.split("::", 1)[0] for name in self.added_tests})
            out.append(
                {
                    "file": files[0],
                    "files": files,
                    "severity": "major",
                    "issue": (
                        "the tests added this run pass against the code as it was before "
                        "the run, so none of them exercises the change: "
                        + ", ".join(self.added_tests[:6])
                    ),
                    "failure": evidence,
                    "fix": (
                        "Add or rewrite a test so it FAILS on the pre-run code and passes on "
                        "the current code: drive the public entry point named in the goal and "
                        "assert the behaviour that changed, at the boundary the change moved. "
                        "Do not weaken or delete any existing test, and do not edit the "
                        "source."
                    ),
                }
            )
        if (
            self.killed_before is not None
            and self.killed_after is not None
            and self.killed_after < self.killed_before
        ):
            edited = sorted(k for k, v in self.test_edits.items() if v in ("rewritten", "reduced"))
            out.append(
                {
                    "file": edited[0],
                    "files": edited,
                    "severity": "major",
                    "issue": (
                        f"the edited test file(s) {', '.join(edited)} now catch "
                        f"{self.killed_after} of {self.mutants_tried} small code changes, "
                        f"down from {self.killed_before} before this run"
                    ),
                    "failure": evidence,
                    "fix": (
                        "Restore what the earlier version of the test file checked, keeping "
                        "any expectation the goal genuinely changed. A test may be updated "
                        "because behaviour was meant to change; it may not be loosened."
                    ),
                }
            )
        return out

    def as_dict(self) -> dict:
        return {
            "ran": self.ran,
            "skipped": self.skipped,
            "added_tests": self.added_tests,
            "added_tests_pass_on_old_code": self.added_tests_pass_on_old_code,
            "mutants_tried": self.mutants_tried,
            "survivors": self.survivors,
            "test_edits": self.test_edits,
            "killed_before": self.killed_before,
            "killed_after": self.killed_after,
            "seconds": round(self.seconds, 1),
            "findings": self.findings,
        }


# -- reading tests ---------------------------------------------------------


def _is_test(rel: str) -> bool:
    return langtools.is_test_file(langtools.PYTHON, rel)


def _assert_count(node: ast.AST) -> int:
    total = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            total += 1
        elif isinstance(child, ast.Call):
            name = ast.unparse(child.func)
            if name in ("pytest.raises", "pytest.warns") or name.startswith("self.assert"):
                total += 1
    return total


def tests_in(source: str | None) -> dict[str, tuple[int, str]]:
    """{qualified test name: (assertion count, normalised body)}."""
    if not source:
        return {}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    found: dict[str, tuple[int, str]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}::")
            elif isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef
            ) and child.name.startswith("test"):
                found[f"{prefix}{child.name}"] = (_assert_count(child), ast.dump(child))

    walk(tree, "")
    return found


def classify_test_edit(before: str | None, after: str | None) -> str:
    """How a test file changed: created, deleted, unchanged, additive,
    rewritten (a test's body changed but it kept as many assertions) or reduced
    (a test vanished, or one lost assertions)."""
    if before is None:
        return "created"
    if after is None:
        return "deleted"
    old, new = tests_in(before), tests_in(after)
    if old == new:
        return "unchanged"
    if any(name not in new or new[name][0] < old[name][0] for name in old):
        return "reduced"
    if any(new[name][1] != old[name][1] for name in old):
        return "rewritten"
    return "additive"


def added_test_names(before: str | None, after: str | None) -> list[str]:
    return sorted(set(tests_in(after)) - set(tests_in(before)))


# -- mutation --------------------------------------------------------------


def _function_spans(tree: ast.AST) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                spans.append(
                    (
                        f"{prefix}{child.name}",
                        child.lineno,
                        getattr(child, "end_lineno", child.lineno) or child.lineno,
                    )
                )
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return spans


def _sites(tree: ast.AST) -> list[tuple[str, ast.AST, int | None]]:
    """Mutable spots, in source order. The index into this list is stable
    across two parses of the same source, which is how a mutant is rebuilt."""
    found: list[tuple[str, ast.AST, int | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            found += [
                ("comparison", node, i)
                for i, op in enumerate(node.ops)
                if type(op) in _COMPARE_FLIP
            ]
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SWAP:
            found.append(("operator", node, None))
        elif (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and not isinstance(node.value, bool)
        ):
            found.append(("constant", node, None))
        elif isinstance(node, ast.BoolOp):
            found.append(("and/or", node, None))
    found.sort(key=lambda s: (getattr(s[1], "lineno", 0), getattr(s[1], "col_offset", 0)))
    return found


def _apply(kind: str, node: ast.AST, index: int | None) -> None:
    if kind == "comparison" and isinstance(node, ast.Compare) and index is not None:
        node.ops[index] = _COMPARE_FLIP[type(node.ops[index])]()
    elif kind == "operator" and isinstance(node, ast.BinOp):
        node.op = _BINOP_SWAP[type(node.op)]()
    elif kind == "constant" and isinstance(node, ast.Constant) and isinstance(node.value, int):
        node.value = node.value + 1
    elif kind == "and/or" and isinstance(node, ast.BoolOp):
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()


def mutants_for(rel: str, before: str | None, after: str) -> list[Mutant]:
    """Mutants of the functions this run changed in one source file.

    Only sites on changed lines, and only inside a function: a mutant of an
    untouched line says nothing about this run's work, and module-level
    constants are mostly configuration that a mutant would "kill" trivially.
    """
    try:
        tree = ast.parse(after)
    except (SyntaxError, ValueError):
        return []
    changed = changed_lines(before, after)
    spans = _function_spans(tree)
    out: list[Mutant] = []
    for position, (kind, node, _index) in enumerate(_sites(tree)):
        line = getattr(node, "lineno", 0)
        if line not in changed:
            continue
        owner = [name for name, first, last in spans if first <= line <= last]
        if not owner:
            continue
        rebuilt = ast.parse(after)
        rebuilt_kind, rebuilt_node, rebuilt_index = _sites(rebuilt)[position]
        _apply(rebuilt_kind, rebuilt_node, rebuilt_index)
        out.append(Mutant(rel, line, kind, owner[-1], ast.unparse(rebuilt)))
    return out


def _spread(items: list[Mutant], limit: int) -> list[Mutant]:
    """At most `limit`, spread across the list rather than the first few, so a
    long function does not use the whole budget on its opening lines."""
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


# -- scratch workspace -----------------------------------------------------


def copy_workspace(root: Path, destination: Path) -> None:
    """Copy the workspace for throwaway use, skipping caches, VCS and huge files,
    and symlinking the real .venv so imports resolve as they would in place."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {n for n in names if n in IGNORED_DIRS or n in (".harness", ".git")}
        for name in names:
            candidate = Path(directory) / name
            try:
                if candidate.is_file() and candidate.stat().st_size > MAX_COPY_FILE_BYTES:
                    skipped.add(name)
            except OSError:
                skipped.add(name)
        return skipped

    shutil.copytree(root, destination, ignore=ignore, symlinks=True)
    venv = root / ".venv"
    if venv.is_dir():
        with contextlib.suppress(OSError):
            (destination / ".venv").symlink_to(venv, target_is_directory=True)


def _pytest(scratch: Path, targets: list[str], deadline: float) -> int:
    """Exit code of one pytest invocation, or -1 when the audit's own time is
    spent. A timeout counts as a detection (124), the usual convention: a
    mutant that hangs the suite has plainly been noticed."""
    remaining = deadline - time.time()
    if remaining <= 1:
        return -1
    code, _, _ = toolchain.run(
        "pytest",
        [*_PYTEST_FLAGS, *targets],
        scratch,
        timeout=int(min(PER_RUN_TIMEOUT_S, remaining)),
    )
    return code


def _covering_targets(scratch: Path, module_stem: str) -> list[str]:
    """Test files that mention the mutated module, else the whole suite."""
    pattern = re.compile(rf"\b(?:import|from)\s+{re.escape(module_stem)}\b")
    hits = []
    for path in scratch.rglob("*.py"):
        rel = path.relative_to(scratch).as_posix()
        if _is_test(rel) and not any(part in IGNORED_DIRS for part in path.parts):
            try:
                if pattern.search(path.read_text(encoding="utf-8")):
                    hits.append(rel)
            except (OSError, UnicodeDecodeError):
                continue
    return sorted(hits) or ["."]


# -- the audit -------------------------------------------------------------


def audit(
    root: Path,
    baseline: Baseline,
    touched: list[str],
    max_mutants: int = MAX_MUTANTS,
    budget_s: float = AUDIT_BUDGET_S,
) -> TestAudit:
    """Run all three checks against the finished workspace. Never raises: a
    failure to audit is a note on the result, not a failure of the run."""
    started = time.time()
    result = TestAudit()
    deadline = started + budget_s
    try:
        _run_audit(root, baseline, touched, max_mutants, deadline, result)
    except Exception as exc:  # telemetry must not fail a run
        result.skipped = f"audit failed: {type(exc).__name__}: {exc}"
    result.seconds = time.time() - started
    return result


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except (OSError, UnicodeDecodeError):
        return None


def _run_audit(
    root: Path,
    baseline: Baseline,
    touched: list[str],
    max_mutants: int,
    deadline: float,
    result: TestAudit,
) -> None:
    snapshots = dict(baseline.snapshots)
    rels = sorted({*touched, *snapshots})
    sources = [r for r in rels if r.endswith(".py") and not _is_test(r)]
    tests = [r for r in rels if r.endswith(".py") and _is_test(r)]

    changed_sources = [
        r for r in sources if _read(root / r) is not None and snapshots.get(r) != _read(root / r)
    ]
    if not changed_sources:
        result.skipped = "no Python source changed"
        return

    for rel in tests:
        kind = classify_test_edit(snapshots.get(rel), _read(root / rel))
        if kind in ("additive", "rewritten", "reduced"):
            result.test_edits[rel] = kind
        result.added_tests += [
            f"{rel}::{name}" for name in added_test_names(snapshots.get(rel), _read(root / rel))
        ]

    with tempfile.TemporaryDirectory(prefix="harness-testaudit-") as tmp:
        scratch = Path(tmp) / "ws"
        copy_workspace(root, scratch)
        result.ran = True
        _revert_check(scratch, snapshots, changed_sources, result, deadline)
        _mutation_and_ledger(
            scratch, root, snapshots, changed_sources, max_mutants, deadline, result
        )


def _revert_check(
    scratch: Path, snapshots: dict, changed_sources: list[str], result: TestAudit, deadline: float
) -> None:
    if not result.added_tests:
        return
    # A run that only *created* source has no earlier behaviour to revert to;
    # its added tests failing there would say nothing.
    if not any(snapshots.get(r) is not None for r in changed_sources):
        return
    for rel in changed_sources:
        before = snapshots.get(rel)
        target = scratch / rel
        if before is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(before, encoding="utf-8")
    code = _pytest(scratch, result.added_tests, deadline)
    if code == 0:
        result.added_tests_pass_on_old_code = True
    elif code not in (-1, 5):
        result.added_tests_pass_on_old_code = False
    # The mutation pass that follows starts by writing the finished sources
    # back over these reverted ones, so nothing needs undoing here.


def _mutants_to_try(
    scratch: Path, root: Path, snapshots: dict, changed_sources: list[str], max_mutants: int
) -> list[Mutant]:
    """Copy the changed sources into the scratch tree and pick the mutants to run."""
    for rel in changed_sources:
        current = _read(root / rel)
        if current is not None:
            (scratch / rel).write_text(current, encoding="utf-8")

    all_mutants: list[Mutant] = []
    for rel in changed_sources:
        after = _read(root / rel)
        if after is not None:
            all_mutants += mutants_for(rel, snapshots.get(rel), after)
    return _spread(all_mutants, max_mutants)


def _killed_by_old_tests(
    scratch: Path, targets: list[str], deadline: float, old_tests: dict, new_tests: dict
) -> bool:
    """Would the tests as they were before this run's edits have caught the
    mutant currently on disk? The new tests are put back before returning."""
    for rel, text in old_tests.items():
        (scratch / rel).write_text(text or "", encoding="utf-8")
    killed = _pytest(scratch, targets, deadline) not in (0, -1)
    for rel, text in new_tests.items():
        (scratch / rel).write_text(text or "", encoding="utf-8")
    return killed


def _try_mutant(
    scratch: Path,
    mutant: Mutant,
    targets: list[str],
    deadline: float,
    result: TestAudit,
    old_tests: dict,
    new_tests: dict,
) -> tuple[bool, bool] | None:
    """Run the suite against one mutant. (killed by the current tests, killed by
    the old ones), or None when the deadline ran out before a verdict."""
    target = scratch / mutant.file
    original = target.read_text(encoding="utf-8")
    target.write_text(mutant.source, encoding="utf-8")
    try:
        code = _pytest(scratch, targets, deadline)
        if code == -1:
            return None
        result.mutants_tried += 1
        killed_now = code != 0
        if not killed_now:
            result.survivors.append(mutant.describe())
        killed_then = bool(old_tests) and _killed_by_old_tests(
            scratch, targets, deadline, old_tests, new_tests
        )
        return killed_now, killed_then
    finally:
        target.write_text(original, encoding="utf-8")


def _mutation_and_ledger(
    scratch: Path,
    root: Path,
    snapshots: dict,
    changed_sources: list[str],
    max_mutants: int,
    deadline: float,
    result: TestAudit,
) -> None:
    mutants = _mutants_to_try(scratch, root, snapshots, changed_sources, max_mutants)
    if not mutants:
        return

    # The unmutated suite has to pass in the scratch copy first, or every
    # "kill" below is just the suite failing for its own reasons.
    stems = {Path(m.file).stem for m in mutants}
    targets = sorted({t for stem in stems for t in _covering_targets(scratch, stem)})
    if _pytest(scratch, targets, deadline) != 0:
        result.skipped = "the suite did not pass in the scratch copy, so no mutation ran"
        return

    edited = [r for r, k in result.test_edits.items() if k in ("rewritten", "reduced")]
    old_tests = {r: snapshots.get(r) for r in edited if snapshots.get(r) is not None}
    new_tests = {r: _read(root / r) for r in old_tests}

    killed_now = killed_then = 0
    for mutant in mutants:
        verdict = _try_mutant(scratch, mutant, targets, deadline, result, old_tests, new_tests)
        if verdict is None:
            break
        killed_now += verdict[0]
        killed_then += verdict[1]
    if old_tests and result.mutants_tried:
        result.killed_before, result.killed_after = killed_then, killed_now
