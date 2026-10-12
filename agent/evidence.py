"""A bug fix's own evidence must show the reported problem going away (DESIGN-LOG §120).

Settings.bugfix_evidence. On django 11790 and 11951 every failed run had the right idea in
the wrong place: `maxlength` set in `UsernameField.widget_attrs()`, which runs before
`AuthenticationForm.__init__` gives the field its max_length, or batch_size capped in
`bulk_create` instead of where `_batched_insert` computes it. The runs' checks could not see
it: one release run tested `widget_attrs()` directly, which already returned `maxlength`
before the fix; others asserted only that a bulk insert succeeded, which it did before the
fix too. The runs that recovered from a misplaced fix (§94 r3, §107 notail r3, §117 step r1)
did so because their own test went through the form, failed, and moved the fix.

So when an execute step that changed existing source calls finish_step, the harness checks,
once a run each, the two pieces of evidence the run already has:

1. **The plan's reproduction.** When the planner's `reproduce` snippet failed at plan time
   (the problem was real), it is run again on the code as it is now. Still failing: the
   edits have not changed the behaviour the goal reports.
2. **The run's new tests, on the code before the run.** Test functions the run added are run
   in a scratch copy with every source file the run changed put back as it was. If they all
   pass there, they cannot tell a fix in the wrong place from a right one.

Either finding holds finish_step once with what was seen. Nothing here calls a model: two
subprocess runs at most, and model calls only when a hold sends the step back to work. A
hard test-first gate cost fixture accuracy (§49.1); this asks for nothing the run has not
already written, and only once.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

from . import toolchain
from .failsig import failure_details
from .testcmd import build_argv, pytest_shaped
from .tools.base import ToolContext
from .tools.checks import _is_project_test_file, as_test_labels, run_in_scratch

EVENT = "evidence_held"
# The scratch run of the new tests: the step's own budget is not charged for it.
OLD_CODE_TIMEOUT_S = 180
MAX_SHOWN = 1_500
# pytest's exit code, and testcmd's for a project runner, when no test ran at all.
NOTHING_RAN = 5
_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)\s*\(", re.MULTILINE)

REPRO_STILL_FAILS = (
    "[harness] Step NOT finished yet: the plan's reproduction of the reported problem still "
    "fails on the code as it is now. The snippet:\n```python\n{snippet}\n```\nIts output:\n"
    "{output}\n"
    "So the edits so far do not change the behaviour the goal reports; a test of your own that "
    "passes does not change that. Find where that behaviour is actually produced, through the "
    "entry point the goal names, and move or extend the fix there; run the snippet with "
    "run_python to check. If the snippet itself is wrong, say why in finish_step's notes and "
    "finish."
)
TESTS_PASS_ON_OLD_CODE = (
    "[harness] Step NOT finished yet: your new test(s) {names} also pass on the code as it "
    "was before this run (the harness ran them with your source edits undone), so they do not "
    "show the reported problem, and they cannot tell a fix in the wrong place from a right "
    "one. Change a test so it checks what the goal reports, through the entry point the goal "
    "names (the call a user of the code makes), so that it fails without your fix; then check "
    "that your fix makes it pass, moving the fix if it does not."
)


def _is_test_path(rel: str, label_root: str) -> bool:
    root = label_root.strip("/")
    parts = rel.split("/")
    return (
        _is_project_test_file(rel, root)
        or (bool(root) and rel.startswith(root + "/"))
        or "tests" in parts[:-1]
        or "test" in parts[:-1]
    )


def _current(ctx: ToolContext, rel: str) -> str | None:
    path = ctx.workspace.root / rel
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except (OSError, UnicodeDecodeError):
        return None


def changed_sources(ctx: ToolContext) -> dict[str, str | None]:
    """{path: content before the run} for every non-test Python file the run changed."""
    with ctx.baseline._lock:
        snapshots = dict(ctx.baseline.snapshots)
    return {
        rel: before
        for rel, before in snapshots.items()
        if rel.endswith(".py")
        and not _is_test_path(rel, ctx.test_label_root)
        and before != _current(ctx, rel)
    }


def fixes_existing_code(changed: dict[str, str | None]) -> bool:
    """The run changed a source file that existed before it: a fix, not new work."""
    return any(before is not None for before in changed.values())


def new_tests(ctx: ToolContext) -> dict[str, list[str]]:
    """{test file: names of test functions it has now and did not have before the run}."""
    with ctx.baseline._lock:
        snapshots = dict(ctx.baseline.snapshots)
    out: dict[str, list[str]] = {}
    for rel, before in snapshots.items():
        if not rel.endswith(".py") or not _is_project_test_file(rel, ctx.test_label_root):
            continue
        now = _current(ctx, rel)
        if now is None:
            continue
        old = set(_TEST_DEF.findall(before or ""))
        added = [n for n in dict.fromkeys(_TEST_DEF.findall(now)) if n not in old]
        if added:
            out[rel] = added
    return out


def _scratch_env(scratch: Path) -> dict[str, str]:
    """The workspace's environment for the copy, with the copy first on the path: an
    editable install (django's) would otherwise import the real workspace's code."""
    env = toolchain.workspace_env(scratch)
    env["PYTHONPATH"] = str(scratch)
    return env


def _run_tests_in(ctx: ToolContext, scratch: Path, files: list[str]) -> tuple[int, str] | None:
    """(exit code, pytest-shaped output) of `files` run in `scratch`, or None when it could
    not run at all."""
    if ctx.test_command:
        argv = build_argv(ctx.test_command, as_test_labels(files, ctx.test_label_root))
        result = toolchain.run_capturing(
            argv, cwd=scratch, timeout=OLD_CODE_TIMEOUT_S, env=_scratch_env(scratch)
        )
        if result.timed_out or (result.returncode == 127 and not result.stdout):
            return None
        return pytest_shaped(result.output, result.returncode, result.elapsed_s, scratch)
    code, out, err = toolchain.run(
        "pytest",
        ["-v", "--no-header", "--tb=line", "-p", "no:cacheprovider", "-p", "no:randomly", *files],
        scratch,
        timeout=OLD_CODE_TIMEOUT_S,
        extra_env={"PYTHONPATH": str(scratch)},
    )
    if code in (124, 127):
        return None
    return code, f"{out}\n{err}"


def verdict_on_old_code(code: int, output: str, names: list[str]) -> str:
    """'red' when a new test failed (or the old code could not even load it), 'green' when
    every new test ran and passed, 'unknown' when the output does not say (pytest's exit 5,
    nothing collected, included)."""
    if code == NOTHING_RAN:
        return "unknown"
    ids, _, _ = failure_details(output)
    if any(name in tid for tid in ids for name in names):
        return "red"
    if code != 0 and not any("test" in tid.rsplit(".", 1)[-1] for tid in ids):
        return "red"  # an import or collection failure: the old code cannot run the test
    if code == 0 or ids:
        ran = all(re.search(rf"\b{re.escape(n)}\b", output) for n in names)
        return "green" if ran else "unknown"
    return "unknown"


def scratch_target(scratch: Path, rel: str) -> Path | str:
    """`scratch / rel` when writing or deleting it stays inside the scratch copy, else why
    not. copy_workspace keeps symlinks, so a symlinked file or parent directory in the copy
    still points wherever it pointed in the workspace, and writing through it would put the
    old code back in the real workspace (or anywhere else) before the run is scored."""
    if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return f"refused path {rel!r}: absolute or climbs out"
    root = scratch.resolve()
    target = scratch / rel
    probe = scratch
    for part in Path(rel).parts:
        probe = probe / part
        if probe.is_symlink():
            return f"refused path {rel!r}: {probe.relative_to(scratch)} is a symlink"
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        return f"refused path {rel!r}: resolves outside the scratch copy"
    return target


def _restore_old_code(scratch: Path, changed: dict[str, str | None]) -> str:
    """Put every changed source file in `scratch` back as it was; '' or why it refused.
    Every path is checked before any is touched, so a refusal changes nothing."""
    targets = {}
    for rel in changed:
        target = scratch_target(scratch, rel)
        if isinstance(target, str):
            return target
        targets[rel] = target
    for rel, before in changed.items():
        target = targets[rel]
        if before is None:
            target.unlink(missing_ok=True)
            continue
        # O_NOFOLLOW: the last component may not have become a symlink since the check.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(before)
    return ""


def run_new_tests_on_old_code(
    ctx: ToolContext, changed: dict[str, str | None], tests: dict[str, list[str]]
) -> tuple[str, str, str]:
    """(verdict, output, refusal): the run's new tests in a copy of the workspace whose
    changed source files are put back as they were before the run. A refusal (a path that
    would write outside the copy) gives "unknown" and says why."""
    from .testcheck import copy_workspace

    names = [n for found in tests.values() for n in found]
    with tempfile.TemporaryDirectory(prefix="harness-evidence-") as tmp:
        scratch = Path(tmp) / "ws"
        copy_workspace(ctx.workspace.root, scratch)
        refused = _restore_old_code(scratch, changed)
        if refused:
            return "unknown", "", refused
        ran = _run_tests_in(ctx, scratch, sorted(tests))
    if ran is None:
        return "unknown", "", ""
    code, output = ran
    return verdict_on_old_code(code, output, names), output, ""


class EvidenceGate:
    """Holds an execute step's finish_step once a run for each kind of missing evidence.

    Run-scoped through ctx.evidence_holds, like PremiseGate's holds: a step retried, or the
    next step, does not get the same hold twice."""

    event = EVENT

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def observe(self, call: dict) -> None:
        pass

    def hold(self, call: dict) -> str | None:
        if (call.get("function") or {}).get("name") != "finish_step":
            return None
        holds = self.ctx.evidence_holds
        if {"repro", "tests"} <= holds:
            return None
        changed = changed_sources(self.ctx)
        if not fixes_existing_code(changed):
            return None
        started = time.monotonic()
        held = None
        if "repro" not in holds and self.ctx.repro_reproduced:
            held = self._repro()
        if held is None and "tests" not in holds:
            held = self._tests(changed)
        self.ctx.trace.event(
            "evidence_check", held=held is not None, seconds=round(time.monotonic() - started, 2)
        )
        return held

    def _repro(self) -> str | None:
        ran = run_in_scratch(self.ctx, self.ctx.repro_reproduced or "")
        if isinstance(ran, str):
            self.ctx.trace.event("evidence_repro", outcome="not_run", detail=ran[:200])
            return None
        code, output = ran
        outcome = "fixed" if code == 0 else "still_fails"
        self.ctx.trace.event("evidence_repro", outcome=outcome, exit_code=code)
        if code == 0:
            return None
        self.ctx.evidence_holds.add("repro")
        shown = output[-MAX_SHOWN:] if output else "(no output)"
        snippet = (self.ctx.repro_reproduced or "").strip()[:MAX_SHOWN]
        return REPRO_STILL_FAILS.format(snippet=snippet, output=shown)

    def _tests(self, changed: dict[str, str | None]) -> str | None:
        tests = new_tests(self.ctx)
        if not tests:
            return None
        verdict, output, refused = run_new_tests_on_old_code(self.ctx, changed, tests)
        names = [n for found in tests.values() for n in found]
        self.ctx.trace.event(
            "evidence_tests",
            verdict=verdict,
            tests=names[:10],
            files=sorted(tests),
            refused=refused,
            tail=output[-400:] if verdict == "unknown" else "",
        )
        if verdict != "green":
            return None
        self.ctx.evidence_holds.add("tests")
        return TESTS_PASS_ON_OLD_CODE.format(names=", ".join(names[:6]))
