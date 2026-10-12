"""Validation tools: Ruff, ty, and a structured pytest runner.

These wrap the same binaries the model could reach through run_bash, but they return
parsed, bounded output instead of a wall of terminal text, and they do not spend the
step's shell budget.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .. import langtools, toolchain, validate
from ..approval import Denied
from ..failsig import failure_details, pytest_summary
from ..projectconfig import uses_a_formatter
from ..text import truncate
from ..workspace import walk_files
from .base import Tool, ToolContext, mark_external_writes, obj, snapshot_before_write
from .web import code_denied

MAX_FAILURE_CHARS = 6_000
# complexipy's own default ceiling per function. Treated here as a ceiling
# rather than a target: most functions should be well under it.
COMPLEXITY_LIMIT = 15


def _target_paths(ctx: ToolContext, paths: list[str] | None) -> list[str]:
    """Default to the Python files this run has actually touched."""
    if paths:
        resolved: list[str] = []
        for rel in paths:
            try:
                resolved.append(str(ctx.workspace.resolve(rel)))
            except Exception:
                continue
        return resolved
    touched = [str(ctx.workspace.resolve(p)) for p in ctx.touched if p.endswith(".py")]
    return touched or [str(ctx.workspace.root)]


def check_complexity(
    ctx: ToolContext, paths: list[str] | None = None, max_allowed: int = COMPLEXITY_LIMIT
) -> str:
    """Cognitive complexity per function, via complexipy.

    Cognitive complexity — unlike cyclomatic — weights nesting and flow
    breaks, which is what actually makes code hard to change. Generated code
    accretes branches a case at a time and each addition looks locally
    reasonable, so having a number to point at is the difference between
    "this feels tangled" and a claim that can be checked before and after a
    refactor.

    `complexipy` only ever covers `.py` files — `_target_paths` filters to
    them explicitly — so a Node/Go/Rust workspace got no complexity signal
    at all before this existed, not because nobody wanted one but because
    nothing else was wired up. When the caller didn't name specific paths
    and nothing touched this run is Python, but the workspace's detected
    language isn't either, fall back to `scc`'s per-file cyclomatic
    complexity (`--by-file`) instead — coarser than complexipy's
    cognitive-complexity-per-function, but real, and it's already computing
    this from the same `scc` call `langtools.detect_profile` uses as its own
    fallback. `scc` is a counter, not a reviewer: this fills the
    complexity-metric gap specifically, not a stand-in for a future
    non-Python linter.
    """
    if (
        paths is None
        and ctx.language_profile not in (None, langtools.PYTHON)
        and not any(p.endswith(".py") for p in ctx.touched)
    ):
        return _check_complexity_scc(ctx, max_allowed)
    targets = _target_paths(ctx, paths)
    if not targets:
        return "nothing to analyse"
    code, out, err = toolchain.run(
        "complexipy", ["--output-format", "json", "-q", *targets], ctx.workspace.root, timeout=120
    )
    if code == 127:
        return (
            "error: complexipy is not installed in this workspace. "
            "`uv add --dev complexipy` first, or skip this check."
        )

    report = ctx.workspace.root / "complexipy-results.json"
    try:
        entries = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"error: complexipy produced no readable report ({(out or err)[:300]})"
    finally:
        # It writes its report into the working directory; leaving it behind
        # would show up as an untracked file the agent then tries to explain.
        report.unlink(missing_ok=True)

    scored = sorted(
        (
            (int(e.get("complexity", 0)), str(e.get("path", "")), str(e.get("function_name", "")))
            for e in entries
            if isinstance(e, dict)
        ),
        reverse=True,
    )
    if not scored:
        return "no functions found to analyse"
    over = [row for row in scored if row[0] > max_allowed]
    ctx.trace.event(
        "check_complexity", functions=len(scored), over_limit=len(over), worst=scored[0][0]
    )

    lines = [f"{len(scored)} function(s) analysed, {len(over)} over the limit of {max_allowed}."]
    if over:
        lines.append("")
        lines.append("Over the limit (worst first):")
        lines.extend(f"  {score:>3}  {path}::{name}" for score, path, name in over[:20])
        lines.append("")
        lines.append(
            "Reduce nesting first — a guard clause that lets the body dedent "
            "buys more than anything else. Extract a block only when you can "
            "name what it means."
        )
    else:
        lines.append(f"Worst is {scored[0][0]} ({scored[0][1]}::{scored[0][2]}).")
    return "\n".join(lines)


SCC_TIMEOUT_S = 30


def _check_complexity_scc(ctx: ToolContext, max_allowed: int) -> str:
    """The non-Python fallback: per-file cyclomatic complexity via `scc`."""
    scope = [str(ctx.workspace.resolve(p)) for p in ctx.touched] or [str(ctx.workspace.root)]
    result = toolchain.run_capturing(
        ["scc", "--by-file", "--format", "json", *scope], timeout=SCC_TIMEOUT_S
    )
    if result.returncode == 127:
        return (
            "error: scc is not installed in this environment, so there is no "
            "complexity check available for a non-Python file."
        )
    try:
        breakdown = json.loads(result.stdout)
    except json.JSONDecodeError:
        return f"error: scc produced no readable report ({result.output[:300]})"

    scored = sorted(
        (
            (int(f.get("Complexity", 0)), str(f.get("Location", "")))
            for row in breakdown
            if isinstance(row, dict)
            for f in row.get("Files", [])
            if isinstance(f, dict)
        ),
        reverse=True,
    )
    if not scored:
        return "no files found to analyse"
    over = [row for row in scored if row[0] > max_allowed]
    ctx.trace.event(
        "check_complexity",
        functions=len(scored),
        over_limit=len(over),
        worst=scored[0][0],
        tool="scc",
    )

    lines = [
        f"{len(scored)} file(s) analysed (scc, per-file, not per-function), "
        f"{len(over)} over the limit of {max_allowed}."
    ]
    if over:
        lines.append("")
        lines.append("Over the limit (worst first):")
        lines.extend(f"  {score:>3}  {path}" for score, path in over[:20])
    else:
        lines.append(f"Worst is {scored[0][0]} ({scored[0][1]}).")
    return "\n".join(lines)


def check_code(
    ctx: ToolContext, paths: list[str] | None = None, types: bool = False, fix: bool = False
) -> str:
    """Syntax, then Ruff, then optionally ty."""
    if fix and ctx.writable is not None:
        return (
            "error: ruff --fix is disabled while steps run in parallel; "
            "use file tools for declared writes or run this step sequentially"
        )
    targets = _target_paths(ctx, paths)
    root = ctx.workspace.root
    sections: list[str] = []

    if fix:
        # ruff --fix/format writes files behind the tools' backs the same
        # way a shell command can -- run_bash tracks exactly this with the
        # same two calls (agent/tools/shell.py); without them, a step's own
        # lint-fix never appears in ctx.touched and is invisible to
        # review/diff/finalcheck.
        before = snapshot_before_write(ctx)
        _, out, err = toolchain.run("ruff", ["check", "--fix", "--quiet", *targets], root)
        formats = uses_a_formatter(root)
        if formats:
            toolchain.run("ruff", ["format", "--quiet", *targets], root)
        mark_external_writes(ctx, before)
        ctx.cache.invalidate()
        done = (
            "applied ruff --fix and ruff format"
            if formats
            else (
                "applied ruff --fix (not ruff format: this project does not configure ruff "
                "or black formatting, so reformatting would restyle code the goal never touched)"
            )
        )
        sections.append(f"{done}\n{(out or err).strip()[:800]}".strip())

    broken = []
    for target in targets:
        path = Path(target)
        if path.is_file():
            diagnostic = validate.syntax_check(path)
            if diagnostic:
                broken.append(diagnostic)
    if broken:
        sections.append("SYNTAX ERRORS (fix these first):\n" + validate.render(broken))
        return "\n\n".join(sections)

    lint = validate.ruff_check(root, targets)
    sections.append(f"ruff: {len(lint)} issue(s)" + (f"\n{validate.render(lint)}" if lint else ""))

    if types:
        scope = targets if paths else None
        type_errors = validate.ty_check(root, scope)
        # With errors, show ty's full output instead of the one-line form: it
        # names where the other side of the mismatch is defined, which is the
        # half of a cross-file defect the fixer cannot see from its own file.
        detail = validate.ty_context(root, scope) if type_errors else ""
        sections.append(
            f"ty: {len(type_errors)} issue(s)"
            + (f"\n{detail or validate.render(type_errors)}" if type_errors else "")
        )
    else:
        sections.append("ty: not run (pass types=true for cross-file type checking)")

    ctx.trace.event(
        "check_code",
        paths=[ctx.workspace.rel(Path(t)) for t in targets],
        ruff_issues=len(lint),
        types=types,
        fixed=fix,
    )
    return "\n\n".join(sections)


def check_code_readonly(
    ctx: ToolContext, paths: list[str] | None = None, types: bool = False
) -> str:
    """check_code with no `fix` parameter at all, for phases that must not write.

    Omitting `fix` from the Tool schema is not enough on its own — _dispatch()
    calls tool.fn(ctx, **args) straight from the model's JSON with no schema
    validation, so a schema that merely doesn't *mention* `fix` would still
    accept it if a model sent it anyway. This wrapper's signature has no `fix`
    parameter to receive it: a model that tries gets an ordinary "bad
    arguments" TypeError back, not a write.
    """
    return check_code(ctx, paths=paths, types=types, fix=False)


_summarise = pytest_summary


def _default_test_targets(ctx: ToolContext) -> list[str]:
    """Touched files matching this workspace's language's test-file naming
    convention — the same touched-files defaulting `_target_paths` already
    uses for check_code/check_complexity, applied here too.

    A traced fix loop called `run_tests(path=".")` — the old hardcoded
    default — to check one file it had just edited, and re-ran the whole
    suite to do it; replayed against that project's own venv directly,
    `pytest test_eval_and_compare.py` measured 1.90s against `pytest .`'s
    9.96s on the same workspace (both pytest's own reported time), 5.2x for
    the exact case that motivated this. Falls back to `["."]` when nothing
    touched yet looks like a test file: early in a step, or a step that
    isn't test work, there is nothing narrower to target than the whole
    suite, and running nothing would be wrong, not faster.
    """
    profile = ctx.language_profile or langtools.PYTHON
    touched = [p for p in ctx.touched if langtools.is_test_file(profile, p)]
    return touched or ["."]


def as_test_labels(paths: list[str], label_root: str) -> list[str]:
    """Test files under `label_root` as dotted module labels relative to it.

    django's runner takes `pagination.tests`, not `tests/pagination/tests.py`,
    and rejects the path outright (§70). Anything else, a label already, a
    directory, or a file outside the root, passes through unchanged.
    """
    root = label_root.strip("/")
    if not root:
        return list(paths)
    out = []
    for path in paths:
        # A pytest node id (`tests/x/test_y.py::Class::test`) is what the model
        # reaches for by habit; the runner wants `x.test_y.Class.test` (§70).
        rel, _, rest = path.removeprefix("./").partition("::")
        rel = rel.rstrip("/")
        if rel.startswith(root + "/"):
            # A file or a package directory (`tests/pagination` -> `pagination`).
            rel = rel[len(root) + 1 :].removesuffix(".py").replace("/", ".")
            rel = rel.removesuffix(".__init__")
        out.append(".".join([rel, *rest.split("::")]) if rest else rel)
    return out


# Existing test modules that import a source file this run changed, run by the final
# check beside the tests the run touched (see project_test_targets). A cap, since a
# widely imported module (a base class) can be imported by most of a large suite.
IMPORTING_TEST_MODULES_MAX = 8


def _is_project_test_file(rel: str, label_root: str = "") -> bool:
    """A test module by name: `test_*.py`, `*_test.py` or `tests.py`. django names its
    modules `tests.py`, and matching only `test_*` left the final check with nothing
    to run on every django task (§75). Not by place: a test app's `models.py` or
    `admin.py` under the label root is a helper, and passing `admin_views.admin` to
    django's runner as a label failed its system checks before any test ran, which
    cost two runs their fix rounds (§75.1)."""
    name = rel.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "tests.py"


def _module_of(rel: str) -> str:
    return rel.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def tests_importing(root: Path, sources: list[str], label_root: str = "") -> list[str]:
    """Test modules (under the label root when there is one) that import any of
    `sources`, as `from pkg.mod import`, `import pkg.mod`, or `from pkg import mod`.

    django-11885 changed `django/db/models/deletion.py` and broke two tests in
    `tests/delete/tests.py`, which imports `Collector` from it (§75).
    """
    patterns = []
    for rel in sources:
        module = _module_of(rel)
        parent, _, name = module.rpartition(".")
        patterns.append(re.compile(rf"^\s*(?:from|import)\s+{re.escape(module)}\b", re.M))
        if parent:
            patterns.append(
                re.compile(rf"^\s*from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{name}\b", re.M)
            )
    if not patterns:
        return []
    base = root / label_root if label_root else root
    found = []
    # walk_files prunes .venv, node_modules and the rest before descending: an
    # rglob that filtered afterwards cost 0.97 s against 0.007 s on a tree with a
    # virtualenv, on every run_tests call (code review of 2026-09-30, finding 3).
    for path in sorted(p for p in walk_files(base) if p.suffix == ".py"):
        rel = str(path.relative_to(root))
        if not _is_project_test_file(rel, label_root):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(p.search(text) for p in patterns):
            found.append(rel)
    return found


def project_test_targets(ctx: ToolContext) -> list[str]:
    """What the project's own test command runs by default: the test files this run
    touched, then (up to IMPORTING_TEST_MODULES_MAX) existing test modules that
    import a source file it changed, so a change that breaks tests the run never
    opened is still seen."""
    label_root = ctx.test_label_root.strip("/")
    touched = [p for p in ctx.touched if p.endswith(".py")]
    tests = [p for p in touched if _is_project_test_file(p, label_root)]
    sources = [p for p in touched if p not in tests]
    extra = [p for p in tests_importing(ctx.workspace.root, sources, label_root) if p not in tests][
        :IMPORTING_TEST_MODULES_MAX
    ]
    return tests + extra


def run_project_tests(
    ctx: ToolContext, targets: list[str], pattern: str, timeout: float
) -> tuple[int, str]:
    """Settings.test_command on `targets`, read back in pytest's shape.

    The command's own output is kept; agent/testcmd.py appends the FAILED
    lines, frames and summary banner that failsig and the gates parse, and
    maps the exit code onto pytest's (0 pass, 1 fail, 2 could not start,
    5 nothing ran).
    """
    from .. import testcmd

    args = (["-k", pattern] if pattern else []) + as_test_labels(targets, ctx.test_label_root)
    argv = testcmd.build_argv(ctx.test_command, args)
    result = toolchain.run_capturing(
        argv,
        cwd=ctx.workspace.root,
        timeout=timeout,
        env=toolchain.workspace_env(ctx.workspace.root),
    )
    if result.timed_out:
        return 124, f"the test command timed out after {timeout:.0f}s: {' '.join(argv)}"
    if result.returncode == 127 and not result.stdout:
        return 127, f"could not run the test command {' '.join(argv)}: {result.stderr}"
    return testcmd.pytest_shaped(
        result.output, result.returncode, result.elapsed_s, ctx.workspace.root
    )


def _invoke_tests(
    ctx: ToolContext, path: str, pattern: str, verbose: bool, budget: float
) -> tuple[int, str, str, list[str]] | str:
    """(code, out, err, targets) from pytest or the project's own command, or an
    error string for the model when there is nothing it can usefully run."""
    timeout = int(max(10, min(300, budget)))
    if ctx.test_command:
        touched = project_test_targets(ctx)
        # "." is pytest's whole suite; a project runner's whole suite is what the
        # message below exists to avoid, so it is treated as no target at all.
        targets = [path] if path and path.strip() not in (".", "./") else touched
        if not targets:
            # The whole of a large project's suite (django's takes minutes) is
            # not a useful default, and pytest's "." means nothing to its runner.
            return (
                "error: this project's tests run with its own command "
                f"(`{ctx.test_command}`); name what to run in `path`, e.g. a test "
                "module label such as `pagination.tests` or a test file."
            )
        code, text = run_project_tests(ctx, targets, pattern, timeout)
        return code, text, "", targets
    targets = [path] if path else _default_test_targets(ctx)
    # no:randomly, same as the gate's own suite invocation: deterministic
    # ordering here too, and it is a no-op when the plugin is not installed.
    args = ["-q", "--no-header", "--tb=short", "-p", "no:cacheprovider", "-p", "no:randomly"]
    args.append("-v" if verbose else "-x")
    if pattern:
        args.extend(["-k", pattern])
    args.extend(targets)
    code, out, err = toolchain.run("pytest", args, ctx.workspace.root, timeout=timeout)
    return code, out, err, targets


def _source(ctx: ToolContext) -> dict:
    """`source="auto"` on a run_tests event the harness made itself (check_after_edit)."""
    source = getattr(ctx, "test_source", "")
    return {"source": source} if source else {}


def run_tests(ctx: ToolContext, path: str = "", pattern: str = "", verbose: bool = False) -> str:
    """Run the tests (pytest, or Settings.test_command) and return the parts of
    their output that are actionable."""
    budget = ctx.guard.remaining_seconds()
    if budget <= 0:
        return ctx.guard.check_subprocess_budget() or "error: no shell budget left this step"

    started = time.time()
    invoked = _invoke_tests(ctx, path, pattern, verbose, budget)
    if isinstance(invoked, str):
        return invoked
    code, out, err, targets = invoked
    ctx.guard.record_subprocess(time.time() - started)
    ctx.cache.invalidate()

    if code == 127:
        return f"error: {err}"
    if code == 5:
        return "no tests were collected. Check the path and that files are named test_*.py."

    combined = f"{out}\n{err}".strip()
    summary = _summarise(combined, code)
    targets_str = " ".join(targets)

    # Exit 2/3/4 are not test failures: pytest stopped before (or instead of)
    # running anything — almost always an import error in a test module, a
    # missing dependency, or a bad argument. Reporting that as "3 errors" the
    # way a failing assertion is reported sends the model looking for a bug in
    # logic that never ran; one real run spent two whole attempts and 30 tool
    # calls on packaging archaeology because of exactly this.
    if code in (2, 3, 4):
        ctx.trace.event(
            "run_tests",
            path=targets_str,
            exit_code=code,
            summary=summary,
            collection_error=True,
            **_source(ctx),
        )
        return (
            f"COLLECTION ERROR (pytest exit {code}) — pytest could not even "
            f"collect the tests, so none of them ran. This is an import or "
            f"configuration problem, not a failing assertion: look for the "
            f"ImportError/ModuleNotFoundError below and fix that first. "
            f"Re-running this command unchanged will produce the same output.\n\n"
            f"{truncate(combined, MAX_FAILURE_CHARS, keep='tail')}"
        )

    if code == 0 and not pattern:
        # Remember what passed and against which workspace, so the step's own
        # verify command — very often this same invocation — does not have to
        # run it again seconds later. A `-k` pattern is excluded because it
        # ran a subset, which proves nothing about the whole.
        from .base import workspace_state

        ctx.tests_green = (workspace_state(ctx.workspace.root), tuple(targets))
    if code == 0:
        ctx.trace.event(
            "run_tests", path=targets_str, exit_code=code, summary=summary, **_source(ctx)
        )
        return f"PASS — {summary}"

    # Which tests, and where the failure surfaced, so a trace can say whether
    # the next edit went to the frame the failure named. Until this was
    # recorded the trace held only "2 failed", and that question could not be
    # asked of any run after the fact.
    ids, _, where = failure_details(combined)
    ctx.trace.event(
        "run_tests",
        path=targets_str,
        exit_code=code,
        summary=summary,
        failing=sorted(ids)[:10],
        where=where,
        **_source(ctx),
        # A failure no id could be read from is the case nobody can diagnose
        # afterwards: django-12708's run recorded "1 error" with no id and no
        # output, and the same test, rerun, parsed fine (DESIGN-LOG §71).
        **({"unparsed_tail": combined[-UNPARSED_TAIL_CHARS:]} if not ids else {}),
    )

    # On failure the traceback is the payload; keep the tail, which holds the assertion.
    return f"FAIL — {summary}\n\n{truncate(combined, MAX_FAILURE_CHARS, keep='tail')}"


# Output kept on a run_tests event whose failures named no test id.
UNPARSED_TAIL_CHARS = 1_500


def run_python(ctx: ToolContext, code: str, timeout: int = 60) -> str:
    """Execute a Python snippet against the workspace.

    Measured on real runs, a third of all shell commands were `python3 -c "..."`
    verification snippets. Routing them through a tool avoids shell quoting —
    the code arrives as a string, not as something the model has to escape
    twice — but it is exactly as capable as a shell command (shutil, os,
    subprocess, sockets are all reachable from it), so it goes through the same
    approval gate as run_bash rather than skipping it.
    """
    if not code.strip():
        return "error: code must not be empty"
    denied = code_denied(ctx, "run_python", code)
    if denied:
        return denied
    if ctx.writable is not None:
        return (
            "error: arbitrary Python is disabled while steps run in parallel; "
            "use file tools for declared writes or run this step sequentially"
        )
    try:
        ctx.approver.check_run(code, f"run this Python:\n{code}")
    except Denied as exc:
        ctx.trace.event("run_python_denied", reason=str(exc))
        return f"error: {exc}"
    budget = ctx.guard.remaining_seconds()
    if budget <= 0:
        return ctx.guard.check_subprocess_budget() or "error: no shell budget left this step"

    # Unique per call: with --step-concurrency > 1, concurrent steps share this
    # workspace, and a fixed name let one step's run_python execute another's
    # code, or delete the file out from under a sibling's still-running
    # subprocess. Stays inside the workspace root (not a system temp dir) so
    # `python script.py` still puts the workspace on sys.path[0] for imports.
    # A snippet can create or rewrite workspace files just as easily as
    # run_bash can (shutil, os, subprocess are all reachable from it, per
    # this function's own docstring) -- tracked the same way run_bash tracks
    # its own external writes, so a file this snippet wrote is not invisible
    # to review/diff/finalcheck the way it was before this existed.
    before = snapshot_before_write(ctx)
    scratch = ctx.workspace.root / f"_agent_snippet_{uuid.uuid4().hex[:12]}.py"
    scratch.write_text(code, encoding="utf-8")
    argv = toolchain.resolve("python", str(ctx.workspace.root)) or ["python3"]
    try:
        result = toolchain.run_capturing(
            [*argv, scratch.name],
            cwd=ctx.workspace.root,
            timeout=max(5, min(timeout, budget)),
            env=toolchain.workspace_env(ctx.workspace.root),
        )
    finally:
        scratch.unlink(missing_ok=True)
        mark_external_writes(ctx, before)
    ctx.guard.record_subprocess(result.elapsed_s)
    if result.timed_out:
        return f"error: the snippet timed out after {timeout}s"
    if result.returncode == 127 and not result.stdout:
        return f"error: could not run Python: {result.stderr}"
    ctx.cache.invalidate()

    output = truncate(result.output.strip(), MAX_FAILURE_CHARS)
    ctx.trace.event("run_python", exit_code=result.returncode, chars=len(code))
    return f"exit={result.returncode}\n{output or '(no output)'}"


def run_in_scratch(ctx: ToolContext, code: str, timeout: int = 30) -> tuple[int, str] | str:
    """Run `code` against a fresh throwaway copy of the workspace.

    Returns (exit code, output), or an "error: ..." string when it could not be
    run at all (denied, timed out, no interpreter). A fresh copy per call means
    nothing the snippet writes reaches the real workspace or the next call, and
    the copy is cheap because `copy_workspace` skips caches and symlinks the
    venv. It still runs arbitrary code on this machine, so it goes through the
    same approval gate as run_python.
    """
    import tempfile

    from ..testcheck import copy_workspace

    if not code.strip():
        return "error: code must not be empty"
    denied = code_denied(ctx, "run_in_scratch", code)
    if denied:
        return denied
    try:
        ctx.approver.check_run(code, f"run this Python (in a scratch copy):\n{code}")
    except Denied as exc:
        ctx.trace.event("probe_python_denied", reason=str(exc))
        return f"error: {exc}"
    with tempfile.TemporaryDirectory(prefix="harness-probe-") as tmp:
        scratch = Path(tmp) / "ws"
        started = time.monotonic()
        copy_workspace(ctx.workspace.root, scratch)
        copy_s = time.monotonic() - started
        (scratch / "_agent_probe.py").write_text(code, encoding="utf-8")
        argv = toolchain.resolve("python", str(scratch)) or ["python3"]
        result = toolchain.run_capturing(
            [*argv, "_agent_probe.py"],
            cwd=scratch,
            timeout=max(5, min(timeout, 60)),
            env=toolchain.workspace_env(scratch),
        )
    ctx.trace.event(
        "scratch_run", exit_code=result.returncode, chars=len(code), copy_s=round(copy_s, 3)
    )
    if result.timed_out:
        return f"error: timed out after {timeout}s"
    if result.returncode == 127 and not result.stdout:
        return f"error: could not run Python: {result.stderr}"
    return result.returncode, truncate(result.output.strip(), MAX_FAILURE_CHARS)


def probe_python(ctx: ToolContext, code: str, timeout: int = 30) -> str:
    """Run a snippet against a throwaway copy of the workspace, for the planner.

    The planner's kit is read-only, which is why it could not reproduce a
    reported symptom before naming a cause (DESIGN-LOG §44-45): 32% of 1,699
    bench plans were submitted with no look at all, 40% after one call.
    `run_in_scratch` keeps that property: nothing a probe writes survives.
    """
    ran = run_in_scratch(ctx, code, timeout)
    if isinstance(ran, str):
        return ran
    exit_code, output = ran
    return f"exit={exit_code}\n{output or '(no output)'}"


PROBE_PYTHON = Tool(
    "probe_python",
    "Run a Python snippet against a throwaway copy of the workspace, with the workspace "
    "importable, and see its output. Use it to reproduce what the goal reports before you "
    "decide the cause: call the entry point the report names with the input it describes and "
    "print what comes out. Nothing it writes survives; each call gets a fresh copy.",
    obj(
        {
            "code": {"type": "string", "description": "Python source. Use print() for output."},
            "timeout": {
                "type": "integer",
                "description": "Seconds before it is killed, default 30.",
            },
        },
        ["code"],
    ),
    probe_python,
)


MAX_TRACE_LINES = 200

# Runs ahead of the model's own snippet, inside the same subprocess. Traces
# every executed line under the workspace root (stdlib/site-packages/venv
# excluded -- that machinery would otherwise dominate a trace of the model's
# own code) and shows only the locals that changed since that function's
# frame last reported, so a tight loop reads as one line per iteration with
# just what moved, not a repeated full dump. Pure sys.settrace -- no
# dependency beyond the interpreter itself.
_TRACER_PREAMBLE = """
import sys as __sys, os as __os

__WORKSPACE_ROOT = "__WORKSPACE_ROOT_PLACEHOLDER__"
__MAX_LINES = __MAX_LINES_PLACEHOLDER__
__trace_lines = []
__last_locals = {}
__src_cache = {}

def __tracer(frame, event, arg):
    if event != "line":
        return __tracer
    filename = frame.f_code.co_filename
    # A synthetic frame (<frozen importlib._bootstrap>, <string>, <stdin>) is
    # never a real file -- relpath against it produces a path that happens to
    # pass the filters below, which is how import machinery itself first
    # showed up in a trace meant to be about the model's own code.
    if filename.startswith("<"):
        return __tracer
    try:
        rel = __os.path.relpath(filename, __WORKSPACE_ROOT)
    except ValueError:
        return __tracer
    if rel.startswith("..") or "site-packages" in filename or ".venv" in filename:
        return __tracer
    if len(__trace_lines) >= __MAX_LINES:
        __sys.settrace(None)
        return None
    lineno = frame.f_lineno
    func = frame.f_code.co_name
    key = (filename, func, id(frame))

    def __safe_repr(v):
        # A local's own __repr__ can raise (seen live: a half-constructed
        # import-machinery object) -- that must never take down the
        # snippet being traced, only cost this one variable's display.
        try:
            return repr(v)[:80]
        except Exception:
            return f"<unrepr'able {type(v).__name__}>"

    now = {k: __safe_repr(v) for k, v in frame.f_locals.items() if not k.startswith("__")}
    changed = {k: v for k, v in now.items() if __last_locals.get(key, {}).get(k) != v}
    __last_locals[key] = now
    if filename not in __src_cache:
        try:
            with open(filename) as __f:
                __src_cache[filename] = __f.readlines()
        except OSError:
            __src_cache[filename] = []
    lines = __src_cache[filename]
    src = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
    entry = f"{rel}:{lineno} [{func}] {src}"
    if changed:
        entry += "  # " + ", ".join(f"{k}={v}" for k, v in changed.items())
    __trace_lines.append(entry)
    return __tracer

__sys.settrace(__tracer)
"""

_TRACER_POSTAMBLE = """
__sys.settrace(None)
__sys.stdout.write("\\n--- TRACE (" + str(len(__trace_lines)) + " line(s), "
                   "workspace files only) ---\\n")
for __l in __trace_lines:
    __sys.stdout.write(__l + "\\n")
if len(__trace_lines) >= __MAX_LINES:
    __sys.stdout.write("(stopped at the line cap; narrow the snippet to trace less)\\n")
__sys.exit(__exit_code)
"""


def trace_python(ctx: ToolContext, code: str, timeout: int = 60) -> str:
    """Run a Python snippet with a line-by-line execution trace of the
    workspace code it touches, instead of requiring print() statements added
    by hand and re-run across several calls to see one more variable each
    time.

    Traced, not guessed: a large share of loop_exhausted step failures were
    the model using run_python as an interactive debugger, adding one print
    at a time and re-running to see the next variable -- one bench trace
    stepped through four nested method calls (_peek, _parse_factor,
    _parse_number, _parse_term) via five separate run_python calls, each
    adding a single new print(). This tool answers that same question in one
    call. It complements run_python rather than replacing it -- ordinary
    execution (does this return the right value?) doesn't need a trace, and
    a trace of a long-running loop is capped and truncated, not a substitute
    for a targeted snippet.
    """
    if not code.strip():
        return "error: code must not be empty"
    denied = code_denied(ctx, "trace_python", code)
    if denied:
        return denied
    if ctx.writable is not None:
        return (
            "error: arbitrary Python is disabled while steps run in parallel; "
            "use file tools for declared writes or run this step sequentially"
        )
    try:
        ctx.approver.check_run(code, f"run this Python (traced):\n{code}")
    except Denied as exc:
        ctx.trace.event("trace_python_denied", reason=str(exc))
        return f"error: {exc}"
    budget = ctx.guard.remaining_seconds()
    if budget <= 0:
        return ctx.guard.check_subprocess_budget() or "error: no shell budget left this step"

    before = snapshot_before_write(ctx)
    scratch = ctx.workspace.root / f"_agent_snippet_{uuid.uuid4().hex[:12]}.py"
    root_str = str(ctx.workspace.root).replace("\\", "\\\\").replace('"', '\\"')
    preamble = _TRACER_PREAMBLE.replace("__WORKSPACE_ROOT_PLACEHOLDER__", root_str).replace(
        "__MAX_LINES_PLACEHOLDER__", str(MAX_TRACE_LINES)
    )
    # try/finally, not a bare concatenation: the snippet raising must still
    # print whatever trace was collected up to that point, the same way a
    # human reaches for a debugger *because* something is about to break.
    wrapped = (
        preamble
        + "try:\n"
        + "\n".join(f"    {line}" for line in code.splitlines())
        + "\nexcept BaseException:\n"
        + "    import traceback as __traceback\n"
        + "    __traceback.print_exc()\n"
        + "    __exit_code = 1\n"
        + "else:\n"
        + "    __exit_code = 0\n"
        + "finally:\n"
        + "\n".join(f"    {line}" for line in _TRACER_POSTAMBLE.splitlines())
        + "\n"
    )
    scratch.write_text(wrapped, encoding="utf-8")
    argv = toolchain.resolve("python", str(ctx.workspace.root)) or ["python3"]
    try:
        result = toolchain.run_capturing(
            [*argv, scratch.name],
            cwd=ctx.workspace.root,
            timeout=max(5, min(timeout, budget)),
            env=toolchain.workspace_env(ctx.workspace.root),
        )
    finally:
        scratch.unlink(missing_ok=True)
        mark_external_writes(ctx, before)
    ctx.guard.record_subprocess(result.elapsed_s)
    if result.timed_out:
        return f"error: the snippet timed out after {timeout}s"
    if result.returncode == 127 and not result.stdout:
        return f"error: could not run Python: {result.stderr}"
    ctx.cache.invalidate()

    # keep="tail", not run_python's default "head": the load-bearing part of
    # a trace is what happened *last* -- the state closest to a crash, or
    # the "(stopped at the line cap)" notice -- the same reasoning truncate()
    # already documents for a traceback.
    output = truncate(result.output.strip(), MAX_FAILURE_CHARS, keep="tail")
    ctx.trace.event("trace_python", exit_code=result.returncode, chars=len(code))
    return f"exit={result.returncode}\n{output or '(no output)'}"


# check_code's schema, defined once. The review phase gets the same tool minus
# its `fix` parameter, and building that variant from this rather than writing
# it out a second time is the point: the two copies that used to exist had
# already drifted apart in their descriptions, which is the model-facing half
# and the half nobody notices going stale.
_CHECK_PARAMS = {
    "paths": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Files to check. Omit for everything changed this run.",
    },
    "types": {
        "type": "boolean",
        "description": "Also run ty. Slower (~1s) but catches signature mismatches.",
    },
}
_CHECK_DESC = (
    "Validate Python with Ruff (and optionally ty for cross-file type errors). "
    "Defaults to the files changed so far. This is cheaper and clearer than "
    "running ruff through run_bash."
)
_FIX_DESC = " Set fix=true to apply Ruff's safe autofixes and formatting."

RUN_TESTS = Tool(
    "run_tests",
    "Run pytest and return a parsed summary plus the failing assertions. Prefer this over "
    "calling pytest through run_bash.",
    obj(
        {
            "path": {
                "type": "string",
                "description": "File or directory. Omit to target the test file(s) you've touched "
                "this run; pass '.' explicitly to run the whole suite.",
            },
            "pattern": {"type": "string", "description": "pytest -k expression."},
            "verbose": {
                "type": "boolean",
                "description": "Run every test verbosely instead of stopping at the first failure. "
                'Send the literal true or false, not the strings "true"/"false".',
            },
        }
    ),
    run_tests,
)

CHECK_TOOLS = [
    Tool(
        "run_python",
        "Run a Python snippet in the workspace, with the workspace importable. Use this to "
        "check behaviour instead of `run_bash` with python -c: no shell quoting to get wrong. "
        "Subject to the same approval gate as run_bash. Print what you want to see.",
        obj(
            {
                "code": {"type": "string", "description": "Python source. Use print() for output."},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before it is killed, default 60.",
                },
            },
            ["code"],
        ),
        run_python,
    ),
    Tool(
        "trace_python",
        "Run a Python snippet with a line-by-line execution trace of the workspace code it "
        "touches (which lines ran, and which local variables changed) -- instead of "
        "run_python plus print() statements added one at a time across several calls to see "
        "one more variable each time. Use it when you are stepping through *how* existing "
        "code behaves (which branch fired, what a nested call actually received) rather than "
        "just checking a return value. Same approval gate as run_python; capped to "
        f"{MAX_TRACE_LINES} traced lines, stdlib/site-packages excluded.",
        obj(
            {
                "code": {
                    "type": "string",
                    "description": "Python source to run and trace. No print() needed for "
                    "the trace itself -- use print() only for anything beyond "
                    "what the trace already shows.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before it is killed, default 60.",
                },
            },
            ["code"],
        ),
        trace_python,
    ),
    Tool(
        "check_complexity",
        "Measure cognitive complexity per function with complexipy. Use it when "
        "complexity is the actual question — a function you suspect is tangled, or a "
        "refactor you want to justify with a before and after — not reflexively after "
        "every edit. Cognitive complexity weights nesting and flow breaks, so it "
        "tracks how hard code is to change rather than how many tests it needs.",
        obj(
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to analyse. Omit for everything changed this run.",
                },
                "max_allowed": {
                    "type": "integer",
                    "description": "Per-function ceiling, default 15.",
                },
            }
        ),
        check_complexity,
    ),
    Tool(
        "check_code",
        _CHECK_DESC + _FIX_DESC,
        obj(
            {
                **_CHECK_PARAMS,
                "fix": {
                    "type": "boolean",
                    "description": "Apply ruff --fix and ruff format before reporting.",
                },
            }
        ),
        check_code,
    ),
    RUN_TESTS,
]

# For phases that must not write: no run_python, and check_code has no fix
# parameter to accept at all rather than one merely undocumented. run_tests is
# the same object in both registries — it writes nothing either way.
READ_ONLY_CHECK_TOOLS = [
    Tool("check_code", _CHECK_DESC, obj(dict(_CHECK_PARAMS)), check_code_readonly),
    RUN_TESTS,
]
