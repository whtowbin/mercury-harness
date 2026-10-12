"""Validation, caching, guards and diffs — the layer that gives the model feedback."""

from __future__ import annotations

import pytest

from agent import validate
from agent.approval import Approver
from agent.diffs import Baseline
from agent.guards import GuardTripped, ResourceGuard, advise, workspace_size_mb
from agent.tools import ToolContext
from agent.tools.fs import edit_file, read_file, write_file
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- syntax checking ------------------------------------------------------


def test_syntax_check_reports_line_and_context(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def f():\n    return (1\n")
    diagnostic = validate.syntax_check(path)
    assert diagnostic is not None
    assert diagnostic.code == "SyntaxError"
    assert "return (1" in diagnostic.message


def test_syntax_check_passes_valid_code(tmp_path):
    path = tmp_path / "ok.py"
    path.write_text("def f():\n    return 1\n")
    assert validate.syntax_check(path) is None


def test_syntax_check_ignores_non_python(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("this is (((not python")
    assert validate.syntax_check(path) is None


def test_write_file_reports_syntax_error_in_the_same_result(ctx):
    result = write_file(ctx, "bad.py", "def f(:\n    pass\n")
    assert "SYNTAX ERROR" in result
    # The file is still written; the model is told, not overruled.
    assert (ctx.workspace.root / "bad.py").exists()


def test_edit_file_reports_syntax_error_it_introduced(ctx):
    write_file(ctx, "m.py", "def f():\n    return 1\n")
    result = edit_file(ctx, "m.py", "    return 1", "    return (1")
    assert "SYNTAX ERROR" in result


def test_clean_write_reports_nothing_extra(ctx):
    result = write_file(ctx, "clean.py", "x = 1\n")
    assert "SYNTAX ERROR" not in result
    assert "ruff:" not in result


def test_autolint_can_be_disabled(ctx):
    ctx.autolint = False
    result = write_file(ctx, "unused.py", "import os\n")
    assert "ruff:" not in result


# -- ruff / ty parsing ----------------------------------------------------


def test_ruff_check_finds_undefined_name(ctx):
    path = ctx.workspace.root / "u.py"
    path.write_text("def f():\n    return undefined_thing\n")
    diagnostics = validate.ruff_check(ctx.workspace.root, [str(path)])
    codes = {d.code for d in diagnostics}
    assert "F821" in codes, f"expected F821, got {codes}"


def test_render_truncates_long_diagnostic_lists():
    many = [validate.Diagnostic("f.py", i, 1, "F401", "unused") for i in range(30)]
    rendered = validate.render(many, limit=5)
    assert rendered.count("\n") == 5
    assert "and 25 more" in rendered


def test_ty_concise_parser_extracts_location(tmp_path):
    line = "error[invalid-argument-type] src/app.py:12:5: Argument is incorrect"
    parsed = validate._parse_concise(tmp_path, line)
    assert parsed is not None
    assert parsed.line == 12
    assert parsed.column == 5
    assert parsed.code == "invalid-argument-type"


def test_ty_concise_parser_skips_noise(tmp_path):
    assert validate._parse_concise(tmp_path, "All checks passed!") is None
    assert validate._parse_concise(tmp_path, "") is None


# -- read cache -----------------------------------------------------------


def test_cache_serves_repeat_reads(ctx):
    write_file(ctx, "c.py", "x = 1\n")
    ctx.cache.invalidate()
    read_file(ctx, "c.py")
    hits_before = ctx.cache.hits
    read_file(ctx, "c.py")
    assert ctx.cache.hits == hits_before + 1


def test_cache_invalidates_when_the_file_changes_underneath(ctx):
    write_file(ctx, "c.py", "x = 1\n")
    read_file(ctx, "c.py")
    # Simulate a shell command rewriting the file.
    path = ctx.workspace.root / "c.py"
    path.write_text("x = 2\nextra = True\n")
    assert "x = 2" in read_file(ctx, "c.py")


def test_edit_refreshes_the_cache(ctx):
    write_file(ctx, "c.py", "x = 1\n")
    edit_file(ctx, "c.py", "x = 1", "x = 99")
    assert "x = 99" in read_file(ctx, "c.py")


# -- guards ---------------------------------------------------------------


def test_write_size_cap_refuses_oversized_content(ctx):
    ctx.guard.max_file_write_mb = 0.001
    result = write_file(ctx, "big.txt", "x" * 5000)
    assert result.startswith("error:")
    assert not (ctx.workspace.root / "big.txt").exists()


def test_shell_budget_blocks_once_spent():
    guard = ResourceGuard(step_subprocess_budget_s=10)
    assert guard.check_subprocess_budget() is None
    guard.record_subprocess(11)
    assert "shell budget" in (guard.check_subprocess_budget() or "")


def test_reset_step_clears_the_clock_and_errors():
    guard = ResourceGuard()
    guard.record_subprocess(50)
    guard.record_error()
    guard.reset_step()
    assert guard.subprocess_seconds == 0
    assert guard.step_errors == 0


def test_error_budget_warns_then_trips():
    guard = ResourceGuard(max_step_errors=4)
    assert guard.record_error() is None
    assert "error budget" in (guard.record_error() or "")  # halfway
    guard.record_error()
    with pytest.raises(GuardTripped):
        guard.record_error()


def test_workspace_size_ignores_virtualenvs(tmp_path):
    (tmp_path / "real.txt").write_text("x" * 1000)
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "huge.bin").write_text("y" * 500_000)
    assert workspace_size_mb(tmp_path) < 0.01


def test_advise_flags_pip_install():
    assert "uv add" in (advise("pip install requests") or "")
    assert advise("uv add requests") is None


# -- baseline diffs -------------------------------------------------------


def test_baseline_diff_reports_a_new_file(ctx):
    write_file(ctx, "new.py", "x = 1\n")
    diff = ctx.baseline.run_diff(ctx.workspace.root)
    assert "[new file] new.py" in diff
    assert "+x = 1" in diff


def test_baseline_diff_reports_a_modification(ctx):
    (ctx.workspace.root / "m.py").write_text("a = 1\n")
    edit_file(ctx, "m.py", "a = 1", "a = 2")
    diff = ctx.baseline.run_diff(ctx.workspace.root)
    assert "[modified] m.py" in diff
    assert "-a = 1" in diff and "+a = 2" in diff


def test_baseline_keeps_only_the_original_state(ctx):
    (ctx.workspace.root / "m.py").write_text("v = 0\n")
    edit_file(ctx, "m.py", "v = 0", "v = 1")
    edit_file(ctx, "m.py", "v = 1", "v = 2")
    diff = ctx.baseline.run_diff(ctx.workspace.root)
    # The diff is against the pre-run state, not the previous edit.
    assert "-v = 0" in diff and "+v = 2" in diff
    assert "v = 1" not in diff


def test_baseline_stat_counts_lines(ctx):
    write_file(ctx, "s.py", "a = 1\nb = 2\nc = 3\n")
    assert "+3/-0 (new)" in ctx.baseline.stat(ctx.workspace.root)


def test_baseline_is_empty_before_any_write(ctx):
    assert Baseline().run_diff(ctx.workspace.root) == "(no changes)"


# -- the workspace cap, which used to be advertised and never applied -----


def test_the_workspace_cap_is_actually_enforced_on_writes(tmp_path):
    """`--max-workspace-mb` is a documented flag, plumbed all the way into
    ResourceGuard, and check_workspace_size had no callers: the cap was
    believed and never applied. This pins the call path, not just the maths."""
    from agent.guards import WRITES_BETWEEN_SIZE_CHECKS, ResourceGuard

    guard = ResourceGuard(max_workspace_mb=0.0001)
    (tmp_path / "big.bin").write_bytes(b"x" * 200_000)

    # Amortised: the first writes are free, then one measurement.
    for _ in range(WRITES_BETWEEN_SIZE_CHECKS - 1):
        assert guard.check_before_write(tmp_path) is None
    problem = guard.check_before_write(tmp_path)
    assert problem is not None
    assert "over the" in problem and "generating files in a loop" in problem


def test_a_workspace_under_the_cap_is_never_refused(tmp_path):
    from agent.guards import WRITES_BETWEEN_SIZE_CHECKS, ResourceGuard

    guard = ResourceGuard(max_workspace_mb=500.0)
    (tmp_path / "small.txt").write_text("hello")
    for _ in range(WRITES_BETWEEN_SIZE_CHECKS * 2):
        assert guard.check_before_write(tmp_path) is None


def test_write_file_refuses_once_the_workspace_is_over_the_cap(tmp_path):
    """End to end through the tool the model actually calls."""
    from agent.approval import Approver
    from agent.guards import WRITES_BETWEEN_SIZE_CHECKS, ResourceGuard
    from agent.tools import ToolContext
    from agent.tools.fs import write_file
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    root.mkdir()
    (root / "hog.bin").write_bytes(b"x" * 300_000)
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        guard=ResourceGuard(max_workspace_mb=0.0001),
    )
    for _ in range(WRITES_BETWEEN_SIZE_CHECKS - 1):
        write_file(ctx, "ok.py", "x = 1\n")
    out = write_file(ctx, "blocked.py", "x = 2\n")
    assert out.startswith("error:")
    assert not (root / "blocked.py").exists(), "the refused write must not land"
