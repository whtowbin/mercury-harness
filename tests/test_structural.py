"""ast_grep_search and ast_grep_rewrite: structural search and rewrite.

Like test_grep_rg.py, the "tool is missing" tests force that condition
explicitly rather than depending on whether ast-grep happens to be on the
machine running the suite.
"""

from __future__ import annotations

import pytest

from agent import toolchain
from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.fs import write_file
from agent.tools.structural import ast_grep_rewrite, ast_grep_search
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _no_ast_grep(monkeypatch):
    real = toolchain.run_capturing

    def fake(argv, **kwargs):
        if argv[0] == "ast-grep":
            return toolchain.ProcResult(127, "", "ast-grep: command not found", 0.0)
        return real(argv, **kwargs)

    monkeypatch.setattr(toolchain, "run_capturing", fake)


# -- ast_grep_search --------------------------------------------------------


def test_ast_grep_search_finds_a_call_regardless_of_argument_layout(ctx):
    write_file(ctx, "a.py", "old_name(1, 2)\n\nold_name(\n    a,\n    b,\n)\n")
    result = ast_grep_search(ctx, "old_name($$$ARGS)")
    assert result.count("a.py") == 2


def test_ast_grep_search_does_not_match_a_different_call(ctx):
    write_file(ctx, "a.py", "other_name(1, 2)\n")
    result = ast_grep_search(ctx, "old_name($$$ARGS)")
    assert "no matches" in result


def test_ast_grep_search_reports_missing_binary(ctx, monkeypatch):
    _no_ast_grep(monkeypatch)
    write_file(ctx, "a.py", "old_name(1)\n")
    result = ast_grep_search(ctx, "old_name($$$ARGS)")
    assert result.startswith("error:")
    assert "not installed" in result


def test_ast_grep_search_ignores_denylisted_directories(ctx):
    (ctx.workspace.root / "__pycache__").mkdir()
    (ctx.workspace.root / "__pycache__" / "a.py").write_text("old_name(1)\n")
    write_file(ctx, "b.py", "old_name(2)\n")
    result = ast_grep_search(ctx, "old_name($$$ARGS)")
    assert "b.py" in result
    assert "__pycache__" not in result


# -- ast_grep_rewrite: dry run by default -----------------------------------


def test_ast_grep_rewrite_defaults_to_a_dry_run(ctx):
    write_file(ctx, "a.py", "old_name(1, 2)\n")
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)")
    assert "preview only, nothing written" in result
    assert "old_name(1, 2)" in (ctx.workspace.root / "a.py").read_text()


def test_ast_grep_rewrite_preview_reports_no_matches_without_writing(ctx):
    write_file(ctx, "a.py", "other_name(1)\n")
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)")
    assert "no matches" in result
    assert "other_name(1)" in (ctx.workspace.root / "a.py").read_text()


def test_ast_grep_rewrite_missing_binary_in_preview(ctx, monkeypatch):
    _no_ast_grep(monkeypatch)
    write_file(ctx, "a.py", "old_name(1)\n")
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)")
    assert result.startswith("error:")
    assert "not installed" in result


# -- ast_grep_rewrite: apply=true writes for real ---------------------------


def test_ast_grep_rewrite_applies_across_every_matching_file(ctx):
    write_file(ctx, "a.py", "old_name(1)\n")
    write_file(ctx, "b.py", "x = old_name(2)\n")
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)", apply=True)
    assert "applied to 2 file(s)" in result
    assert (ctx.workspace.root / "a.py").read_text() == "new_name(1)\n"
    assert (ctx.workspace.root / "b.py").read_text() == "x = new_name(2)\n"


def test_ast_grep_rewrite_marks_touched_and_baselines_changed_files(ctx):
    """A file that already existed before this run touched anything — not
    one write_file itself just created, whose baseline is correctly None —
    so the pre-rewrite content this asserts is really the run's starting
    state, the thing `snapshot_before_write` exists to capture."""
    (ctx.workspace.root / "a.py").write_text("old_name(1)\n")
    ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)", apply=True)
    assert "a.py" in ctx.touched
    assert ctx.baseline.snapshots.get("a.py") == "old_name(1)\n"


def test_ast_grep_rewrite_apply_with_no_matches_writes_nothing(ctx):
    write_file(ctx, "a.py", "other_name(1)\n")
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)", apply=True)
    assert "no matches" in result
    assert (ctx.workspace.root / "a.py").read_text() == "other_name(1)\n"


# -- jail awareness: refuses outright during a concurrent batch ------------


def test_ast_grep_rewrite_apply_is_disabled_during_concurrent_steps(ctx):
    write_file(ctx, "a.py", "old_name(1)\n")
    ctx.writable = frozenset({"a.py"})
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)", apply=True)
    assert "disabled while steps run in parallel" in result
    assert (ctx.workspace.root / "a.py").read_text() == "old_name(1)\n"


def test_ast_grep_rewrite_preview_is_unaffected_by_the_jail(ctx):
    """A dry run only reads, so it stays available even mid-batch — refusing
    it too would take away the one safe way to plan a rewrite from inside a
    concurrent step."""
    write_file(ctx, "a.py", "old_name(1)\n")
    ctx.writable = frozenset({"a.py"})
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)")
    assert "preview only" in result


def test_ast_grep_search_is_unaffected_by_the_jail(ctx):
    write_file(ctx, "a.py", "old_name(1)\n")
    ctx.writable = frozenset({"a.py"})
    result = ast_grep_search(ctx, "old_name($$$ARGS)")
    assert "a.py" in result


# -- approval gating on apply=true -------------------------------------------


def test_ast_grep_rewrite_apply_is_gated_when_not_auto_approved(tmp_path):
    """apply=true writes files exactly as capable of damage as run_python,
    so it must not run unconditionally just because it isn't shell syntax."""
    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    write_file(gated_ctx, "a.py", "old_name(1)\n")
    # No TTY in a test process: the gate must deny rather than silently run.
    result = ast_grep_rewrite(gated_ctx, "old_name($$$ARGS)", "new_name($$$ARGS)", apply=True)
    assert result.startswith("error:")
    assert (tmp_path / "a.py").read_text() == "old_name(1)\n"


def test_ast_grep_rewrite_preview_needs_no_approval(tmp_path):
    """Nothing is written by a preview, so it should never block on approval
    the way apply=true does."""
    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    write_file(gated_ctx, "a.py", "old_name(1)\n")
    result = ast_grep_rewrite(gated_ctx, "old_name($$$ARGS)", "new_name($$$ARGS)")
    assert "preview only" in result
