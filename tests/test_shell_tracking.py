"""run_bash's writes are no longer invisible to touched/baseline tracking.

Review chunking, syntax checks, changed-line coverage, the no-tests
behavioural-change gate, run diffs, and workspace_fingerprint all derive from
ctx.touched. A shell command used to be able to create or rewrite a file and
leave every one of those blind to it.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.shell import run_bash
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def test_a_file_created_by_shell_is_marked_touched(ctx):
    run_bash(ctx, "echo 'x = 1' > created.py")
    assert "created.py" in ctx.touched


def test_a_file_modified_by_shell_is_marked_touched(ctx):
    (ctx.workspace.root / "existing.py").write_text("VALUE = 1\n")
    run_bash(ctx, "echo 'VALUE = 2' > existing.py")
    assert "existing.py" in ctx.touched


def test_an_unrelated_file_the_command_never_touches_is_not_marked(ctx):
    (ctx.workspace.root / "untouched.py").write_text("VALUE = 1\n")
    run_bash(ctx, "echo hi")
    assert "untouched.py" not in ctx.touched


def test_a_file_modified_by_shell_gets_a_correct_pre_run_baseline(ctx):
    """The baseline must be the content before the shell command ran, not
    (as a naive post-hoc capture would produce) the content after."""
    (ctx.workspace.root / "existing.py").write_text("VALUE = 1\n")
    run_bash(ctx, "echo 'VALUE = 2' > existing.py")
    diff = ctx.baseline.diff_for("existing.py", ctx.workspace.root)
    assert "-VALUE = 1" in diff
    assert "+VALUE = 2" in diff


def test_a_file_deleted_by_shell_is_marked_touched(ctx):
    (ctx.workspace.root / "doomed.py").write_text("VALUE = 1\n")
    run_bash(ctx, "rm doomed.py")
    assert "doomed.py" in ctx.touched


def test_cdpath_in_the_parent_environment_cannot_redirect_an_approved_cd(tmp_path, monkeypatch):
    """`cd tests && cat secret.txt` is approved as workspace-local, but with CDPATH
    exported the shell's `cd tests` can land in another directory first. Real /bin/sh,
    real directories: the workspace's own tests/ must be the one read."""
    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "secret.txt").write_text("WORKSPACE-OWN\n")
    outside = tmp_path / "outside"
    (outside / "tests").mkdir(parents=True)
    (outside / "tests" / "secret.txt").write_text("OUTSIDE-SECRET\n")
    monkeypatch.setenv("CDPATH", str(outside))
    ctx = ToolContext(
        workspace=Workspace(ws),
        approver=Approver(auto_approve=True, workspace_root=ws, unattended=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    result = run_bash(ctx, "cd tests && cat secret.txt")
    assert "WORKSPACE-OWN" in result
    assert "OUTSIDE-SECRET" not in result
