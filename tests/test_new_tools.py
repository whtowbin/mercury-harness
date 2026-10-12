"""move_file, delete_file, git_add/git_commit, and the web tools."""

from __future__ import annotations

import subprocess

import pytest

from agent.approval import Approver
from agent.tools import ToolContext, execution_registry, planning_registry, review_registry
from agent.tools.fs import delete_file, move_file, read_file, write_file
from agent.tools.vcs import git_add, git_commit
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def git_repo(ctx) -> None:
    """A real repo, since _is_repo shells out to the real git binary."""
    root = str(ctx.workspace.root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


# -- move_file --------------------------------------------------------------


def test_move_file_moves_the_content(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    result = move_file(ctx, "a.py", "b.py")
    assert "moved a.py -> b.py" in result
    assert not (ctx.workspace.root / "a.py").is_file()
    assert (ctx.workspace.root / "b.py").read_text() == "x = 1\n"


def test_move_file_refuses_to_overwrite_an_existing_destination(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    write_file(ctx, "b.py", "x = 2\n")
    result = move_file(ctx, "a.py", "b.py")
    assert result.startswith("error:")
    assert (ctx.workspace.root / "a.py").is_file()
    assert (ctx.workspace.root / "b.py").read_text() == "x = 2\n"


def test_move_file_missing_source_errors(ctx):
    assert move_file(ctx, "nope.py", "b.py").startswith("error:")


def test_move_file_creates_intermediate_directories(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    move_file(ctx, "a.py", "sub/dir/b.py")
    assert (ctx.workspace.root / "sub" / "dir" / "b.py").is_file()


def test_move_file_updates_the_baseline_for_both_sides(ctx):
    # Written directly, not via write_file: the baseline only records a
    # file's *pre-run* state, so a.py must already exist before this run's
    # own bookkeeping starts for its move to show up as a deletion — a file
    # created and moved within the same run nets to "a.py never existed,
    # b.py does", which is the correct run-level diff, not a bug.
    (ctx.workspace.root / "a.py").write_text("x = 1\n")
    move_file(ctx, "a.py", "b.py")
    diff = ctx.baseline.run_diff(ctx.workspace.root)
    assert "a.py" in diff and "deleted" in diff.lower()
    assert "b.py" in diff and "new file" in diff.lower()


def test_move_file_catches_syntax_errors_in_the_moved_file(ctx):
    write_file(ctx, "a.py", "def broken(:\n")
    result = move_file(ctx, "a.py", "b.py")
    assert "SYNTAX ERROR" in result


# -- delete_file --------------------------------------------------------------


def test_delete_file_removes_it(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    result = delete_file(ctx, "a.py")
    assert "deleted a.py" in result
    assert not (ctx.workspace.root / "a.py").is_file()


def test_delete_file_missing_file_errors(ctx):
    assert delete_file(ctx, "nope.py").startswith("error:")


def test_delete_file_shows_up_in_the_diff_as_deleted(ctx):
    # Same reasoning as the move_file baseline test above: written directly
    # so the baseline sees it as pre-existing, not created this run.
    (ctx.workspace.root / "a.py").write_text("x = 1\n")
    delete_file(ctx, "a.py")
    diff = ctx.baseline.run_diff(ctx.workspace.root)
    assert "deleted" in diff.lower()


def test_delete_file_invalidates_the_read_cache(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    read_file(ctx, "a.py")  # populate the cache
    delete_file(ctx, "a.py")
    assert ctx.cache.get(ctx.workspace.root / "a.py") is None


# -- git_add / git_commit -----------------------------------------------------


def test_git_add_and_commit_outside_a_repo_explain_themselves(ctx):
    assert "not a git repository" in git_add(ctx)
    assert "not a git repository" in git_commit(ctx, "message")


def test_git_add_then_commit(ctx):
    git_repo(ctx)
    write_file(ctx, "a.py", "x = 1\n")
    assert "staged" in git_add(ctx).lower() or git_add(ctx) == ""
    result = git_commit(ctx, "add a.py")
    assert "add a.py" in result or result == ""
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=str(ctx.workspace.root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "add a.py" in log.stdout


def test_git_commit_can_stage_specific_paths_itself(ctx):
    git_repo(ctx)
    write_file(ctx, "a.py", "x = 1\n")
    write_file(ctx, "b.py", "x = 2\n")
    git_commit(ctx, "just a.py", paths=["a.py"])
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ctx.workspace.root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "?? b.py" in status.stdout  # b.py was never staged or committed


def test_git_commit_message_needs_no_shell_quoting(ctx):
    """The actual point of a dedicated tool over run_bash: this message has
    quotes, a backtick and a newline — all of which would need careful
    shell-escaping through run_bash, and none of which need any escaping
    here since the message is a real argv element, never a shell string."""
    git_repo(ctx)
    write_file(ctx, "a.py", "x = 1\n")
    message = 'fix: handle "quoted" input, `backticks`, and\na second line'
    git_commit(ctx, message, paths=["a.py"])
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=str(ctx.workspace.root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'handle "quoted" input, `backticks`, and' in log.stdout
    assert "a second line" in log.stdout


def test_git_commit_with_nothing_staged_says_so_clearly(ctx):
    git_repo(ctx)
    result = git_commit(ctx, "nothing to see here")
    assert "nothing to commit" in result.lower()


def test_git_commit_rejects_an_empty_message(ctx):
    git_repo(ctx)
    write_file(ctx, "a.py", "x = 1\n")
    assert git_commit(ctx, "   ").startswith("error:")


# -- write tools are execution-only, never planning or review ----------------


def test_move_delete_and_git_write_tools_are_execution_only():
    exec_names = execution_registry().names()
    for tool in ("move_file", "delete_file", "git_add", "git_commit"):
        assert tool in exec_names
        assert tool not in planning_registry().names()
        assert tool not in review_registry().names()


def test_web_tools_are_available_everywhere_read_only_included(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    for registry in (execution_registry(), planning_registry(), review_registry()):
        names = registry.names()
        assert "fetch_url" in names
        assert "web_search" in names


def test_web_search_is_not_offered_without_a_backend(monkeypatch):
    """A tool that can only return an error is a wasted schema and round trip."""
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    for registry in (
        execution_registry(),
        execution_registry("look up the api docs for this library"),
        planning_registry(),
        review_registry(),
    ):
        names = registry.names()
        assert "web_search" not in names
        assert "fetch_url" in names
