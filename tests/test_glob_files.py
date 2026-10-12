"""glob_files: the ripgrep-backed file listing, and the pathlib rglob
fallback it must never regress -- same structure and the same reason as
test_grep_rg.py's split between the two paths.
"""

from __future__ import annotations

import pytest

from agent import toolchain
from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.fs import glob_files, write_file
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _no_rg(monkeypatch):
    real = toolchain.run_capturing

    def fake(argv, **kwargs):
        if argv[0] == "rg":
            return toolchain.ProcResult(127, "", "rg: command not found", 0.0)
        return real(argv, **kwargs)

    monkeypatch.setattr(toolchain, "run_capturing", fake)


# -- basic correctness, whichever path answers ----------------------------


def test_glob_files_finds_a_match_by_extension(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    write_file(ctx, "a.txt", "x = 1\n")
    result = glob_files(ctx, "*.py")
    assert "a.py" in result and "a.txt" not in result


def test_glob_files_matches_at_any_depth_regardless_of_leading_star_star(ctx):
    write_file(ctx, "top.py", "x = 1\n")
    write_file(ctx, "sub/dir/nested.py", "x = 1\n")
    bare = glob_files(ctx, "*.py")
    starred = glob_files(ctx, "**/*.py")
    assert "top.py" in bare and "sub/dir/nested.py" in bare
    assert sorted(bare.splitlines()) == sorted(starred.splitlines())


def test_glob_files_can_be_scoped_to_a_subdirectory(ctx):
    write_file(ctx, "agent/tools/a.py", "x = 1\n")
    write_file(ctx, "other/b.py", "x = 1\n")
    result = glob_files(ctx, "agent/tools/*.py")
    assert "agent/tools/a.py" in result and "other/b.py" not in result


def test_glob_files_reports_no_matches(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    assert "no files match" in glob_files(ctx, "*.zzznope")


def test_glob_files_ignores_denylisted_directories(ctx):
    (ctx.workspace.root / "__pycache__").mkdir()
    (ctx.workspace.root / "__pycache__" / "a.py").write_text("x = 1\n")
    write_file(ctx, "b.py", "x = 1\n")
    result = glob_files(ctx, "*.py")
    assert "b.py" in result
    assert "__pycache__" not in result


# -- the fallback path specifically ----------------------------------------


def test_glob_files_falls_back_to_python_when_rg_is_missing(ctx, monkeypatch):
    _no_rg(monkeypatch)
    write_file(ctx, "a.py", "x = 1\n")
    write_file(ctx, "a.txt", "x = 1\n")
    result = glob_files(ctx, "*.py")
    assert "a.py" in result and "a.txt" not in result


def test_glob_files_fallback_matches_the_same_set_as_the_fast_path(ctx, monkeypatch):
    write_file(ctx, "top.py", "x = 1\n")
    write_file(ctx, "sub/dir/nested.py", "x = 1\n")
    write_file(ctx, "sub/dir/test_nested.py", "x = 1\n")
    fast = glob_files(ctx, "**/*.py")
    _no_rg(monkeypatch)
    slow = glob_files(ctx, "**/*.py")
    assert sorted(fast.splitlines()) == sorted(slow.splitlines())
