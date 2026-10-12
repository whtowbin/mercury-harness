"""agent.checkpoints: real git snapshots via a throwaway index.

These exercise actual git subprocess calls (like tests/test_ty_diagnostics.py
does for ty), so they are skipped rather than mocked when git is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from agent import checkpoints

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="git not installed")


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _init_repo(root):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "a@b.c")
    _git(root, "config", "user.name", "Test")
    (root / "README.md").write_text("hello\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")


def test_not_a_repo_returns_none(tmp_path):
    assert checkpoints.is_repo(tmp_path) is False
    assert checkpoints.create(tmp_path, "run1", "step1", "msg") is None


def test_creates_a_commit_without_moving_head_or_branch(tmp_path):
    _init_repo(tmp_path)
    head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    branch_before = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    (tmp_path / "new.py").write_text("print(1)\n")
    sha = checkpoints.create(tmp_path, "run1", "step-a", "checkpoint after step-a")
    assert sha is not None

    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    branch_after = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head_after == head_before
    assert branch_after == branch_before
    # The real index/staging area must be untouched by the throwaway one.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    assert "new.py" in status  # still untracked, not staged


def test_the_checkpoint_commit_contains_the_untracked_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "new.py").write_text("print(1)\n")
    sha = checkpoints.create(tmp_path, "run1", "step-a", "msg")
    assert sha is not None
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", sha],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "new.py" in listing
    assert "README.md" in listing


def test_checkpoint_is_reachable_via_its_ref(tmp_path):
    _init_repo(tmp_path)
    sha = checkpoints.create(tmp_path, "run1", "step-a", "msg")
    pairs = checkpoints.list_checkpoints(tmp_path, "run1")
    assert (f"{checkpoints.REF_PREFIX}/run1/step-a", sha) in pairs


def test_list_checkpoints_filters_by_run_id(tmp_path):
    _init_repo(tmp_path)
    checkpoints.create(tmp_path, "run1", "step-a", "msg")
    checkpoints.create(tmp_path, "run2", "step-a", "msg")
    assert len(checkpoints.list_checkpoints(tmp_path, "run1")) == 1
    assert len(checkpoints.list_checkpoints(tmp_path, "run2")) == 1
    assert len(checkpoints.list_checkpoints(tmp_path)) == 2


def test_a_workspace_nested_inside_a_host_repo_is_not_checkpointed(tmp_path):
    _init_repo(tmp_path)
    inner = tmp_path / "sandbox"
    inner.mkdir()
    assert checkpoints.is_repo(inner) is False
    assert checkpoints.create(inner, "run1", "step-a", "msg") is None
