"""Automatic per-step git snapshots, so a run can be inspected or reverted
without anyone having had to think to commit along the way.

Claude Code and Cursor both checkpoint the working tree as steps complete;
this harness had no equivalent, so a run that went sideways three steps in
had no anchor to compare against except a full read of the trace. A
checkpoint here is a git commit object holding every file in the workspace at
that moment, including untracked ones — but it is not on the user's branch:
it is staged through a private, throwaway index, so the user's real staging
area, HEAD, and current branch are never touched. Restoring is left to the
user (or the model, via run_bash/git_diff, both already available): a bad
checkpoint sha is inert until someone chooses to check it out.

Best-effort throughout: a checkpoint is a convenience, not a guarantee, so
anything that goes wrong — no git, not a repo, a workspace nested inside a
larger project, a mid-command failure — returns None/[] rather than raising.
That last case matters most: a workspace nested inside an unrelated host
project must not have commits written into the host's own .git, which is why
this insists the workspace *is* a repo's toplevel, not merely inside one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import toolchain

CHECKPOINT_TIMEOUT = 30
REF_PREFIX = "refs/harness-checkpoints"


def _run(root: Path, args: list[str], env: dict[str, str] | None = None):
    return toolchain.run_capturing(
        ["git", *args],
        cwd=root,
        timeout=CHECKPOINT_TIMEOUT,
        env=env if env is not None else toolchain.clean_env(),
    )


def is_repo(root: Path) -> bool:
    """True only when `root` is a git repo's own toplevel.

    Deliberately stricter than "inside a work tree": git searches upward for
    a repo, so a workspace nested inside a larger project would otherwise
    have its checkpoints written into that outer repo's objects and refs.
    """
    result = _run(root, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == root.resolve()
    except OSError:
        return False


def _head(root: Path) -> str | None:
    result = _run(root, ["rev-parse", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else None


def create(root: Path, run_id: str, label: str, message: str) -> str | None:
    """Snapshot the working tree into a commit under `refs/harness-checkpoints/
    <run_id>/<label>`, touching neither HEAD, the current branch, nor the
    user's real index. Returns the commit sha, or None if this is not a repo
    (by its own toplevel), git is unavailable, or the snapshot failed.
    """
    if not is_repo(root):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(toolchain.clean_env())
        env["GIT_INDEX_FILE"] = os.path.join(tmp, "index")
        head = _head(root)
        if head and _run(root, ["read-tree", head], env=env).returncode != 0:
            return None
        if _run(root, ["add", "-A"], env=env).returncode != 0:
            return None
        tree = _run(root, ["write-tree"], env=env)
        if tree.returncode != 0 or not tree.stdout.strip():
            return None
        commit_args = ["commit-tree", tree.stdout.strip(), "-m", message]
        if head:
            commit_args += ["-p", head]
        commit = _run(root, commit_args, env=env)
        if commit.returncode != 0 or not commit.stdout.strip():
            return None
        sha = commit.stdout.strip()
    ref = f"{REF_PREFIX}/{run_id}/{label}"
    if _run(root, ["update-ref", ref, sha]).returncode != 0:
        return None
    return sha


def list_checkpoints(root: Path, run_id: str | None = None) -> list[tuple[str, str]]:
    """(ref, sha) pairs, newest-created last, for one run or every run this
    workspace has ever checkpointed."""
    pattern = f"{REF_PREFIX}/{run_id}" if run_id else REF_PREFIX
    result = _run(root, ["for-each-ref", "--format=%(refname) %(objectname)", pattern])
    if result.returncode != 0:
        return []
    pairs = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs
