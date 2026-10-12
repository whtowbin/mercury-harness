"""Baseline snapshots and unified diffs.

git_diff is only available when the workspace is a repository, which a scratch
directory usually is not. So the harness keeps its own baseline: the first time a
file is touched, its original bytes are stashed, and the run's complete diff can be
reconstructed at any point. The reviewer gets that diff, which is the thing it
actually needs — final file contents do not say what changed.
"""

from __future__ import annotations

import difflib
import threading
from dataclasses import dataclass, field
from pathlib import Path

MAX_DIFF_LINES = 4_000


def _stat_row(root: Path, rel_path: str, before: str | None) -> str | None:
    """One `stat()` line for a single file, or None if it did not change."""
    path = root / rel_path
    try:
        after = path.read_text(encoding="utf-8") if path.is_file() else None
    except (OSError, UnicodeDecodeError):
        return None
    if before == after:
        return None
    added = removed = 0
    for line in difflib.unified_diff(
        (before or "").splitlines(), (after or "").splitlines(), lineterm="", n=0
    ):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    tag = "new" if before is None else ("deleted" if after is None else "mod")
    return f"  {rel_path}  +{added}/-{removed} ({tag})"


@dataclass
class Baseline:
    """Original contents of every file the agent has touched. None means it did not exist.

    Locked internally rather than trusting every caller to coordinate: with
    --step-concurrency > 1, one thread's write triggers capture() while another
    thread's _step_prompt() is mid-iteration over stat()/run_diff() reading the
    same dict — an unguarded "dictionary changed size during iteration" crash,
    not just a theoretical race. The lock is reentrant so run_diff() can call
    diff_for() (which also locks) from the same thread without deadlocking.
    """

    snapshots: dict[str, str | None] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def capture(self, rel_path: str, path: Path) -> None:
        with self._lock:
            if rel_path in self.snapshots:
                return  # only the pre-run state matters
            try:
                self.snapshots[rel_path] = (
                    path.read_text(encoding="utf-8") if path.is_file() else None
                )
            except (OSError, UnicodeDecodeError):
                self.snapshots[rel_path] = None

    def diff_for(self, rel_path: str, root: Path, context: int = 3) -> str:
        with self._lock:
            before = self.snapshots.get(rel_path)
        path = root / rel_path
        try:
            after = path.read_text(encoding="utf-8") if path.is_file() else None
        except (OSError, UnicodeDecodeError):
            return f"--- {rel_path}: unreadable"
        if before == after:
            return ""
        if before is None:
            header, before_lines = "new file", []
        else:
            header, before_lines = "modified", before.splitlines()
        if after is None:
            header, after_lines = "deleted", []
        else:
            after_lines = after.splitlines()
        body = "\n".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                lineterm="",
                n=context,
            )
        )
        return f"[{header}] {rel_path}\n{body}" if body else ""

    def run_diff(
        self,
        root: Path,
        paths: list[str] | None = None,
        context: int = 3,
        max_lines: int = MAX_DIFF_LINES,
    ) -> str:
        with self._lock:
            targets = paths if paths is not None else list(self.snapshots)
        chunks = [d for d in (self.diff_for(p, root, context) for p in targets) if d]
        if not chunks:
            return "(no changes)"
        text = "\n\n".join(chunks)
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines]) + f"\n... diff truncated at {max_lines} lines"
        return text

    def stat(self, root: Path) -> str:
        """A `git diff --stat` style summary, cheap enough to show every step."""
        with self._lock:
            items = list(self.snapshots.items())
        rows = [
            row
            for rel_path, before in items
            if (row := _stat_row(root, rel_path, before)) is not None
        ]
        return "\n".join(rows) if rows else "  (no changes)"
