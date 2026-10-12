"""Resource guards. These exist because the failure modes that cost real money are
not bad code, they are loops: a step that shells out forever, a test run that fills
the disk, a model that makes the same failing call twenty times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .workspace import walk_files

# Writes between full workspace measurements. See check_before_write.
WRITES_BETWEEN_SIZE_CHECKS = 25

# Commands worth a second look before they run, even when allowlisted.
EXPENSIVE_PATTERNS = (
    (
        re.compile(r"(?<!uv )\bpip\s+install\b"),
        "use `uv add` so the dependency is recorded in pyproject.toml",
    ),
    (
        re.compile(r"\buv\s+pip\b"),
        "`uv pip` targets an ambient virtualenv; `uv add` targets this project",
    ),
    (
        re.compile(r"\bnpm\s+(install|i)\b"),
        "npm install can take minutes and write thousands of files",
    ),
    (re.compile(r"\bdocker\b"), "docker pulls are slow and may not be available"),
    (re.compile(r"\bcurl\b|\bwget\b"), "network fetches are slow and may hang"),
    (
        re.compile(r"\b(find|grep|rg)\s+/\s"),
        "searching from / scans the whole filesystem; scope it to the workspace",
    ),
    (re.compile(r"\bwhile\s+true\b|\byes\b\s*\|"), "this looks like an unbounded loop"),
)


class GuardTripped(Exception):
    """A hard limit was hit. The step stops; the run may continue."""


@dataclass
class ResourceGuard:
    """Per-run limits, with the subprocess clock reset at each step boundary."""

    # Cumulative wall time a single step may spend in subprocesses.
    step_subprocess_budget_s: float = 90.0
    # Total workspace size. Catches runaway file generation.
    max_workspace_mb: float = 500.0
    # Largest single file the agent may write.
    max_file_write_mb: float = 5.0
    # Tool calls that returned an error before a step is abandoned.
    max_step_errors: int = 12

    subprocess_seconds: float = 0.0
    step_errors: int = 0
    # Writes since the workspace was last measured. Walking the tree costs a
    # stat() per file, so the cap is checked on a cadence rather than on every
    # write: runaway generation is a loop, so it trips within a few writes
    # either way, while an ordinary step pays for one walk or none.
    writes_since_size_check: int = 0

    def reset_step(self) -> None:
        self.subprocess_seconds = 0.0
        self.step_errors = 0

    # -- subprocess time -------------------------------------------------
    def remaining_seconds(self) -> float:
        return max(0.0, self.step_subprocess_budget_s - self.subprocess_seconds)

    def check_subprocess_budget(self) -> str | None:
        if self.remaining_seconds() <= 0:
            return (
                f"error: this step has used its entire {self.step_subprocess_budget_s:.0f}s "
                "shell budget. Stop running commands and call finish_step with what you have."
            )
        return None

    def record_subprocess(self, seconds: float) -> None:
        self.subprocess_seconds += seconds

    # -- writes ----------------------------------------------------------
    def check_write_size(self, num_bytes: int) -> str | None:
        cap = self.max_file_write_mb * 1024 * 1024
        if num_bytes > cap:
            return (
                f"error: refusing to write {num_bytes / 1048576:.1f}MB; the per-file cap is "
                f"{self.max_file_write_mb:.0f}MB. Write something smaller."
            )
        return None

    def check_workspace_size(self, root: Path) -> str | None:
        used_mb = workspace_size_mb(root)
        if used_mb > self.max_workspace_mb:
            return (
                f"error: the workspace is {used_mb:.0f}MB, over the "
                f"{self.max_workspace_mb:.0f}MB cap. Something is generating files in a loop."
            )
        return None

    def check_before_write(self, root: Path) -> str | None:
        """The workspace cap, amortised over WRITES_BETWEEN_SIZE_CHECKS writes.

        This exists because the cap was, until now, entirely unenforced.
        `--max-workspace-mb` is a documented CLI flag, plumbed through Settings
        into this dataclass and then read by nothing: check_workspace_size had
        no callers at all. A limit the user is offered and the harness never
        applies is worse than no limit, because it is believed.
        """
        self.writes_since_size_check += 1
        if self.writes_since_size_check < WRITES_BETWEEN_SIZE_CHECKS:
            return None
        self.writes_since_size_check = 0
        return self.check_workspace_size(root)

    # -- errors ----------------------------------------------------------
    def record_error(self) -> str | None:
        self.step_errors += 1
        if self.step_errors == self.max_step_errors // 2:
            return (
                "\n\n[harness] Half this step's error budget is gone. Re-read the files "
                "you are editing instead of retrying variations of the same call."
            )
        if self.step_errors >= self.max_step_errors:
            raise GuardTripped(f"{self.step_errors} tool errors in one step; abandoning it")
        return None


def workspace_size_mb(root: Path) -> float:
    total = 0
    for path in walk_files(root):
        try:
            if not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total / 1048576


def advise(command: str) -> str | None:
    """A note prepended to the command's output, not a block. The model still runs it."""
    for pattern, reason in EXPENSIVE_PATTERNS:
        if pattern.search(command):
            return reason
    return None
