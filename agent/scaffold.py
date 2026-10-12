"""A new project's starting layout: `uv init`, a Ruff config, pytest and ruff as dev tools.

Request (2026-10-01): new projects use uv for their basic structure and have Ruff formatting on
by default (DESIGN-LOG §104). A Ruff config is also what opts a project into the harness's
automatic formatting (`projectconfig.uses_a_formatter`). Only an empty workspace with a goal
that does not name another language is scaffolded; an existing project is never touched.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .trace import Trace

# Files and folders a workspace may hold and still count as new: the harness's own.
_HARNESS_ENTRIES = frozenset({".harness", "traces", "failed-chats", ".git", ".DS_Store"})
# A goal naming one of these is not a Python project.
_OTHER_LANGUAGE = re.compile(
    r"\b(javascript|typescript|node(\.js)?|react|vue|svelte|rust|cargo|golang|go module|"
    r"java|kotlin|swift|c\+\+|c#|\.net|ruby|php|html page|static site)\b",
    re.IGNORECASE,
)
RUFF_SECTION = "\n[tool.ruff]\nline-length = 100\n"
# uv init is local; uv add reaches the package index, so it gets longer.
INIT_TIMEOUT_S = 60
ADD_TIMEOUT_S = 120


def is_new_project(root: Path) -> bool:
    """Nothing in the workspace but the harness's own files."""
    try:
        return all(entry.name in _HARNESS_ENTRIES for entry in root.iterdir())
    except OSError:
        return False


def wants_python(goal: str) -> bool:
    return not _OTHER_LANGUAGE.search(goal)


def _name(root: Path) -> str:
    """A package name uv accepts, from the folder name."""
    name = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-")
    return name or "project"


def scaffold(root: Path, trace: Trace) -> bool:
    """Set up a new Python project in `root`. True when the layout was created."""
    init = _run(["uv", "init", "--app", "--name", _name(root), "."], root, INIT_TIMEOUT_S)
    if init is None or init.returncode != 0:
        trace.event("scaffold_failed", step="uv init", error=_tail(init))
        return False
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    if "[tool.ruff" not in text:
        pyproject.write_text(text.rstrip("\n") + "\n" + RUFF_SECTION, encoding="utf-8")
    added = _run(["uv", "add", "--dev", "pytest", "ruff"], root, ADD_TIMEOUT_S)
    dev_tools = added is not None and added.returncode == 0
    files = sorted(p.name for p in root.iterdir() if p.name not in _HARNESS_ENTRIES)
    trace.event(
        "scaffolded", files=files, dev_tools=dev_tools, error="" if dev_tools else _tail(added)
    )
    trace.say(
        "new project: set up with uv (pyproject.toml, main.py, Ruff formatting"
        + (", pytest and ruff as dev tools)" if dev_tools else "; dev tools not added, offline?)")
    )
    return True


def _run(argv: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _tail(result: subprocess.CompletedProcess | None) -> str:
    if result is None:
        return "did not run (uv missing or timed out)"
    return (result.stderr or result.stdout or "").strip()[-300:]
