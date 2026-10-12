"""Say so when a failing run overlapped a change to the shared virtualenv.

Several Claude sessions work in this repo at once, and they share
`.venv`. The tests that start pytest in a subprocess (the audit in
agent/testcheck.py, run_tests, the finalcheck gate) run the `pytest` on PATH,
which is this `.venv`'s. When another session runs `uv sync` or `uv add`,
packages are removed and rewritten under a running suite: reinstalling pytest
in a loop while the suite ran broke 1 run in 8 with
`ModuleNotFoundError: No module named '_pytest._py.error'` (DESIGN-LOG §66.2).
Two intermittent fast-suite failures could not be reproduced by CPU load in 46
loaded runs; this makes the next one say whether the environment moved.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

import pytest

_STARTED_AT: float | None = None


def site_packages_mtime() -> float | None:
    """Newest modification time of this interpreter's site-packages directories.

    Installing or removing a package adds or renames an entry in the
    directory, which moves the directory's own mtime; one stat per directory,
    no walk.
    """
    times = []
    for path in site.getsitepackages() if hasattr(site, "getsitepackages") else []:
        try:
            times.append(Path(path).stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def venv_changed_since(started: float | None, now: float | None) -> bool:
    return started is not None and now is not None and now > started


def pytest_sessionstart(session: pytest.Session) -> None:
    global _STARTED_AT
    _STARTED_AT = site_packages_mtime()


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if exitstatus == 0:
        return
    if venv_changed_since(_STARTED_AT, site_packages_mtime()):
        terminalreporter.write_line(
            f"WARNING: {sys.prefix}'s installed packages changed during this run "
            "(another session's `uv sync`/`uv add`?). Failures that import from the "
            "environment may be environmental; rerun before debugging them.",
            yellow=True,
            bold=True,
        )
