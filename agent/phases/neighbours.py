"""The files that import a step's files: its tests and its callers.

A step's declared files are preloaded because every file that arrives in the
prompt is a round trip saved (DESIGN-LOG §7). But the files a change has to
stay compatible with are usually *other* files: the test that pins the
behaviour, the caller that depends on the signature. §44 found plans
committing to a fix after a single read; this is the deterministic half of
closing that gap, with no model call and no retrieval index.

Pure `ast` over the workspace's Python files. A name match is by module
stem, so two modules both called `utils` can over-select; for preloaded
context that costs a few hundred tokens, not correctness.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ..outline import _python_files

# Bounds on the scan, not on what is preloaded: a repository big enough to
# hit these is one where the executor should be grepping anyway.
MAX_SCANNED_FILES = 500
MAX_SCANNED_BYTES = 200_000
_TEST_PATH = re.compile(r"(^|/)(test_[^/]*|[^/]*_test|conftest)\.py$")


def module_stem(rel: str) -> str:
    """The name a Python file is imported by, last component only."""
    path = Path(rel)
    if path.suffix != ".py":
        return ""
    return path.parent.name if path.name == "__init__.py" else path.stem


def imported_names(source: str) -> set[str]:
    """Every module name a source file's imports could refer to, by last
    component. `from pkg import pricing` yields both `pkg` and `pricing`,
    because `pricing` may be a module rather than a name inside `pkg`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
            names.update(alias.name for alias in node.names)
    return names


def importers(root: Path, targets: list[str]) -> list[str]:
    """Workspace-relative paths of Python files importing any of `targets`,
    tests first, each group in path order. Targets themselves are excluded."""
    stems = {module_stem(t) for t in targets} - {""}
    if not stems:
        return []
    exclude = {Path(t).as_posix() for t in targets}
    tests: list[str] = []
    callers: list[str] = []
    for path in _python_files(root)[:MAX_SCANNED_FILES]:
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        try:
            if path.stat().st_size > MAX_SCANNED_BYTES:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if imported_names(source) & stems:
            (tests if _TEST_PATH.search(rel) else callers).append(rel)
    return tests + callers
