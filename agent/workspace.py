"""Path jail. Every file tool resolves through here."""

from __future__ import annotations

import os
from pathlib import Path

# Directories no traversal in the harness should ever descend into: build
# artefacts, caches, vendored dependencies, and the harness's own trace
# output. This was previously spelled out at five call sites (the workspace
# tree, workspace_size_mb, list_dir, grep and glob_files) which had already
# drifted apart — glob_files was still scanning .pytest_cache and
# .mypy_cache, and workspace_size_mb was counting traces/ against the
# workspace size cap. One definition means one behaviour.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
        "traces",
    }
)


def is_ignored(path: Path) -> bool:
    """True when any component of `path` is a directory we never descend into."""
    return any(part in IGNORED_DIRS for part in path.parts)


def exclude_globs(flag: str) -> list[str]:
    """`IGNORED_DIRS` as external-tool exclude-glob arguments, one per
    directory — e.g. `exclude_globs("--glob")` for ripgrep, or
    `exclude_globs("--globs")` for ast-grep. Both tools understand the same
    gitignore-style `!name/**` exclude syntax; only the flag's name differs
    between them, which is the whole reason this takes one as a parameter
    instead of being two near-identical list comprehensions living in two
    different tool modules (which is what it used to be, before a third
    call site made the duplication worth naming).
    """
    return [f"{flag}=!{name}/**" for name in IGNORED_DIRS]


def walk_files(root: Path, skip_hidden: bool = False):
    """Yield every file under `root`, pruning IGNORED_DIRS (and, if asked,
    any other dot-directory) *before* descending into them.

    `root.rglob("*")` followed by `is_ignored()` -- the shape every call site
    below used before this existed -- does the opposite: it opens, reads and
    stats every file inside a `.venv`/`node_modules` first, and only then
    throws the result away. Measured against a workspace with a real 2,000-
    file `.venv` present (the investigation's own stated common case, DESIGN-LOG §20,
    confirmed rather than assumed): `workspace_manifest()` alone dropped
    from 13.1ms to 0.1ms, a 130x difference, once it used this instead.
    `os.walk`'s `dirnames[:] = ...` idiom is what makes the difference --
    removing a name from `dirnames` in place stops `os.walk` from ever
    opening that directory at all, rather than filtering what it already
    opened. `Workspace.tree()` and `list_dir()` already did this; everything
    else that walked the whole tree did not.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in IGNORED_DIRS and not (skip_hidden and d.startswith("."))
        ]
        for name in filenames:
            yield Path(dirpath) / name


class WorkspaceError(Exception):
    pass


class NestedWorkspaceError(WorkspaceError):
    """The workspace sits inside another Python project.

    This matters because build tools walk *upward*. `uv pip install` with no
    virtualenv in the workspace resolves to the enclosing project's .venv and
    installs there — the path jail does not constrain subprocesses, so an agent
    running ordinary commands can modify the project that is hosting it.
    """


class Workspace:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")

    def enclosing_project(self) -> Path | None:
        """The nearest ancestor above the workspace that owns a Python project."""
        for parent in self.root.parents:
            if (parent / "pyproject.toml").is_file() or (parent / ".venv").is_dir():
                return parent
        return None

    def venv(self) -> Path:
        return self.root / ".venv"

    def resolve(self, rel: str) -> Path:
        """Resolve a user/model supplied path, refusing anything outside root."""
        candidate = (self.root / rel).expanduser()
        # resolve() follows symlinks, so a symlink escape is caught too.
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"path escapes workspace: {rel}")
        return resolved

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def tree(self, max_entries: int = 400) -> str:
        """A compact repo map for orienting the model at low token cost."""
        lines: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")
            )
            rel_dir = os.path.relpath(dirpath, self.root)
            depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
            if rel_dir != ".":
                lines.append("  " * (depth - 1) + f"{os.path.basename(dirpath)}/")
            for name in sorted(filenames):
                if name.startswith(".") or name.endswith((".pyc", ".lock")):
                    continue
                lines.append("  " * depth + name)
            if len(lines) > max_entries:
                lines.append(f"... truncated at {max_entries} entries")
                break
        return "\n".join(lines) or "(empty workspace)"
