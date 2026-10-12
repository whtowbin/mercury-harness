"""Which language a workspace is, and what its lint/test/repair commands are.

Before this existed, "the toolchain" meant `ruff`/`ty`/`pytest`/`uv sync`
hardcoded directly into plan.py and envfail.py -- correct for the harness's
own Python-only history, but it meant a Node or Go workspace got none of
check_code's cheap-verify benefit and none of envfail's environmental-failure
protection, not because nobody wanted that but because there was nowhere to
put a second language without duplicating both files wholesale.

One `LanguageProfile` per language instead: a marker file that identifies a
workspace as owning that language's project, its lint/test command
templates, and what "repair the environment" means for it. Python is the
first row, unchanged in behaviour from before this module existed -- proving
the shape doesn't cost anything, not just that it could theoretically hold
more than one language. Node is the second, added deliberately thin: its own
dependency-drip signature parity (the `envfail._declared_dependencies`
equivalent) and its own crash-signature taxonomy get built from real traced
Node failures later, the same way Python's were built from real traces and
not speculation -- inventing them now, with no Node trace to check them
against, would be exactly the kind of unmeasured guess this project's own
conventions argue against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import toolchain

SCC_TIMEOUT_S = 10


@dataclass(frozen=True)
class LanguageProfile:
    name: str
    marker: str  # file at the workspace's own root
    test_suffixes: tuple[str, ...]  # how this language names its test files
    lint_cmd: str | None  # "{files}" placeholder, or None
    test_cmd: str  # "{files}" placeholder
    repair_argv: tuple[str, ...]  # run when the environment looks broken
    # but the project file itself is intact


PYTHON = LanguageProfile(
    name="python",
    marker="pyproject.toml",
    test_suffixes=("test_", "_test.py"),
    lint_cmd="uv run ruff check {files}",
    test_cmd="uv run pytest -q {files}",
    repair_argv=("uv", "sync"),
)
NODE = LanguageProfile(
    name="node",
    marker="package.json",
    test_suffixes=(".test.js", ".test.ts", ".spec.js", ".spec.ts"),
    lint_cmd="npx --no-install eslint {files}",
    test_cmd="npm test -- {files}",
    repair_argv=("npm", "install"),
)
GO = LanguageProfile(
    name="go",
    marker="go.mod",
    test_suffixes=("_test.go",),
    # go vet ships with the toolchain itself, unlike eslint/clippy -- no
    # "is it even installed" question for a project that has `go` at all.
    lint_cmd="go vet {files}",
    test_cmd="go test {files}",
    repair_argv=("go", "mod", "download"),
)
RUST = LanguageProfile(
    name="rust",
    marker="Cargo.toml",
    # Rust has no file-naming convention for tests the way Python/JS/Go do --
    # they live inline behind #[cfg(test)] or under a shared tests/
    # directory, neither of which a filename alone can tell apart from
    # ordinary source. Left empty rather than guessed: `is_test_file` simply
    # never matches for Rust, same detect-don't-assume convention as
    # everywhere else here, not a bug to fix later.
    test_suffixes=(),
    lint_cmd="cargo clippy --message-format=short {files}",
    test_cmd="cargo test {files}",
    repair_argv=("cargo", "fetch"),
)
LANGUAGE_PROFILES: tuple[LanguageProfile, ...] = (PYTHON, NODE, GO, RUST)

_SCC_NAME_TO_PROFILE = {
    "Python": PYTHON,
    "JavaScript": NODE,
    "TypeScript": NODE,
    "Go": GO,
    "Rust": RUST,
}


def is_test_file(profile: LanguageProfile, rel_path: str) -> bool:
    """Whether `rel_path`'s own name matches this profile's test convention."""
    name = rel_path.rsplit("/", 1)[-1]
    return any(name.startswith(s) or name.endswith(s) for s in profile.test_suffixes)


def detect_profile(root: Path) -> LanguageProfile | None:
    """The one profile whose marker file exists at this workspace's own root.

    Root only, never an ancestor: an upward scan is exactly the escape route
    `Workspace.enclosing_project()` already exists to refuse for Python (see
    agent/workspace.py), and a language profile must not reopen that hole for
    Node by scanning up for a `package.json` some other project owns.

    Falls back to `scc`'s language breakdown when no marker file matches -- a
    workspace of, say, JavaScript source with no package.json yet is
    invisible to marker detection but obvious from what's actually in it.
    Skipped silently, same detect-don't-assume convention as everywhere else
    in this codebase, when `scc` isn't on PATH: marker detection (or None) is
    exactly what happens without this module.
    """
    for profile in LANGUAGE_PROFILES:
        if (root / profile.marker).is_file():
            return profile
    return _detect_by_scc(root)


def _detect_by_scc(root: Path) -> LanguageProfile | None:
    result = toolchain.run_capturing(["scc", "--format", "json", str(root)], timeout=SCC_TIMEOUT_S)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        breakdown = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    known = [
        row
        for row in breakdown
        if isinstance(row, dict) and row.get("Name") in _SCC_NAME_TO_PROFILE
    ]
    if not known:
        return None
    dominant = max(known, key=lambda row: row.get("Count", 0))
    return _SCC_NAME_TO_PROFILE[dominant["Name"]]
