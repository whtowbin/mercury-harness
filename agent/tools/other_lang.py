"""Lint/test for a detected non-Python workspace, and a data-format check the
model can call directly mid-loop rather than only getting it synthesized
into a step's `verify`.

`lint_project`/`test_project` exist because `check_code`/`run_tests` are
Python-specific by construction (they parse Ruff's and pytest's own output
shapes) and stay that way — see agent/langtools.py's module docstring for
why rewriting them to be generic risked destabilising a heavily-used,
heavily-tested path for a case that, so far, has no real Node trace to
design against. These are the thin, separate, lazily-loaded tools that give
a detected non-Python workspace *something* instead of nothing, gated the
same way toolchain.py already resolves ruff/ty/pytest: try the real binary,
report the 127-missing-tool case plainly, never guess.
"""

from __future__ import annotations

import shlex

from .. import dataformats, langtools, toolchain
from .base import Tool, ToolContext, obj

PROJECT_TOOL_TIMEOUT_S = 120
VALIDATE_TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 6_000


def _missing_profile(ctx: ToolContext) -> str | None:
    profile = ctx.language_profile
    if profile is None:
        return (
            "error: no language profile detected for this workspace (no "
            "pyproject.toml, package.json, or scc-recognisable majority "
            "language). Nothing to lint or test against."
        )
    if profile is langtools.PYTHON:
        return "error: this is a Python workspace — use check_code/run_tests instead."
    return None


def _run_profile_command(ctx: ToolContext, template: str, files: list[str]) -> str:
    argv = shlex.split(template.format(files=" ".join(files)))
    result = toolchain.run_capturing(
        argv,
        cwd=ctx.workspace.root,
        timeout=PROJECT_TOOL_TIMEOUT_S,
        env=toolchain.workspace_env(ctx.workspace.root),
    )
    if result.returncode == 127:
        return f"error: `{argv[0]}` is not installed in this environment."
    output = result.output.strip()[:MAX_OUTPUT_CHARS] or "(no output)"
    return f"exit={result.returncode}\n{output}"


def lint_project(ctx: ToolContext, files: list[str]) -> str:
    """Lint files in a detected non-Python workspace (e.g. eslint for Node)."""
    problem = _missing_profile(ctx)
    if problem:
        return problem
    profile = ctx.language_profile
    if profile is None or profile.lint_cmd is None:
        return f"error: no lint command configured for {profile.name if profile else 'this'}"
    return _run_profile_command(ctx, profile.lint_cmd, files)


def test_project(ctx: ToolContext, files: list[str]) -> str:
    """Run tests in a detected non-Python workspace (e.g. `npm test`)."""
    problem = _missing_profile(ctx)
    if problem:
        return problem
    profile = ctx.language_profile
    assert profile is not None
    return _run_profile_command(ctx, profile.test_cmd, files)


def validate_data_file(ctx: ToolContext, files: list[str]) -> str:
    """Check that JSON/YAML/TOML/XML/CSV files parse, mid-loop rather than
    only at the end of a step. Wraps agent.dataformats.build_validator —
    see its docstring for exactly which formats get a real check and which
    (Markdown, plain text) fall outside its scope.

    Always run through a shell (`build_validator` can join several tools
    with `&&`, e.g. the jq/tomllib fallback for a mix of JSON and TOML
    files) rather than split into one argv — a shell also gives the same
    127-for-a-missing-binary convention every other tool here relies on, so
    no separate detection is needed.
    """
    command = dataformats.build_validator(files)
    if not command:
        return (
            "no validator available for these files (either none of them "
            "have a recognised structured-data extension, or nothing to "
            "check them with — yq, jq, or xmllint — is installed)"
        )
    result = toolchain.run_capturing(
        ["/bin/sh", "-c", command], cwd=ctx.workspace.root, timeout=VALIDATE_TIMEOUT_S
    )
    if result.returncode == 127:
        return f"error: {command.split()[0]} is not installed in this environment."
    if result.returncode == 0:
        return "valid — every file parsed cleanly"
    return f"invalid:\n{result.output.strip()[:MAX_OUTPUT_CHARS]}"


LANGTOOLS_TOOLS = [
    Tool(
        "lint_project",
        "Lint files in a workspace whose language isn't Python (Python has check_code "
        "instead). Only usable when a non-Python language was detected for this "
        "workspace.",
        obj(
            {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to lint.",
                }
            },
            ["files"],
        ),
        lint_project,
    ),
    Tool(
        "test_project",
        "Run tests in a workspace whose language isn't Python (Python has run_tests "
        "instead). Only usable when a non-Python language was detected for this "
        "workspace.",
        obj(
            {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Test files to run.",
                }
            },
            ["files"],
        ),
        test_project,
    ),
    Tool(
        "validate_data_file",
        "Check that JSON/YAML/TOML/XML/CSV files parse correctly. Use this right "
        "after writing a structured-data file, instead of waiting for the step's "
        "own verify command to catch a syntax mistake.",
        obj(
            {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to validate.",
                }
            },
            ["files"],
        ),
        validate_data_file,
    ),
]
