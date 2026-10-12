"""Plan repairs the harness makes deterministically rather than asking for.

The planner prompt already forbids everything checked here. A traced run broke
two of those rules in a single step — one called "add pytest", verified with
`uv run pytest --version` — so the rules that can be enforced are enforced
instead of merely stated.
"""

from __future__ import annotations

from agent.phases.plan import is_tool_probe

# -- commands that pass however wrong the code is -------------------------


def test_the_command_from_the_traced_run_is_recognised():
    assert is_tool_probe("uv run pytest --version")


def test_version_and_help_probes_are_recognised():
    for command in ("pytest --version", "ruff -V", "uv run ty --help", "python -m pytest -h"):
        assert is_tool_probe(command), command


def test_presence_checks_are_recognised():
    for command in (
        "which pytest",
        "command -v uv",
        "uv show pytest",
        "pip show pygame",
        "uv pip list",
    ):
        assert is_tool_probe(command), command


# -- commands that actually test the work: none of these may be rejected --


def test_a_real_test_run_is_not_a_probe():
    for command in (
        "uv run pytest -q",
        "uv run pytest -q tests/test_game.py",
        "uv run pytest -q -k scoring",
        "pytest --no-header -q",
    ):
        assert not is_tool_probe(command), command


def test_a_real_lint_or_type_check_is_not_a_probe():
    for command in (
        "uv run ruff check game.py",
        "uv run ty check .",
        "uv run ruff format --check .",
    ):
        assert not is_tool_probe(command), command


def test_a_command_that_merely_mentions_a_version_is_not_a_probe():
    """`-V` must match as a flag, not inside a word or a path."""
    assert not is_tool_probe("uv run python scripts/check_VERSION_bump.py")
    assert not is_tool_probe("uv run pytest -q tests/test_version.py")


def test_an_empty_verify_is_not_a_probe():
    assert not is_tool_probe("")
