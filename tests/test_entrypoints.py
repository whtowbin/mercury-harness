"""The console scripts, and the ways they can break silently.

`harness-tui` is a thin argv rewrite onto the existing `run` command, so what
needs pinning is not its behaviour but the two things that broke while it was
being written: reaching into Typer's internals for the subcommand, and
importing a package Typer vendors rather than installs.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_declared_entry_points_all_resolve():
    """A console script naming a function that does not exist installs fine
    and fails only when a user runs it."""
    import importlib

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert "harness-tui" in scripts
    for name, target in scripts.items():
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr, None)), f"{name} -> {target} is not callable"


def test_the_cli_imports_nothing_typer_merely_vendors():
    """Typer 0.27 vendors click as typer._click and does not depend on it, so
    `import click` works in this repo's dev environment and raises
    ModuleNotFoundError inside the installed tool. The narrowing in tui_main
    must use typer.core.TyperGroup, never click.Group."""
    import ast

    tree = ast.parse((ROOT / "agent" / "cli.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "click" not in imported, "click is vendored by Typer, not installed"


def test_tui_main_reaches_the_run_command():
    import typer.core
    import typer.main

    from agent.cli import app

    group = typer.main.get_command(app)
    assert isinstance(group, typer.core.TyperGroup)
    assert "run" in group.commands, "tui_main dispatches to the 'run' command by name"


def test_harness_tui_help_runs_in_a_clean_subprocess():
    """End to end: the entry point actually executes, and its help names
    itself rather than the subcommand it delegates to."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv=['harness-tui','--help'];"
            "from agent.cli import tui_main; tui_main()",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "COLUMNS": "200", "HOME": str(Path.home())},
    )
    assert result.returncode == 0, result.stderr
    assert "harness-tui [OPTIONS]" in result.stdout
    assert "harness-tui run" not in result.stdout


def test_the_tui_flag_is_added_when_absent():
    source = (ROOT / "agent" / "cli.py").read_text()
    assert '"--tui" not in argv' in source, "tui_main must force the dashboard on"


def test_no_test_file_defines_the_same_test_name_twice():
    """A redefined test silently replaces the first one, which then never runs
    and never reports anything — the suite still goes green with a hole in it.

    Found for real: `test_the_budget_stops_it_repeating` and
    `test_nothing_is_appended_when_the_last_message_is_not_a_tool_result` each
    existed twice in test_batching.py, once for the batching nudge and once
    for the no-progress nudge, so both batching cases had been silently
    dropped since the no-progress tests were added.
    """
    import ast
    from collections import Counter
    from pathlib import Path

    offenders = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scopes = [("", tree.body)] + [
            (f"{node.name}.", node.body) for node in tree.body if isinstance(node, ast.ClassDef)
        ]
        for prefix, body in scopes:
            names = [
                n.name
                for n in body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                and n.name.startswith("test_")
            ]
            offenders += [
                f"{path.name}::{prefix}{name}"
                for name, count in Counter(names).items()
                if count > 1
            ]
    assert not offenders, f"duplicate test names shadow the earlier one: {offenders}"
