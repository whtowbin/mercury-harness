"""Getting optional tools used: whole-word triggers and the nudges that widen a kit.

The measurement behind this (DESIGN-LOG §62): in 3,915 bench runs an optional
tool was called on 4 of the ~3,100 step kits that carried one, and most of
those kits carried it by accident ("move" inside "remove", "version" inside
"conversion"). Offering a tool does not get it used; a nudge at the moment
the by-hand version is costing something is what has moved this model before.
"""

from __future__ import annotations

import json

import pytest

from agent.kitnudges import (
    KitWatch,
    nudge_about_shell_duplicates,
)
from agent.tools import execution_registry, execution_tool_names


class _Trace:
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))


def _call(name, **args):
    return {
        "id": f"c{abs(hash(json.dumps(args, sort_keys=True))) % 10000}",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _edit(path, old, new):
    return _call("edit_file", path=path, old_string=old, new_string=new)


def _results(n, text="ok"):
    return [{"role": "tool", "content": text} for _ in range(n)]


# -- whole-word triggers ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "absent"),
    [
        ("remove the stale cache entry", "move_file"),
        ("fix the currency conversion", "pypi_info"),
        ("add a docstring to parse()", "web_search"),
        ("validate_input returns early", "git_commit"),
    ],
)
def test_a_trigger_inside_another_word_does_not_fire(text, absent):
    assert absent not in execution_tool_names(text)


@pytest.mark.parametrize(
    ("text", "present"),
    [
        ("move helpers into util.py", "move_file"),
        ("fix the broken imports and commit", "git_commit"),
        ("rename make_line and update its callers", "move_file"),
        ("bump the pinned versions", "pypi_info"),
        ("add the missing dependencies", "pypi_info"),
        ("fix the lint errors in app.ts", "lint_project"),
        ("write a config.json file", "validate_data_file"),
    ],
)
def test_a_trigger_still_fires_as_a_word_or_its_plural(text, present):
    assert present in execution_tool_names(text)


# -- the shell redirect nudge ---------------------------------------------


def test_pytest_through_run_bash_points_at_run_tests_once():
    registry = execution_registry("fix the bug")
    watch, trace = KitWatch(), _Trace()
    call = [_call("run_bash", command="uv run pytest -q test_x.py")]
    messages = _results(1, "1 failed")
    assert nudge_about_shell_duplicates(watch, call, messages, registry, trace, "execute:s") == 1
    assert "run_tests" in messages[0]["content"]
    again = _results(1, "1 failed")
    assert nudge_about_shell_duplicates(watch, call, again, registry, trace, "execute:s") == 0
    assert again[0]["content"] == "1 failed"


def test_ruff_through_run_bash_points_at_check_code():
    registry = execution_registry("fix the bug")
    call = [_call("run_bash", command="uvx --quiet ruff check throttle.py")]
    messages = _results(1)
    nudge_about_shell_duplicates(KitWatch(), call, messages, registry, _Trace(), "execute:s")
    assert "check_code" in messages[0]["content"]


def test_other_shell_commands_are_left_alone():
    registry = execution_registry("fix the bug")
    call = [_call("run_bash", command="ls -la && git status")]
    messages = _results(1)
    fired = nudge_about_shell_duplicates(
        KitWatch(), call, messages, registry, _Trace(), "execute:s"
    )
    assert fired == 0
    assert messages[0]["content"] == "ok"


def test_the_redirect_lands_on_the_run_bash_result_not_a_neighbour():
    registry = execution_registry("fix the bug")
    calls = [_call("read_file", path="a.py"), _call("run_bash", command="pytest -q")]
    messages = [
        {"role": "tool", "content": "file"},
        {"role": "tool", "content": "passed"},
    ]
    nudge_about_shell_duplicates(KitWatch(), calls, messages, registry, _Trace(), "execute:s")
    assert messages[0]["content"] == "file"
    assert "run_tests" in messages[1]["content"]


# -- the probe nudge now widens too -------------------------------------


def test_the_probe_nudge_adds_trace_python_to_an_execute_kit():
    """Gated on the kit, it fired once in 3,915 bench runs: the debug trigger
    is on 0.3% of steps."""
    from collections import Counter

    from agent.runner import RUN_PYTHON_PROBE_LIMIT, _nudge_about_repeated_probing

    registry = execution_registry("fix the bug")
    assert "trace_python" not in registry
    messages = _results(1)
    fired = _nudge_about_repeated_probing(
        messages,
        Counter({"expr": RUN_PYTHON_PROBE_LIMIT}),
        registry,
        _Trace(),
        "execute:s",
        budget=2,
    )
    assert fired == 1
    assert "trace_python" in registry


def test_a_sanctioned_signature_change_is_not_dishonest(tmp_path):
    """rename-callers asks for make_line to go; integrity must not count the
    change the goal requested as breaking the public interface."""
    import shutil as sh

    from bench import integrity, tasks

    task = tasks.get("rename-callers")
    sh.copytree(task.seed_dir, tmp_path, dirs_exist_ok=True)
    pricing = tmp_path / "pricing.py"
    pricing.write_text(
        pricing.read_text().replace(
            "def make_line(qty, price, name)", "def line_item(name, qty, price)"
        )
    )
    assert integrity.check(task, tmp_path).signature_changes == []
    cart = tmp_path / "cart.py"
    cart.write_text(
        cart.read_text().replace("def cart_total(items)", "def cart_total(items, tax=0)")
    )
    assert integrity.check(task, tmp_path).signature_changes == [
        "cart.py::cart_total(items -> items, tax)"
    ]


def test_a_trigger_ending_in_e_matches_its_ing_form_and_still_not_inside_another_word():
    from agent.tools import execution_tool_names

    assert "move_file" in execution_tool_names("moving the helpers into utils.py")
    assert "move_file" in execution_tool_names("move the helpers into utils.py")
    assert "move_file" not in execution_tool_names("removing the dead branch")
