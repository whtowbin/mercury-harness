"""The symbol map: what is defined where, cheaply.

The planner declares which files each step touches, and a traced plan put four
steps on the same two files and then had them duplicate each other's work. The
tree it was given says which files exist and nothing about what is in them, so
"where does scoring live" could only be answered by reading. Measured on three
real workspaces the map costs 6-8% of the source it summarises.
"""

from __future__ import annotations

from pathlib import Path

from agent.outline import MAX_METHODS, outline_source, symbol_map

SOURCE = '''
"""A module."""
import os

SCREEN_WIDTH = 480
lowercase_ignored = 1


def helper(a, b=2, *rest, **kw):
    return a


class Game:
    """Docstring."""

    def __init__(self, prefs=None):
        self.prefs = prefs

    def update(self):
        pass

    async def tick(self):
        pass
'''


def test_functions_appear_with_their_signature():
    lines = outline_source(SOURCE)
    assert any("def helper(a, b, *rest, **kw)" in line for line in lines)


def test_classes_list_their_methods():
    lines = outline_source(SOURCE)
    game = next(line for line in lines if "class Game" in line)
    assert "__init__(self, prefs)" in game
    assert "update(self)" in game
    assert "tick(self)" in game, "async methods count too"


def test_module_constants_appear_and_lowercase_names_do_not():
    lines = outline_source(SOURCE)
    assert any(line.strip() == "SCREEN_WIDTH" for line in lines)
    assert not any("lowercase_ignored" in line for line in lines)


def test_bodies_are_not_included():
    """It answers "where does this live", not "how does it work" — that is the
    whole reason it is 6% of the source rather than 100%."""
    assert "self.prefs = prefs" not in "\n".join(outline_source(SOURCE))


def test_a_syntactically_broken_file_is_skipped_not_raised():
    assert outline_source("def oops(:\n") == []


def test_a_huge_class_is_elided_rather_than_dumped():
    body = "\n".join(f"    def m{i}(self): pass" for i in range(40))
    lines = outline_source(f"class Big:\n{body}\n")
    assert f"+{40 - MAX_METHODS} more" in lines[0]


# -- across a workspace ---------------------------------------------------


def test_the_map_names_each_file_and_its_symbols(tmp_path: Path):
    (tmp_path / "game.py").write_text("class Bird:\n    def tap(self): pass\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("def helper():\n    pass\n")
    text = symbol_map(tmp_path)
    assert "game.py" in text and "class Bird" in text
    assert "pkg/util.py" in text and "def helper()" in text


def test_ignored_directories_are_skipped(tmp_path: Path):
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "dep.py").write_text("def vendored(): pass\n")
    (tmp_path / "mine.py").write_text("def mine(): pass\n")
    text = symbol_map(tmp_path)
    assert "mine" in text
    assert "vendored" not in text


def test_a_workspace_with_no_python_yields_nothing(tmp_path: Path):
    (tmp_path / "README.md").write_text("# hi\n")
    assert symbol_map(tmp_path) == ""


def test_the_map_is_capped(tmp_path: Path):
    for i in range(50):
        (tmp_path / f"m{i}.py").write_text("def f():\n    pass\n" * 40)
    text = symbol_map(tmp_path, char_cap=500)
    assert len(text) < 900
    assert "truncated" in text


def test_the_map_is_a_small_fraction_of_the_source(tmp_path: Path):
    """The property that makes it worth sending every run."""
    for i in range(6):
        (tmp_path / f"m{i}.py").write_text(
            "\n".join(
                f"def f{j}(a, b):\n    x = a + b\n    y = x * 2\n    return y\n" for j in range(10)
            )
        )
    source = sum(len((tmp_path / f"m{i}.py").read_text()) for i in range(6))
    assert len(symbol_map(tmp_path)) < source * 0.30


def test_the_planner_prompt_carries_the_map(tmp_path: Path):
    """Wiring, not just the function."""
    import inspect

    from agent.phases import plan

    source = inspect.getsource(plan.make_plan)
    assert "outline.symbol_map" in source
    assert "symbols" in source
