"""Preloading the files that import a step's files (its tests and callers)."""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.phases.execute import _preload
from agent.phases.neighbours import imported_names, importers
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "pricing.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def test_imported_names_covers_every_import_form():
    source = (
        "import pricing\nimport pkg.cart as c\nfrom pkg import tax\n"
        "from .ledger import post\nfrom . import util\n"
    )
    names = imported_names(source)
    assert {"pricing", "cart", "pkg", "tax", "ledger", "post", "util"} <= names
    assert imported_names("def broken(:\n") == set()


def test_importers_lists_tests_first_and_excludes_the_targets(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "pricing.py").write_text("def price(): ...\n")
    (tmp_path / "app.py").write_text("from pkg.pricing import price\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_pricing.py").write_text("from pkg import pricing\n")
    (tmp_path / "unrelated.py").write_text("import os\n")
    assert importers(tmp_path, ["pkg/pricing.py"]) == ["tests/test_pricing.py", "app.py"]
    assert importers(tmp_path, ["README.md"]) == []


def test_preload_adds_importers_only_when_asked(ctx):
    (ctx.workspace.root / "test_pricing.py").write_text("from pricing import x\n")
    without = _preload(ctx, ["pricing.py"], 4_000)
    assert "test_pricing.py" not in without
    with_neighbours = _preload(ctx, ["pricing.py"], 4_000, neighbours=True)
    assert with_neighbours.startswith(without)
    assert "--- test_pricing.py (imports this step's code; context only" in with_neighbours


def test_neighbours_get_only_what_the_declared_files_leave(ctx):
    (ctx.workspace.root / "test_pricing.py").write_text("from pricing import x\n" * 200)
    tight = _preload(ctx, ["pricing.py"], 50, neighbours=True)
    assert "test_pricing.py" not in tight
