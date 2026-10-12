"""New projects are set up with uv and Ruff formatting; the files a run touched are formatted
when a step finishes, in projects that opt in (DESIGN-LOG §104). Real `uv init` and real
`ruff format`; only `uv add`, which reaches the package index, is stubbed."""

from __future__ import annotations

import json
import subprocess

from agent import scaffold
from agent.approval import Approver
from agent.autoformat import format_touched
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


def _events(trace: Trace) -> list[dict]:
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def _offline_add(monkeypatch):
    real = scaffold._run

    def run(argv, cwd, timeout):
        if argv[:2] == ["uv", "add"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return real(argv, cwd, timeout)

    monkeypatch.setattr(scaffold, "_run", run)


def test_an_empty_workspace_becomes_a_uv_project_with_ruff_on(tmp_path, monkeypatch):
    _offline_add(monkeypatch)
    ws = tmp_path / "My Game!"
    ws.mkdir()
    (ws / ".harness").mkdir()
    assert scaffold.is_new_project(ws)
    trace = Trace(str(tmp_path / "t"), echo=False)
    assert scaffold.scaffold(ws, trace)
    pyproject = (ws / "pyproject.toml").read_text()
    assert 'name = "my-game"' in pyproject and "[tool.ruff]" in pyproject
    assert (ws / "main.py").is_file()
    (done,) = [e for e in _events(trace) if e["kind"] == "scaffolded"]
    assert "pyproject.toml" in done["files"] and done["dev_tools"] is True


def test_an_existing_project_is_not_new(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    assert not scaffold.is_new_project(tmp_path)


def test_a_goal_in_another_language_is_left_alone():
    assert scaffold.wants_python("make a flappy bird clone")
    assert not scaffold.wants_python("build a React todo app in TypeScript")
    assert not scaffold.wants_python("write a Rust CLI")


def _ctx(root) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(root / "t"), echo=False),
    )


UGLY = "def f( a,b ):\n    return {'x':a,'y':b}\n"


def test_touched_files_are_formatted_in_a_project_that_opts_in(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "p"\n\n[tool.ruff]\n')
    (tmp_path / "game.py").write_text(UGLY)
    (tmp_path / "other.py").write_text(UGLY)
    ctx = _ctx(tmp_path)
    ctx.touched.append("game.py")
    changed = format_touched(ctx, ctx.trace, "execute:s")
    assert changed == ["game.py"]
    assert (tmp_path / "game.py").read_text() == 'def f(a, b):\n    return {"x": a, "y": b}\n'
    assert (tmp_path / "other.py").read_text() == UGLY  # not touched by the run


def test_a_project_without_a_formatter_keeps_its_style(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "p"\n')
    (tmp_path / "game.py").write_text(UGLY)
    ctx = _ctx(tmp_path)
    ctx.touched.append("game.py")
    assert format_touched(ctx, ctx.trace, "execute:s") == []
    assert (tmp_path / "game.py").read_text() == UGLY
