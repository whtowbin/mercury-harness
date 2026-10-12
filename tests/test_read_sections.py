"""read_file reads several parts of a file in one call, widens a narrow window to the function
around it, and answers an identical re-read of unchanged text in one line (DESIGN-LOG §104)."""

from __future__ import annotations

from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.fs import read_file
from agent.trace import Trace
from agent.workspace import Workspace

GAME = """import pygame


class Game:
    def __init__(self):
        self.camera_x = 0
        self.score = 0

    def update(self):
        self.camera_x += 1
        self.score += 1
        if self.score > 10:
            self.score = 0
        return self.score

    def draw(self, screen):
        screen.fill((0, 0, 0))


def main():
    Game().update()
"""


def _ctx(tmp_path) -> ToolContext:
    (tmp_path / "main.py").write_text(GAME)
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def test_sections_by_name_and_range_come_back_together(tmp_path):
    ctx = _ctx(tmp_path)
    out = read_file(ctx, "main.py", sections="Game.update, draw, 1-2")
    assert "--- Game.update: lines 9-14 ---" in out
    assert "9\t    def update(self):" in out and "14\t        return self.score" in out
    assert "--- draw: lines 16-17 ---" in out  # a unique short name resolves too
    assert "--- 1-2: lines 1-2 ---" in out and "1\timport pygame" in out
    assert "camera_x = 0" not in out  # __init__ was not asked for


def test_an_unknown_name_lists_the_real_ones(tmp_path):
    out = read_file(_ctx(tmp_path), "main.py", sections="Game.render")
    assert "no function or class named 'Game.render'" in out and "Game.update" in out


def test_a_narrow_window_widens_to_the_function_around_it(tmp_path):
    out = read_file(_ctx(tmp_path), "main.py", offset=11, limit=2)
    assert out.startswith("(widened to lines 9-14, the whole `Game.update`)")
    assert "9\t    def update(self):" in out and "14\t" in out


def test_a_whole_file_read_is_not_widened(tmp_path):
    assert not read_file(_ctx(tmp_path), "main.py").startswith("(widened")


def test_an_identical_reread_of_unchanged_text_is_one_line(tmp_path):
    ctx = _ctx(tmp_path)
    first = read_file(ctx, "main.py")
    again = read_file(ctx, "main.py")
    assert "1\timport pygame" in first
    assert again.startswith("(lines 1-21 of main.py are unchanged since you read them")


def test_a_changed_file_or_a_new_loop_reads_in_full_again(tmp_path):
    ctx = _ctx(tmp_path)
    read_file(ctx, "main.py")
    (tmp_path / "main.py").write_text(GAME + "\n# changed\n")
    ctx.cache.invalidate()
    assert "1\timport pygame" in read_file(ctx, "main.py")
    ctx.read_epoch += 1  # what run_tool_loop does at a new loop or a compaction
    assert "1\timport pygame" in read_file(ctx, "main.py")


def test_the_tool_loop_starts_each_conversation_fresh(tmp_path):
    import json

    from agent.llm import Reply, Usage
    from agent.runner import run_tool_loop
    from agent.tools import execution_registry

    read = {
        "id": "r",
        "type": "function",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "main.py"})},
    }
    done = {
        "id": "f",
        "type": "function",
        "function": {"name": "finish_step", "arguments": json.dumps({"summary": "ok"})},
    }

    class Scripted:
        def __init__(self, turns):
            self.turns, self.sent = list(turns), []

        def chat(self, model, messages, **kwargs):
            self.sent.append([dict(m) for m in messages])
            return Reply(content="", tool_calls=self.turns.pop(0), usage=Usage(10, 5, 0.0, 0.1))

    ctx = _ctx(tmp_path)
    results = []
    for turns in ([[read], [read], [done]], [[read], [done]]):
        client = Scripted(turns)
        ctx.done = False
        run_tool_loop(
            client=client,
            model="m",
            phase="execute:s",
            system="s",
            user="u",
            registry=execution_registry(),
            ctx=ctx,
            trace=ctx.trace,
            max_iters=len(turns),
        )
        results.append([m["content"] for m in client.sent[-1] if m.get("role") == "tool"])
    first_loop, second_loop = results
    assert "1\timport pygame" in first_loop[0] and first_loop[1].startswith("(lines 1-21")
    assert "1\timport pygame" in second_loop[0]
