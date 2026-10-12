"""Short follow-up goals tied to the last run (clarify.resolve_follow_up, DESIGN-LOG §92.3).

The goals are invented, in the shape of real app-run follow-ups; the memory is a real
`.harness/memory.md` in a temporary workspace.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.interact import Interactor
from agent.llm import Reply, Usage
from agent.memory import MemoryStore
from agent.phases.clarify import gather_clarifications, is_follow_up
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace

LAST_RUN = """## [2026-09-30 02:10] confirm the newest changes appear in the game
keywords: check, edits
outcome: FAILED
- [fix-imports] pygame fails to build: SDL2 headers not found on Python 3.14
unresolved:
  - tests still failing: pytest exited 1
"""


@pytest.mark.parametrize(
    "goal",
    [
        "please sort out those errors?",
        "alright, run it again",
        "that failed, the labels still wrap past the edge",
        "The icons are still stacked on top of each other",
    ],
)
def test_follow_ups_are_detected(goal):
    assert is_follow_up(goal)


@pytest.mark.parametrize(
    "goal",
    [
        "add a shield pickup: the player gains a shield on contact and it absorbs one hit. "
        "Add tests for the shield pickup",
        "Make a basic top-down maze game",
        "add a --verbose flag",
    ],
)
def test_new_goals_are_not(goal):
    assert not is_follow_up(goal)


class Recording:
    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, **kwargs):
        self.calls += 1
        return Reply(content="Q: Which issues?", usage=Usage(10, 5, 0.0, 0.1))


def run(tmp_path, answer: str, flag: bool = True):
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "memory.md").write_text(LAST_RUN)
    asked: list[str] = []

    def ask(question, options=None):
        asked.append(question)
        return answer

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        interactor=Interactor(ask=ask, enabled=True),
    )
    ctx.memory = MemoryStore(tmp_path)
    settings = Settings(
        workspace=str(tmp_path), goal="please sort out those errors?", resolve_followups=flag
    )
    client = Recording()
    block = gather_clarifications(client, settings, ctx, ctx.trace)
    return block, client, asked, ctx


def test_a_confirmed_follow_up_carries_the_last_run_and_skips_the_clarify_call(tmp_path):
    block, client, asked, _ = run(tmp_path, "yes, continue from it")
    assert client.calls == 0
    assert "SDL2 headers not found" in block and "pytest exited 1" in block
    assert asked[0].startswith('Continue from the last run ("confirm the newest changes')


def test_declining_falls_back_to_clarify_with_the_question_given_back(tmp_path):
    block, client, asked, _ctx = run(tmp_path, "no, something else")
    assert client.calls == 1  # the normal clarify call runs
    assert "follows up the previous run" not in block
    assert asked[0].startswith("Continue from the last run")


def test_turned_off_nothing_changes(tmp_path):
    _, client, asked, _ = run(tmp_path, "yes", flag=False)
    assert client.calls == 1
    assert not any(q.startswith("Continue from the last run") for q in asked)
