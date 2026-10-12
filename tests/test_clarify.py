"""Phase 0: asking about an underspecified goal before planning it."""

from __future__ import annotations

import dataclasses

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.interact import Interactor
from agent.llm import Reply, Usage
from agent.phases.clarify import _parse, gather_clarifications
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


class FakeClient:
    """Returns canned replies and records what it was asked."""

    def __init__(self, content: str = "CLEAR", raises: Exception | None = None):
        self.content = content
        self.raises = raises
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return Reply(content=self.content, usage=Usage())


@pytest.fixture
def ctx(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def settings(**overrides) -> Settings:
    return dataclasses.replace(
        Settings(workspace=".", goal="make it faster", clarify=True, ask_enabled=True), **overrides
    )


def answering(ctx, answers: list[str]) -> list[str]:
    """Point the context's interactor at canned answers; return questions asked."""
    asked: list[str] = []
    replies = iter(answers)

    def ask(question: str, options=None) -> str:
        asked.append(question)
        return next(replies, "")

    ctx.interactor = Interactor(ask=ask, enabled=True)
    return asked


# -- parsing the model's reply -------------------------------------------


def test_clear_means_no_questions():
    assert _parse("CLEAR") == []


def test_clear_with_trailing_noise_still_means_clear():
    assert _parse("CLEAR\n") == []


def test_questions_are_pulled_off_their_prefix():
    # (question, options) now: a question may carry answer buttons.
    assert _parse("Q: One?\nQ: Two?") == [("One?", []), ("Two?", [])]


def test_at_most_five_questions_are_taken():
    """Clarify's own budget, separate from the run's later questions (DESIGN-LOG §101)."""
    assert len(_parse("\n".join(f"Q: question {i}?" for i in range(9)))) == 5


def test_prose_around_the_questions_is_ignored():
    assert _parse("Here are my questions:\nQ: Only this one?\nThanks") == [("Only this one?", [])]


def test_an_empty_reply_asks_nothing():
    assert _parse("   ") == []


# -- the phase itself -----------------------------------------------------


def test_a_clear_goal_asks_nothing_and_adds_no_context(ctx):
    asked = answering(ctx, [])
    result = gather_clarifications(FakeClient("CLEAR"), settings(), ctx, ctx.trace)
    assert result == ""
    assert asked == []


def test_questions_are_asked_and_their_answers_reach_the_planner(ctx):
    asked = answering(ctx, ["Redis", "JSON"])
    client = FakeClient("Q: Which cache?\nQ: Which format?")
    result = gather_clarifications(client, settings(), ctx, ctx.trace)
    assert asked == ["Which cache?", "Which format?"]
    assert "Which cache?" in result
    assert "Redis" in result
    assert "JSON" in result


def test_clarify_disabled_makes_no_call_at_all(ctx):
    client = FakeClient("Q: Anything?")
    assert gather_clarifications(client, settings(clarify=False), ctx, ctx.trace) == ""
    assert client.calls == []


def test_ask_disabled_makes_no_call_either(ctx):
    """There is no point paying for a question nobody can answer."""
    client = FakeClient("Q: Anything?")
    assert gather_clarifications(client, settings(ask_enabled=False), ctx, ctx.trace) == ""
    assert client.calls == []


def test_an_unanswered_question_is_dropped_rather_than_forwarded(ctx):
    """The interactor's no-answer fallback is prose aimed at the model. Passing
    it on as though a human had said it is worse than passing nothing."""
    answering(ctx, [])
    ctx.interactor = Interactor(ask=None, enabled=False)
    result = gather_clarifications(FakeClient("Q: Which cache?"), settings(), ctx, ctx.trace)
    assert result == ""


def test_a_failed_call_degrades_to_planning_as_before(ctx):
    answering(ctx, [])
    result = gather_clarifications(
        FakeClient(raises=RuntimeError("network")), settings(), ctx, ctx.trace
    )
    assert result == ""


def test_the_call_runs_at_low_reasoning_not_the_planner_default(ctx):
    """planner_reasoning defaults to "high"; paying that for a triage question
    on every interactive run is the most expensive way to buy the least."""
    answering(ctx, [])
    client = FakeClient("CLEAR")
    gather_clarifications(client, settings(planner_reasoning="high"), ctx, ctx.trace)
    assert client.calls[0]["reasoning_effort"] == "low"


def test_the_workspace_map_is_sent_so_it_cannot_ask_about_visible_files(ctx):
    (ctx.workspace.root / "cache.py").write_text("x = 1\n")
    answering(ctx, [])
    client = FakeClient("CLEAR")
    gather_clarifications(client, settings(), ctx, ctx.trace)
    sent = client.calls[0]["messages"][1]["content"]
    assert "cache.py" in sent
