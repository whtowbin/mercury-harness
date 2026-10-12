"""try_quick_answer: the cheap pre-planning classifier, and the bug in a
traced run where it decided blind.

A run's goal was literally "yes" -- accepting the previous run's suggested
next step, "Do you want to restore 'P_kbar' to the feature list?" -- and
try_quick_answer, seeing nothing but the bare word, answered "How can I help
you with this workspace?" The run recorded a session summary of "Answered
'yes' directly without planning" and changed nothing; the suggestion it was
accepting was silently discarded. `_plan_until_accepted` already recalls
memory before planning, so a goal that made it that far would have worked
correctly -- the gap was specifically that try_quick_answer decided whether
to even reach planning without ever seeing that context.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent.config import Settings
from agent.llm import Reply, Usage
from agent.loop import _reads_as_a_hedge, reads_as_a_question, try_quick_answer
from agent.trace import Trace


class FakeClient:
    """Returns canned replies and records what it was asked."""

    def __init__(self, content: str = "NEEDS_PLAN", finish_reason: str = "stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return Reply(content=self.content, usage=Usage(), finish_reason=self.finish_reason)


def settings_for(**overrides):
    base = Settings(workspace=".", goal="yes")
    return dataclasses.replace(base, **overrides)


@pytest.fixture
def trace(tmp_path):
    return Trace(str(tmp_path / "t"), echo=False)


def _user_message(client: FakeClient) -> str:
    return client.calls[0]["messages"][1]["content"]


# -- the question sent to the model includes what was passed in ------------


def test_the_bare_goal_alone_is_the_question_by_default(trace):
    client = FakeClient()
    try_quick_answer(client, settings_for(goal="what does this function do?"), trace)
    assert _user_message(client) == "what does this function do?"


def test_memory_context_is_folded_into_the_question(trace):
    client = FakeClient()
    try_quick_answer(
        client,
        settings_for(goal="yes"),
        trace,
        memory_context="next: Do you want to restore 'P_kbar'?",
    )
    message = _user_message(client)
    assert "yes" in message
    assert "restore 'P_kbar'" in message
    assert "Recalled context" in message


def test_workspace_map_and_memory_context_can_both_be_present(trace):
    client = FakeClient()
    try_quick_answer(
        client,
        settings_for(goal="yes"),
        trace,
        workspace_map="a.py\nb.py",
        memory_context="next: do the thing",
    )
    message = _user_message(client)
    assert "a.py" in message
    assert "do the thing" in message


def test_no_memory_context_leaves_the_question_unchanged_from_before(trace):
    """The traced bug's exact shape: goal="yes", nothing else -- this must
    not regress to always appending an empty "Recalled context:" section."""
    client = FakeClient()
    try_quick_answer(client, settings_for(goal="yes"), trace)
    assert _user_message(client) == "yes"
    assert "Recalled context" not in _user_message(client)


# -- the actual traced failure, replayed --------------------------------


def test_a_bare_yes_with_no_context_gets_answered_not_planned(trace):
    """This is the traced bug itself, reproduced: without memory_context,
    the model (here standing in for the traced one) has no way to resolve
    "yes" against anything, so a real model reasonably treats it as an
    unclear question rather than NEEDS_PLAN. Fixing this is what
    memory_context is for -- see the next test."""
    client = FakeClient(content="How can I help you with this workspace?")
    answer = try_quick_answer(client, settings_for(goal="yes"), trace)
    assert answer == "How can I help you with this workspace?"


def test_needs_plan_is_still_recognised_when_memory_context_is_present(trace):
    """memory_context changes what the model sees, not how its answer is
    parsed: NEEDS_PLAN must still fall through to the full pipeline."""
    client = FakeClient(content="NEEDS_PLAN")
    answer = try_quick_answer(
        client, settings_for(goal="yes"), trace, memory_context="next: restore P_kbar"
    )
    assert answer == ""


# -- a hedged "quick answer" falls through to planning, not to the person ---
# (two real runs of the platformer app asked "why do fixes keep failing?" and got
# exactly this: a paragraph of hypotheses ending "Should I proceed?", which
# the caller then printed as if it were a finished answer and ended the run
# having read nothing and changed nothing -- see DESIGN-LOG for the traces.)


def test_a_hedged_answer_falls_through_to_planning(trace):
    client = FakeClient(
        content=(
            "Based on the logs, there are a few structural problems. "
            "Do you want me to proceed with that analysis?"
        )
    )
    answer = try_quick_answer(client, settings_for(goal="why do the fixes keep failing?"), trace)
    assert answer == ""


def test_hedged_answer_falling_through_is_traced(trace):
    client = FakeClient(content="I can help with that. Should I proceed?")
    try_quick_answer(client, settings_for(goal="what is wrong in the traces?"), trace)
    with open(trace.path) as f:
        events = [json.loads(line) for line in f]
    assert any(e.get("kind") == "quick_answer_hedged" for e in events)


def test_a_genuine_answer_that_is_not_a_hedge_still_returns(trace):
    client = FakeClient(content="This project uses pygame for rendering and pytest for tests.")
    answer = try_quick_answer(client, settings_for(goal="what does this project use?"), trace)
    assert answer == "This project uses pygame for rendering and pytest for tests."


@pytest.mark.parametrize(
    "text",
    [
        "Here's what I found. Should I proceed with that analysis?",
        "I can read these files to check. Do you want me to proceed?",
        "Would you like me to go ahead and look into this?",
        "Let me know if you want me to dig further.",
    ],
)
def test_reads_as_a_hedge_catches_the_observed_phrasings(text):
    assert _reads_as_a_hedge(text)


@pytest.mark.parametrize(
    "text",
    [
        "The project uses pygame and pytest.",
        "main.py, player.py, and test_game.py.",
        "Is this a rhetorical question about design, or one you want answered?",
    ],
)
def test_reads_as_a_hedge_leaves_real_answers_alone(text):
    assert not _reads_as_a_hedge(text)


# -- failure handling is unaffected by the new parameter --------------------


def test_a_client_exception_still_returns_empty_not_a_crash(trace):
    class Blows:
        def chat(self, **kwargs):
            raise RuntimeError("network is down")

    answer = try_quick_answer(
        Blows(), settings_for(goal="yes"), trace, memory_context="next: restore P_kbar"
    )
    assert answer == ""


# -- agent.loop.run(): the memory lookup actually reaches try_quick_answer --


def test_run_passes_recalled_memory_into_try_quick_answer(tmp_path, monkeypatch):
    import agent.loop as loop_module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    harness = workspace / ".harness"
    harness.mkdir()
    (harness / "memory.md").write_text(
        "\n## [2026-01-01 00:00] add P_kbar to the feature list\n"
        "keywords: kbar, feature\n"
        "outcome: ok\n"
        "next: Do you want to restore 'P_kbar' to the feature list?\n"
    )

    settings = settings_for(
        workspace=str(workspace),
        goal="yes",
        quick_answers=True,
        trace_dir=str(tmp_path / "traces"),
        failure_log_dir=str(tmp_path / "failed"),
        final_check=False,
        clarify=False,
    )

    seen = {}

    def fake_quick_answer(client, settings, trace, workspace_map="", memory_context=""):
        seen["memory_context"] = memory_context
        return "used the recalled context"

    monkeypatch.setattr(loop_module, "try_quick_answer", fake_quick_answer)
    loop_module.run(settings)

    assert "restore 'P_kbar'" in seen["memory_context"]


def test_run_passes_no_memory_context_when_memory_is_disabled(tmp_path, monkeypatch):
    import agent.loop as loop_module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    harness = workspace / ".harness"
    harness.mkdir()
    (harness / "memory.md").write_text(
        "\n## [2026-01-01 00:00] add P_kbar\nkeywords: kbar\nnext: restore it?\n"
    )

    settings = settings_for(
        workspace=str(workspace),
        goal="yes",
        quick_answers=True,
        memory=False,
        trace_dir=str(tmp_path / "traces"),
        failure_log_dir=str(tmp_path / "failed"),
        final_check=False,
        clarify=False,
    )

    seen = {}

    def fake_quick_answer(client, settings, trace, workspace_map="", memory_context=""):
        seen["memory_context"] = memory_context
        return "answered"

    monkeypatch.setattr(loop_module, "try_quick_answer", fake_quick_answer)
    loop_module.run(settings)

    assert seen["memory_context"] == ""


def test_an_answer_cut_off_at_the_token_cap_falls_through_to_planning(trace):
    """A SWE-bench issue report came back as 595 tokens of workarounds cut off
    mid-code, and the run ended having changed nothing (DESIGN-LOG §69)."""
    client = FakeClient(
        content="This is a known issue in Django 1.11... **Workarounds:** 1. Remove",
        finish_reason="length",
    )
    # Asked as a question: the statement form of this report no longer reaches the model
    # at all (reads_as_a_question), which is the stronger fix for the traced run.
    goal = "Why does a migration crash deleting an index_together with a unique_together?"
    assert try_quick_answer(client, settings_for(goal=goal), trace) == ""
    with open(trace.path) as f:
        assert any(json.loads(line).get("kind") == "quick_answer_truncated" for line in f)


@pytest.mark.parametrize(
    "goal",
    [
        "Write solution.py, a single Python module implementing the code below.",
        "please implement a CSV export",
        "Fix the crash in parse_date",
        "  add a --verbose flag",
    ],
)
def test_a_goal_that_opens_with_a_work_verb_never_reaches_the_quick_answer_model(trace, goal):
    """Two SciCode goals ("Write solution.py ...") came back as code in a chat reply
    and no file was written (DESIGN-LOG §76)."""
    client = FakeClient(content="def f(): ...")
    assert try_quick_answer(client, settings_for(goal=goal), trace) == ""
    assert client.calls == []


@pytest.mark.parametrize("goal", ["what does this project use?", "why is writing slow here?"])
def test_a_question_still_reaches_it(trace, goal):
    client = FakeClient(content="An answer.")
    assert try_quick_answer(client, settings_for(goal=goal), trace) == "An answer."


# Vague SWE-bench reports, verbatim openings: 4 of 20 were answered as questions, wrongly
# ("This is actually already supported in Django!"), and the runs changed nothing (§78).
REPORTS = [
    "When I delete a record that other records point to, Django sends a separate delete",
    "I'm using the pagination feature to go through all the pages of my results",
    "The username box on the login form no longer limits how many characters you can type.",
    'Traceback (most recent call last):\n  File "main.py", line 3',
]


@pytest.mark.parametrize("goal", REPORTS)
def test_a_statement_is_work_and_never_reaches_the_quick_answer_model(trace, goal):
    client = FakeClient(content="This is expected behaviour.")
    assert try_quick_answer(client, settings_for(goal=goal), trace) == ""
    assert client.calls == []


@pytest.mark.parametrize(
    "goal",
    [
        "explain the auth flow",
        "Which files are in this folder",
        "list the tests that cover parse_date",
        "does this project use pytest",
        "is the cache thread-safe? I see it shared across workers.",
    ],
)
def test_questions_without_a_question_mark_still_read_as_questions(goal):
    assert reads_as_a_question(goal)


@pytest.mark.parametrize("goal", REPORTS)
def test_reports_do_not_read_as_questions(goal):
    assert not reads_as_a_question(goal)
