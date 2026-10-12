"""Making the agent's reading of a request visible, and answers authoritative.

The traced failure. The goal said lamps "can appear inside or next to
the gap of a wall". The clarifier asked whether they should align with the gap
or spawn near it. The person answered that lamps should stay away
from the gaps. The plan came back as "fix lamp spawn to align vertically with
wall gaps" — the opposite of the answer — and nobody saw until the game was
run, because the draft was narrated as one joined line of four notes.

Reasoning was not the cause: the planner already runs at "high", the top of
the ladder.
"""

from __future__ import annotations

import inspect

from agent.phases.clarify import gather_clarifications
from agent.phases.plan import Plan, Step, _narrate_understanding
from agent.prompts import PLANNER_SYSTEM


class _Trace:
    def __init__(self):
        self.said: list[str] = []

    def say(self, message: str) -> None:
        self.said.append(message)

    def event(self, *a, **k) -> None:
        pass


def _plan(draft):
    return Plan(goal="g", draft=draft, steps=[Step(id="s", title="t", intent="i", acceptance="a")])


# -- the reading gets its own line ---------------------------------------


def test_the_restatement_is_shown_on_its_own_line():
    trace = _Trace()
    _narrate_understanding(
        _plan(
            [
                "understood: 1) stop the player changing colour 2) keep lamps out of wall gaps",
                "read game.py",
                "fix spawn_lamp",
                "add tests",
            ]
        ),
        trace,
    )
    first = trace.said[0]
    assert "understood as:" in first
    assert "keep lamps out of wall gaps" in first
    assert "read game.py" not in first, "the rest of the draft is a separate line"


def test_the_remaining_notes_still_appear():
    trace = _Trace()
    _narrate_understanding(_plan(["understood: do the thing", "read x", "edit y"]), trace)
    joined = "\n".join(trace.said)
    assert "draft: read x | edit y" in joined


def test_a_plan_without_a_restatement_still_narrates_its_draft():
    """Older plans, and any run where the planner ignores the instruction."""
    trace = _Trace()
    _narrate_understanding(_plan(["read x", "edit y"]), trace)
    assert trace.said == ["draft: read x | edit y"]


def test_an_empty_draft_says_nothing():
    trace = _Trace()
    _narrate_understanding(_plan([]), trace)
    assert trace.said == []


# -- answers beat the goal where they disagree ---------------------------


def test_the_answers_block_says_the_answer_wins():
    source = inspect.getsource(gather_clarifications)
    assert "authoritative" in source
    assert "contradicts the goal" in source


def test_the_answers_block_explains_a_short_answer_needs_its_question():
    """ "neither" means nothing on its own; it means something against the two
    options the question offered, and it rejects both."""
    source = inspect.getsource(gather_clarifications)
    assert "rejects them both" in source


def test_the_planner_is_asked_for_the_restatement():
    assert "understood:" in PLANNER_SYSTEM
    assert "every separate thing asked for" in PLANNER_SYSTEM
