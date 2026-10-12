"""Chat checks with a person present: counter-questions answered, assumptions confirmed (§111).

The answers below are the three counter-questions found in 91 app runs' clarify rounds.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent import chatflow
from agent.approval import Approver
from agent.config import Settings
from agent.interact import Interactor
from agent.llm import Reply, Usage
from agent.loop import _plan_until_accepted
from agent.phases.clarify import gather_clarifications
from agent.phases.plan import Plan, Step
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace

# Shaped like the real answers in app-run traces (a statement then a question, two questions,
# a list ending in one), not quoted from them.
ANSWERS = [
    "not sure, it ran fine yesterday. should I revert the last commit?",
    "on the start screen. Which checks are still red? why are they not fixed yet?",
    "linear, quadratic, spline, any other options worth trying?",
]


def test_the_questions_inside_answers_are_found():
    assert chatflow.counter_questions(ANSWERS[0]) == ["should I revert the last commit?"]
    assert chatflow.counter_questions(ANSWERS[1]) == [
        "Which checks are still red?",
        "why are they not fixed yet?",
    ]
    assert chatflow.counter_questions(ANSWERS[2]) == [
        "linear, quadratic, spline, any other options worth trying?"
    ]
    assert chatflow.counter_questions("no, keep the lamps away from the gaps") == []


@pytest.fixture
def ctx(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def person(**overrides) -> Settings:
    return dataclasses.replace(
        Settings(workspace=".", goal="g", clarify=True, ask_enabled=True), **overrides
    )


def test_only_runs_with_a_person_present_get_the_checks():
    assert chatflow.enabled(person())
    assert not chatflow.enabled(person(clarify=False))  # every benchmark driver
    assert not chatflow.enabled(person(ask_enabled=False))
    assert not chatflow.enabled(person(chat_checks=False))


class FakeClient:
    def __init__(self, content: str):
        self.content = content

    def chat(self, **kwargs):
        return Reply(content=self.content, usage=Usage())


def test_a_question_asked_back_is_named_for_the_planner(ctx):
    replies = iter([ANSWERS[2]])
    ctx.interactor = Interactor(ask=lambda q, o=None: next(replies, ""), enabled=True)
    text = gather_clarifications(FakeClient("Q: Which fitting methods?"), person(), ctx, ctx.trace)
    assert "they also asked you: linear, quadratic, spline, any other options worth trying?" in text


# -- the planner's message ----------------------------------------------------------


def _planner_message(ctx, settings, monkeypatch) -> str:
    import agent.phases.plan as plan_module

    seen: dict = {}

    def fake_loop(**kwargs):
        seen["user"] = kwargs["user"]
        return {"draft": [], "steps": []}, []

    monkeypatch.setattr(plan_module, "run_tool_loop", fake_loop)
    plan_module.make_plan(None, settings, ctx, ctx.trace)
    return seen["user"]


def test_the_planner_is_asked_for_answers_and_assumptions_last(ctx, monkeypatch):
    user = _planner_message(ctx, person(workspace=str(ctx.workspace.root)), monkeypatch)
    assert user.endswith(chatflow.PLANNER_CLOSING)


def test_an_unattended_planner_message_is_unchanged(ctx, monkeypatch):
    bench = person(workspace=str(ctx.workspace.root), clarify=False, ask_enabled=False)
    user = _planner_message(ctx, bench, monkeypatch)
    assert "assume:" not in user and "answer:" not in user


# -- the plan gate --------------------------------------------------------------------


def plan_with(draft: list[str]) -> Plan:
    return Plan(goal="g", draft=draft, steps=[Step(id="s", title="t", intent="i", acceptance="a")])


@pytest.fixture
def planning(monkeypatch):
    """make_plan returns the queued drafts in turn and records each prompt's context."""
    import agent.loop as loop_module

    drafts: list[list[str]] = []
    made: list[str] = []

    def fake_make_plan(client, settings, ctx, trace, extra_context=""):
        made.append(extra_context)
        return plan_with(drafts[min(len(made), len(drafts)) - 1])

    monkeypatch.setattr(loop_module, "make_plan", fake_make_plan)
    monkeypatch.setattr(loop_module, "gather_clarifications", lambda *a, **k: "")
    return drafts, made


def _person_answers(ctx, answers: list[str]) -> list[str]:
    asked: list[str] = []
    replies = iter(answers)

    def ask(question, options=None, recommended=None):
        asked.append(question)
        return next(replies, recommended or "")

    ctx.interactor = Interactor(ask=ask, enabled=True, max_questions=0)
    return asked


def _events(ctx) -> list[dict]:
    return [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]


WING = "the wing is the polygon drawn at game.py:40"


def test_assumptions_are_confirmed_with_one_keystroke(planning, ctx):
    drafts, made = planning
    drafts.append(["understood: fix the wing", f"assume: {WING}"])
    asked = _person_answers(ctx, [chatflow.GO_AHEAD])  # what Enter returns
    plan = _plan_until_accepted(None, person(), ctx, ctx.trace, None)
    assert plan is not None and len(made) == 1
    assert len(asked) == 1 and WING in asked[0]
    assert "assumptions_confirmed" in [e["kind"] for e in _events(ctx)]


def test_a_correction_replans_with_it(planning, ctx):
    drafts, made = planning
    drafts += [[f"assume: {WING}"], ["assume: the wing is the ellipse at game.py:52"]]
    _person_answers(ctx, ["no, that polygon is the beak; the wing is the ellipse", "yes"])
    plan = _plan_until_accepted(None, person(), ctx, ctx.trace, None)
    assert plan is not None and len(made) == 2
    assert "the wing is the ellipse" in made[1] and WING in made[1]


def test_a_plan_with_no_assumptions_asks_nothing(planning, ctx):
    drafts, made = planning
    drafts.append(["understood: x"])
    asked = _person_answers(ctx, [])
    assert _plan_until_accepted(None, person(), ctx, ctx.trace, None) is not None
    assert asked == [] and len(made) == 1


def test_an_unattended_run_never_asks(planning, ctx):
    drafts, _made = planning
    drafts.append([f"assume: {WING}"])
    asked = _person_answers(ctx, [])
    _plan_until_accepted(None, person(clarify=False), ctx, ctx.trace, None)
    assert asked == []


def test_the_planner_s_answer_to_a_counter_question_is_shown(planning, ctx):
    drafts, _made = planning
    drafts.append(["answer: gradient boosting and k-nearest neighbours are worth a try"])
    _person_answers(ctx, [])
    _plan_until_accepted(None, person(), ctx, ctx.trace, None)
    shown = [e for e in _events(ctx) if e["kind"] == "counter_answered"]
    assert shown and "gradient boosting" in shown[0]["answers"][0]


def test_the_confirmation_draws_on_no_question_budget(ctx):
    asked = _person_answers(ctx, ["x"])  # max_questions=0: ask() would refuse
    assert ctx.interactor.ask("q").startswith("(no answer")
    assert ctx.interactor.confirm("assumes ...", [chatflow.GO_AHEAD]) == "x"
    assert asked == ["assumes ..."]
