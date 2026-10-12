"""The clarifying-question tool: Interactor and ask_user."""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.interact import Interactor
from agent.tools import ToolContext, execution_registry, planning_registry, review_registry
from agent.tools.ask import ask_user
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- Interactor -------------------------------------------------------------


def test_a_custom_ask_callback_is_used_when_provided():
    interactor = Interactor(ask=lambda q, opts: "custom answer")
    assert interactor.ask("anything?") == "custom answer"
    assert interactor.asked == 1
    assert interactor.answered == 1


def test_disabled_interactor_never_blocks():
    interactor = Interactor(enabled=False)
    answer = interactor.ask("anything?")
    assert "no answer available" in answer
    assert "turned off" in answer
    assert interactor.answered == 0


def test_no_tty_falls_back_without_blocking(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    interactor = Interactor()
    answer = interactor.ask("anything?")
    assert "no answer available" in answer
    assert "no TTY" in answer


def test_a_scripted_free_text_answer(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "use option A")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    interactor = Interactor()
    assert interactor.ask("which design?") == "use option A"


def test_a_scripted_numeric_choice_resolves_to_the_option_text(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    interactor = Interactor()
    answer = interactor.ask("which?", options=["alpha", "beta", "gamma"])
    assert answer == "beta"


def test_an_out_of_range_choice_falls_back_to_the_raw_text(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "99")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    interactor = Interactor()
    answer = interactor.ask("which?", options=["alpha", "beta"])
    assert answer == "99"


def test_an_empty_answer_falls_back_without_blocking(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    interactor = Interactor()
    assert "no answer available" in interactor.ask("which?")


# -- ask_user tool ------------------------------------------------------


def test_ask_user_rejects_an_empty_question(ctx):
    assert ask_user(ctx, "   ").startswith("error:")


def test_ask_user_uses_the_contexts_interactor(ctx):
    ctx.interactor = Interactor(ask=lambda q, opts: f"answered: {q}")
    assert ask_user(ctx, "pick one") == "answered: pick one"


def test_ask_user_truncates_and_forwards_options(ctx):
    seen = {}

    def capture(question, options):
        seen["question"] = question
        seen["options"] = options
        return "ok"

    ctx.interactor = Interactor(ask=capture)
    ask_user(ctx, "which?", options=["a", "b", "c", "d", "e", "f", "g", "h"])
    assert len(seen["options"]) == 6  # MAX_OPTIONS


def test_ask_user_logs_the_question_and_answer(ctx):
    ctx.interactor = Interactor(ask=lambda q, opts: "yes")
    ask_user(ctx, "should I do X?")
    kinds = [e for e in _events(ctx) if e["kind"] in ("ask_user", "ask_user_answered")]
    assert len(kinds) == 2
    assert kinds[1]["answer"] == "yes"


def _events(ctx):
    import json

    return [json.loads(line) for line in ctx.trace.path.read_text().splitlines() if line]


# -- registry placement ---------------------------------------------------


def test_ask_user_is_available_to_the_planner_and_executor():
    assert "ask_user" in planning_registry().names()
    assert "ask_user" in execution_registry().names()


def test_ask_user_is_not_available_to_the_reviewer():
    """The reviewer critiques; it does not make decisions that need a human."""
    assert "ask_user" not in review_registry().names()


# -- the question cap -----------------------------------------------------


def test_a_run_can_only_ask_so_many_questions():
    """A real trace shows a planner asking seven questions in a row and never
    producing a plan, most of them notes to itself rather than questions."""
    asked = []
    interactor = Interactor(
        ask=lambda q, o=None: (asked.append(q), "sure")[1], enabled=True, max_questions=3
    )
    for i in range(6):
        interactor.ask(f"question {i}?")
    assert len(asked) == 3, "only the first three should reach a human"


def test_questions_past_the_cap_tell_the_model_to_proceed():
    interactor = Interactor(ask=lambda q, o=None: "sure", enabled=True, max_questions=1)
    interactor.ask("first?")
    answer = interactor.ask("second?")
    assert "no answer available" in answer
    assert "Decide it yourself" in answer


def test_the_cap_still_counts_every_attempt():
    interactor = Interactor(ask=lambda q, o=None: "sure", enabled=True, max_questions=1)
    interactor.ask("first?")
    interactor.ask("second?")
    assert interactor.asked == 2
    assert interactor.answered == 1


# -- a recommended option -------------------------------------------------


def test_the_recommendation_reaches_a_custom_asker():
    seen = {}

    def ask(question, options=None, recommended=None):
        seen["recommended"] = recommended
        return "alpha"

    Interactor(ask=ask).ask("which?", ["alpha", "beta"], recommended="beta")
    assert seen["recommended"] == "beta"


def test_an_asker_that_only_takes_two_arguments_still_works():
    """The TUI's callback grew a third parameter; a simpler one must not break."""
    Interactor(ask=lambda q, o=None, r=None: "alpha").ask("which?", ["alpha"])


def test_a_two_argument_asker_is_never_handed_a_third():
    """Checked by signature, not by catching TypeError: a TypeError raised
    inside the callback would otherwise look like an arity mismatch and the
    retry would ask the human the same question twice."""
    calls = []

    def two_arg(question, options=None):
        calls.append(question)
        return "alpha"

    answer = Interactor(ask=two_arg).ask("which?", ["alpha"], recommended="alpha")
    assert answer == "alpha"
    assert len(calls) == 1, "it must not have been called a second time"
