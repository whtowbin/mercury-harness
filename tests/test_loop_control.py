"""agent.loop.run()'s stop/cancel/failure-log wiring.

Every case here cancels or interrupts *before* the client would ever be
asked to do real work, so a fake API key is enough — client.chat() is never
called, only agent.loop's own try/except/finish plumbing is exercised.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import agent.loop as loop_module
from agent.cancel import CancelToken
from agent.config import Settings


def settings_for(tmp_path, **overrides):
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    base = Settings(
        workspace=str(tmp_path / "ws"),
        goal="build a thing",
        trace_dir=str(tmp_path / "traces"),
        failure_log_dir=str(tmp_path / "failed-chats"),
        suggest_next_step=False,
        # Every other test in this file cancels/interrupts before a plan
        # exists — without this, quick_answers=True (the real default) would
        # make try_quick_answer's own real network call first, on a fake
        # API key, for every one of them.
        quick_answers=False,
        # Same reasoning: the final check shells out to pytest and ty against
        # the workspace, which is seconds of subprocess time per test for a
        # phase none of these tests is about. tests/test_finalcheck.py
        # exercises it directly instead.
        final_check=False,
        clarify=False,
    )
    return dataclasses.replace(base, **overrides)


def test_an_already_cancelled_token_stops_before_any_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = settings_for(tmp_path)
    token = CancelToken()
    token.cancel()

    summary = loop_module.run(settings, cancel=token)

    assert summary["error"].startswith("stopped:")
    assert summary["failure_log_path"]


def test_cancelling_writes_a_failure_log_with_the_goal_and_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = settings_for(tmp_path, goal="catalog my books")
    token = CancelToken()
    token.cancel()

    summary = loop_module.run(settings, cancel=token)

    log_path = summary["failure_log_path"]
    assert log_path
    text = Path(log_path).read_text(encoding="utf-8")
    assert "catalog my books" in text
    assert "stopped" in text


def test_a_disabled_failure_log_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = settings_for(tmp_path, failure_log_dir=None)
    token = CancelToken()
    token.cancel()

    summary = loop_module.run(settings, cancel=token)

    assert summary["error"].startswith("stopped:")
    assert summary["failure_log_path"] == ""
    assert not (tmp_path / "failed-chats").exists()


def test_keyboard_interrupt_during_planning_is_logged_then_reraised(tmp_path, monkeypatch):
    """A caller relying on KeyboardInterrupt reaching it (the CLI's exit code
    130) must still see it — but the run's progress and the reason are saved
    first, not discarded, the same as every other way a run can stop."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_make_plan(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(loop_module, "make_plan", fake_make_plan)
    settings = settings_for(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        loop_module.run(settings)

    saved = list((tmp_path / "failed-chats").glob("*.md"))
    assert saved, "the failure log must be written before the interrupt re-raises"
    assert "interrupted by user" in saved[0].read_text(encoding="utf-8")


def test_stop_after_plan_never_touches_the_failure_log(tmp_path, monkeypatch):
    """--stop-after plan is an ordinary early return, not a failure."""
    from agent.phases.plan import Plan, Step

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_make_plan(*_args, **_kwargs):
        return Plan(
            goal="build a thing",
            draft=["do it"],
            steps=[Step(id="only-step", title="do it", intent="do it", acceptance="it is done")],
        )

    monkeypatch.setattr(loop_module, "make_plan", fake_make_plan)
    settings = settings_for(tmp_path)

    summary = loop_module.run(settings, stop_after="plan")

    assert not summary.get("error")
    assert not (tmp_path / "failed-chats").exists()


# -- quick_answers -----------------------------------------------------------


def test_a_quick_answer_skips_planning_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        loop_module, "try_quick_answer", lambda *a, **k: "Because it's simpler and needs no state."
    )
    monkeypatch.setattr(
        loop_module,
        "make_plan",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("make_plan must not be called")),
    )
    settings = settings_for(tmp_path, quick_answers=True, goal="why prefer a tuple?")

    summary = loop_module.run(settings)

    assert summary["quick_answer"] == "Because it's simpler and needs no state."
    assert summary["plan"] is None
    assert not summary.get("error")


def test_needs_plan_falls_through_to_the_normal_pipeline(tmp_path, monkeypatch):
    from agent.phases.plan import Plan, Step

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(loop_module, "try_quick_answer", lambda *a, **k: "")
    called = []

    def fake_make_plan(*_a, **_k):
        called.append(1)
        return Plan(
            goal="build a thing",
            draft=["do it"],
            steps=[Step(id="s", title="s", intent="s", acceptance="s")],
        )

    monkeypatch.setattr(loop_module, "make_plan", fake_make_plan)
    settings = settings_for(tmp_path, quick_answers=True)

    summary = loop_module.run(settings, stop_after="plan")

    assert called == [1]
    assert summary["quick_answer"] == ""
    assert summary["plan"] is not None


def test_quick_answers_off_never_calls_the_classifier(tmp_path, monkeypatch):
    from agent.phases.plan import Plan, Step

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def blows_up(*_a, **_k):
        raise AssertionError("try_quick_answer must not be called when quick_answers=False")

    monkeypatch.setattr(loop_module, "try_quick_answer", blows_up)
    monkeypatch.setattr(
        loop_module,
        "make_plan",
        lambda *a, **k: Plan(
            goal="g", draft=[], steps=[Step(id="s", title="s", intent="s", acceptance="s")]
        ),
    )
    settings = settings_for(tmp_path, quick_answers=False)

    loop_module.run(settings, stop_after="plan")  # must not raise


def test_a_saved_plan_path_skips_the_quick_answer_check(tmp_path, monkeypatch):
    """--plan is an explicit request to re-run a saved plan — that's already
    a decision to do real work, not a question to classify."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def blows_up(*_a, **_k):
        raise AssertionError("try_quick_answer must not run when a saved plan is given")

    monkeypatch.setattr(loop_module, "try_quick_answer", blows_up)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        '{"plan": {"goal": "g", "draft": [], "steps": '
        '[{"id": "s", "title": "s", "intent": "s", "acceptance": "s"}]}}',
        encoding="utf-8",
    )
    settings = settings_for(tmp_path, quick_answers=True)

    loop_module.run(settings, plan_path=str(plan_file), stop_after="plan")  # must not raise


# -- show_stats ---------------------------------------------------------------


def test_show_stats_false_omits_the_totals_narration(tmp_path, monkeypatch):
    from agent.phases.plan import Plan, Step

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        loop_module,
        "make_plan",
        lambda *a, **k: Plan(
            goal="g", draft=[], steps=[Step(id="s", title="s", intent="s", acceptance="s")]
        ),
    )
    logs: list[str] = []
    settings = settings_for(tmp_path, show_stats=False)

    loop_module.run(
        settings,
        stop_after="plan",
        on_event=lambda r: logs.append(r.get("message", "")) if r.get("kind") == "log" else None,
    )

    assert not any("=== totals ===" in m for m in logs)


def test_show_stats_true_includes_the_totals_narration(tmp_path, monkeypatch):
    from agent.phases.plan import Plan, Step

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        loop_module,
        "make_plan",
        lambda *a, **k: Plan(
            goal="g", draft=[], steps=[Step(id="s", title="s", intent="s", acceptance="s")]
        ),
    )
    logs: list[str] = []
    settings = settings_for(tmp_path, show_stats=True)

    loop_module.run(
        settings,
        stop_after="plan",
        on_event=lambda r: logs.append(r.get("message", "")) if r.get("kind") == "log" else None,
    )

    assert any("=== totals ===" in m for m in logs)
