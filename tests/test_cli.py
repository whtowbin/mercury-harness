"""The Typer CLI: argument parsing, error handling, and the config/skills
subcommands. Uses typer.testing.CliRunner so nothing here makes a network call.

Note: `run`'s option *defaults* are computed from persisted user preferences at
import time (agent/cli.py's `_prefs = userconfig.load()`), so that specific
wiring is exercised live rather than here — monkeypatching after import can't
reach it. Everything that reads config at call time (config/skills subcommands)
is isolated from the real config dir below.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from agent import userconfig
from agent.cli import app

runner = CliRunner()


def flat(output: str) -> str:
    """`result.output`, whitespace-collapsed to one line.

    Click/Rich wrap CliRunner's output to whatever terminal width it detects
    at the moment a test runs, which is not fixed: a wide interactive
    terminal and a narrow one (or a pytest-xdist worker with no controlling
    terminal at all, which is what actually surfaced this) can wrap the same
    message across a different line break. `"already exists" in flat(result.output)`
    passed on a wide terminal and failed under `-n auto`, because the message
    wrapped to "...dup.md already\nexists" there -- a real width-dependent
    flake, not a workspace/state race. Asserting against the flattened text
    is immune to wherever Click happened to wrap it."""
    return re.sub(r"\s+", " ", output)


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(userconfig, "config_dir", lambda: tmp_path / "config")


# -- top-level ---------------------------------------------------------


def test_help_lists_all_three_command_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "config" in result.output
    assert "skills" in result.output


def test_run_with_no_goal_and_no_plan_errors_cleanly():
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "give the agent a goal" in flat(result.output)


def test_run_help_does_not_crash():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--workspace" in result.output
    assert "--tui" in result.output


# -- config --------------------------------------------------------------


def test_config_show_when_nothing_is_persisted():
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "nothing persisted" in flat(result.output)


def test_config_set_then_show():
    result = runner.invoke(app, ["config", "set", "executor_model", "some/model"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["config", "show"])
    assert "some/model" in result.output


def test_config_set_unknown_key_fails_with_a_clear_message():
    result = runner.invoke(app, ["config", "set", "not_a_field", "x"])
    assert result.exit_code == 2
    assert "unknown setting" in result.output


def test_config_unset_then_show_reverts():
    runner.invoke(app, ["config", "set", "executor_model", "some/model"])
    result = runner.invoke(app, ["config", "unset", "executor_model"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["config", "show"])
    assert "nothing persisted" in flat(result.output)


def test_config_path_prints_the_json_file_location():
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert "config.json" in result.output


def test_config_reset_requires_confirmation_without_yes(monkeypatch):
    runner.invoke(app, ["config", "set", "executor_model", "some/model"])
    result = runner.invoke(app, ["config", "reset"], input="n\n")
    assert result.exit_code != 0
    result = runner.invoke(app, ["config", "show"])
    assert "some/model" in result.output  # not reset


def test_config_reset_with_yes_skips_confirmation():
    runner.invoke(app, ["config", "set", "executor_model", "some/model"])
    result = runner.invoke(app, ["config", "reset", "-y"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["config", "show"])
    assert "nothing persisted" in flat(result.output)


# -- skills ----------------------------------------------------------


def test_skills_list_shows_the_bundled_skills():
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "uv" in result.output
    assert "ruff" in result.output
    assert "bundled" in result.output


def test_skills_new_scaffolds_a_file_and_it_is_then_listed():
    result = runner.invoke(app, ["skills", "new", "my-test-skill"])
    assert result.exit_code == 0
    assert "my-test-skill" in result.output

    result = runner.invoke(app, ["skills", "list"])
    assert "my-test-skill" in result.output
    assert "yours" in result.output


def test_skills_new_refuses_to_overwrite_an_existing_skill():
    runner.invoke(app, ["skills", "new", "dup"])
    result = runner.invoke(app, ["skills", "new", "dup"])
    assert result.exit_code != 0
    assert "already exists" in flat(result.output)


def test_skills_add_copies_a_valid_skill_file(tmp_path):
    source = tmp_path / "custom.md"
    source.write_text(
        "---\nname: custom\ndescription: A custom skill.\ntriggers: [custom]\n---\n\nBody text.\n"
    )
    result = runner.invoke(app, ["skills", "add", str(source)])
    assert result.exit_code == 0
    assert "custom" in result.output

    result = runner.invoke(app, ["skills", "list"])
    assert "custom" in result.output


def test_skills_add_rejects_a_file_with_no_frontmatter(tmp_path):
    source = tmp_path / "bad.md"
    source.write_text("just some text, no frontmatter")
    result = runner.invoke(app, ["skills", "add", str(source)])
    assert result.exit_code != 0
    assert "no valid frontmatter" in flat(result.output)


def test_skills_path_prints_the_user_skills_directory():
    result = runner.invoke(app, ["skills", "path"])
    assert result.exit_code == 0
    assert "skills" in result.output


# -- interactive continue loop --------------------------------------------


def test_continue_loop_is_a_noop_without_a_tty(monkeypatch):
    """The core guarantee: a script or pipe gets exactly the pre-existing
    one-shot behavior, regardless of --continue's default."""
    from agent.cli import _interactive_continue

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    settings = _make_settings()
    summary = {"suggested_next_step": "add tests"}
    result = _interactive_continue(settings, summary, "review")
    assert result is summary  # untouched, no prompt was ever issued


def test_continue_loop_quits_on_q(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: "q")
    settings = _make_settings()
    summary = {"suggested_next_step": "add tests"}
    result = cli_module._interactive_continue(settings, summary, "review")
    assert result is summary


def test_continue_loop_accepts_the_suggestion_on_empty_enter(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["", "q"])  # accept the suggestion, then stop
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: next(answers))

    calls = []

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        calls.append(settings.goal)
        return {"suggested_next_step": "", "totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    settings = _make_settings()
    summary = {"suggested_next_step": "add input validation"}
    cli_module._interactive_continue(settings, summary, "review")
    assert calls == ["add input validation"]


def test_continue_loop_overrides_the_suggestion_with_typed_text(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["do something else entirely", "q"])
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: next(answers))

    calls = []

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        calls.append(settings.goal)
        return {"suggested_next_step": "", "totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    settings = _make_settings()
    summary = {"suggested_next_step": "the model's suggestion"}
    cli_module._interactive_continue(settings, summary, "review")
    assert calls == ["do something else entirely"]


def test_continue_loop_stops_when_nothing_typed_and_no_suggestion(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: "")
    called = {"n": 0}

    def fake_run_agent(*a, **k):
        called["n"] += 1
        return {}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    settings = _make_settings()
    summary = {"suggested_next_step": ""}
    cli_module._interactive_continue(settings, summary, "review")
    assert called["n"] == 0


def test_continue_loop_stops_on_a_failed_followup_run(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # Only one answer is consumed: the loop must stop after the failure,
    # not prompt a second time.
    answers = iter(["next goal"])
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: next(answers))

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        return {
            "error": "boom",
            "suggested_next_step": "",
            "totals": {"by_model": {}},
            "trace_path": "t",
        }

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    settings = _make_settings()
    result = cli_module._interactive_continue(settings, {"suggested_next_step": ""}, "review")
    assert result["error"] == "boom"


def _make_settings():
    from agent.config import Settings

    return Settings(workspace=".", goal="original goal")


# -- prompting for a missing goal ------------------------------------------


def test_no_goal_with_a_tty_prompts_for_one(monkeypatch):
    # Called directly, not via CliRunner: CliRunner substitutes its own
    # non-TTY stdin stream at invocation time, so a monkeypatched
    # sys.stdin.isatty set beforehand never reaches code running inside it.
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: "build a thing")
    seen = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        seen["goal"] = settings.goal
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    # keep_going=False: this test is about goal-prompting, not the separate
    # continuation loop, which would otherwise also see isatty=True and spin
    # forever feeding the same mocked "build a thing" answer to itself.
    cli_module.run_command(keep_going=False)
    assert seen["goal"] == "build a thing"


def test_no_goal_with_a_tty_but_an_empty_answer_still_errors(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(cli_module.typer, "prompt", lambda *a, **k: "   ")
    with pytest.raises(cli_module.typer.Exit) as exc_info:
        cli_module.run_command()
    assert exc_info.value.exit_code == 2


def test_no_goal_without_a_tty_still_errors_without_prompting(monkeypatch):
    """The pre-existing script/pipe behavior must be unaffected."""
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    prompted = {"called": False}

    def fake_prompt(*a, **k):
        prompted["called"] = True
        return ""

    monkeypatch.setattr(cli_module.typer, "prompt", fake_prompt)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "give the agent a goal" in flat(result.output)
    assert not prompted["called"]


def test_tui_with_no_goal_does_not_prompt_on_the_console(monkeypatch):
    """--tui skips the console prompt entirely — the TUI asks for the goal
    itself once it starts, not before the screen even launches."""
    from agent import cli as cli_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompted = {"called": False}

    def fake_prompt(*a, **k):
        prompted["called"] = True
        return "should not be used"

    monkeypatch.setattr(cli_module.typer, "prompt", fake_prompt)

    def fake_watch_run(settings, plan_path, stop_after):
        assert settings.goal == ""  # empty: the TUI is responsible for it
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    import agent.tui

    monkeypatch.setattr(agent.tui, "watch_run", fake_watch_run)
    result = runner.invoke(app, ["run", "--tui"])
    assert result.exit_code == 0
    assert not prompted["called"]


def test_the_printed_summary_names_the_real_goal_not_empty(monkeypatch):
    """summary has no top-level "goal" key -- it lives at summary["plan"]["goal"],
    and _print_session_summary used to look up the wrong one, so the boxed
    terminal summary said "Ran without a goal." on every single run
    regardless of what was actually asked. Caught live, not by a unit test:
    the stored session_summary text (written to the trace) was always
    correct, only the CLI's own separate Rich re-render was wrong."""
    from agent import cli as cli_module

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        return {
            "ok": True,
            "session_summary": "Goal: do the thing. 1 step ran.",
            "plan": {"goal": "do the thing"},
            "totals": {"by_model": {}},
            "trace_path": "t",
        }

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "do the thing"])
    assert result.exit_code == 0
    assert "Ran without a goal" not in result.output
    assert "do the thing" in result.output


def test_a_goal_still_works_normally_with_no_prompting(monkeypatch):
    from agent import cli as cli_module

    prompted = {"called": False}
    monkeypatch.setattr(
        cli_module.typer, "prompt", lambda *a, **k: prompted.__setitem__("called", True) or "x"
    )

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "a real goal"])
    assert result.exit_code == 0
    assert not prompted["called"]


def test_auto_approve_python_flag_reaches_settings(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["auto_approve_python"] = settings.auto_approve_python
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "--auto-approve-python", "a goal"])
    assert result.exit_code == 0
    assert captured["auto_approve_python"] is True


def test_auto_approve_python_defaults_to_false(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["auto_approve_python"] = settings.auto_approve_python
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 0
    assert captured["auto_approve_python"] is False


# -- --goal-loop ------------------------------------------------------------


def test_goal_loop_and_tui_are_rejected_together():
    result = runner.invoke(app, ["run", "--goal-loop", "--tui", "a goal"])
    assert result.exit_code == 2
    assert "--goal-loop" in result.output and "--tui" in result.output


def test_goal_loop_flag_dispatches_to_run_until_done(monkeypatch):
    from agent import cli as cli_module
    from agent import goal as goal_module

    calls = []

    def fake_run_until_done(settings, max_goal_iterations, **_kwargs):
        calls.append((settings.goal, max_goal_iterations))
        return {
            "ok": True,
            "totals": {"by_model": {}},
            "trace_path": "t",
            "goal_loop": {"iterations": [], "status": "done", "reason": "x"},
        }

    monkeypatch.setattr(goal_module, "run_until_done", fake_run_until_done)
    monkeypatch.setattr(
        cli_module, "run_agent", lambda *a, **k: pytest.fail("run_agent should not be called")
    )
    result = runner.invoke(app, ["run", "--goal-loop", "--max-goal-iterations", "5", "a goal"])
    assert result.exit_code == 0
    assert calls == [("a goal", 5)]


def test_goal_loop_off_by_default_uses_run_agent(monkeypatch):
    from agent import cli as cli_module
    from agent import goal as goal_module

    monkeypatch.setattr(
        goal_module,
        "run_until_done",
        lambda *a, **k: pytest.fail("run_until_done should not be called"),
    )

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 0


def test_approval_mode_manual_forces_both_flags_off(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["auto_approve"] = settings.auto_approve
        captured["auto_approve_python"] = settings.auto_approve_python
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    # -y and --auto-approve-python both on, but --approval-mode manual wins.
    result = runner.invoke(
        app, ["run", "-y", "--auto-approve-python", "--approval-mode", "manual", "a goal"]
    )
    assert result.exit_code == 0
    assert captured == {"auto_approve": False, "auto_approve_python": False}


def test_approval_mode_auto_python_only_approves_python(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["auto_approve"] = settings.auto_approve
        captured["auto_approve_python"] = settings.auto_approve_python
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "--approval-mode", "auto-python", "a goal"])
    assert result.exit_code == 0
    assert captured == {"auto_approve": False, "auto_approve_python": True}


def test_approval_mode_auto_plan_approves_everything(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["auto_approve"] = settings.auto_approve
        captured["auto_approve_python"] = settings.auto_approve_python
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "--approval-mode", "auto-plan", "a goal"])
    assert result.exit_code == 0
    assert captured == {"auto_approve": True, "auto_approve_python": True}


def test_approval_mode_omitted_leaves_yes_and_auto_approve_python_as_given(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["auto_approve"] = settings.auto_approve
        captured["auto_approve_python"] = settings.auto_approve_python
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "--auto-approve-python", "a goal"])
    assert result.exit_code == 0
    assert captured == {"auto_approve": False, "auto_approve_python": True}


def test_narrate_quick_answers_and_stats_flags_reach_settings(monkeypatch):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["narrate"] = settings.narrate
        captured["quick_answers"] = settings.quick_answers
        captured["show_stats"] = settings.show_stats
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "--no-narrate", "--no-quick-answers", "--stats", "a goal"])
    assert result.exit_code == 0
    assert captured == {"narrate": False, "quick_answers": False, "show_stats": True}


def test_narrate_quick_answers_and_stats_default_on_narrate_and_quick_answers_off_stats(
    monkeypatch,
):
    """narrate and quick_answers default on; show_stats defaults off — the
    literal ask was "hold off on stats unless requested"."""
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["narrate"] = settings.narrate
        captured["quick_answers"] = settings.quick_answers
        captured["show_stats"] = settings.show_stats
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 0
    assert captured == {"narrate": True, "quick_answers": True, "show_stats": False}


def test_stats_off_by_default_does_not_print_the_totals_table(monkeypatch):
    from agent import cli as cli_module

    def fake_run_agent(*_a, **_k):
        return {
            "totals": {
                "by_model": {
                    "m": {"calls": 1, "completion_tokens": 10, "cost_usd": 0.01, "latency_s": 1.0}
                }
            },
            "trace_path": "t",
        }

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 0
    assert "totals" not in result.output


def test_stats_flag_prints_the_totals_table(monkeypatch):
    from agent import cli as cli_module

    def fake_run_agent(*_a, **_k):
        return {
            "totals": {
                "by_model": {
                    "m": {"calls": 1, "completion_tokens": 10, "cost_usd": 0.01, "latency_s": 1.0}
                }
            },
            "trace_path": "t",
        }

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    result = runner.invoke(app, ["run", "--stats", "a goal"])
    assert result.exit_code == 0
    assert "totals" in result.output


def test_tui_forces_show_stats_off_regardless_of_the_flag(monkeypatch):
    """--stats is meaningless for --tui (it has its own stats panel instead)
    — passing it must not make the TUI path try to print a Rich table."""
    import agent.tui as tui_module

    captured = {}

    def fake_watch_run(settings, plan_path=None, stop_after="review"):
        captured["show_stats"] = settings.show_stats
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(tui_module, "watch_run", fake_watch_run)
    result = runner.invoke(app, ["run", "--tui", "--stats", "a goal"])
    assert result.exit_code == 0
    assert captured["show_stats"] is False


# -- the new planning/verification flags ---------------------------------


def _capture(monkeypatch, summary=None):
    """Patch run_agent, returning the dict of Settings it was handed."""
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path=None, stop_after="review", **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return (
            summary
            if summary is not None
            else {"ok": True, "totals": {"by_model": {}}, "trace_path": "t"}
        )

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    return captured


def test_clarify_is_on_by_default(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].clarify is True


def test_no_clarify_turns_it_off(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--no-clarify", "a goal"]).exit_code == 0
    assert captured["settings"].clarify is False


def test_final_check_is_on_by_default(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].final_check is True


def test_no_final_check_turns_it_off(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--no-final-check", "a goal"]).exit_code == 0
    assert captured["settings"].final_check is False


def test_the_s105_switches_default_as_measured(monkeypatch):
    """DESIGN-LOG §105.2: read assists and the type check on, edit_blocks off."""
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    s = captured["settings"]
    assert (s.read_assist, s.types_after_edit, s.edit_blocks) == (True, True, False)
    assert runner.invoke(app, ["run", "--no-types-after-edit", "--edit-blocks", "g"]).exit_code == 0
    s = captured["settings"]
    assert (s.types_after_edit, s.edit_blocks) == (False, True)


def test_check_after_edit_is_on_by_default(monkeypatch):
    """DESIGN-LOG §99.3."""
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].check_after_edit is True
    assert runner.invoke(app, ["run", "--no-check-after-edit", "a goal"]).exit_code == 0
    assert captured["settings"].check_after_edit is False


def test_review_is_off_by_default_and_review_turns_it_on(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].review is False
    assert captured["settings"].final_check is True
    assert runner.invoke(app, ["run", "--review", "a goal"]).exit_code == 0
    assert captured["settings"].review is True
    assert captured["settings"].final_check is True


def test_review_plan_is_off_by_default(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].review_plan is False


def test_review_plan_passes_no_gate_without_a_terminal(monkeypatch):
    """CliRunner has no real stdin, which is exactly the unattended case: the
    gate must not be installed, or the run would block on a prompt forever."""
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--review-plan", "a goal"]).exit_code == 0
    assert captured["kwargs"].get("on_plan") is None


# -- exit codes reflect whether the work actually passed -----------------


def test_a_run_whose_steps_failed_verification_exits_non_zero(monkeypatch):
    _capture(
        monkeypatch,
        summary={
            "ok": False,
            "error": "",
            "steps": {"total": 2, "failed": ["s2"], "abandoned": [], "unverified": []},
            "final_check": None,
            "totals": {"by_model": {}},
            "trace_path": "t",
        },
    )
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 1
    assert "did not pass its own checks" in flat(result.output)
    assert "s2" in result.output


def test_a_failing_final_check_exits_non_zero_and_says_why(monkeypatch):
    _capture(
        monkeypatch,
        summary={
            "ok": False,
            "error": "",
            "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
            "final_check": {
                "ok": False,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "1 failed, 3 passed",
                "verify_failures": [],
            },
            "totals": {"by_model": {}},
            "trace_path": "t",
        },
    )
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 1
    assert "1 failed, 3 passed" in flat(result.output)


def test_a_clean_run_exits_zero_and_says_nothing_extra(monkeypatch):
    _capture(
        monkeypatch,
        summary={
            "ok": True,
            "error": "",
            "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
            "final_check": {"ok": True},
            "totals": {"by_model": {}},
            "trace_path": "t",
        },
    )
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 0
    assert "did not pass its own checks" not in flat(result.output)


def test_an_abandoned_step_is_reported(monkeypatch):
    _capture(
        monkeypatch,
        summary={
            "ok": False,
            "error": "",
            "steps": {"total": 2, "failed": [], "abandoned": ["s1"], "unverified": []},
            "final_check": None,
            "totals": {"by_model": {}},
            "trace_path": "t",
        },
    )
    result = runner.invoke(app, ["run", "a goal"])
    assert "abandoned" in result.output


def test_a_harness_error_still_reports_through_the_error_path(monkeypatch):
    """An outright failure already prints its own message; the outcome block
    must not duplicate it."""
    _capture(
        monkeypatch,
        summary={
            "ok": False,
            "error": "RuntimeError: boom",
            "steps": {"total": 0, "failed": [], "abandoned": [], "unverified": []},
            "totals": {"by_model": {}},
            "trace_path": "t",
        },
    )
    result = runner.invoke(app, ["run", "a goal"])
    assert result.exit_code == 1
    assert "did not pass its own checks" not in flat(result.output)


# -- approval_mode is persistable now ------------------------------------


def test_approval_mode_can_be_persisted():
    result = runner.invoke(app, ["config", "set", "approval_mode", "auto-python"])
    assert result.exit_code == 0
    assert userconfig.load().get("approval_mode") == "auto-python"


def test_an_unrecognised_persisted_approval_mode_is_ignored(monkeypatch):
    """A config file is hand-editable; a typo in it should cost a default,
    not the ability to run the tool."""
    from agent import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_prefs", userconfig.UserPreferences({"approval_mode": "nonsense"})
    )
    assert cli_module._read_approval_mode_pref() is None


def test_a_valid_persisted_approval_mode_is_read_back(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_prefs", userconfig.UserPreferences({"approval_mode": "auto-plan"})
    )
    assert cli_module._read_approval_mode_pref() is cli_module.ApprovalMode.auto_plan


def test_fix_existing_failures_is_on_by_default(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].fix_existing_failures is True


def test_no_fix_existing_failures_turns_it_off(monkeypatch):
    captured = _capture(monkeypatch)
    result = runner.invoke(app, ["run", "--no-fix-existing-failures", "a goal"])
    assert result.exit_code == 0
    assert captured["settings"].fix_existing_failures is False


def test_the_test_audit_is_on_by_default_and_the_repair_round_is_opt_in(monkeypatch):
    """The audit is report-only and measured at 0.4s, so it is on. Acting on it
    cost ~4 more model calls a run on a confirmation sweep and its honesty gain
    was not established there, so the repair round is off until asked for."""
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].test_audit is True
    assert captured["settings"].fix_weak_tests is False


def test_the_audit_flags_reach_settings(monkeypatch):
    captured = _capture(monkeypatch)
    result = runner.invoke(app, ["run", "--no-test-audit", "--fix-weak-tests", "a goal"])
    assert result.exit_code == 0
    assert captured["settings"].test_audit is False
    assert captured["settings"].fix_weak_tests is True


def test_require_tests_is_on_by_default(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].require_tests is True


def test_no_require_tests_turns_it_off(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--no-require-tests", "a goal"]).exit_code == 0
    assert captured["settings"].require_tests is False


def test_check_coverage_is_on_by_default(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].check_coverage is True


def test_no_check_coverage_turns_it_off(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--no-check-coverage", "a goal"]).exit_code == 0
    assert captured["settings"].check_coverage is False


def test_draft_style_defaults_to_terse_and_reaches_settings(monkeypatch):
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].draft_style == "terse"
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--draft-style", "prose", "a goal"]).exit_code == 0
    assert captured["settings"].draft_style == "prose"


def test_an_unknown_draft_style_is_rejected(monkeypatch):
    _capture(monkeypatch)
    result = runner.invoke(app, ["run", "--draft-style", "poetry", "a goal"])
    assert result.exit_code == 2
    assert "terse" in flat(result.output) and "prose" in flat(result.output)


def test_plan_probe_defaults_to_repro_and_reaches_settings(monkeypatch):
    """ "repro" by default: +0.13 calls a run over 19 tasks (§54.2)."""
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert captured["settings"].plan_probe == "repro"
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--plan-probe", "require", "a goal"]).exit_code == 0
    assert captured["settings"].plan_probe == "require"
    captured = _capture(monkeypatch)
    assert runner.invoke(app, ["run", "--plan-probe", "repro", "a goal"]).exit_code == 0
    assert captured["settings"].plan_probe == "repro"


def test_an_unknown_plan_probe_mode_is_rejected(monkeypatch):
    _capture(monkeypatch)
    result = runner.invoke(app, ["run", "--plan-probe", "always", "a goal"])
    assert result.exit_code == 2
    assert "require" in flat(result.output)


# -- the end-of-run suggestion (DESIGN-LOG §122) ------------------------------


def _captured_suggest(monkeypatch, args, tty):
    from agent import cli as cli_module

    captured = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        captured["suggest"] = settings.suggest_next_step
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    # CliRunner swaps sys.stdin/stdout inside invoke, so the terminal check is stubbed instead;
    # the real check is exercised in a subprocess below.
    monkeypatch.setattr(cli_module, "_at_a_terminal", lambda: tty)
    result = runner.invoke(app, ["run", *args, "a goal"])
    assert result.exit_code == 0, result.output
    return captured["suggest"]


def test_a_scripted_run_without_memory_skips_the_suggestion_call(monkeypatch):
    # The benchmark drivers' shape: no terminal, --no-memory. Nobody reads the suggestion.
    assert _captured_suggest(monkeypatch, ["--no-memory"], tty=False) is False


def test_the_suggestion_is_kept_wherever_something_reads_it(monkeypatch):
    assert _captured_suggest(monkeypatch, ["--no-memory"], tty=True) is True  # a person
    assert _captured_suggest(monkeypatch, [], tty=False) is True  # memory notes read it


def test_the_tui_and_the_goal_loop_keep_the_suggestion(monkeypatch):
    from agent import cli as cli_module

    monkeypatch.setattr(cli_module, "_at_a_terminal", lambda: False)
    assert cli_module._anyone_reads_the_suggestion(tui=True, goal_loop=False, memory=False)
    assert cli_module._anyone_reads_the_suggestion(tui=False, goal_loop=True, memory=False)
    assert not cli_module._anyone_reads_the_suggestion(tui=False, goal_loop=False, memory=False)


def test_a_driver_logging_stdout_to_a_file_is_not_at_a_terminal(tmp_path):
    # The drivers' real shape: the harness's stdout is a log file. A real process, not a stub.
    import subprocess
    import sys

    log = tmp_path / "run.log"
    with open(log, "w") as out:
        subprocess.run(
            [sys.executable, "-c", "from agent.cli import _at_a_terminal; print(_at_a_terminal())"],
            stdout=out, stdin=subprocess.DEVNULL, check=True,
        )  # fmt: skip
    assert log.read_text().strip() == "False"


def test_stop_after_idle_fix_round_flag_reaches_settings_and_defaults_off(monkeypatch):
    from agent import cli as cli_module

    seen = []

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        seen.append(settings.stop_after_idle_fix_round)
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    assert runner.invoke(app, ["run", "a goal"]).exit_code == 0
    assert runner.invoke(app, ["run", "--stop-after-idle-fix-round", "a goal"]).exit_code == 0
    assert seen == [False, True]


def test_version_prints_the_package_version():
    from agent import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"mercury-harness {__version__}"


def test_a_missing_extra_says_how_to_install_it(monkeypatch, tmp_path):
    """`trace sql` without DuckDB, as from an install without the analysis extra: an install
    command, not a ModuleNotFoundError."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "duckdb" else real(name, *a)
    )
    result = runner.invoke(app, ["trace", "sql", str(tmp_path), "select 1"])
    assert result.exit_code == 2
    assert "mercury-harness[analysis]" in flat(result.output)


def test_no_expected_result_gate_reaches_settings(monkeypatch, tmp_path):
    """§131.9: the sealed benchmark arms set it from the command line."""
    from agent import cli as cli_module

    seen = []

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        seen.append(settings.expected_result_gate)
        return {"suggested_next_step": "", "totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    for flags, want in (([], False), (["--expected-result-gate"], True)):
        result = CliRunner().invoke(
            app, ["run", "goal", "-w", str(tmp_path), "-y", "--no-memory", "--no-animate", *flags]
        )
        assert result.exit_code == 0, result.output
        assert seen[-1] is want


def test_runaway_step_down_reaches_settings(monkeypatch, tmp_path):
    """§131.12: sealed arms set it from the command line."""
    from agent import cli as cli_module

    seen = []

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        seen.append(settings.runaway_effort_step_down)
        return {"suggested_next_step": "", "totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli_module, "run_agent", fake_run_agent)
    for flags, want in (([], True), (["--no-runaway-step-down"], False)):
        result = CliRunner().invoke(
            app, ["run", "goal", "-w", str(tmp_path), "-y", "--no-memory", "--no-animate", *flags]
        )
        assert result.exit_code == 0, result.output
        assert seen[-1] is want
