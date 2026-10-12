"""`bench scout`: a cheap first pass read for failures and for which tasks separate arms."""

from __future__ import annotations

import argparse
import json

import pytest

from bench import cli, scout


@pytest.fixture(autouse=True)
def search_up(monkeypatch):
    """The scout command checks web search before it spends anything (bench/cli.py
    _search_ready). These tests examine the scout's planning and report, not that check, so
    the health probe answers "up" instead of needing a local SearXNG."""
    monkeypatch.setattr(cli.web, "search_health", lambda: (True, "stubbed for tests"))


LLM = "LLMError: request failed after 7 attempts or 420s: stream open past its 180s deadline"


def run(config="a", task="expr", accuracy=1.0, honest=True, calls=10, **extra):
    return {
        "config": config,
        "task": task,
        "repeat": 0,
        "accuracy": accuracy,
        "honest": honest,
        "calls": calls,
        "cost_usd": 0.005,
        "agent_error": "",
        "timed_out": False,
        "wall_s": 30.0,
        **extra,
    }


# -- the guard: drop a config that times out, stop on real trouble -------------------


def test_a_config_that_times_out_is_dropped_and_the_scout_continues():
    guard = scout.ScoutGuard()
    slow = run("a", agent_error="BenchTimeout: exceeded the 300s limit", timed_out=True)
    assert guard(slow) is None
    assert guard.skip("a") and not guard.skip("b")


def test_timeouts_in_two_configs_stop_the_scout():
    guard = scout.ScoutGuard()
    for config in ("a", "b"):
        reason = guard(run(config, agent_error="BenchTimeout: x", timed_out=True))
    assert reason is not None and "time limit" in reason


def test_repeated_provider_stalls_stop_the_scout():
    guard = scout.ScoutGuard()
    reasons = [guard(run(agent_error=LLM, cost_usd=0.0)) for _ in range(scout.SCOUT_STALL_LIMIT)]
    assert reasons[:-1] == [None] * (scout.SCOUT_STALL_LIMIT - 1)
    assert "provider" in (reasons[-1] or "")


def test_a_stall_does_not_drop_the_config_it_happened_on():
    guard = scout.ScoutGuard()
    guard(run("a", agent_error=LLM, cost_usd=0.0))
    assert not guard.skip("a")


def test_an_account_refusal_stops_at_once():
    guard = scout.ScoutGuard()
    assert "account" in (guard(run(agent_error="LLMError: HTTP 402: out of credits")) or "")


def test_a_config_that_crashes_twice_is_dropped():
    guard = scout.ScoutGuard()
    crash = run("a", agent_error="worker process exited unexpectedly (code 1)")
    guard(crash)
    assert not guard.skip("a")
    guard(crash)
    assert guard.skip("a")


# -- the reading -------------------------------------------------------------------


def arms(**by_config):
    return {name: scout._arm(rows, seeded=True) for name, rows in by_config.items()}


def test_every_arm_perfect_and_honest_is_a_ceiling():
    assert scout.reading(arms(a=[run()] * 2, b=[run()] * 2)).startswith("ceiling")


def test_every_arm_failing_is_a_floor():
    rows = [run(accuracy=0.0, honest=False)] * 2
    assert scout.reading(arms(a=rows, b=rows)).startswith("floor")


def test_a_gap_on_honesty_separates_even_at_the_accuracy_ceiling():
    text = scout.reading(arms(a=[run()] * 2, b=[run(honest=False)] * 2))
    assert text.startswith("separates") and "honest" in text


def test_a_calls_gap_separates():
    text = scout.reading(arms(a=[run(calls=10)] * 2, b=[run(calls=30)] * 2))
    assert "calls" in text


def test_a_calls_gap_within_run_to_run_noise_does_not_separate():
    """2x is what the same config produces on its own 8% of the time at 2 runs."""
    assert scout.reading(arms(a=[run(calls=10)] * 2, b=[run(calls=20)] * 2)).startswith("ceiling")


def test_mid_range_with_no_gap_moves():
    rows = [run(), run(accuracy=0.5)]
    assert scout.reading(arms(a=rows, b=rows)).startswith("moves")


def test_a_single_config_still_gets_ceiling_or_floor():
    assert scout.reading(arms(a=[run()] * 2)).startswith("ceiling")


def test_the_report_lists_failures_first_and_drops_extremes_from_the_long_run():
    records = [
        run("a", "expr"),
        run("b", "expr"),
        run("a", "no-bug", honest=True),
        run("b", "no-bug", honest=False),
        run("b", "semver", agent_error="BenchTimeout: x", timed_out=True, cost_usd=0.01),
        run("a", "semver", agent_error=LLM, cost_usd=0.0),
    ]
    text = scout.render(records, {"b": "hit the per-run time limit"})
    assert text.index("== failures ==") < text.index("== discrimination")
    assert "timeout" in text and "stream_stall" in text and "DROPPED b" in text
    assert "expr: ceiling" in text and "no-bug: separates on" in text
    assert scout.surviving_tasks(records) == ["no-bug"]


# -- the command -------------------------------------------------------------------


def test_the_scout_command_runs_skips_a_dropped_config_and_prints_the_report(
    monkeypatch, capsys, tmp_path
):
    started = []

    def fake_run(config, task, repeat, *a, **k):
        started.append((config.name, task.name))
        if config.name == "mercury-legacy-order":
            return run(config.name, task.name, agent_error="BenchTimeout: x", timed_out=True)
        return run(config.name, task.name)

    monkeypatch.setattr(cli, "run_one_with_timeout", fake_run)
    monkeypatch.setattr(cli, "_keep_awake", lambda: None)
    args = cli.build_parser().parse_args(
        [
            "scout",
            "--configs",
            "mercury-only,mercury-legacy-order",
            "--tasks",
            "expr,semver",
            "--parallel",
            "1",
            "--out",
            str(tmp_path / "s.jsonl"),
            "-y",
        ]
    )
    assert args.repeats == 2 and args.timeout == 300
    assert args.func(args) == 0
    out = capsys.readouterr().out
    # The legacy config timed out on its first job and was not started again.
    assert sum(1 for c, _ in started if c == "mercury-legacy-order") == 1
    assert "DROPPED mercury-legacy-order" in out
    assert "for the long run:" in out


def test_the_scout_view_rerenders_any_results_file(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [run("a"), run("b")]))
    args = argparse.Namespace(
        view="scout", include_stalls=False, by="config", a="", b="", since="", paths=[str(path)]
    )
    assert cli.cmd_report(args) == 0
    assert "expr: ceiling" in capsys.readouterr().out


def test_keep_awake_is_a_noop_off_macos(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._keep_awake() is None


def test_a_sleep_stops_the_scout_at_once():
    guard = scout.ScoutGuard()
    reason = guard({"config": "a", "task": "t", "slept_s": 400.0, "agent_error": "LLMError: stall"})
    assert reason and "slept" in reason


def test_a_planning_only_scout_shows_the_check_not_accuracy_readings(monkeypatch, capsys, tmp_path):
    """Planning-only runs are not scored; reading them for accuracy called every
    greenfield task a floor."""
    from bench import cli

    def fake_execute(jobs, args, log, stop, skip=None):
        for config, task, repeat in jobs:
            log.append(
                {
                    "config": config.name,
                    "task": task.name,
                    "repeat": repeat,
                    "plan_only": True,
                    "accuracy": 0.0,
                    "calls": 3,
                    "diag_repro_passed": 1,
                }
            )
        return None

    monkeypatch.setattr(cli, "_execute", fake_execute)
    args = cli.build_parser().parse_args(
        [
            "scout",
            "--configs",
            "mercury-only",
            "--tasks",
            "expr,no-bug",
            "--plan-only",
            "-y",
            "--out",
            str(tmp_path / "s.jsonl"),
        ]
    )
    assert cli.cmd_scout(args) == 0
    out = capsys.readouterr().out
    assert "reproduce check" in out and "== failures ==" in out
    assert "floor" not in out and "for the long run" not in out
