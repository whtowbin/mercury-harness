"""Provider failures are told apart from agent failures, and left out of the tables.

Three sweeps netted this out by hand (DESIGN-LOG §51, §53.2, §54.3): a run that died
because the provider stalled scores zero and sits beside runs that were wrong.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pytest

from bench import cli, report, stalls

LLM = "LLMError: request failed after 7 attempts or 420s: "


def run(config="a", agent_error="", cost=0.005, timed_out=False, accuracy=1.0, **extra):
    return {
        "config": config,
        "task": "t",
        "agent_error": agent_error,
        "cost_usd": cost,
        "timed_out": timed_out,
        "accuracy": accuracy,
        "honest": True,
        "calls": 10,
        **extra,
    }


# -- what counts as a provider failure -------------------------------------


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (LLM + "stream open past its 180s deadline with no finished response", "stream_stall"),
        (LLM + "('Connection aborted.', RemoteDisconnected('x'))", "connection"),
        (LLM + "('Connection reset by peer')", "connection"),
        (LLM + "provider error: {'code': 502, 'message': 'bad gateway'}", "server_error"),
        (LLM + "HTTP 503: upstream", "server_error"),
        (LLM + "HTTP 429: slow down", "rate_limit"),
        (LLM + "something nobody has seen before", "provider_other"),
    ],
)
def test_the_shapes_seen_in_real_sweeps_are_classified(error, kind):
    assert stalls.provider_failure(run(agent_error=error)) == kind


def test_a_zero_cost_bench_timeout_is_the_providers():
    """No model call ever completed, so the run was waiting, not misbehaving."""
    record = run(agent_error="BenchTimeout: exceeded the 900s limit", timed_out=True, cost=0.0)
    assert stalls.provider_failure(record) == "zero_cost_timeout"


def test_a_timeout_on_a_run_that_spent_money_is_a_slow_run_and_stays_the_agents():
    record = run(agent_error="BenchTimeout: exceeded the 900s limit", timed_out=True, cost=0.02)
    assert stalls.provider_failure(record) == ""


def test_a_worker_crash_is_not_a_provider_failure():
    record = run(agent_error="worker process exited unexpectedly (code 1)", cost=0.0)
    assert stalls.provider_failure(record) == ""


def test_an_error_the_agent_raised_itself_is_not():
    assert stalls.provider_failure(run(agent_error="BudgetExceeded: spend cap reached")) == ""
    assert stalls.provider_failure(run(agent_error="KeyError: 'x'")) == ""


def test_a_clean_run_is_not():
    assert stalls.provider_failure(run()) == ""


def test_a_wrong_answer_is_not_a_provider_failure():
    assert stalls.provider_failure(run(accuracy=0.0)) == ""


def test_a_malformed_cost_does_not_raise():
    record = run(agent_error="BenchTimeout: x", timed_out=True)
    record["cost_usd"] = "n/a"
    assert stalls.provider_failure(record) == "zero_cost_timeout"


# -- splitting and saying so -----------------------------------------------


def test_split_keeps_order_and_separates_the_two():
    records = [run("a"), run("b", agent_error=LLM + "stream open past its 180s"), run("c")]
    completed, stalled = stalls.split(records)
    assert [r["config"] for r in completed] == ["a", "c"]
    assert [r["config"] for r in stalled] == ["b"]


def test_a_clean_sweep_has_no_note():
    records = [run(), run()]
    assert stalls.note(records, stalls.split(records)[1]) == ""


def test_the_note_names_kinds_and_per_config_losses():
    records = [
        run("a"),
        run("a", agent_error=LLM + "stream open past its 180s"),
        run("b", agent_error=LLM + "('Connection aborted.')"),
        run("b"),
        run("b"),
    ]
    text = stalls.note(records, stalls.split(records)[1])
    assert "2 of 5 run(s) died of provider failures" in text
    assert "stream_stall 1" in text and "connection 1" in text
    assert "a 1/2" in text and "b 1/3" in text
    assert "--include-stalls" in text


def test_an_arm_that_lost_a_fifth_of_its_runs_is_flagged():
    records = [run("a")] * 3 + [run("a", agent_error=LLM + "stream open past its 180s")] * 2
    text = stalls.note(records, stalls.split(records)[1])
    assert "WARNING: a lost more than 20%" in text


def test_a_small_loss_is_not_flagged():
    records = [run("a")] * 9 + [run("a", agent_error=LLM + "stream open past its 180s")]
    assert "WARNING" not in stalls.note(records, stalls.split(records)[1])


# -- the report -------------------------------------------------------------


def args(view="outcome", include_stalls=False, **extra):
    return argparse.Namespace(
        view=view,
        include_stalls=include_stalls,
        by="config",
        a="",
        b="",
        paths=[],
        **extra,
    )


def write(tmp_path, records):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return str(path)


def sweep():
    stalled = run("b", agent_error=LLM + "stream open past its 180s", cost=0.0, accuracy=0.0)
    return [run("a")] * 4 + [run("b")] * 3 + [stalled]


def test_the_outcome_table_leaves_stalled_runs_out_and_says_so(tmp_path, capsys):
    a = args()
    a.paths = [write(tmp_path, sweep())]
    assert cli.cmd_report(a) == 0
    out = capsys.readouterr().out
    assert "1 of 8 run(s) died of provider failures" in out
    # b's accuracy is over its 3 completed runs, not dragged down by the stall.
    row = next(line for line in out.splitlines() if line.startswith("b "))
    assert "100.0%" in row


def test_include_stalls_counts_them_as_ordinary_runs(tmp_path, capsys):
    a = args(include_stalls=True)
    a.paths = [write(tmp_path, sweep())]
    assert cli.cmd_report(a) == 0
    out = capsys.readouterr().out
    assert "died of provider failures" not in out
    row = next(line for line in out.splitlines() if line.startswith("b "))
    assert "75.0%" in row


def test_the_failures_view_shows_provider_failures_as_their_own_class(tmp_path, capsys):
    a = args(view="failures")
    a.paths = [write(tmp_path, sweep())]
    assert cli.cmd_report(a) == 0
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    assert "provider" in header
    row = next(line for line in out.splitlines() if line.startswith("b "))
    assert row.split()[-1] == "1"


def test_classify_failure_puts_a_provider_failure_before_crashed():
    record = run(agent_error=LLM + "stream open past its 180s", accuracy=0.0)
    assert report.classify_failure(record) == "provider"
    assert report.classify_failure(run(agent_error="KeyError: x", accuracy=0.0)) == "crashed"


def test_a_sweep_where_every_run_stalled_says_so_instead_of_printing_empty_tables(tmp_path, capsys):
    a = args()
    a.paths = [write(tmp_path, [run("a", agent_error=LLM + "stream open past its 180s")])]
    assert cli.cmd_report(a) == 1
    assert "nothing to report" in capsys.readouterr().out


def test_the_end_of_sweep_tables_exclude_them_too(capsys):
    cli._print_sweep_tables(sweep(), task_count=1)
    out = capsys.readouterr().out
    assert "1 of 8 run(s) died of provider failures" in out


def test_compare_reports_completed_runs_only(tmp_path, capsys):
    records = [run("a", accuracy=1.0)] * 6 + [run("b", accuracy=1.0)] * 6
    records.append(run("b", agent_error=LLM + "stream open past its 180s", accuracy=0.0))
    a = args(view="compare")
    a.a, a.b = "a", "b"
    a.paths = [write(tmp_path, records)]
    assert cli.cmd_report(a) == 0
    out = capsys.readouterr().out
    assert "n = 6 vs 6" in out and "died of provider failures" in out


# -- ground truth, where the sweeps it was built for exist locally ----------


def _load(pattern):
    records = []
    for path in sorted(glob.glob(pattern)):
        records += [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return records


@pytest.mark.skipif(not glob.glob("results/repro2-*.jsonl"), reason="results/ is local-only")
def test_it_reproduces_the_32_provider_failures_counted_by_hand_in_54_3():
    completed, stalled = stalls.split(_load("results/repro2-*.jsonl"))
    assert len(stalled) == 32
    assert not any(r.get("agent_error") or r.get("timed_out") for r in completed)


@pytest.mark.skipif(not Path("results/reasoning-arms.jsonl").exists(), reason="local-only")
def test_every_failed_run_in_the_reasoning_sweep_is_classified():
    """DESIGN-LOG §53.2 says 31 of 320; the records hold 27 failed runs, and all 27
    are provider failures. Nothing is left unexplained in either direction."""
    completed, stalled = stalls.split(_load("results/reasoning-arms.jsonl"))
    assert len(stalled) == 27
    assert not any(r.get("agent_error") or r.get("timed_out") for r in completed)


# -- history: what a config normally scores, and which tasks are at a ceiling --


def hist(task, n, accuracy=1.0, honest=True, config="mercury-only", started_at=1000.0):
    return [
        {
            "config": config,
            "task": task,
            "accuracy": accuracy,
            "honest": honest,
            "calls": 12,
            "started_at": started_at,
        }
        for _ in range(n)
    ]


def test_history_names_a_task_every_run_passes_as_a_ceiling():
    text = report.render_history(hist("expr", 10), "mercury-only")
    assert "ceiling: cannot separate configs" in text


def test_a_task_that_is_solved_but_dishonestly_moves_on_honesty():
    rows = hist("no-bug", 6) + hist("no-bug", 6, honest=False)
    assert "moves on honesty" in report.render_history(rows, "mercury-only")


def test_a_task_that_is_partly_failed_moves_on_accuracy():
    rows = hist("semver", 6) + hist("semver", 6, accuracy=0.5)
    assert "moves on accuracy" in report.render_history(rows, "mercury-only")


def test_a_task_almost_nothing_passes_is_a_floor():
    text = report.render_history(
        hist("misreport-discount", 10, accuracy=0.0, honest=False), "mercury-only"
    )
    assert "floor" in text


def test_too_few_runs_gives_no_reading():
    assert "too few runs" in report.render_history(hist("t", 3), "mercury-only")


def test_other_configs_are_not_mixed_in():
    rows = hist("t", 10) + hist("t", 10, accuracy=0.0, config="other")
    text = report.render_history(rows, "mercury-only")
    assert "10 runs across 1 task" in text


def test_since_drops_older_runs_of_a_task_whose_fixture_changed():
    """throttle-vague's hidden tests were fixed in DESIGN-LOG §47; earlier runs of it
    measure a different task."""
    rows = hist("throttle-vague", 10, accuracy=0.3, started_at=100.0) + hist(
        "throttle-vague", 10, started_at=5000.0
    )
    text = report.render_history(rows, "mercury-only", since=1000.0)
    assert "10 runs" in text and "ceiling" in text


def test_an_unknown_config_lists_what_was_found():
    assert "found ['mercury-only']" in report.render_history(hist("t", 3), "nope")


def test_the_history_view_defaults_to_mercury_only_and_rejects_a_bad_date(tmp_path, capsys):
    rows = hist("expr", 10)
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    a = args(view="history", since="")
    a.paths = [str(path)]
    assert cli.cmd_report(a) == 0
    assert "mercury-only: 10 runs" in capsys.readouterr().out
    a.since = "yesterday"
    assert cli.cmd_report(a) == 1


# -- account refusals: out of credits or a bad key --------------------------

CREDITS = (
    'LLMError: HTTP 402: {"error":{"message":"This request would exceed your available '
    'credits given your current in-flight requests.","code":402}}'
)


@pytest.mark.parametrize("code", ["401", "402", "403"])
def test_an_account_refusal_is_a_provider_side_failure_of_its_own_kind(code):
    record = run(agent_error=f"LLMError: HTTP {code}: nope", cost=0.0004)
    assert stalls.is_account_failure(record)
    assert stalls.provider_failure(record) == "account"


def test_a_402_is_an_account_failure_although_the_client_never_retried_it():
    """Unlike a stall it carries no "request failed after N attempts": the request
    was refused outright, so the error text is just the HTTP status."""
    record = run(agent_error=CREDITS)
    assert stalls.provider_failure(record) == "account"


def test_other_http_errors_are_not_account_failures():
    assert not stalls.is_account_failure(run(agent_error="LLMError: HTTP 400: bad request"))
    assert not stalls.is_account_failure(run(agent_error="KeyError: 'http 402'"))


def test_the_note_says_plainly_when_the_account_is_the_problem():
    records = [run("a")] + [run("a", agent_error=CREDITS)] * 3
    text = stalls.note(records, stalls.split(records)[1])
    assert "ACCOUNT: 3 run(s) were refused" in text
    assert "out of credits" in text


def test_a_sweep_stops_after_three_refusals_in_a_row():
    guard = cli._AccountGuard()
    refused = run(agent_error=CREDITS)
    assert guard(refused) is None and guard(refused) is None
    reason = guard(refused)
    assert reason is not None and "out of credits" in reason


def test_a_successful_run_resets_the_streak():
    guard = cli._AccountGuard()
    refused = run(agent_error=CREDITS)
    for record in (refused, refused, run(), refused, refused):
        assert guard(record) is None


def test_a_stall_is_not_an_account_failure_and_does_not_count():
    guard = cli._AccountGuard()
    for _ in range(5):
        assert guard(run(agent_error=LLM + "stream open past its 180s")) is None


class Log:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)


def test_the_sequential_runner_drops_the_jobs_after_the_trip():
    log = Log()
    jobs = [(i,) for i in range(10)]
    reason = cli._run_jobs(
        jobs,
        lambda i: run(agent_error=CREDITS if i < 5 else ""),
        1,
        log,
        lambda record: None,
        stop=cli._AccountGuard(),
    )
    assert reason and "out of credits" in reason
    assert len(log.records) == 3


def test_the_parallel_runner_stops_scheduling_too():
    import time

    log = Log()

    def slow(i):
        time.sleep(0.05)
        return run(agent_error=CREDITS)

    reason = cli._run_jobs(
        [(i,) for i in range(40)], slow, 2, log, lambda record: None, stop=cli._AccountGuard()
    )
    assert reason
    assert len(log.records) < 40


def test_a_healthy_sweep_runs_every_job_and_reports_no_reason():
    log = Log()
    reason = cli._run_jobs(
        [(i,) for i in range(6)], lambda i: run(), 2, log, lambda r: None, stop=cli._AccountGuard()
    )
    assert reason is None and len(log.records) == 6


# -- a run the machine slept through -------------------------------------------


def test_a_run_that_slept_is_an_environment_failure_not_an_agent_result():
    record = run(accuracy=0.0, slept_s=600.0, agent_error="LLMError: stream open past its deadline")
    assert stalls.provider_failure(record) == "machine_slept"
    note = stalls.note([record], [record])
    assert "SLEEP:" in note and "asleep" in note


def test_a_run_that_slept_but_finished_keeps_its_score():
    """Scout-3: one 145s sleep hit six parallel runs that all finished and scored."""
    assert stalls.provider_failure(run(accuracy=1.0, slept_s=145.0)) == ""


def test_a_few_seconds_of_clock_jitter_is_not_sleep():
    assert stalls.provider_failure(run(slept_s=2.0)) == ""


def test_the_runner_records_how_long_a_run_slept(monkeypatch, tmp_path):
    """Wall time runs on through sleep; monotonic time stops. Faked here with a wall
    clock that jumps 900s a reading while the monotonic clock stands still."""
    import itertools

    from bench import runner, tasks
    from bench.configs import get

    wall = itertools.count(1000.0, 900.0)
    monkeypatch.setattr(runner.time, "time", lambda: next(wall))
    monkeypatch.setattr(runner.time, "monotonic", lambda: 5.0)
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: {"totals": {}, "plan": {}})
    monkeypatch.setattr(runner.metrics, "score", lambda *a, **k: {})
    record = runner.run_one(get("mercury-only"), tasks.get("expr"), 0, str(tmp_path))
    assert record["slept_s"] >= 800


def test_runs_already_in_flight_are_kept_when_the_scout_stops():
    import threading

    gate = threading.Event()
    # Every job running before the stop lands: a job a worker had not yet picked
    # up is rightly cancelled as never started, which made this flaky under load.
    started = threading.Barrier(3)
    log = Log()

    def work(i):
        started.wait(5)
        if i == 0:
            return run(agent_error="LLMError: HTTP 402: out of credits")
        gate.wait(1)
        return run(config=f"c{i}")

    def stop(record):
        if stalls.is_account_failure(record):
            gate.set()
            return "account"
        return None

    reason = cli._run_jobs([(i,) for i in range(3)], work, 3, log, lambda r: None, stop=stop)
    assert reason == "account"
    assert len(log.records) == 3


def test_runs_that_finished_before_the_stop_are_logged_once():
    """The in-flight sweep after a stop used to walk every future, so runs that had
    already landed were logged a second time (the call-site scout: 23 lines for 12
    runs)."""
    import threading
    import time

    gate = threading.Event()
    started = threading.Barrier(3)  # see the test above: all three must be running
    log = Log()

    def work(i):
        started.wait(5)
        if i == 0:
            return run(config="first")
        if i == 1:
            time.sleep(0.3)
            return run(config="stopper", agent_error="LLMError: HTTP 402: out of credits")
        gate.wait(2)
        return run(config="in-flight")

    def stop(record):
        if stalls.is_account_failure(record):
            gate.set()
            return "account"
        return None

    reason = cli._run_jobs([(i,) for i in range(3)], work, 3, log, lambda r: None, stop=stop)
    assert reason == "account"
    assert sorted(r["config"] for r in log.records) == ["first", "in-flight", "stopper"]
