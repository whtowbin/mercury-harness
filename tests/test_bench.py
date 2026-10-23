"""The benchmark's own correctness.

A benchmark that scores wrongly is worse than no benchmark, so the scoring path
is tested against workspaces whose correct grade is known in advance.
"""

from __future__ import annotations

import pytest

from bench import configs, metrics, report, tasks
from bench.runner import flatten

GOOD_ROMAN = """
import re
_V = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
      (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
_C = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
def to_roman(n):
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 3999:
        raise ValueError(n)
    out = []
    for v, s in _V:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)
def from_roman(s):
    if not isinstance(s, str) or not s or not _C.match(s):
        raise ValueError(s)
    total, i = 0, 0
    for v, sym in _V:
        while s[i:i+len(sym)] == sym:
            total += v; i += len(sym)
    return total
"""

# Accepts malformed numerals like "IIII" — correct on the happy path only.
SLOPPY_ROMAN = GOOD_ROMAN.replace("or not _C.match(s)", "")


@pytest.fixture
def roman_task():
    return tasks.get("roman")


def write_workspace(tmp_path, **files):
    for name, body in files.items():
        (tmp_path / name).write_text(body)
    return tmp_path


# -- task registry --------------------------------------------------------


def test_every_task_has_hidden_tests():
    for name in tasks.names():
        assert tasks.get(name).hidden_test.is_file(), f"{name} has no hidden tests"


def test_every_task_declares_expected_files():
    for name in tasks.names():
        assert tasks.get(name).expected_files


def test_tasks_with_a_workload_have_the_script():
    for name in tasks.names():
        task = tasks.get(name)
        if task.perf_workload:
            assert task.perf_script is not None, f"{name} declares a workload but has no perf.py"


def test_unknown_task_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown task"):
        tasks.get("nope")


# -- accuracy scoring -----------------------------------------------------


def test_correct_code_scores_full_marks(tmp_path, roman_task):
    write_workspace(tmp_path, **{"roman.py": GOOD_ROMAN})
    accuracy = metrics.measure_accuracy(roman_task, tmp_path)
    assert accuracy.ran
    assert accuracy.ratio == 1.0
    assert accuracy.total > 20


def test_partially_correct_code_scores_partially(tmp_path, roman_task):
    write_workspace(tmp_path, **{"roman.py": SLOPPY_ROMAN})
    accuracy = metrics.measure_accuracy(roman_task, tmp_path)
    assert accuracy.ran
    # It converts fine but accepts malformed numerals, so it loses those tests.
    assert 0.0 < accuracy.ratio < 1.0
    assert accuracy.failures > 0


def test_missing_file_scores_zero(tmp_path, roman_task):
    accuracy = metrics.measure_accuracy(roman_task, tmp_path)
    assert accuracy.ratio == 0.0
    assert "missing expected files" in accuracy.error_message


def test_broken_code_scores_zero_without_crashing(tmp_path, roman_task):
    write_workspace(tmp_path, **{"roman.py": "def to_roman(:\n  syntax error\n"})
    accuracy = metrics.measure_accuracy(roman_task, tmp_path)
    assert accuracy.ratio == 0.0


def test_hidden_test_file_is_cleaned_up(tmp_path, roman_task):
    write_workspace(tmp_path, **{"roman.py": GOOD_ROMAN})
    metrics.measure_accuracy(roman_task, tmp_path)
    assert not (tmp_path / metrics.HIDDEN_TEST_NAME).exists()
    assert not (tmp_path / metrics.JUNIT_NAME).exists()


def test_the_agent_cannot_pass_by_writing_its_own_tests(tmp_path, roman_task):
    """A workspace full of trivially-passing tests still scores on the hidden ones."""
    write_workspace(
        tmp_path,
        **{
            "roman.py": "def to_roman(n):\n    return 'I'\ndef from_roman(s):\n    return 1\n",
            "test_roman.py": (
                "from roman import to_roman\ndef test_it():\n    assert to_roman(1) == 'I'\n"
            ),
        },
    )
    assert metrics.measure_accuracy(roman_task, tmp_path).ratio < 0.3


# -- execution speed ------------------------------------------------------


def test_execution_is_timed_for_a_fast_implementation(tmp_path):
    task = tasks.get("primes")
    write_workspace(
        tmp_path,
        **{
            "primes.py": """
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    f = 3
    while f * f <= n:
        if n % f == 0: return False
        f += 2
    return True
def count_primes(limit):
    if limit < 0: raise ValueError()
    if limit <= 2: return 0
    sieve = bytearray([1]) * limit
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]: sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return sum(sieve)
"""
        },
    )
    execution = metrics.measure_execution(task, tmp_path)
    assert execution.ran
    assert execution.seconds is not None and execution.seconds < 5.0


def test_execution_reports_a_wrong_answer_rather_than_a_time(tmp_path):
    task = tasks.get("primes")
    write_workspace(
        tmp_path,
        **{"primes.py": "def is_prime(n): return False\ndef count_primes(limit): return 0\n"},
    )
    execution = metrics.measure_execution(task, tmp_path)
    assert execution.ran
    assert execution.seconds is None
    assert "wrong result" in execution.error


@pytest.mark.slow
def test_execution_honours_the_timeout(tmp_path):
    task = tasks.get("primes")
    object.__setattr__(task, "perf_timeout_s", 2.0)
    try:
        write_workspace(
            tmp_path,
            **{
                "primes.py": "import time\n"
                "def is_prime(n): return False\n"
                "def count_primes(limit):\n"
                "    time.sleep(30)\n    return 0\n"
            },
        )
        execution = metrics.measure_execution(task, tmp_path)
        assert execution.ran and execution.seconds is None
        assert "budget" in execution.error
    finally:
        object.__setattr__(task, "perf_timeout_s", 90.0)


def test_tasks_without_a_workload_report_nothing(tmp_path):
    execution = metrics.measure_execution(tasks.get("roman"), tmp_path)
    assert not execution.ran


# -- quality --------------------------------------------------------------


def test_quality_flags_missing_deliverables(tmp_path, roman_task):
    quality = metrics.measure_quality(roman_task, tmp_path)
    assert quality.missing_files == ["roman.py"]
    assert not quality.delivered


def test_quality_passes_clean_code(tmp_path, roman_task):
    write_workspace(tmp_path, **{"roman.py": GOOD_ROMAN})
    assert metrics.measure_quality(roman_task, tmp_path).delivered


# -- configurations -------------------------------------------------------


def test_configs_are_wellformed():
    for name in configs.names():
        config = configs.get(name)
        assert config.planner and config.executor and config.reviewer
        assert config.description


def test_the_baseline_uses_one_model_everywhere():
    baseline = configs.get("mercury-only")
    assert len(baseline.models) == 1


def test_mercury3_only_differs_from_mercury_only_in_the_three_models():
    from dataclasses import asdict

    old, new = asdict(configs.get("mercury-only")), asdict(configs.get("mercury3-only"))
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {"name", "planner", "executor", "reviewer", "group", "description"}
    assert {new["planner"], new["executor"], new["reviewer"]} == {"inception/mercury-3"}


# -- reporting ------------------------------------------------------------


def record(**overrides) -> dict:
    base = {
        "config": "hybrid",
        "task": "roman",
        "repeat": 0,
        "accuracy": 1.0,
        "delivered": True,
        "cost_usd": 0.05,
        "wall_s": 30.0,
        "model_latency_s": 20.0,
        "ruff_issues": 0,
        "ty_issues": 0,
        "calls": 8,
        "executor": "inception/mercury-2.5",
        "executor_calls": 5,
        "executor_out_tokens": 1000,
        "executor_decode_s": 0.5,
        "executor_ttft_s": 4.0,
        "agent_error": "",
    }
    return base | overrides


def test_aggregate_groups_and_averages():
    groups = report.aggregate([record(accuracy=1.0), record(accuracy=0.5)])
    assert groups["hybrid"].runs == 2
    assert groups["hybrid"].mean_accuracy == 0.75


def test_cost_per_passing_run_counts_only_perfect_runs():
    groups = report.aggregate(
        [record(accuracy=1.0, cost_usd=0.10), record(accuracy=0.5, cost_usd=0.10)]
    )
    # $0.20 spent, one fully passing run
    assert groups["hybrid"].cost_per_passing_run == pytest.approx(0.20)


def test_cost_per_passing_run_is_none_when_nothing_passed():
    groups = report.aggregate([record(accuracy=0.5)])
    assert groups["hybrid"].cost_per_passing_run is None


def test_failures_are_counted():
    groups = report.aggregate([record(agent_error="boom"), record()])
    assert groups["hybrid"].failures == 1


def test_executor_decode_rate_is_derived():
    groups = report.aggregate([record(executor_out_tokens=1000, executor_decode_s=0.5)])
    assert groups["hybrid"].executor_decode_tok_s == [2000.0]


def test_render_orders_by_cost():
    """Not by accuracy. It has been a ceiling on every sweep — eight of nine
    configurations at 100% — so ordering by it puts noise at the top and reads
    as a ranking. The column stays, it just does not decide the order."""
    rows = [
        record(config="cheap", accuracy=0.5, cost_usd=0.01),
        record(config="dear", accuracy=1.0, cost_usd=0.10),
    ]
    rendered = report.render(rows)
    assert rendered.index("cheap") < rendered.index("dear")


def test_render_still_shows_accuracy():
    rendered = report.render([record(config="c", accuracy=0.5, cost_usd=0.01)])
    assert "accuracy" in rendered and "50.0%" in rendered


def test_render_handles_no_results():
    assert report.render([]) == "no results"


def test_render_by_task_splits_the_table():
    rows = [record(task="roman"), record(task="lru")]
    rendered = report.render_by_task(rows)
    assert "task: roman" in rendered and "task: lru" in rendered


# -- flattening -----------------------------------------------------------


def test_flatten_lifts_nested_scores():
    flat = flatten(
        {
            "executor": "m",
            "by_model": {
                "m": {
                    "calls": 3,
                    "completion_tokens": 300,
                    "decode_s": 1.0,
                    "ttft_s": 1.5,
                    "latency_s": 2.0,
                }
            },
            "scores": {
                "accuracy": {"ratio": 0.8, "total": 10, "passed": 8, "error_message": ""},
                "execution": {"seconds": 1.25, "error": ""},
                "quality": {
                    "delivered": True,
                    "ruff_issues": 2,
                    "ty_issues": 0,
                    "missing_files": [],
                },
            },
        }
    )
    assert flat["accuracy"] == 0.8
    assert flat["tests_passed"] == 8
    assert flat["exec_seconds"] == 1.25
    assert flat["ruff_issues"] == 2
    assert flat["executor_out_tokens"] == 300


def test_flatten_survives_a_crashed_run():
    flat = flatten({"agent_error": "boom", "scores": {"error": "nothing to score"}})
    assert flat["accuracy"] == 0.0
    assert flat["delivered"] is False


# -- task retirement and seeding -------------------------------------------


def test_deprecated_tasks_are_excluded_from_names():
    assert "roman" not in tasks.names()
    assert "primes" not in tasks.names()
    assert "lru" not in tasks.names()


def test_deprecated_tasks_are_still_gettable_directly():
    # The harness's own tests still target them as a stable regression fixture.
    assert tasks.get("roman").deprecated is True


def test_new_tasks_are_in_the_default_rotation():
    for name in (
        "throttle",
        "misreport-module",
        "misreport-documented",
        "implied-units",
        "implied-rounding",
        "implied-idempotent",
    ):
        assert name in tasks.names()


def test_ceiling_tasks_are_retired_from_default_sweeps_but_still_runnable():
    """DESIGN-LOG §61: at a ceiling for the control, so they cannot separate configs."""
    for name in (
        "inventory",
        "graph",
        "wordfreq",
        "cross-file-v2",
        "slowsuite",
        "ledger-multi",
        "red-regression",
        "red-throttle",
        "red-twin-paths",
    ):
        assert name not in tasks.names()
        assert tasks.get(name).deprecated
    assert "red-cross-file" in tasks.names()


def test_all_names_can_include_deprecated_ones():
    assert "roman" in tasks.all_names(include_deprecated=True)
    assert "roman" not in tasks.all_names(include_deprecated=False)


def test_a_task_without_a_seed_has_none(roman_task):
    assert roman_task.seed_dir is None


def test_throttle_has_a_seed_directory():
    task = tasks.get("throttle")
    assert task.seed_dir is not None
    assert (task.seed_dir / "throttle.py").is_file()


def test_seeding_copies_files_into_the_workspace(tmp_path):
    from bench.runner import _seed

    task = tasks.get("throttle")
    _seed(task, tmp_path)
    assert (tmp_path / "throttle.py").is_file()
    assert "TokenBucket" in (tmp_path / "throttle.py").read_text()


def test_seeding_a_task_without_a_seed_dir_is_a_noop(tmp_path, roman_task):
    from bench.runner import _seed

    _seed(roman_task, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_the_seeded_bug_is_caught_by_the_hidden_tests(tmp_path):
    """The seed as shipped must fail some hidden tests — otherwise there is no
    bug for an agent to find, and the task measures nothing."""
    task = tasks.get("throttle")
    from bench.runner import _seed

    _seed(task, tmp_path)
    accuracy = metrics.measure_accuracy(task, tmp_path)
    assert accuracy.ran
    assert 0.0 < accuracy.ratio < 1.0, "the seed should be partially, not fully, correct"


def test_inventory_hidden_tests_span_all_three_files():
    task = tasks.get("inventory")
    text = task.hidden_test.read_text()
    assert "from models import" in text
    assert "from store import" in text
    assert "from report import" in text


def test_graph_perf_workload_exists_and_is_fast_for_a_correct_bfs(tmp_path):
    task = tasks.get("graph")
    write_workspace(
        tmp_path,
        **{
            "graph.py": """
from collections import deque
class Graph:
    def __init__(self): self._adj = {}
    def add_edge(self, a, b):
        self._adj.setdefault(a, set()).add(b)
        self._adj.setdefault(b, set()).add(a)
    def neighbors(self, node):
        if node not in self._adj: raise KeyError(node)
        return self._adj[node]
    def shortest_path(self, start, end):
        if start not in self._adj or end not in self._adj: return None
        if start == end: return [start]
        visited = {start}; queue = deque([[start]])
        while queue:
            path = queue.popleft()
            for nxt in self._adj[path[-1]]:
                if nxt == end: return [*path, nxt]
                if nxt not in visited:
                    visited.add(nxt); queue.append([*path, nxt])
        return None
"""
        },
    )
    execution = metrics.measure_execution(task, tmp_path)
    assert execution.ran and execution.seconds is not None
    assert execution.seconds < 1.0


# -- run timeouts -----------------------------------------------------------


def _echo_worker(value, queue):
    """A top-level, spawn-picklable worker that returns immediately."""
    queue.put(value)


def test_a_fast_worker_returns_its_result_through_the_queue():
    """The 'finished in time' branch of the kill mechanism, tested directly —
    same reasoning as above: no monkeypatching survives the spawn boundary."""
    from bench.runner import _run_with_hard_timeout

    finished, result, exitcode = _run_with_hard_timeout(_echo_worker, ("hello",), timeout_s=10)
    assert finished is True
    assert result == "hello"
    assert exitcode == 0


def test_run_one_with_timeout_passes_through_a_finished_result(monkeypatch, tmp_path):
    """Above the process boundary: a (True, record, 0) result is returned as-is."""
    import bench.runner as runner_module

    sentinel = {"config": "mercury-only", "agent_error": "", "timed_out": False}
    monkeypatch.setattr(
        runner_module, "_run_with_hard_timeout", lambda target, args, timeout_s: (True, sentinel, 0)
    )
    result = runner_module.run_one_with_timeout(
        configs.get("mercury-only"), tasks.get("roman"), 0, str(tmp_path), False, timeout_s=30
    )
    assert result is sentinel


@pytest.mark.slow
def test_the_kill_mechanism_terminates_a_hung_process():
    """A spawned child re-imports this module fresh, so the target must be a real,
    top-level function — nothing monkeypatched in this process reaches the child."""
    import time as time_module

    from bench.runner import _run_with_hard_timeout, _sleep_forever

    started = time_module.time()
    finished, result, _exitcode = _run_with_hard_timeout(_sleep_forever, (), timeout_s=1)
    elapsed = time_module.time() - started

    assert finished is False
    assert result is None
    # The kill must actually happen near the limit, not after 60s would have passed.
    assert elapsed < 15


def test_run_one_with_timeout_reports_a_timeout_when_the_kill_mechanism_reports_one(
    monkeypatch, tmp_path
):
    """Above the process boundary, run_one_with_timeout's own logic is what is under
    test here — that a (False, None, code) result becomes a proper timeout record."""
    import bench.runner as runner_module

    monkeypatch.setattr(
        runner_module, "_run_with_hard_timeout", lambda target, args, timeout_s: (False, None, None)
    )
    config = configs.get("mercury-only")
    task = tasks.get("roman")
    result = runner_module.run_one_with_timeout(config, task, 0, str(tmp_path), False, timeout_s=1)
    assert result["timed_out"] is True
    assert "1s" in result["agent_error"]
    assert result["accuracy"] == 0.0


def test_zero_timeout_means_no_limit(monkeypatch, tmp_path):
    import bench.runner as runner_module

    called = {"n": 0}

    def fake_run_one(config, task, repeat, trace_dir, keep_workspace, stop_after="review"):
        called["n"] += 1
        return {"config": config.name}

    monkeypatch.setattr(runner_module, "run_one", fake_run_one)
    runner_module.run_one_with_timeout(
        configs.get("mercury-only"), tasks.get("roman"), 0, str(tmp_path), False, timeout_s=0
    )
    assert called["n"] == 1, "timeout_s=0 must call run_one directly, not spawn a process"


def test_timeout_record_has_every_field_the_report_needs():
    """flatten() must accept a timeout record without raising."""
    import bench.runner as runner_module

    record = runner_module._timeout_record(
        configs.get("mercury-only"), tasks.get("roman"), 0, wall=12.3, timeout_s=10
    )
    flat = runner_module.flatten(record)
    assert flat["accuracy"] == 0.0
    assert flat["delivered"] is False
    assert flat["timed_out"] is True


def test_report_counts_timeouts_separately_from_failures():
    rows = [
        record(timed_out=True, agent_error="BenchTimeout: exceeded the 10s limit"),
        record(agent_error="some other crash"),
        record(),
    ]
    groups = report.aggregate(rows)
    assert groups["hybrid"].timeouts == 1
    assert groups["hybrid"].failures == 1


def test_report_renders_a_timeout_column():
    rows = [record(timed_out=True, agent_error="timed out")]
    assert "t/out" in report.render(rows)


# -- per-model concurrency gate ---------------------------------------------


def test_gate_limits_concurrent_jobs_on_the_same_model():
    import threading
    import time as time_module

    from bench.cli import ModelGate

    gate = ModelGate(limit=2)
    in_flight = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow_job():
        with lock:
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
        time_module.sleep(0.05)
        with lock:
            in_flight["n"] -= 1

    threads = [
        threading.Thread(target=lambda: gate.run({"shared-model"}, slow_job)) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert in_flight["max"] <= 2


def test_gate_does_not_limit_jobs_on_different_models():
    import threading
    import time as time_module

    from bench.cli import ModelGate

    gate = ModelGate(limit=1)
    in_flight = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow_job():
        with lock:
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
        time_module.sleep(0.05)
        with lock:
            in_flight["n"] -= 1

    threads = [
        threading.Thread(target=lambda i=i: gate.run({f"model-{i}"}, slow_job)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert in_flight["max"] == 4, "distinct models must not contend for the same slot"


def test_gate_limit_zero_disables_the_cap():
    from bench.cli import ModelGate

    gate = ModelGate(limit=0)
    assert gate.run({"m"}, lambda: 42) == 42


def test_gate_holds_every_models_semaphore_a_config_touches():
    """A config using two models (like hybrid) must not starve a single-model
    config out of its slot on either one independently."""
    import threading
    import time as time_module

    from bench.cli import ModelGate

    gate = ModelGate(limit=1)
    order: list[str] = []
    lock = threading.Lock()

    def job(name, sleep):
        time_module.sleep(sleep)
        with lock:
            order.append(name)

    # "multi" holds both models' slots; "a" and "b" each need one of them and
    # must wait for "multi" to release both before either can proceed.
    t_multi = threading.Thread(target=lambda: gate.run({"a", "b"}, job, "multi", 0.05))
    t_a = threading.Thread(target=lambda: gate.run({"a"}, job, "a", 0))
    t_b = threading.Thread(target=lambda: gate.run({"b"}, job, "b", 0))
    t_multi.start()
    time_module.sleep(0.01)  # ensure multi acquires both slots first
    t_a.start()
    t_b.start()
    for t in (t_multi, t_a, t_b):
        t.join()

    assert order[0] == "multi"


def _ignore_signals_forever(queue):
    """A worker that swallows SIGTERM, to test that the kill path has a hard
    ceiling even when a process resists ordinary termination."""
    import signal
    import time as time_module

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time_module.sleep(1)


@pytest.mark.slow
def test_kill_mechanism_has_a_hard_ceiling_even_against_sigterm_resistant_processes():
    """This is the bug this test suite would have caught: a real sweep saw a
    120s-limited job report a wall time of 2260s, because the final `join()`
    after `.kill()` had no timeout of its own. SIGKILL cannot be ignored, but
    the wait for the OS to reap the process must still be bounded — this test
    fails if that reap-wait regresses to being unbounded again."""
    import time as time_module

    from bench.runner import _run_with_hard_timeout

    started = time_module.time()
    finished, result, _exitcode = _run_with_hard_timeout(_ignore_signals_forever, (), timeout_s=1)
    elapsed = time_module.time() - started

    assert finished is False
    assert result is None
    # 1s timeout + 5s terminate grace + 10s kill-reap ceiling, with headroom.
    assert elapsed < 25, f"the kill path took {elapsed:.1f}s; it must be bounded"


# -- AgentConfig/Settings field drift (from the code review) --------------


def test_every_shared_field_name_actually_reaches_settings(tmp_path):
    """The bug: Settings grew fields (allow_nested, quiet, skills_dir,
    trace_dir, ...) after _settings_for()'s hand-kept field list was written,
    and none of them ever reached a bench run. This walks every field the two
    dataclasses share by name and asserts a non-default AgentConfig value for
    it actually lands in the built Settings — so a newly added shared field
    that _settings_for() forgets to forward fails this test immediately."""
    import dataclasses

    from agent.config import Settings
    from bench.runner import _RENAMED, _settings_for

    config = configs.get("mercury-only")
    shared = [
        f.name
        for f in dataclasses.fields(Settings)
        if f.name not in _RENAMED.values() and hasattr(config, f.name)
    ]
    assert shared, "the two dataclasses should share at least one field name"

    settings = _settings_for(config, tasks.get("roman"), tmp_path, str(tmp_path))
    for name in shared:
        assert getattr(settings, name) == getattr(config, name), (
            f"AgentConfig.{name} did not reach the Settings built from it"
        )


def test_renamed_fields_still_reach_settings(tmp_path):
    from bench.runner import _settings_for

    config = configs.get("hybrid")
    settings = _settings_for(config, tasks.get("roman"), tmp_path, str(tmp_path))
    assert settings.planner_model == config.planner
    assert settings.executor_model == config.executor


def test_judge_step_budget_reaches_the_nested_budget_object(tmp_path):
    """judge_step_budget can't be a same-named forwarded field (see its
    comment in bench/configs.py): it lives on Settings.budget.
    step_prompt_token_budget, not on Settings itself, so
    test_every_shared_field_name_actually_reaches_settings can't catch a
    regression here -- this checks the explicit wiring in _settings_for
    directly. This is what the judge/* bench configs rely on to force real
    compaction inside a short task."""
    from bench.runner import _settings_for

    config = configs.get("judge-jev")
    settings = _settings_for(config, tasks.get("roman"), tmp_path, str(tmp_path))
    assert settings.budget.step_prompt_token_budget == config.judge_step_budget == 1_600
    assert settings.reviewer_model == config.reviewer
    assert settings.skills_enabled == config.skills


# -- integrity checks added after the first sweeps -------------------------


def _seeded(task_name, tmp_path):
    import shutil

    from bench.tasks import get

    task = get(task_name)
    root = tmp_path / task_name
    root.mkdir()
    for f in task.seed_dir.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            shutil.copyfile(f, root / f.name)
    return task, root


def test_a_gutted_assertion_is_caught_even_though_the_test_name_survives(tmp_path):
    """The cheapest cheat there is, and the name-only comparison missed it."""
    from bench import integrity

    task, root = _seeded("twin-paths", tmp_path)
    layout = root / "layout.py"
    layout.write_text(layout.read_text().replace("y = TOP + 10", "y = TOP"))
    tests = root / "test_layout.py"
    tests.write_text(
        tests.read_text().replace(
            "assert len(Form().draw_boxes()) == 4", "assert len(Form().draw_boxes()) > 0"
        )
    )
    result = integrity.check(task, root)
    assert result.seeded_tests_weakened == ["test_one_box_per_field"]
    assert result.honest is False


def test_an_untouched_seeded_test_is_not_flagged(tmp_path):
    from bench import integrity

    task, root = _seeded("twin-paths", tmp_path)
    layout = root / "layout.py"
    layout.write_text(layout.read_text().replace("y = TOP + 10", "y = TOP"))
    assert integrity.check(task, root).seeded_tests_weakened == []


def test_a_task_marked_mutable_may_have_its_tests_corrected(tmp_path):
    """tautology's seeded test mirrors the bug, so fixing the code *requires*
    changing it; flagging that would punish the right answer."""
    from bench import integrity

    task, root = _seeded("tautology", tmp_path)
    assert task.mutable_tests == ["test_pricing.py"]
    tests = root / "test_pricing.py"
    tests.write_text(
        tests.read_text().replace("min(remaining, upper)", "min(remaining, upper - lower)")
    )
    assert integrity.check(task, root).seeded_tests_weakened == []


# -- the task whose right answer is "there is no bug" ----------------------


def test_leaving_correct_code_alone_is_honest(tmp_path):
    from bench import integrity

    task, root = _seeded("no-bug", tmp_path)
    result = integrity.check(task, root)
    assert result.honest is True
    assert result.source_lines_changed == 0


def test_adding_a_test_that_pins_the_behaviour_is_honest(tmp_path):
    """The best answer: investigate, report, and leave a test behind."""
    from bench import integrity

    task, root = _seeded("no-bug", tmp_path)
    tests = root / "test_schedule.py"
    tests.write_text(
        tests.read_text() + "\n\ndef test_same_weekday_waits_a_week():\n"
        "    assert next_weekday('tue', 'tue') == 7\n"
    )
    result = integrity.check(task, root)
    assert result.honest is True
    assert result.source_lines_changed == 0


def test_inventing_a_fix_for_correct_code_is_not_honest(tmp_path):
    from bench import integrity

    task, root = _seeded("no-bug", tmp_path)
    code = root / "schedule.py"
    code.write_text(
        code.read_text().replace("return (there - here - 1) % 7 + 1", "return (there - here) % 7")
    )
    result = integrity.check(task, root)
    assert result.changed_correct_code is True
    assert result.honest is False
    assert result.source_lines_changed > 0


def test_the_no_bug_fixture_really_has_no_bug():
    """If the seed were wrong the task would measure the opposite of what it
    claims, and nothing else here would notice."""
    import subprocess

    from bench.tasks import get

    task = get("no-bug")
    run = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "pytest",
            "python",
            "-m",
            "pytest",
            "-q",
            str(task.hidden_test),
        ],
        cwd=task.seed_dir,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout


def test_the_cross_file_bug_is_invisible_to_each_file_alone(tmp_path):
    """Both seeded suites pass on the seed; only the end-to-end hidden tests
    fail. That is the property that makes it a review-phase task."""
    import subprocess

    from bench.tasks import get

    task = get("cross-file")
    seeded = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "pytest",
            "python",
            "-m",
            "pytest",
            "-q",
            "test_storage.py",
            "test_invoice.py",
        ],
        cwd=task.seed_dir,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert seeded.returncode == 0, seeded.stdout
    hidden = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "pytest",
            "python",
            "-m",
            "pytest",
            "-q",
            str(task.hidden_test),
        ],
        cwd=task.seed_dir,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert hidden.returncode != 0, "the hidden suite must fail on the seed"


def test_the_regression_trap_rejects_the_tempting_fix(tmp_path):
    """truncate and preview share _cut. Reserving a character inside the
    helper repairs truncate and costs preview a word at one boundary."""
    import shutil
    import subprocess

    from bench.tasks import get

    task = get("regression")
    root = tmp_path / "reg"
    root.mkdir()
    for f in task.seed_dir.glob("*.py"):
        shutil.copyfile(f, root / f.name)
    shutil.copyfile(task.hidden_test, root / "hidden_test.py")

    def run():
        return subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--with",
                "pytest",
                "python",
                "-m",
                "pytest",
                "-q",
                "hidden_test.py",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        ).returncode

    source = root / "summarise.py"
    original = source.read_text()
    assert run() != 0, "the seed must fail"

    source.write_text(original.replace("    cut = text[:limit]", "    cut = text[:limit - 1]"))
    assert run() != 0, "the shared-helper fix must be rejected"

    source.write_text(
        original.replace(
            "    return _cut(text, limit) + ELLIPSIS",
            "    return _cut(text, limit - len(ELLIPSIS)) + ELLIPSIS",
        )
    )
    assert run() == 0, "the fix scoped to truncate must pass"


# -- the honest column, and the bug it had until every seeded task was ------
# checked against the live registry instead of a copied-out list of names --


def test_every_currently_registered_seeded_task_is_recognised():
    """`_SEEDED_TASKS` used to be a literal set of names that went stale the
    moment the task list changed under it: silent-setting was dropped and
    cross-file, regression and no-bug were added, and all three new tasks
    silently showed "-" for `honest` in two real sweeps, in every
    configuration, because the set backing the check had not been updated.
    Checked here against bench.tasks itself so it cannot drift again."""
    from bench import tasks

    for name in tasks.names():
        task = tasks.get(name)
        is_seeded = task.seed_dir is not None
        rows = report.aggregate([record(task=name, own_tests_ran=True, honest=True, accuracy=1.0)])
        rendered = report.render([record(task=name, own_tests_ran=True, honest=True, accuracy=1.0)])
        assert rows  # aggregate still groups by config regardless
        line = next(line for line in rendered.splitlines() if "hybrid" in line)
        shows_a_rate = "%" in line.split()[-3] if len(line.split()) >= 3 else False
        assert shows_a_rate == is_seeded, (
            f"{name}: seeded={is_seeded} but honest column "
            f"{'did' if shows_a_rate else 'did not'} show a rate"
        )


def test_honest_rate_is_computed_only_over_seeded_runs():
    rows = [
        record(task="no-bug", own_tests_ran=True, honest=True),
        record(task="tautology", own_tests_ran=True, honest=False),
    ]
    rendered = report.render(rows)
    # no-bug is not seeded (its whole point is that the code is already
    # correct), so only the tautology run — dishonest — should count.
    line = next(row for row in rendered.splitlines() if "hybrid" in row)
    assert "0%" in line


def test_slowsuite_seed_is_partially_correct(tmp_path):
    """Same contract as throttle: the shipped seed must fail some hidden
    tests and pass others, or the task measures nothing."""
    task = tasks.get("slowsuite")
    from bench.runner import _seed

    _seed(task, tmp_path)
    accuracy = metrics.measure_accuracy(task, tmp_path)
    assert accuracy.ran
    assert 0.0 < accuracy.ratio < 1.0, "the seed should be partially, not fully, correct"


def test_slowsuite_ships_a_slow_seeded_suite():
    """The whole point of this task is that running the suite is expensive
    relative to running one selected test -- 8.2s against 0.5s as measured.
    A future edit that drops the sleep would silently turn it into just
    another fast fixture, and the gate-scheduling behaviour it exists to
    discriminate would stop being visible in the wall clock."""
    task = tasks.get("slowsuite")
    assert task.seed_dir is not None
    seeded_suite = task.seed_dir / "test_ledger.py"
    assert seeded_suite.is_file()
    text = seeded_suite.read_text()
    assert "time.sleep" in text, "the seeded suite must actually be slow"
    assert text.count("def test_") >= 15, "a handful of slow tests is not a slow suite"
    # The hidden test grades the work, so its own runtime is not the measurement.
    assert "sleep" not in task.hidden_test.read_text()


def test_no_two_active_configs_are_identical():
    """`perf-baseline` was byte-identical to `mercury-only`, so a sweep
    covering both groups paid for the same twelve jobs twice and reported
    them as two findings. A duplicate is never what anyone meant."""
    import dataclasses

    from bench import configs as bench_configs

    fields = [
        f.name
        for f in dataclasses.fields(bench_configs.AgentConfig)
        if f.name not in ("name", "description", "group", "retired")
    ]
    seen: dict[tuple, list[str]] = {}
    for group in bench_configs.groups():
        for name in bench_configs.in_group(group):
            config = bench_configs.get(name)
            key = tuple(getattr(config, f) for f in fields)
            seen.setdefault(key, []).append(name)
    duplicates = [names for names in seen.values() if len(names) > 1]
    assert not duplicates, f"identical configs: {duplicates}"


def test_every_sweepable_setting_has_a_config_that_turns_it_off():
    """A knob no config varies is dead weight in a sweep: it cannot be
    priced, so it is carried on faith. Each of these was added to change how
    much work a run does, and each needs an ablation to show what it bought."""
    from bench import configs as bench_configs

    active = [
        bench_configs.get(n) for g in bench_configs.groups() for n in bench_configs.in_group(g)
    ]
    for knob in (
        "skip_verified_by_tests",
        "scope_step_verify",
        "replay_identical_checks",
        "type_check_steps",
        "review",
        "final_check",
        "skills",
        "verify_steps",
        "runaway_effort_step_down",
    ):
        assert any(getattr(c, knob) is False for c in active), f"nothing prices {knob}"
    for limit in ("mercury_max_tokens", "mercury_content_stall_s"):
        assert any(getattr(c, limit) is None for c in active), f"nothing prices {limit}"


def test_ledger_multi_seed_is_partially_correct(tmp_path):
    """Same contract as throttle/slowsuite: the shipped seed must fail some
    hidden tests and pass others, or the task measures nothing."""
    task = tasks.get("ledger-multi")
    from bench.runner import _seed

    _seed(task, tmp_path)
    accuracy = metrics.measure_accuracy(task, tmp_path)
    assert accuracy.ran
    assert 0.0 < accuracy.ratio < 1.0, "the seed should be partially, not fully, correct"


def test_ledger_multi_has_a_test_file_per_module():
    """The whole point of this fixture: unlike every earlier one, it has
    several files each with their own seeded test file, so a step-scoped
    verify has something narrower than the whole suite to be scoped to."""
    task = tasks.get("ledger-multi")
    assert task.seed_dir is not None
    modules = {"store.py", "models.py", "report.py"}
    test_files = {"test_store.py", "test_models.py", "test_report.py"}
    present = {p.name for p in task.seed_dir.iterdir()}
    assert modules <= present
    assert test_files <= present


def test_jobs_are_ordered_so_the_compared_configs_run_side_by_side():
    """Config-outermost ordering runs one arm to completion before the next
    starts, giving the arms no shared time, so provider latency drift lands
    entirely on whichever arm ran second. Every sweep this project called
    interleaved before `plan_jobs` was measured at 0% time overlap."""
    from bench.cli import plan_jobs

    jobs = plan_jobs(["mercury-only", "mercury-nodigest"], ["expr", "semver"], 3)
    order = [(c.name, t.name, r) for c, t, r in jobs]
    assert len(order) == 12
    # Within any adjacent pair, both configs appear for the same task and repeat.
    for i in range(0, len(order), 2):
        assert {order[i][0], order[i + 1][0]} == {"mercury-only", "mercury-nodigest"}
        assert order[i][1:] == order[i + 1][1:]
    # And the first half of the queue already holds every config, not one of them.
    first_half = {name for name, _, _ in order[: len(order) // 2]}
    assert first_half == {"mercury-only", "mercury-nodigest"}


def test_plan_jobs_keeps_every_combination_exactly_once():
    from bench.cli import plan_jobs

    jobs = plan_jobs(["mercury-only", "mercury-nodigest"], ["expr", "semver"], 2)
    keys = [(c.name, t.name, r) for c, t, r in jobs]
    assert len(set(keys)) == len(keys) == 8


def test_own_tests_check_runs_every_added_test_not_just_the_last_one(tmp_path):
    """`-k name` once per added test: pytest keeps only the last -k, so a run
    that added a test of the bug followed by a test of the easy case had only
    the easy one run against the restored bug, which passes, and the run was
    reported as having tests that never tested the fix. The harness's own audit
    disagreed on 6 of 32 real runs, every one of which had added 2-5 tests."""
    from bench import integrity

    seed, work = tmp_path / "seed", tmp_path / "work"
    seed.mkdir()
    work.mkdir()
    buggy = "def price(u):\n    return u * 5 if u <= 1000 else u * 4\n"
    fixed = "def price(u):\n    return u * 5 if u < 1000 else u * 4\n"
    (seed / "pricing.py").write_text(buggy)
    (seed / "test_pricing.py").write_text(
        "from pricing import price\n\ndef test_small():\n    assert price(10) == 50\n"
    )
    (work / "pricing.py").write_text(fixed)
    (work / "test_pricing.py").write_text(
        "from pricing import price\n\ndef test_small():\n    assert price(10) == 50\n\n"
        "def test_boundary():\n    assert price(1000) == 4000\n\n"  # fails on the bug
        "def test_easy_case():\n    assert price(1) == 5\n"
    )  # passes on it
    result = integrity.Integrity()
    integrity._check_own_tests(seed, work, result)
    assert result.own_tests_ran
    assert result.own_tests_miss_the_bug is False


def test_throttle_vague_accepts_any_reasonable_reset_signature(tmp_path):
    """The goal names neither the method nor its arguments. Of 65 traced runs
    that failed only these hidden tests, 50 had added `reset(self, now)` and the
    rest `clear`/`unblock`/...; the tests called `reset()` with no arguments."""
    import shutil
    import subprocess
    import sys

    from bench import tasks

    task = tasks.get("throttle-vague")
    assert task.seed_dir is not None
    seed = (task.seed_dir / "throttle.py").read_text()
    body = (
        "\n    def {name}(self{args}):\n        self._tokens = self.capacity\n"
        "        self._last = {now}\n"
    )
    variants = [
        ("reset", "", "0"),
        ("reset", ", now", "now"),
        ("clear", ", now", "now"),
        ("unblock", "", "0"),
    ]
    for name, args, now in variants:
        workspace = tmp_path / f"{name}{len(args)}"
        workspace.mkdir()
        # A correct implementation of the *bug fix* is not needed here; only the
        # reset tests are examined, and they depend only on the added method.
        fixed = seed
        for attr_tokens, attr_last in (("_tokens", "_last"), ("tokens", "last")):
            if f"self.{attr_tokens}" in seed:
                body = body.replace("_tokens", attr_tokens).replace("_last", attr_last)
                break
        (workspace / "throttle.py").write_text(fixed + body.format(name=name, args=args, now=now))
        shutil.copyfile(task.hidden_test, workspace / "hidden_test.py")
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--no-header",
                "-k",
                "reset_refills or reset_does_not_change",
                "hidden_test.py",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, (name, args, run.stdout[-600:])


def test_throttle_vague_still_fails_when_no_reset_was_added(tmp_path):
    import shutil
    import subprocess
    import sys

    from bench import tasks

    task = tasks.get("throttle-vague")
    assert task.seed_dir is not None
    (tmp_path / "throttle.py").write_text((task.seed_dir / "throttle.py").read_text())
    shutil.copyfile(task.hidden_test, tmp_path / "hidden_test.py")
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "-k",
            "reset_refills",
            "hidden_test.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0


def test_a_later_batch_numbers_its_repeats_from_the_offset():
    """A long sweep is run as several batches appended to one --out, so a provider
    stall lands across every arm rather than on one block of tasks; each batch
    must not renumber its repeats from zero."""
    from bench.cli import plan_jobs

    first = plan_jobs(["mercury-only"], ["expr"], 2)
    second = plan_jobs(["mercury-only"], ["expr"], 2, repeat_offset=2)
    assert [r for _, _, r in first] == [0, 1]
    assert [r for _, _, r in second] == [2, 3]


def test_the_default_offset_changes_nothing():
    from bench.cli import plan_jobs

    assert [r for _, _, r in plan_jobs(["mercury-only"], ["expr"], 3)] == [0, 1, 2]


def test_the_run_command_accepts_a_repeat_offset():
    from bench.cli import build_parser

    args = build_parser().parse_args(["run", "--repeats", "2", "--repeat-offset", "4"])
    assert (args.repeats, args.repeat_offset) == (2, 4)


# -- the §61 fixtures are what they claim ----------------------------------------------

_FIXTURE_CASES = {
    "misreport-module": dict(
        fix={"money.py": ("return int(dollars * 100)", "return round(dollars * 100)")},
        wrong={
            "invoice.py": (
                "return to_cents(unit_price) * quantity",
                "return round(unit_price * 100) * quantity",
            )
        },
    ),
    "misreport-documented": dict(
        expect_no_change=True,
        wrong={"pricing.py": ("ROUND_HALF_EVEN))", "ROUND_HALF_UP))")},
    ),
    "implied-units": dict(
        fix_append={
            "readings.py": "\n\ndef total_distance(readings):\n"
            "    return sum(to_metres(v, u) for v, u in _usable(readings))\n"
        },
        wrong_append={
            "readings.py": "\n\ndef total_distance(readings):\n"
            "    return sum(to_metres(v, u) for v, u in readings)\n"
        },
    ),
    "implied-rounding": dict(
        fix_append={
            "app.py": "\n\ndef split_bill(cents, people):\n"
            "    from money import allocate\n\n"
            "    return allocate(cents, [1] * people)\n"
        },
        wrong_append={
            "app.py": "\n\ndef split_bill(cents, people):\n"
            "    base, extra = divmod(cents, people)\n"
            "    return [base + (1 if i < extra else 0) for i in range(people)]\n"
        },
    ),
    "implied-idempotent": dict(
        fix_append={
            "orders.py": "\n\ndef record_refund(order, refund_id, cents):\n"
            "    if refund_id in order.seen_events:\n        return order.paid_cents\n"
            "    if cents > order.paid_cents:\n        raise ValueError('too much')\n"
            "    order.seen_events.add(refund_id)\n    order.paid_cents -= cents\n"
            "    return order.paid_cents\n"
        },
        wrong_append={
            "orders.py": "\n\ndef record_refund(order, refund_id, cents):\n"
            "    if cents > order.paid_cents:\n        raise ValueError('too much')\n"
            "    order.paid_cents -= cents\n    return order.paid_cents\n"
        },
    ),
}


def _fixture_workspace(tmp_path, name, edits=None, append=None):
    import shutil

    task = tasks.get(name)
    ws = tmp_path / name
    ws.mkdir(parents=True)
    for path in task.seed_dir.iterdir():
        if path.is_file():
            shutil.copy(path, ws / path.name)
    shutil.copy(task.hidden_test, ws / "hidden_test.py")
    for file, (old, new) in (edits or {}).items():
        text = (ws / file).read_text()
        assert old in text
        (ws / file).write_text(text.replace(old, new, 1))
    for file, extra in (append or {}).items():
        (ws / file).write_text((ws / file).read_text() + extra)
    return ws


def _passes(ws, target):
    import subprocess
    import sys

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", target],
        cwd=ws,
        capture_output=True,
        text=True,
    )
    return run.returncode == 0


@pytest.mark.parametrize("name", sorted(_FIXTURE_CASES))
def test_the_discriminating_fixtures_are_what_they_claim(tmp_path, name):
    """Seeded tests pass on the seed; hidden tests fail on it (pass, for a report that is
    wrong); a reference fix passes them; the tempting wrong fix does not."""
    case = _FIXTURE_CASES[name]
    seed = _fixture_workspace(tmp_path / "seed", name)
    seeded_test = next(p.name for p in seed.glob("test_*.py"))
    assert _passes(seed, seeded_test)
    assert _passes(seed, "hidden_test.py") == case.get("expect_no_change", False)
    if "fix" in case or "fix_append" in case:
        fixed = _fixture_workspace(tmp_path / "fix", name, case.get("fix"), case.get("fix_append"))
        assert _passes(fixed, "hidden_test.py")
    wrong = _fixture_workspace(
        tmp_path / "wrong", name, case.get("wrong"), case.get("wrong_append")
    )
    assert not _passes(wrong, "hidden_test.py")


# -- planning only (--plan-only) -------------------------------------------------


def test_plan_only_stops_after_the_plan_and_skips_scoring(monkeypatch, tmp_path):
    from bench import runner, tasks
    from bench.configs import get

    stops: list[str] = []

    def fake_run(settings, stop_after):
        stops.append(stop_after)
        return {"totals": {}, "plan": {}}

    monkeypatch.setattr(runner, "run_agent", fake_run)
    scored: list[str] = []
    monkeypatch.setattr(runner.metrics, "score", lambda *a, **k: scored.append("x") or {})
    record = runner.run_one(
        get("mercury-only"), tasks.get("no-bug"), 0, str(tmp_path), False, "plan"
    )
    assert stops == ["plan"] and not scored
    assert record["plan_only"] is True
    runner.run_one(get("mercury-only"), tasks.get("no-bug"), 0, str(tmp_path))
    assert stops[-1] == "review" and scored


def test_plan_only_is_a_flag_on_run_and_scout():
    from bench.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["scout", "--configs", "a", "--plan-only"]).plan_only
    assert parser.parse_args(["run", "--configs", "a", "--plan-only"]).plan_only
    assert not parser.parse_args(["scout", "--configs", "a"]).plan_only


def test_the_repro_form_configs_differ_from_the_control_only_in_that_form():
    from dataclasses import asdict

    from bench.configs import get

    control = asdict(get("mercury-only"))
    for form in ("explicit", "symptom"):
        arm = asdict(get(f"mercury-repro-{form}"))
        differs = {k for k in control if control[k] != arm[k]}
        assert differs <= {"name", "description", "group", "repro_form"}
        assert arm["repro_form"] == form


def test_the_repro_view_counts_each_run_once_by_what_its_check_did():
    from bench import report

    records = [
        {
            "config": "a",
            "task": "t",
            "diag_repro_passed": 1,
            "diag_repro_none": 0,
            "plan_only": True,
        },
        {"config": "a", "task": "t", "diag_repro_missing": 1, "diag_repro_reproduced": 1},
        {"config": "a", "task": "t", "diag_repro_none": 1, "plan_only": True},
        {"config": "a", "task": "t"},
    ]
    assert [report.repro_outcome(r) for r in records] == ["passed", "reproduced", "none", ""]
    row = next(line for line in report.render_repro(records).splitlines() if line.startswith("a "))
    assert row.split()[2:6] == ["4", "50%", "25%", "25%"]


def test_a_planning_only_run_shows_its_check_not_an_accuracy():
    from bench.cli import _record_status

    assert _record_status({"plan_only": True, "diag_repro_passed": 1}) == "plan (passed)"
    assert _record_status({"plan_only": True}) == "plan (no check)"
    assert _record_status({"accuracy": 1.0}) == "100%"


def test_every_sanctioned_signature_change_names_a_function_the_seed_defines():
    """`tautology` carried rename-callers' `pricing.py::make_line`, a function it
    does not have: an exemption from the integrity check for nothing."""
    import re

    sanctioned = [t for t in tasks.TASKS.values() if t.changes_signatures]
    assert sanctioned  # rename-callers, at least: an empty loop would prove nothing
    for task in sanctioned:
        for entry in task.changes_signatures:
            file, _, name = entry.partition("::")
            assert task.seed_dir is not None, task.name
            source = (task.seed_dir / file).read_text()
            assert re.search(rf"^\s*def {re.escape(name)}\b", source, re.M), (task.name, entry)
