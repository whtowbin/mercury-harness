"""The sweep's run time limit and guards (scripts/runlimits.py, scripts/sweepguard.py; §96.2).

The time limit is checked against real processes: a run that starts a long-lived child must
leave nothing alive. The old drivers' limits killed only the direct child, and the harness
underneath kept running.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runlimits = _load("runlimits", ROOT / "scripts" / "runlimits.py")
guard = _load("sweepguard", ROOT / "scripts" / "sweepguard.py")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0) until reaped; ps says whether it is really running.
    ps = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = ps.stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_the_limit_stops_the_run_and_the_child_it_started(tmp_path):
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "run.sh"
    script.write_text(f"#!/bin/sh\nsleep 300 &\necho $! > {pidfile}\nwait\n")
    script.chmod(0o755)
    log = (tmp_path / "run.log").open("w")
    started = time.monotonic()
    finished = runlimits.run([str(script)], timeout_s=1.0, log=log)
    log.close()
    assert not finished
    assert time.monotonic() - started < runlimits.GRACE_S + 10
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child), "the run's child survived the limit"
    assert runlimits.KILLED_MARK in (tmp_path / "run.log").read_text()


def test_a_run_inside_its_limit_is_left_alone(tmp_path):
    log = (tmp_path / "run.log").open("w")
    assert runlimits.run([sys.executable, "-c", "print('ok')"], timeout_s=30, log=log)
    log.close()
    assert runlimits.KILLED_MARK not in (tmp_path / "run.log").read_text()


# -- the watchdog's decision ---------------------------------------------------------------

LIMITS = guard.Limits(total_usd=10, brave_max=2600, stall_s=1200)


def test_the_watchdog_stops_on_each_limit_and_not_before():
    stage = ("gaia", 6.0)
    assert guard.verdict(stage, 5.9, 9.0, 2500, 600, LIMITS) == ""
    assert "gaia" in guard.verdict(stage, 6.1, 9.0, 0, 0, LIMITS)
    assert "sweep spent" in guard.verdict(stage, 1.0, 10.5, 0, 0, LIMITS)
    assert "Brave" in guard.verdict(stage, 1.0, 1.0, 2601, 0, LIMITS)
    assert "stalled" in guard.verdict(stage, 1.0, 1.0, 0, 1300, LIMITS)
    # One unreadable poll is retried; five in a row stop the sweep (a cap it cannot check).
    assert guard.verdict(stage, None, None, 0, 0, LIMITS, unreadable=1) == ""
    assert "unreadable" in guard.verdict(stage, None, None, 0, 0, LIMITS, unreadable=5)


def test_the_balance_is_readable_with_the_real_key():
    """Live: the watchdog's spend cap depends on this read (it once failed silently)."""
    import pytest

    if not guard.api_key():
        pytest.skip("no OpenRouter key on this machine")
    assert guard.balance() is not None


def test_the_current_stage_is_the_last_line_written(tmp_path):
    state = tmp_path / "state"
    state.write_text("fixtures 1.0\nscicode 1.0\n")
    assert guard.current_stage(state) == ("scicode", 1.0)


# -- the scout check ------------------------------------------------------------------------


def _round(tmp_path: Path, logs: dict[str, str], with_totals: set[str]) -> Path:
    for rid, text in logs.items():
        (tmp_path / f"{rid}.log").write_text(text)
        trace = tmp_path / "traces" / rid
        trace.mkdir(parents=True)
        events = [{"kind": "run_start"}]
        if rid in with_totals:
            events.append({"kind": "totals", "cost_usd": 0.01})
        (trace / "run-1.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return tmp_path


def test_scout_classifies_each_way_a_run_can_fail(tmp_path):
    root = _round(
        tmp_path,
        {
            "ok": "done\n",
            "retried": "   ~ plan: 4 attempts over 12s (HTTP 429)\ndone\n",
            "provider": "FAILED: LLMError: provider error: server_error\n",
            "killed": f"working\n{runlimits.KILLED_MARK}\n",
            "unfinished": "working\n",
        },
        with_totals={"ok", "retried", "provider", "killed"},
    )
    why = {r.name: r.failed for r in guard.harness_runs(root)}
    assert why["ok"] == "" and why["retried"] == ""
    assert "LLMError" in why["provider"]
    assert "time limit" in why["killed"]
    assert "never finished" in why["unfinished"]


def test_scout_drops_an_arm_with_too_many_failures_or_too_much_cost():
    ok = [guard.Run(str(i), "", 0.01) for i in range(10)]
    assert guard.scout_verdict(ok, est_usd=0.01, max_fail_frac=0.1)[0]
    one_bad = [*ok[:9], guard.Run("x", "killed at the time limit", 0.01)]
    assert guard.scout_verdict(one_bad, 0.01, 0.1)[0]
    two_bad = [*ok[:8], *[guard.Run(f"x{i}", "crashed", 0.01) for i in range(2)]]
    assert not guard.scout_verdict(two_bad, 0.01, 0.1)[0]
    pricey = [guard.Run(str(i), "", 0.03) for i in range(10)]
    assert not guard.scout_verdict(pricey, 0.01, 0.1)[0]
    assert not guard.scout_verdict([], 0.01, 0.1)[0]


def test_stopping_the_sweep_stops_runs_in_their_own_sessions(tmp_path):
    """The watchdog kills the tree: a run started with runlimits.start sits in a session of its
    own, and killing only the sweep's group left it running with no limit enforced."""
    pidfile = tmp_path / "run.pid"
    sweep = tmp_path / "sweep.py"
    sweep.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "import runlimits\n"
        "p = runlimits.start(['sleep', '300'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    top = subprocess.Popen([sys.executable, str(sweep)], start_new_session=True)
    deadline = time.monotonic() + 10
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    run_pid = int(pidfile.read_text())
    assert os.getpgid(run_pid) != os.getpgid(top.pid)  # the run really is in its own group
    runlimits.kill_tree(top.pid)
    top.wait(timeout=10)
    deadline = time.monotonic() + 5
    while _alive(run_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(run_pid), "a run in its own session survived the sweep's stop"


def test_idle_time_counts_from_the_watchdog_s_start_not_from_old_files():
    """The first smoke was stopped on its first poll: the newest file was from hours before."""
    started = 1_000_000.0
    assert guard.idle_seconds(newest=started - 11_700, started=started, now=started + 60) == 60
    assert guard.idle_seconds(newest=started + 50, started=started, now=started + 60) == 10


def test_failing_searches_stop_the_sweep(tmp_path):
    """A refused Brave search writes no web_search event; GAIA must not run on without search."""
    trace = tmp_path / "q1"
    trace.mkdir()
    events = [{"kind": "tool_call", "tool": "web_search"}] * 3 + [{"kind": "web_search"}]
    (trace / "run-1.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert guard.search_counts([tmp_path]) == (3, 1)
    stage = ("gaia", 8.0)
    assert guard.verdict(stage, 1.0, 1.0, 400, 0, LIMITS, search_failed=20) == ""
    assert guard.verdict(stage, 1.0, 1.0, 400, 0, LIMITS, search_failed=50) == ""  # 12.5%
    assert "refusing" in guard.verdict(stage, 1.0, 1.0, 200, 0, LIMITS, search_failed=40)


def test_an_eperm_from_a_finished_group_is_not_a_crash(tmp_path, monkeypatch):
    """macOS answers EPERM when probing a group whose members exited but are not reaped; that
    crashed swe.py at the sweep's first real kill (§96.3). The run must still be marked killed."""
    real_killpg = os.killpg

    def killpg(pgid, sig):
        if sig == 0:
            raise PermissionError(1, "Operation not permitted")
        return real_killpg(pgid, sig)

    monkeypatch.setattr(runlimits.os, "killpg", killpg)
    log = (tmp_path / "run.log").open("w")
    assert not runlimits.run(["sleep", "30"], timeout_s=0.5, log=log)
    log.close()
    assert runlimits.KILLED_MARK in (tmp_path / "run.log").read_text()
