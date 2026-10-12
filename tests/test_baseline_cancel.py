"""A run that ends before reading its pre-run test baseline stops the baseline's suite
(DESIGN-LOG §91, finding 4).

Real processes throughout: what is under test is that a process and the children it
started are gone, which a mocked subprocess cannot show.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from pathlib import Path

import pytest

import agent.loop as loop_module
from agent.approval import Approver
from agent.cancel import Cancelled
from agent.config import Settings
from agent.toolchain import CANCELLED_RETURNCODE, run_capturing
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(path: Path, seconds: float = 30.0) -> str:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def _parent_with_child(pid_file: Path) -> list[str]:
    """A process that starts a long-lived child, records the child's pid, and sleeps."""
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    return [sys.executable, "-c", code]


def test_cancel_stops_the_command_and_its_children(tmp_path):
    pid_file = tmp_path / "child.pid"
    cancel = threading.Event()
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(run_capturing(_parent_with_child(pid_file), cancel=cancel))
    )
    worker.start()
    child = int(_wait_for(pid_file))
    started = time.time()
    cancel.set()
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert time.time() - started < 10
    assert result[0].returncode == CANCELLED_RETURNCODE
    time.sleep(0.2)
    assert not _alive(child)


def test_a_timeout_on_the_cancellable_path_stops_the_children_too(tmp_path):
    pid_file = tmp_path / "child.pid"
    result = run_capturing(_parent_with_child(pid_file), timeout=2, cancel=threading.Event())
    assert result.timed_out and result.returncode == 124
    time.sleep(0.2)
    assert not _alive(int(pid_file.read_text()))


def test_an_uncancelled_command_runs_as_before(tmp_path):
    result = run_capturing([sys.executable, "-c", "print('hi')"], cancel=threading.Event())
    assert (result.returncode, result.stdout.strip()) == (0, "hi")


def _slow_suite(workspace: Path, pid_file: Path) -> None:
    """A suite whose one test records its process id and then sleeps for a minute."""
    (workspace / "test_slow.py").write_text(
        "import os, time\n\n"
        "def test_slow():\n"
        f"    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
    )


def _baseline_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("baseline") and t.is_alive()]


def test_close_stops_a_real_baseline_suite(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pid_file = tmp_path / "pytest.pid"
    _slow_suite(workspace, pid_file)
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(workspace), approver=Approver(auto_approve=True), trace=trace
    )
    settings = Settings(workspace=str(workspace), goal="g", cache_baseline_across_runs=False)

    baselines = loop_module._start_baselines(ctx, trace, settings)
    pytest_pid = int(_wait_for(pid_file))
    started = time.time()
    baselines.close()
    assert time.time() - started < loop_module.BASELINE_CLOSE_WAIT_S
    assert not _baseline_threads()
    time.sleep(0.2)
    assert not _alive(pytest_pid)


def test_a_run_that_stops_before_reading_the_baseline_leaves_no_suite_running(
    tmp_path, monkeypatch
):
    """The run ends in memory recall, before clarify and before the baseline is read:
    the shape of a plan stopped early. It used to return only after the suite finished."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pid_file = tmp_path / "pytest.pid"
    _slow_suite(workspace, pid_file)

    def stop_once_the_suite_runs(*_args, **_kwargs):
        _wait_for(pid_file)
        raise Cancelled("stopped by the user")

    monkeypatch.setattr(loop_module, "recall_memory", stop_once_the_suite_runs)
    settings = dataclasses.replace(
        Settings(
            workspace=str(workspace),
            goal="build a thing",
            trace_dir=str(tmp_path / "traces"),
            failure_log_dir="",
            suggest_next_step=False,
            quick_answers=False,
            clarify=False,
        ),
        cache_baseline_across_runs=False,
    )

    started = time.time()
    summary = loop_module.run(settings)
    assert "stopped" in (summary.get("error") or "")
    assert time.time() - started < 30  # the suite alone sleeps 60 s
    assert not _baseline_threads()
    time.sleep(0.2)
    assert not _alive(int(pid_file.read_text()))


@pytest.mark.parametrize("read_first", [True])
def test_close_after_the_result_was_read_does_nothing(tmp_path, read_first):
    trace = Trace(str(tmp_path / "t"), echo=False)
    handle = loop_module.Baselines(None, trace)
    handle.result()
    handle.close()  # nothing to stop, must not raise
