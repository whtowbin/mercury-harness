"""A wall-time limit for one benchmark run that actually stops it (DESIGN-LOG §96.2).

The harness has no wall-time flag of its own, and the drivers either waited with no limit
(`sci.py`, `gaia.py`) or killed only their direct child. `round.py`'s 30-minute timeout killed
`swe.py` and left the harness it had started running underneath. Here the run starts in its
own session, and on the limit its whole process group gets SIGTERM, then SIGKILL, so the
harness, its pytest subprocesses and any shell it opened stop together. A killed run writes
`KILLED_MARK` to its log, which the sweep's scout check counts.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

KILLED_MARK = "[runlimits] killed at the run time limit"
GRACE_S = 10.0


def start(argv: list[str], **kwargs) -> subprocess.Popen:
    """Popen in a new session, so the run and everything it starts share one process group."""
    return subprocess.Popen(argv, start_new_session=True, **kwargs)


def kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the run's process group, then SIGKILL whatever is left after GRACE_S."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    for sig, wait_s in ((signal.SIGTERM, GRACE_S), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
        if not _group_alive(pgid):
            return
    # The group leader may be reaped while a child lingers; one last SIGKILL to the group.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def descendants(root_pid: int) -> list[int]:
    """Every live process under root_pid, root included, from one `ps` snapshot."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, todo = [], [root_pid]
    while todo:
        pid = todo.pop()
        found.append(pid)
        todo += children.get(pid, [])
    return found


def kill_tree(root_pid: int) -> None:
    """Stop a whole sweep: every process under root_pid and every process group they lead.

    The runs a sweep starts are in sessions of their own (`start`), so killing the sweep's own
    group would orphan them, and with their driver gone nothing would enforce their limit. The
    tree is read once, before anything dies, because an orphan is re-parented to launchd."""
    pids = descendants(root_pid)
    groups = set()
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            groups.add(os.getpgid(pid))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pgid in groups:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        if sig == signal.SIGTERM:
            time.sleep(GRACE_S)


def _group_alive(pgid: int) -> bool:
    """Whether any process in the group can still be signalled. macOS answers EPERM, not
    ESRCH, for a group whose members have exited but are not yet reaped. That PermissionError
    crashed swe.py at the final sweep's first real kill (django 11885, §96.3) before the kill
    mark was written, so here it means "nothing left to stop"."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def wait(proc: subprocess.Popen, deadline: float, log=None) -> bool:
    """Wait for the run until `deadline` (time.monotonic); kill its group past it. True when it
    finished on its own."""
    try:
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        return True
    except subprocess.TimeoutExpired:
        kill_group(proc)
        if log is not None:
            log.write(f"\n{KILLED_MARK}\n")
            log.flush()
        return False


def run(argv: list[str], timeout_s: float, log=None, **kwargs) -> bool:
    """Run to completion or to `timeout_s`, whichever is first. True when it finished."""
    proc = start(argv, stdout=log, stderr=subprocess.STDOUT if log is not None else None, **kwargs)
    return wait(proc, time.monotonic() + timeout_s, log)


def env_seconds(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default
