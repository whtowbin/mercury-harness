"""Keep a Mac awake while the harness is working.

A sleeping machine runs no timers and holds no connections: a sweep left on a
laptop stalled ~2,500s per run (DESIGN-LOG §61), and every attempt that escaped
the stream bounds during the SWE-bench spike, five of them at 219-536s, sat
inside a pmset sleep, on battery and on AC alike (§71). `bench` has held an
assertion since §61; `harness run` did not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_held: subprocess.Popen | None = None


def keep_awake() -> subprocess.Popen | None:
    """Hold a no-idle-sleep assertion for this process's life, on macOS.

    `caffeinate -i -s -w PID` (-i idle sleep, -s system sleep on power) lasts until
    this process exits and needs no cleanup. Started once per process, so a test
    suite driving the CLI many times does not leave one per invocation. Neither
    flag stops a lid closed on battery; bench records that as `machine_slept`.
    """
    global _held
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    if _held is None or _held.poll() is not None:
        _held = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    return _held
