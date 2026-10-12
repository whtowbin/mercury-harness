"""Cooperative cancellation for a run in progress.

A blocking HTTP call already in flight cannot be interrupted from another
thread — the same reason bench's --timeout runs a job in a whole separate
process rather than trying to stop a thread. This does not attempt that: it
lets something outside the run (a TUI keybinding, a signal handler) request a
stop, and the run itself checks for that request at safe points between
operations — the top of the tool-calling loop's iteration, and between plan
steps — and unwinds cleanly from there rather than being killed mid-write.
"""

from __future__ import annotations

import threading


class Cancelled(Exception):
    """Raised by CancelToken.check() once cancel() has been called."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled("stopped by user")
