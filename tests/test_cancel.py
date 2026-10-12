"""CancelToken: cooperative cancellation checked between operations."""

from __future__ import annotations

import threading

import pytest

from agent.cancel import Cancelled, CancelToken


def test_a_fresh_token_never_raises():
    token = CancelToken()
    token.check()  # must not raise
    assert token.requested is False


def test_cancel_makes_check_raise():
    token = CancelToken()
    token.cancel()
    assert token.requested is True
    with pytest.raises(Cancelled):
        token.check()


def test_cancel_is_idempotent():
    token = CancelToken()
    token.cancel()
    token.cancel()  # must not raise or toggle back off
    assert token.requested is True


def test_cancel_from_another_thread_is_visible():
    token = CancelToken()

    def canceller():
        token.cancel()

    t = threading.Thread(target=canceller)
    t.start()
    t.join(timeout=5)
    assert token.requested is True
    with pytest.raises(Cancelled):
        token.check()
