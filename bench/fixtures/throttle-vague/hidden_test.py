"""Hidden tests for `throttle`. All time is passed explicitly — never real time."""
import inspect

import pytest

from throttle import TokenBucket

# The goal describes the requested behaviour ("snap back to fully available
# right away") and deliberately names neither the method nor its signature, so
# this fixture must not fail an agent for choosing a reasonable one. It did:
# across 102 traced runs whose own gate was green but whose hidden tests failed,
# 65 were this task, and in every one the agent HAD implemented the behaviour —
# 50 as `reset(self, now)` (matching every other method's `now` parameter), the
# rest as `clear`, `unblock`, `clear_block` or `reset_quota` — while the old
# tests here called `bucket.reset()` with no arguments. That measured
# interface-guessing, not the diagnosis the task is for.
_RESET_NAMES = ("reset", "clear", "unblock", "clear_block", "reset_quota", "refill")


def _reset(bucket, now=0):
    """Call whichever reset-like method the agent added, however it takes time."""
    for name in _RESET_NAMES:
        method = getattr(bucket, name, None)
        if not callable(method):
            continue
        params = [p for p in inspect.signature(method).parameters.values()
                  if p.default is inspect.Parameter.empty
                  and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not params:
            return method()
        if len(params) == 1 and params[0].name in ("now", "t", "time", "timestamp"):
            return method(now)
    raise AssertionError("no reset-like method (reset/clear/unblock/...) was added")


def test_starts_full():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    assert bucket.available(0) == 10


def test_allow_consumes_tokens():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    assert bucket.allow(0, cost=3) is True
    assert bucket.available(0) == 7


def test_allow_refuses_when_insufficient():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    bucket.allow(0, cost=10)
    assert bucket.allow(0, cost=1) is False
    assert bucket.available(0) == 0


def test_refusal_does_not_consume_tokens():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    bucket.allow(0, cost=6)
    assert bucket.allow(0, cost=6) is False
    assert bucket.available(0) == 4


def test_refills_linearly_over_time():
    bucket = TokenBucket(capacity=10, refill_rate=2)
    bucket.allow(0, cost=10)
    assert bucket.available(3) == pytest.approx(6.0)


def test_never_exceeds_capacity():
    bucket = TokenBucket(capacity=10, refill_rate=5)
    assert bucket.available(100) == 10


def test_never_goes_negative():
    bucket = TokenBucket(capacity=5, refill_rate=1)
    bucket.allow(0, cost=5)
    assert bucket.available(0) >= 0


# -- the actual bug: repeated observation must not double-credit refill -----

def test_repeated_availability_checks_do_not_inflate_tokens():
    """Calling available() twice at increasing times must accumulate refill
    from the LAST OBSERVED time, not recompute it from scratch each call."""
    bucket = TokenBucket(capacity=10, refill_rate=1)
    bucket.allow(0, cost=10)
    assert bucket.available(5) == pytest.approx(5.0)
    # One more second passed since the LAST check (t=5), not since t=0.
    assert bucket.available(6) == pytest.approx(6.0)


def test_many_small_time_steps_match_one_big_step():
    """Refilling in small increments must equal refilling in one big increment —
    this is the property the original bug violates by re-crediting from t=0
    on every call instead of advancing its internal clock."""
    a = TokenBucket(capacity=100, refill_rate=1)
    a.allow(0, cost=100)
    for t in range(1, 11):
        a.available(t)
    b = TokenBucket(capacity=100, refill_rate=1)
    b.allow(0, cost=100)
    assert a.available(10) == pytest.approx(b.available(10))


def test_allow_and_available_share_the_same_clock():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    bucket.allow(0, cost=10)
    bucket.available(4)          # an observation at t=4: refills to 4, advances the clock
    assert bucket.allow(5, cost=1) is True   # only 1s has passed since t=4, so 5 available
    assert bucket.available(5) == pytest.approx(4.0)


# -- the requested addition --------------------------------------------------

def test_reset_refills_to_full_immediately():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    bucket.allow(0, cost=10)
    _reset(bucket)
    assert bucket.available(0) == 10


def test_reset_does_not_change_capacity_or_rate():
    bucket = TokenBucket(capacity=7, refill_rate=3)
    _reset(bucket)
    assert bucket.capacity == 7
    assert bucket.refill_rate == 3
