"""Hidden tests for `throttle`. All time is passed explicitly — never real time."""
import pytest

from throttle import TokenBucket


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
    bucket.reset()
    assert bucket.available(0) == 10


def test_reset_does_not_change_capacity_or_rate():
    bucket = TokenBucket(capacity=7, refill_rate=3)
    bucket.reset()
    assert bucket.capacity == 7
    assert bucket.refill_rate == 3
