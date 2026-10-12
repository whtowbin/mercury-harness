"""A token-bucket rate limiter."""


class TokenBucket:
    """A token bucket that refills over time.

    capacity: the maximum number of tokens the bucket can hold.
    refill_rate: tokens added per second of elapsed time.

    allow(now, cost=1) consumes `cost` tokens if enough are available and
    returns True, otherwise leaves the bucket unchanged and returns False.
    Tokens never exceed capacity and never go negative.
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last = 0.0

    def _refill(self, now: float) -> None:
        elapsed = now - self.last
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        # BUG: self.last is never updated, so every call recomputes the refill
        # from the same starting point instead of from the last observation.

    def allow(self, now: float, cost: float = 1) -> bool:
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def available(self, now: float) -> float:
        self._refill(now)
        return self.tokens
