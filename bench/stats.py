"""Deciding whether two configs differ, and how many runs it would take to say.

Twice in one session a large gap between two arms at 24 runs a side turned out
to be noise: a planner-prompt change that read as +12 points of honesty, and a
plan audit that read as +21, neither of which survived a larger interleaved
sweep. The tables `bench report` prints show a mean per config and a count, and
leave the reader to decide whether a difference is real; that judgment was made
by eye both times, and wrongly. This makes it a computed answer.

Standard library only. A percentile bootstrap rather than a t-test because the
metrics here are a mix of shapes (a 0/1 honesty flag, a bounded accuracy share,
a heavy-tailed call count) and the bootstrap makes no claim about any of them.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

RESAMPLES = 4000
CONFIDENCE = 0.95
SEED = 20260923
# 80% power at alpha 0.05, two-sided: (z_{1-a/2} + z_{power})^2.
_POWER_FACTOR = (1.96 + 0.84) ** 2
MIN_RUNS_PER_SIDE = 5


@dataclass
class Comparison:
    metric: str
    mean_a: float
    mean_b: float
    diff: float  # b - a
    low: float  # bootstrap interval for diff
    high: float
    n_a: int
    n_b: int
    runs_needed: int | None  # per side, to detect |diff| at 80% power

    @property
    def distinguishable(self) -> bool:
        return (
            self.n_a >= MIN_RUNS_PER_SIDE
            and self.n_b >= MIN_RUNS_PER_SIDE
            and (self.low > 0 or self.high < 0)
        )

    @property
    def verdict(self) -> str:
        if min(self.n_a, self.n_b) < MIN_RUNS_PER_SIDE:
            return f"too few runs (<{MIN_RUNS_PER_SIDE} a side)"
        if self.distinguishable:
            return "differs"
        if self.runs_needed is None:
            return "not distinguishable"
        return f"not distinguishable (~{self.runs_needed} a side to see this gap)"


def bootstrap_diff(a: list[float], b: list[float]) -> tuple[float, float]:
    """Percentile interval for mean(b) - mean(a)."""
    rng = random.Random(SEED)
    diffs = []
    for _ in range(RESAMPLES):
        sample_a = [a[rng.randrange(len(a))] for _ in a]
        sample_b = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(statistics.fmean(sample_b) - statistics.fmean(sample_a))
    diffs.sort()
    tail = (1 - CONFIDENCE) / 2
    return diffs[int(tail * RESAMPLES)], diffs[int((1 - tail) * RESAMPLES) - 1]


def runs_needed(a: list[float], b: list[float]) -> int | None:
    """Runs per side to detect the observed gap at 80% power, from the pooled
    spread. None when the gap is zero (no finite answer) or the spread is."""
    diff = abs(statistics.fmean(b) - statistics.fmean(a))
    if diff == 0:
        return None
    pooled = statistics.fmean([statistics.pvariance(a), statistics.pvariance(b)])
    if pooled == 0:
        return None
    return math.ceil(2 * _POWER_FACTOR * pooled / (diff * diff))


def compare(metric: str, a: list[float], b: list[float]) -> Comparison | None:
    if not a or not b:
        return None
    low, high = bootstrap_diff(a, b)
    return Comparison(
        metric,
        statistics.fmean(a),
        statistics.fmean(b),
        statistics.fmean(b) - statistics.fmean(a),
        low,
        high,
        len(a),
        len(b),
        runs_needed(a, b),
    )


def interleaved(a_started: list[float], b_started: list[float]) -> bool:
    """Whether two arms ran during the same stretch of time. Two sweeps an hour
    apart share no time at all, which is exactly the case where a provider's
    latency drifting reads as a treatment effect."""
    if not a_started or not b_started:
        return False
    return max(min(a_started), min(b_started)) <= min(max(a_started), max(b_started))
