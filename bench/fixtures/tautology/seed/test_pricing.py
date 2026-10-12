"""Existing tests. They pass."""
from pricing import MINIMUM_CHARGE, TIERS, price, tier_for


def _expected(units):
    total, remaining = 0.0, units
    for lower, upper, rate in TIERS:
        if remaining <= 0:
            break
        span = remaining if upper is None else min(remaining, upper)
        total += span * rate
        remaining -= span
    return max(total, MINIMUM_CHARGE)


def test_price_matches_the_tier_table():
    for units in (0, 500, 1000, 5000, 10000, 25000):
        assert price(units) == _expected(units)


def test_minimum_charge_applies_to_small_orders():
    assert price(1) == MINIMUM_CHARGE


def test_tier_lookup():
    assert tier_for(500)[2] == 0.10
    assert tier_for(5000)[2] == 0.05
