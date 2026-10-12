"""Tiered pricing for a usage-based plan."""

TIERS = [(0, 1000, 0.10), (1000, 10000, 0.05), (10000, None, 0.02)]
MINIMUM_CHARGE = 5.00


def tier_for(units):
    """The (lower, upper, rate) tier a unit count falls in."""
    for lower, upper, rate in TIERS:
        if units >= lower and (upper is None or units < upper):
            return (lower, upper, rate)
    return TIERS[-1]


def price(units):
    """Total price for `units`, charged progressively across the tiers.

    Units in the first 1000 cost 0.10 each, the next 9000 cost 0.05 each, and
    anything beyond 10000 costs 0.02 each. The total is never below
    MINIMUM_CHARGE.
    """
    total = 0.0
    remaining = units
    for lower, upper, rate in TIERS:
        if remaining <= 0:
            break
        span = remaining if upper is None else min(remaining, upper)
        total += span * rate
        remaining -= span
    return max(total, MINIMUM_CHARGE)
