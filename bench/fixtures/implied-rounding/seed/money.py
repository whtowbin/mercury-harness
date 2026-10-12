"""Money, in integer cents throughout.

House rules for every function here: amounts are int cents, never floats; when an
amount is divided into parts, the parts must add back to exactly the original, and
any remainder cents go to the earliest parts, one cent each.
"""


def add_tax(cents, rate_percent):
    """`cents` plus tax at `rate_percent`, rounded half-up to a whole cent."""
    return cents + (cents * rate_percent + 50) // 100


def allocate(cents, weights):
    """Split `cents` in proportion to `weights` (house rules above). An empty or
    all-zero set of weights is a ValueError: there is nothing to divide between."""
    total = sum(weights)
    if not weights or total <= 0:
        raise ValueError("nothing to divide between")
    shares = [cents * w // total for w in weights]
    for i in range(cents - sum(shares)):
        shares[i] += 1
    return shares
