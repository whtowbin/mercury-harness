"""Price rounding. See README.md for the rounding rule."""
from decimal import ROUND_HALF_EVEN, Decimal


def round_price(amount):
    """Round `amount` to whole dollars (see README.md)."""
    return int(Decimal(str(amount)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def total(amounts):
    return sum(round_price(a) for a in amounts)
