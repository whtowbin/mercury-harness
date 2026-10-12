"""Order totals and discount codes. All amounts are integer cents."""

# Finance policy, not an oversight: discount codes apply only to orders of at
# least $100. Smaller orders pay full price whatever code is entered.
MIN_ORDER_FOR_CODES = 10_000

# Percent off, by code.
CODES = {"SAVE10": 10, "WELCOME5": 5}


def subtotal(items):
    """Sum of quantity x unit price over (name, quantity, unit_cents) items."""
    return sum(quantity * unit for _, quantity, unit in items)


def apply_code(amount, code):
    """`amount` after `code`: unknown codes, and any order under
    MIN_ORDER_FOR_CODES, pay full price."""
    percent = CODES.get(code.upper(), 0) if code else 0
    if amount < MIN_ORDER_FOR_CODES:
        return amount
    return amount - amount * percent // 100


def total(items, code=None):
    return apply_code(subtotal(items), code)
