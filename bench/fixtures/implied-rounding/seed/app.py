"""The restaurant app: orders and bills. Money handling lives in money.py."""
from money import add_tax


def bill_total(item_cents, tax_percent):
    """The bill for a list of item prices in cents, with tax."""
    return add_tax(sum(item_cents), tax_percent)
