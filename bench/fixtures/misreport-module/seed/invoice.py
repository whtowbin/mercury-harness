"""Invoice totals. All arithmetic is done in whole cents, via money.to_cents."""
from money import format_cents, to_cents


def line_total(unit_price, quantity):
    """Cents for one line: the unit price in cents times the quantity."""
    return to_cents(unit_price) * quantity


def invoice_total(lines):
    """Cents for a list of (unit_price, quantity) lines."""
    return sum(line_total(price, qty) for price, qty in lines)


def render(lines):
    return f"Total: {format_cents(invoice_total(lines))}"
