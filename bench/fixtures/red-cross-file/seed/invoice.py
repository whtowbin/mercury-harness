"""Renders invoices from the store."""

CURRENCY = "$"


def format_money(amount):
    """Format a money amount for display."""
    return f"{CURRENCY}{amount:,.2f}"


def line_for(store, order_id):
    """One invoice line for an order."""
    return f"{order_id}: {format_money(store.amount(order_id))}"


def render(store):
    """The whole invoice, one line per order plus a total."""
    lines = [line_for(store, oid) for oid in store.ids()]
    lines.append(f"TOTAL: {format_money(store.total())}")
    return "\n".join(lines)
