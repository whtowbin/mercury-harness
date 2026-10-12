"""Renders invoices in Euros from the store's US-cent holdings."""
from convert import to_eur_cents

CURRENCY = "€"


def format_money(eur_cents):
    """Format a whole number of Euro cents as a Euro amount."""
    return f"{CURRENCY}{eur_cents / 100:,.2f}"


def line_for(store, order_id):
    """One invoice line for an order, converted to Euros."""
    eur_cents = to_eur_cents(store.amount(order_id))
    return f"{order_id}: {format_money(eur_cents)}"


def render(store):
    """The whole invoice, one line per order plus a total."""
    lines = [line_for(store, oid) for oid in store.ids()]
    lines.append(f"TOTAL: {format_money(to_eur_cents(store.total()))}")
    return "\n".join(lines)
