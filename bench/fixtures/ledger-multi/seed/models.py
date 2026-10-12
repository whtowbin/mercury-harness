"""Formatting for ledger amounts."""


def format_amount(amount: float) -> str:
    """Render a signed amount as e.g. "+$12.50" or "-$3.00"."""
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.2f}"


def summarize(amounts: list[float]) -> str:
    """A one-line total: "3 entries, net +$9.50"."""
    total = sum(amounts)
    noun = "entry" if len(amounts) == 1 else "entries"
    return f"{len(amounts)} {noun}, net {format_amount(total)}"
