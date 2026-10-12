"""Money helpers. Amounts arrive as float dollars from the order form."""


def to_cents(dollars):
    """Convert a dollar amount to whole cents, to the nearest cent."""
    return int(dollars * 100)


def format_cents(cents):
    """'$12.34' for 1234."""
    return f"${cents // 100}.{cents % 100:02d}"
