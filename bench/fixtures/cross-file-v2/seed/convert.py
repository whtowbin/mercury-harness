"""Currency conversion. Every amount in, every amount out, is whole cents."""

USD_TO_EUR_RATE = 0.92


def to_eur_cents(usd_cents, rate=USD_TO_EUR_RATE):
    """Convert a whole number of US cents to a whole number of Euro cents."""
    return round(usd_cents * rate / 100)
