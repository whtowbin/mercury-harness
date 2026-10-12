"""Promotions: buy-N-get-one-free bundles and percentage discounts."""

from pricing import make_line


def bundle(name, qty, price, free_every=3):
    """Every `free_every`-th unit is free, shown as its own zero-price line."""
    free = qty // free_every
    paid = make_line(qty - free, price, name)
    if not free:
        return [paid]
    return [paid, make_line(free, 0, f"{name} (free)")]


def discount_line(pct, subtotal):
    return make_line(1, -(subtotal * pct // 100), f"Discount {pct}%")
