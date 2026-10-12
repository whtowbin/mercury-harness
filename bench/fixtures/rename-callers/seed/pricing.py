"""Line items: the one shape every other module builds its rows from."""


def make_line(qty, price, name):
    """A line item. `price` is the unit price in cents; `total` is qty * price."""
    if qty < 0:
        raise ValueError("qty must be >= 0")
    return {"name": name, "qty": qty, "price": price, "total": qty * price}


def line_total(lines):
    return sum(line["total"] for line in lines)
