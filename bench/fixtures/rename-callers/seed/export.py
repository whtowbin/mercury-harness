"""CSV export and the sample basket used in the docs."""

from pricing import make_line


def to_csv(items):
    """items: (name, qty, price) tuples -> CSV text with a header row."""
    lines = map(lambda t: make_line(t[1], t[2], t[0]), items)
    rows = ["name,qty,price,total"]
    rows += [f"{line['name']},{line['qty']},{line['price']},{line['total']}" for line in lines]
    return "\n".join(rows)


def sample_basket():
    return [
        make_line(2, 150, "Tea"),
        make_line(1, 325, "Scone"),
        make_line(qty=3, price=90, name="Jam"),
    ]
