"""Sales reports: daily rows and the best-selling product."""

from pricing import make_line


def summary_rows(sales):
    """sales: (day, name, qty, price) tuples -> (day, line) pairs."""
    rows = []
    for day, name, qty, price in sales:
        rows.append((day, make_line(qty, price, name)))
    return rows


def best_seller(sales):
    totals = {}
    for _day, name, qty, price in sales:
        line = make_line(qty=qty, price=price, name=name)
        totals[name] = totals.get(name, 0) + line["total"]
    return max(totals, key=totals.get) if totals else None
