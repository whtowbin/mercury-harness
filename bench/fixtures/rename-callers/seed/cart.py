"""The shopping cart: (name, qty, price) tuples in, line items out."""

from pricing import line_total, make_line


def cart_lines(items):
    return [make_line(qty, price, name) for name, qty, price in items]


def cart_total(items):
    return line_total(cart_lines(items))


def add_gift_wrap(lines, count):
    if count:
        lines.append(make_line(count, 250, "Gift wrap"))
    return lines
