"""Invoices: an order's items plus shipping, rendered as text."""

import pricing

SHIPPING_CENTS = 499


def invoice_lines(order):
    lines = [pricing.make_line(i["qty"], i["price"], i["name"]) for i in order["items"]]
    if order.get("ship"):
        lines.append(
            pricing.make_line(
                1,
                SHIPPING_CENTS,
                "Shipping",
            )
        )
    return lines


def render(order):
    lines = invoice_lines(order)
    rows = [f"{line['name']} x{line['qty']}: {line['total']}" for line in lines]
    rows.append(f"TOTAL: {pricing.line_total(lines)}")
    return "\n".join(rows)
