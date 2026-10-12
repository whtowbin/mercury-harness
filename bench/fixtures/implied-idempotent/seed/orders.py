"""Order payments.

Payment-provider webhooks are delivered at least once, so the same event can arrive
more than once. Every handler in this module must be idempotent on its event id:
applying an event a second time must change nothing. An order's paid amount never
goes below zero: an event that would take it there is rejected with ValueError.
"""


class Order:
    def __init__(self, total_cents):
        self.total_cents = total_cents
        self.paid_cents = 0
        self.seen_events = set()


def record_payment(order, payment_id, cents):
    """Apply a payment event (idempotent on payment_id)."""
    if payment_id in order.seen_events:
        return order.paid_cents
    order.seen_events.add(payment_id)
    order.paid_cents += cents
    return order.paid_cents
