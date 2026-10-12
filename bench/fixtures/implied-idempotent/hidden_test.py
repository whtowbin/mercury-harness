"""The goal asks for refunds; the module says every handler is idempotent on its id."""
import pytest

from orders import Order, record_payment, record_refund


def paid(cents):
    order = Order(10_000)
    record_payment(order, "p1", cents)
    return order


def test_a_refund_reduces_what_was_paid():
    order = paid(5000)
    assert record_refund(order, "r1", 1500) == 3500


def test_a_repeated_refund_event_is_applied_once():
    order = paid(5000)
    record_refund(order, "r1", 1500)
    assert record_refund(order, "r1", 1500) == 3500
    assert order.paid_cents == 3500


def test_two_different_refunds_both_apply():
    order = paid(5000)
    record_refund(order, "r1", 1000)
    assert record_refund(order, "r2", 1000) == 3000


def test_a_refund_larger_than_what_was_paid_is_rejected():
    order = paid(1000)
    with pytest.raises(ValueError):
        record_refund(order, "r1", 1500)
