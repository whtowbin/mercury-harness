from orders import Order, record_payment


def test_a_payment_is_recorded():
    order = Order(5000)
    assert record_payment(order, "p1", 2000) == 2000


def test_a_repeated_payment_event_is_applied_once():
    order = Order(5000)
    record_payment(order, "p1", 2000)
    assert record_payment(order, "p1", 2000) == 2000
