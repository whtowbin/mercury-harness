"""The defect is in money.to_cents (it truncates 0.29 * 100 = 28.999... to 28);
invoice.py is correct and only exposes it."""
from invoice import invoice_total, line_total, render
from money import to_cents


def test_to_cents_rounds_to_the_nearest_cent():
    assert to_cents(0.29) == 29
    assert to_cents(19.99) == 1999
    assert to_cents(4.35) == 435


def test_to_cents_of_whole_dollars_is_unchanged():
    assert to_cents(3.0) == 300


def test_a_line_that_lost_a_cent_is_right():
    assert line_total(0.29, 3) == 87


def test_the_invoice_the_customer_saw():
    assert invoice_total([(19.99, 1), (0.29, 2), (4.35, 1)]) == 2492
    assert render([(19.99, 1), (0.29, 2), (4.35, 1)]) == "Total: $24.92"


def test_whole_dollar_lines_still_add_up():
    assert invoice_total([(2.00, 3), (5.00, 1)]) == 1100
