from models import format_amount, summarize


def test_format_amount_positive():
    assert format_amount(12.5) == "+$12.50"


def test_format_amount_negative():
    assert format_amount(-3.0) == "-$3.00"


def test_summarize_singular():
    assert summarize([5.0]) == "1 entry, net +$5.00"


def test_summarize_plural_and_total():
    assert summarize([5.0, -2.0]) == "2 entries, net +$3.00"
