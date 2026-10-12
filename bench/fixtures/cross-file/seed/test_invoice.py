"""Invoice tests. They pass."""
from invoice import format_money


def test_format_money_two_places():
    assert format_money(12.5) == "$12.50"


def test_format_money_thousands():
    assert format_money(1234.5) == "$1,234.50"


def test_format_money_zero():
    assert format_money(0) == "$0.00"
