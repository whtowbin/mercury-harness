"""Invoice tests. They pass."""
from invoice import format_money


def test_format_money_two_places():
    assert format_money(1150) == "€11.50"


def test_format_money_thousands():
    assert format_money(123400) == "€1,234.00"


def test_format_money_zero():
    assert format_money(0) == "€0.00"
