"""The reported "regression" is the documented behaviour: halves round to even."""
from pricing import round_price, total


def test_halves_round_to_even():
    assert round_price(2.5) == 2
    assert round_price(3.5) == 4
    assert round_price(0.5) == 0


def test_ordinary_values_still_round():
    assert round_price(2.4) == 2
    assert round_price(2.6) == 3


def test_total_uses_the_same_rule():
    assert total([2.5, 3.5]) == 6
