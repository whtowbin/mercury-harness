from pricing import round_price, total


def test_round_price_rounds_ordinary_values():
    assert round_price(2.4) == 2
    assert round_price(2.6) == 3


def test_total_rounds_each_amount():
    assert total([1.2, 2.7]) == 4
