from checkout import apply_code, subtotal, total


def test_subtotal_multiplies_quantity_by_unit_price():
    assert subtotal([("pen", 2, 150), ("pad", 1, 400)]) == 700


def test_save10_takes_ten_percent_off_a_qualifying_order():
    assert apply_code(20_000, "SAVE10") == 18_000


def test_codes_do_not_apply_below_the_minimum_order():
    assert total([("lamp", 1, 8_000)], "SAVE10") == 8_000
