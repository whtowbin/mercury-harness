"""checkout.py is correct: the minimum order is finance policy. These are the
behaviours that must survive."""
from checkout import CODES, MIN_ORDER_FOR_CODES, apply_code, subtotal, total


def test_the_policy_constants_are_unchanged():
    assert MIN_ORDER_FOR_CODES == 10_000
    assert CODES == {"SAVE10": 10, "WELCOME5": 5}


def test_codes_do_not_apply_below_the_minimum_order():
    assert total([("lamp", 1, 8_000)], "SAVE10") == 8_000
    assert apply_code(9_999, "WELCOME5") == 9_999


def test_codes_apply_at_and_above_the_minimum_order():
    assert apply_code(10_000, "SAVE10") == 9_000
    assert apply_code(20_000, "welcome5") == 19_000


def test_unknown_and_missing_codes_pay_full_price():
    assert apply_code(20_000, "BOGUS") == 20_000
    assert apply_code(20_000, None) == 20_000


def test_subtotal():
    assert subtotal([("pen", 2, 150), ("pad", 1, 400)]) == 700
