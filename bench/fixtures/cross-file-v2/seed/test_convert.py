"""Convert tests. They pass."""
from convert import to_eur_cents


def test_zero_converts_to_zero():
    assert to_eur_cents(0) == 0


def test_result_is_a_whole_number():
    assert isinstance(to_eur_cents(1234), int)


def test_doubling_the_rate_doubles_the_result():
    assert to_eur_cents(10_000, rate=1.0) == 2 * to_eur_cents(10_000, rate=0.5)
