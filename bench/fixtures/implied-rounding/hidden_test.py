"""The goal asks for an even split in app.py; money.py's house rules say how cents divide."""
import pytest

from app import split_bill


def test_an_even_split():
    assert split_bill(900, 3) == [300, 300, 300]


def test_the_parts_add_back_to_the_total():
    assert sum(split_bill(1000, 3)) == 1000


def test_remainder_cents_go_to_the_earliest_shares():
    assert split_bill(1000, 3) == [334, 333, 333]
    assert split_bill(101, 4) == [26, 25, 25, 25]


def test_shares_are_integer_cents():
    assert all(isinstance(share, int) for share in split_bill(1000, 7))


def test_nobody_to_split_between_is_an_error():
    with pytest.raises(ValueError):
        split_bill(1000, 0)
