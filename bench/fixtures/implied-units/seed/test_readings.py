import pytest

from readings import longest, to_metres


def test_to_metres():
    assert to_metres(250, "cm") == pytest.approx(2.5)


def test_longest_compares_across_units():
    assert longest([(90, "cm"), (1, "m"), (500, "mm")]) == (1, "m")
