import pytest
from pricing import line_total, make_line


def test_a_line_multiplies_qty_by_price():
    assert make_line(3, 200, "Mug")["total"] == 600


def test_negative_qty_is_refused():
    with pytest.raises(ValueError):
        make_line(-1, 200, "Mug")


def test_line_total_sums_lines():
    assert line_total([make_line(1, 100, "a"), make_line(2, 50, "b")]) == 200
