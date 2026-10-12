"""The goal asks for a total; the module's house rules say what a total may count."""
import pytest

from readings import total_distance


def test_readings_in_one_unit():
    assert total_distance([(1, "m"), (2, "m")]) == pytest.approx(3.0)


def test_mixed_units_are_converted_to_metres():
    assert total_distance([(1, "km"), (250, "cm"), (500, "mm")]) == pytest.approx(1003.0)


def test_a_dropped_packet_is_skipped_not_counted_or_crashed_on():
    assert total_distance([(1, "m"), (None, "m"), (2, "m")]) == pytest.approx(3.0)


def test_a_negative_reading_is_a_fault():
    with pytest.raises(ValueError):
        total_distance([(1, "m"), (-2, "m")])


def test_an_empty_list_is_zero():
    assert total_distance([]) == pytest.approx(0.0)
