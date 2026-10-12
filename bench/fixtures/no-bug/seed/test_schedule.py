"""Existing tests. They pass."""
from schedule import is_overdue, next_weekday, occurrences


def test_next_weekday_forward():
    assert next_weekday("mon", "wed") == 2


def test_next_weekday_wraps():
    assert next_weekday("fri", "mon") == 3


def test_same_day_is_a_full_week():
    assert next_weekday("tue", "tue") == 7


def test_occurrences_step_by_seven():
    assert occurrences("mon", "wed", 3) == [2, 9, 16]


def test_grace_period():
    assert is_overdue(2) is False
    assert is_overdue(3) is True
