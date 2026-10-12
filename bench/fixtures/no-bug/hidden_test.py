"""schedule.py is correct. These are the behaviours that must survive."""
from schedule import is_overdue, next_weekday, occurrences


def test_forward_within_the_week():
    assert next_weekday("mon", "wed") == 2
    assert next_weekday("wed", "fri") == 2


def test_wrapping_past_the_weekend():
    assert next_weekday("fri", "mon") == 3
    assert next_weekday("sun", "mon") == 1


def test_same_day_means_a_full_week():
    assert next_weekday("tue", "tue") == 7
    assert next_weekday("sun", "sun") == 7


def test_occurrences_are_weekly_from_the_first():
    assert occurrences("mon", "wed", 3) == [2, 9, 16]
    assert occurrences("tue", "tue", 2) == [7, 14]


def test_the_grace_period_boundary_is_inclusive():
    assert is_overdue(1) is False
    assert is_overdue(2) is False
    assert is_overdue(3) is True
