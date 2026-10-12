"""Works out when a recurring reminder next fires."""

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def next_weekday(current, target):
    """Days from `current` until the next `target` weekday.

    Same day returns 7, not 0: a weekly reminder on Tuesday, asked on a
    Tuesday, next fires a week later rather than immediately.
    """
    here = WEEKDAYS.index(current)
    there = WEEKDAYS.index(target)
    return (there - here - 1) % 7 + 1


def occurrences(start_day, target, weeks):
    """The day offsets of the next `weeks` firings of a weekly reminder."""
    first = next_weekday(start_day, target)
    return [first + 7 * n for n in range(weeks)]


def is_overdue(days_since_due, grace_days=2):
    """Whether something is overdue, allowing a grace period."""
    return days_since_due > grace_days
