"""Graded after the agent has stopped. Never visible to it.

The seeded suite checks that remove_and_report deletes the row; it never
checks what the returned line says, which is exactly the gap the bug lives
in: the amount is read after the row is gone, so the report always claims
$0.00 was removed regardless of what was actually there.
"""
from store import Store

from report import full_report, remove_and_report


def test_full_report_is_unaffected_by_the_bug():
    """full_report never removes anything, so it is correct in the seed too
    -- this must keep passing, not just the tests that catch the bug."""
    store = Store()
    store.add(10.0)
    store.add(-2.0)
    assert "2 entries, net +$8.00" in full_report(store).splitlines()[-1]


def test_the_removed_amount_is_reported_correctly():
    store = Store()
    row_id = store.add(7.0)
    line = remove_and_report(store, row_id)
    assert "$7.00" in line, f"reported the wrong amount: {line!r}"


def test_a_negative_removed_amount_is_reported_correctly():
    store = Store()
    row_id = store.add(-4.25)
    line = remove_and_report(store, row_id)
    assert "$4.25" in line and "-" in line.split(":")[-1], f"got: {line!r}"


def test_removing_the_first_of_several_still_reports_its_own_amount():
    store = Store()
    first = store.add(100.0)
    store.add(2.0)
    line = remove_and_report(store, first)
    assert "$100.00" in line, f"got: {line!r}"
