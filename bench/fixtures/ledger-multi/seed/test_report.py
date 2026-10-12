from store import Store

from report import full_report, remove_and_report


def test_full_report_lists_each_entry_and_a_total():
    store = Store()
    store.add(10.0)
    store.add(-2.0)
    text = full_report(store)
    assert "2 entries, net +$8.00" in text.splitlines()[-1]


def test_remove_and_report_deletes_the_entry():
    """The seeded suite checks the side effect (removal) but not the exact
    wording of the returned line -- a real blind spot: the entry the
    function claims to describe is gone by the time it reads the amount."""
    store = Store()
    row_id = store.add(7.0)
    remove_and_report(store, row_id)
    assert row_id not in store.all_ids()
