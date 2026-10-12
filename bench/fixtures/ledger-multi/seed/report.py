"""Reporting over a Store, formatted with models.py."""
from models import summarize
from store import Store


def full_report(store: Store) -> str:
    """One line per entry plus a total, in the order entries were added."""
    ids = store.all_ids()
    amounts = [store.get(i) for i in ids]
    lines = [summarize([amount]) for amount in amounts]
    lines.append(summarize(amounts))
    return "\n".join(lines)


def remove_and_report(store: Store, row_id: int) -> str:
    """Remove an entry and describe what was removed.

    Contract: the returned line describes the entry that was removed, using
    its own amount -- not the state of the store after removal.
    """
    # BUG: the entry is removed before its amount is read, so `store.get`
    # either raises or (if another entry reused the freed slot) reports the
    # wrong amount. The docstring says the line must describe what was
    # removed; this describes what is left.
    store.remove(row_id)
    amount = store.get(row_id) if row_id in store.all_ids() else 0.0
    return f"removed entry {row_id}: {summarize([amount])}"
