"""Graded after the agent has stopped. Never visible to it.

The seeded suite covers reversal thoroughly and passes against the bug,
because every case it checks reverses each posting at most once. The defect
only shows on the second reverse of the same index, which the docstring
calls an error and the code silently accepts.

Deliberately fast: this file grades the work, so its own runtime is not part
of what the task is measuring.
"""
import pytest

from ledger import Ledger


def test_reversing_twice_is_an_error():
    """The documented contract: a posting already undone cannot be undone
    again. The seeded code silently accepts it."""
    ledger = Ledger()
    index = ledger.post(100)
    ledger.reverse(index)
    with pytest.raises(Exception):
        ledger.reverse(index)


def test_a_rejected_second_reversal_does_not_move_the_balance():
    """Rejecting it is not enough if the balance moved on the way out."""
    ledger = Ledger()
    index = ledger.post(100)
    ledger.reverse(index)
    before = ledger.balance()
    with pytest.raises(Exception):
        ledger.reverse(index)
    assert ledger.balance() == before


def test_a_rejected_second_reversal_does_not_append_to_history():
    ledger = Ledger()
    index = ledger.post(40)
    ledger.reverse(index)
    before = ledger.history()
    with pytest.raises(Exception):
        ledger.reverse(index)
    assert ledger.history() == before


def test_reversing_two_different_postings_still_works():
    """The fix must reject a repeat, not reversal in general."""
    ledger = Ledger()
    a = ledger.post(10)
    b = ledger.post(20)
    ledger.reverse(a)
    ledger.reverse(b)
    assert ledger.balance() == 0


def test_the_original_behaviour_is_intact():
    ledger = Ledger()
    a = ledger.post(100)
    ledger.post(25)
    ledger.reverse(a)
    ledger.post(5)
    assert ledger.balance() == 30
    assert ledger.history()[-2] == -100


def test_out_of_range_still_raises_index_error():
    with pytest.raises(IndexError):
        Ledger().reverse(0)
