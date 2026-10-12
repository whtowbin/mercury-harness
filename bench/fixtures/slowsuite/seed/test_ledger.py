"""The ledger's existing suite.

Every test here passes against the seeded code, and every one of them is
slow on purpose. The sleep stands in for what makes a real project's suite
expensive -- importing a scientific stack, fitting a model, touching a
database -- without needing any of it: the fixture stays stdlib-only and
portable, which is what lets the benchmark run it anywhere.

The point of the slowness is that it changes which of the harness's
decisions matter. In a traced real run, 73% of the wall clock was pytest and
27% was the model, and the expensive mistakes were all about *how often the
suite gets run* rather than how many tokens a prompt cost. A suite that
finishes in 30ms cannot show that; this one can.
"""
import time

import pytest

from ledger import Ledger

# ~0.4s each across 20 tests: about 8 seconds for the whole suite against a
# small fraction of that for a single selected test. Large enough that
# re-running the suite when a selection would do is visible in the wall
# clock, small enough that a run which does it a few times still finishes
# inside the benchmark's timeout.
SLOW = 0.4


@pytest.fixture(autouse=True)
def _expensive_setup():
    time.sleep(SLOW)


def test_starts_empty():
    assert Ledger().balance() == 0


def test_single_posting():
    ledger = Ledger()
    ledger.post(100)
    assert ledger.balance() == 100


def test_several_postings_sum():
    ledger = Ledger()
    for amount in (10, 20, 30):
        ledger.post(amount)
    assert ledger.balance() == 60


def test_negative_posting():
    ledger = Ledger()
    ledger.post(100)
    ledger.post(-40)
    assert ledger.balance() == 60


def test_post_returns_index():
    ledger = Ledger()
    assert ledger.post(5) == 0
    assert ledger.post(5) == 1


def test_history_records_order():
    ledger = Ledger()
    ledger.post(1)
    ledger.post(2)
    assert ledger.history() == [1, 2]


def test_history_is_a_copy():
    ledger = Ledger()
    ledger.post(1)
    ledger.history().append(999)
    assert ledger.history() == [1]


def test_reverse_restores_balance():
    ledger = Ledger()
    index = ledger.post(100)
    ledger.reverse(index)
    assert ledger.balance() == 0


def test_reverse_leaves_other_postings():
    ledger = Ledger()
    keep = ledger.post(50)
    undo = ledger.post(100)
    ledger.reverse(undo)
    assert ledger.balance() == 50
    assert ledger.history()[keep] == 50


def test_reverse_records_the_negation():
    ledger = Ledger()
    index = ledger.post(75)
    ledger.reverse(index)
    assert ledger.history()[-1] == -75


def test_reverse_of_a_negative_posting():
    ledger = Ledger()
    index = ledger.post(-30)
    ledger.reverse(index)
    assert ledger.balance() == 0


def test_reverse_rejects_a_negative_index():
    with pytest.raises(IndexError):
        Ledger().reverse(-1)


def test_reverse_rejects_an_index_past_the_end():
    ledger = Ledger()
    ledger.post(1)
    with pytest.raises(IndexError):
        ledger.reverse(5)


def test_reverse_on_an_empty_ledger():
    with pytest.raises(IndexError):
        Ledger().reverse(0)


def test_zero_posting():
    ledger = Ledger()
    ledger.post(0)
    assert ledger.balance() == 0


def test_float_amounts():
    ledger = Ledger()
    ledger.post(0.1)
    ledger.post(0.2)
    assert ledger.balance() == pytest.approx(0.3)


def test_many_postings():
    ledger = Ledger()
    for _ in range(100):
        ledger.post(1)
    assert ledger.balance() == 100


def test_reverse_the_first_of_many():
    ledger = Ledger()
    first = ledger.post(10)
    ledger.post(20)
    ledger.reverse(first)
    assert ledger.balance() == 20


def test_two_different_reversals():
    ledger = Ledger()
    a = ledger.post(10)
    b = ledger.post(20)
    ledger.reverse(a)
    ledger.reverse(b)
    assert ledger.balance() == 0


def test_balance_after_interleaved_work():
    ledger = Ledger()
    a = ledger.post(100)
    ledger.post(25)
    ledger.reverse(a)
    ledger.post(5)
    assert ledger.balance() == 30
