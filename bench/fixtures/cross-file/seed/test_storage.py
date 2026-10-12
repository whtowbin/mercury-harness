"""Storage tests. They pass."""
import pytest
from storage import Store


def test_add_and_read_back():
    s = Store()
    s.add("A1", 1250)
    assert s.amount("A1") == 1250


def test_total_sums_orders():
    s = Store()
    s.add("A1", 1250)
    s.add("A2", 500)
    assert s.total() == 1750


def test_floats_are_rejected():
    with pytest.raises(TypeError):
        Store().add("A1", 12.5)
