"""The invoice must show what the order was actually worth, converted."""
from invoice import render
from storage import Store


def _store(**orders):
    s = Store()
    for oid, cents in orders.items():
        s.add(oid, cents)
    return s


def test_a_single_order_converts_to_euros():
    # $12.50 at 0.92 -> €11.50
    assert "A1: €11.50" in render(_store(A1=1250))


def test_a_small_order_still_converts_correctly():
    # $0.99 at 0.92 -> €0.91
    assert "A1: €0.91" in render(_store(A1=99))


def test_the_total_matches_the_converted_lines():
    out = render(_store(A1=1250, A2=500))
    assert "A1: €11.50" in out
    assert "A2: €4.60" in out
    assert "TOTAL: €16.10" in out


def test_thousands_separator_on_a_real_total():
    # 123456789 US cents at 0.92 -> 113580246 EUR cents -> €1,135,802.46
    assert "TOTAL: €1,135,802.46" in render(_store(BIG=123456789))


def test_storage_still_refuses_floats():
    import pytest
    with pytest.raises(TypeError):
        Store().add("A1", 12.5)


def test_storage_still_reports_us_cents():
    """Whatever the fix, the store's own contract is unchanged: it holds
    US cents, and the conversion happens downstream of it."""
    assert _store(A1=1250).amount("A1") == 1250
    assert _store(A1=1250, A2=500).total() == 1750
