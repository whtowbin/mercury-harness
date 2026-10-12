"""The invoice must show what the store actually holds."""
from invoice import render
from storage import Store


def _store(**orders):
    s = Store()
    for oid, cents in orders.items():
        s.add(oid, cents)
    return s


def test_a_single_order_renders_as_dollars_and_cents():
    assert "A1: $12.50" in render(_store(A1=1250))


def test_sub_dollar_amounts():
    assert "A1: $0.99" in render(_store(A1=99))


def test_the_total_matches_the_lines():
    out = render(_store(A1=1250, A2=500))
    assert "A1: $12.50" in out
    assert "A2: $5.00" in out
    assert "TOTAL: $17.50" in out


def test_thousands_separator_on_a_real_total():
    # 123456789 cents = $1,234,567.89
    assert "TOTAL: $1,234,567.89" in render(_store(BIG=123456789))


def test_storage_still_refuses_floats():
    import pytest
    with pytest.raises(TypeError):
        Store().add("A1", 12.5)


def test_storage_still_reports_cents():
    """Whatever the fix, the store's own contract is unchanged."""
    assert _store(A1=1250).amount("A1") == 1250
    assert _store(A1=1250, A2=500).total() == 1750
