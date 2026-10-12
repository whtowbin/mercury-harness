"""Hidden tests for `inventory`, spanning models.py, store.py and report.py."""
import pytest

from models import Item
from report import format_report
from store import InventoryStore


# -- models ----------------------------------------------------------------

def test_item_holds_its_fields():
    item = Item(sku="A1", name="Widget", price=9.99, qty=5)
    assert (item.sku, item.name, item.price, item.qty) == ("A1", "Widget", 9.99, 5)


@pytest.mark.parametrize("kwargs", [
    {"sku": "", "name": "x", "price": 1.0, "qty": 1},
    {"sku": "A1", "name": "x", "price": -1.0, "qty": 1},
    {"sku": "A1", "name": "x", "price": 1.0, "qty": -1},
])
def test_item_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        Item(**kwargs)


def test_item_allows_zero_price_and_zero_qty():
    Item(sku="FREE", name="Sample", price=0.0, qty=0)


# -- store -------------------------------------------------------------

def test_add_and_total_value():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=2.0, qty=3))
    store.add_item(Item(sku="B1", name="Gadget", price=5.0, qty=2))
    assert store.total_value() == pytest.approx(16.0)


def test_add_rejects_duplicate_sku():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=1.0, qty=1))
    with pytest.raises(ValueError):
        store.add_item(Item(sku="A1", name="Other", price=2.0, qty=1))


def test_remove_qty_reduces_stock():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=2.0, qty=10))
    store.remove_qty("A1", 4)
    assert store.total_value() == pytest.approx(12.0)


def test_remove_qty_rejects_unknown_sku():
    store = InventoryStore()
    with pytest.raises(ValueError):
        store.remove_qty("NOPE", 1)


def test_remove_qty_rejects_more_than_available():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=2.0, qty=3))
    with pytest.raises(ValueError):
        store.remove_qty("A1", 4)


def test_remove_all_stock_is_allowed():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=2.0, qty=3))
    store.remove_qty("A1", 3)
    assert store.total_value() == pytest.approx(0.0)


def test_low_stock_is_sorted_and_filtered():
    store = InventoryStore()
    store.add_item(Item(sku="C1", name="c", price=1.0, qty=1))
    store.add_item(Item(sku="A1", name="a", price=1.0, qty=0))
    store.add_item(Item(sku="B1", name="b", price=1.0, qty=100))
    assert store.low_stock(5) == ["A1", "C1"]


def test_low_stock_threshold_is_exclusive_of_equal():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="a", price=1.0, qty=5))
    assert store.low_stock(5) == []
    assert store.low_stock(6) == ["A1"]


def test_empty_store_total_value_is_zero():
    assert InventoryStore().total_value() == 0.0


# -- report --------------------------------------------------------------

def test_report_is_sorted_by_sku_with_a_total_line():
    store = InventoryStore()
    store.add_item(Item(sku="B1", name="Gadget", price=5.0, qty=2))
    store.add_item(Item(sku="A1", name="Widget", price=2.5, qty=4))
    lines = format_report(store).strip().splitlines()
    assert lines[0].startswith("A1")
    assert lines[1].startswith("B1")
    assert lines[-1].startswith("TOTAL")


def test_report_values_are_two_decimal_places():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=1.005, qty=1))
    report = format_report(store)
    assert "1.00" in report or "1.01" in report  # rounding direction is not specified
    assert "TOTAL" in report


def test_report_of_an_empty_store_still_has_a_total_line():
    report = format_report(InventoryStore()).strip()
    assert report.splitlines()[-1].startswith("TOTAL")
    assert "0.00" in report


def test_report_reflects_removed_stock():
    store = InventoryStore()
    store.add_item(Item(sku="A1", name="Widget", price=2.0, qty=10))
    store.remove_qty("A1", 6)
    report = format_report(store)
    assert "4" in report.splitlines()[0]
