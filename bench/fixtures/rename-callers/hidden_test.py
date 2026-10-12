"""line_item(name, qty, price) replaces make_line everywhere; nothing else changes."""

import importlib
import inspect

import pytest

MODULES = ("pricing", "cart", "invoice", "promo", "report", "export")


def test_line_item_takes_name_first():
    from pricing import line_item

    assert list(inspect.signature(line_item).parameters)[:3] == ["name", "qty", "price"]
    assert line_item("Mug", 3, 200) == {"name": "Mug", "qty": 3, "price": 200, "total": 600}


def test_line_item_still_refuses_a_negative_qty():
    from pricing import line_item

    with pytest.raises(ValueError):
        line_item("Mug", -1, 200)


@pytest.mark.parametrize("module", MODULES)
def test_make_line_is_gone(module):
    assert not hasattr(importlib.import_module(module), "make_line")


def test_cart():
    from cart import add_gift_wrap, cart_lines, cart_total

    assert cart_total([("Mug", 2, 800), ("Tea", 1, 450)]) == 2050
    lines = add_gift_wrap(cart_lines([("Mug", 1, 800)]), 2)
    assert lines[-1] == {"name": "Gift wrap", "qty": 2, "price": 250, "total": 500}


def test_invoice():
    from invoice import render

    out = render({"items": [{"name": "Mug", "qty": 2, "price": 800}], "ship": True})
    assert out.splitlines() == ["Mug x2: 1600", "Shipping x1: 499", "TOTAL: 2099"]


def test_promo():
    from promo import bundle, discount_line

    paid, free = bundle("Soap", 7, 300)
    assert (paid["name"], paid["qty"], paid["total"]) == ("Soap", 5, 1500)
    assert (free["name"], free["qty"]) == ("Soap (free)", 2)
    assert bundle("Soap", 2, 300)[0]["total"] == 600
    assert discount_line(15, 2000) == {
        "name": "Discount 15%",
        "qty": 1,
        "price": -300,
        "total": -300,
    }


def test_report():
    from report import best_seller, summary_rows

    sales = [("mon", "Tea", 3, 450), ("tue", "Mug", 1, 800), ("wed", "Mug", 1, 800)]
    assert summary_rows(sales)[1] == ("tue", {"name": "Mug", "qty": 1, "price": 800, "total": 800})
    assert best_seller(sales) == "Mug"


def test_export():
    from export import sample_basket, to_csv

    assert to_csv([("Tea", 2, 150), ("Jam", 1, 90)]).splitlines() == [
        "name,qty,price,total",
        "Tea,2,150,300",
        "Jam,1,90,90",
    ]
    assert [(line["name"], line["total"]) for line in sample_basket()] == [
        ("Tea", 300),
        ("Scone", 325),
        ("Jam", 270),
    ]
