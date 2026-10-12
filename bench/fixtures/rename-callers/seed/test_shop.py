from cart import cart_total
from export import sample_basket, to_csv
from invoice import render
from promo import bundle, discount_line
from report import best_seller, summary_rows


def test_cart_total():
    assert cart_total([("Mug", 2, 800), ("Tea", 1, 450)]) == 2050


def test_invoice_with_shipping():
    out = render({"items": [{"name": "Mug", "qty": 1, "price": 800}], "ship": True})
    assert out.splitlines()[-1] == "TOTAL: 1299"


def test_bundle_splits_out_the_free_units():
    paid, free = bundle("Soap", 6, 300)
    assert (paid["qty"], free["qty"], free["total"]) == (4, 2, 0)


def test_discount_is_negative():
    assert discount_line(10, 5000)["total"] == -500


def test_report():
    sales = [("mon", "Tea", 3, 450), ("tue", "Mug", 1, 800)]
    assert summary_rows(sales)[0][1]["total"] == 1350
    assert best_seller(sales) == "Tea"


def test_export():
    assert to_csv([("Tea", 2, 150)]).splitlines()[1] == "Tea,2,150,300"
    assert sum(line["total"] for line in sample_basket()) == 895
