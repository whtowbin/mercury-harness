from app import bill_total


def test_bill_total_adds_tax():
    assert bill_total([1000, 500], 8) == 1620
