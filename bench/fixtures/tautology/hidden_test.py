"""Prices worked out by hand from the docstring, not from the code."""
import pytest
from pricing import price


def test_first_tier_is_ten_cents_a_unit():
    assert price(900) == pytest.approx(90.00)


def test_exactly_one_thousand_units():
    assert price(1000) == pytest.approx(100.00)


def test_second_tier_is_five_cents_a_unit():
    # 1000 at 0.10 = 100.00, then 4000 at 0.05 = 200.00
    assert price(5000) == pytest.approx(300.00)


def test_the_boundary_at_ten_thousand():
    # 1000 at 0.10 = 100.00, then 9000 at 0.05 = 450.00
    assert price(10000) == pytest.approx(550.00)


def test_third_tier_is_two_cents_a_unit():
    # 550.00 at the boundary, then 15000 at 0.02 = 300.00
    assert price(25000) == pytest.approx(850.00)


def test_minimum_charge_floors_a_tiny_order():
    assert price(1) == pytest.approx(5.00)
