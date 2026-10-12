"""scripts/gaia/gaia.py's scorer keeps GAIA's quasi-exact match (the leaderboard's scorer.py).

The cases are made up, in the answer shapes the validation set uses: plain numbers, money and
percentages, comma lists mixing numbers and words, and short strings. None is a GAIA answer:
the dataset may not be reshared outside a gated repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "gaia", Path(__file__).resolve().parents[1] / "scripts" / "gaia" / "gaia.py"
)
assert _spec and _spec.loader
gaia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gaia)


@pytest.mark.parametrize(
    ("answer", "gold"),
    [
        ("42", "42"),
        ("42.0", "42"),
        ("2,500", "2500"),
        ("$12.50", "12.5"),
        ("30%", "30"),
        ("Left", "left"),
        ("St.Louis", "St. Louis"),
        ("the  Rolling Stones", "The Rolling Stones"),
        ("red, blue, green", "red, blue, green"),
        ("5; 6", "5, 6"),
        ("x,y", "x, y"),
    ],
)
def test_matches(answer, gold):
    assert gaia.question_scorer(answer, gold)


@pytest.mark.parametrize(
    ("answer", "gold"),
    [
        ("42 apples", "42"),
        ("43", "42"),
        ("red, blue", "red, blue, green"),
        ("blue, red, green", "red, blue, green"),
        ("St. Louis", "Saint Louis"),
        (None, "42"),
        ("", "Lisbon"),
    ],
)
def test_refuses(answer, gold):
    assert not gaia.question_scorer(answer, gold)
