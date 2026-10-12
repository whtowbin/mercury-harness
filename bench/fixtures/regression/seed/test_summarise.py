"""Existing tests. They pass."""
from summarise import preview, truncate


def test_short_text_is_untouched():
    assert truncate("hello", 20) == "hello"


def test_exactly_at_the_limit_is_untouched():
    assert truncate("hello", 5) == "hello"


def test_words_are_not_split():
    assert truncate("alpha beta gamma", 12) == "alpha beta…"


def test_preview_stops_on_a_word_boundary():
    assert preview("alpha beta gamma", 12) == "alpha beta"


def test_preview_adds_no_ellipsis():
    assert "…" not in preview("alpha beta gamma", 12)
