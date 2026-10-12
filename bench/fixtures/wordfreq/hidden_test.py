"""Hidden accuracy tests for the `wordfreq` task."""
from collections import Counter

from wordfreq import count_words, normalize, top_n


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Hello, World!") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize("a   b\n\tc") == "a b c"


def test_normalize_removes_internal_apostrophes():
    assert normalize("don't stop") == "dont stop"


def test_normalize_of_empty_text():
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_count_words_returns_a_counter():
    counts = count_words("a b a")
    assert isinstance(counts, Counter)
    assert counts["a"] == 2 and counts["b"] == 1


def test_count_words_is_case_insensitive():
    assert count_words("The the THE")["the"] == 3


def test_top_n_orders_by_count_descending():
    assert top_n("a a a b b c", 3) == [("a", 3), ("b", 2), ("c", 1)]


def test_top_n_breaks_ties_alphabetically():
    # every word appears once, so the order is purely alphabetical
    assert top_n("delta charlie bravo alpha", 4) == [
        ("alpha", 1), ("bravo", 1), ("charlie", 1), ("delta", 1)]


def test_top_n_mixed_counts_and_ties():
    text = "zebra zebra apple apple mango kiwi"
    assert top_n(text, 4) == [("apple", 2), ("zebra", 2), ("kiwi", 1), ("mango", 1)]


def test_top_n_zero_returns_empty():
    assert top_n("a b c", 0) == []


def test_top_n_beyond_the_vocabulary_returns_everything():
    assert len(top_n("a b c", 99)) == 3


def test_top_n_on_empty_text():
    assert top_n("", 5) == []


def test_punctuation_does_not_split_or_survive():
    assert count_words("hello, hello. hello!")["hello"] == 3
