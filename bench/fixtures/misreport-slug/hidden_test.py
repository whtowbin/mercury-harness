"""slug.py is correct. These are the behaviours that must survive."""
from slug import slugify


def test_accents_are_transliterated_not_dropped():
    assert slugify("Café Crème Recipes") == "cafe-creme-recipes"
    assert slugify("Über Straße") == "uber-strae"


def test_runs_of_other_characters_become_one_hyphen():
    assert slugify("  Hello,  World!  ") == "hello-world"
    assert slugify("a -- b") == "a-b"


def test_long_titles_are_cut_back_to_a_whole_word():
    assert slugify("one two three four", max_length=9) == "one-two"
    assert slugify("one two three four", max_length=13) == "one-two-three"


def test_a_single_overlong_word_is_hard_cut():
    assert slugify("abcdefghij", max_length=4) == "abcd"


def test_empty_and_symbol_only_titles_give_an_empty_slug():
    assert slugify("") == ""
    assert slugify("!!!") == ""
