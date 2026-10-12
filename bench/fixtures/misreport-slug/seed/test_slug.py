from slug import slugify


def test_spaces_and_punctuation_become_single_hyphens():
    assert slugify("Hello,  World!") == "hello-world"


def test_long_titles_are_cut_at_a_word():
    assert slugify("one two three four", max_length=9) == "one-two"
