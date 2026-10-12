"""truncate must respect its limit; preview must not change."""
from summarise import ELLIPSIS, preview, truncate


# -- the bug: the ellipsis was not counted against the limit --------------

def test_a_long_word_fits_within_the_limit():
    out = truncate("supercalifragilistic", 10)
    assert len(out) <= 10, f"{out!r} is {len(out)} characters"
    assert out.endswith(ELLIPSIS)


def test_every_limit_is_respected():
    # A long unbroken word, because the bug only shows when there is no space
    # in the cut window to shorten the result by accident.
    text = "antidisestablishmentarianism is a long word"
    for limit in range(4, 30):
        out = truncate(text, limit)
        assert len(out) <= limit, f"limit {limit} gave {out!r} ({len(out)})"


def test_text_within_the_limit_is_untouched():
    assert truncate("hello", 5) == "hello"
    assert truncate("hello", 20) == "hello"
    assert ELLIPSIS not in truncate("hello", 20)


def test_shortened_text_still_ends_on_a_whole_word():
    out = truncate("alpha beta gamma", 13)
    assert out.endswith(ELLIPSIS)
    assert not out[:-1].endswith(" ")
    assert out[:-1] in "alpha beta gamma"


# -- what must not change: preview has its own contract ------------------

def test_preview_never_adds_an_ellipsis():
    assert ELLIPSIS not in preview("alpha beta gamma", 12)


def test_preview_uses_its_whole_limit():
    """preview's limit is a hard cap on the text itself, with no ellipsis to
    make room for. A fix that reserves space inside the shared helper takes a
    character away from here."""
    assert preview("alpha beta gamma", 12) == "alpha beta"
    assert preview("hello", 20) == "hello"
    # The boundary case: at limit 11 the text fills the limit exactly up to a
    # space, so a helper that quietly reserves a character for an ellipsis
    # drops a whole word here while looking right everywhere else.
    assert preview("alpha beta gamma", 11) == "alpha beta"


def test_preview_still_stops_on_a_word_boundary():
    assert " " not in preview("alpha beta gamma", 12)[-1:]
    assert preview("alpha beta gamma", 16) == "alpha beta gamma"
