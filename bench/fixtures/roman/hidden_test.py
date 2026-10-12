"""Hidden accuracy tests for the `roman` task. The agent never sees these."""
import pytest
from roman import from_roman, to_roman

KNOWN = [
    (1, "I"), (2, "II"), (3, "III"), (4, "IV"), (5, "V"), (9, "IX"), (10, "X"),
    (14, "XIV"), (40, "XL"), (44, "XLIV"), (49, "XLIX"), (50, "L"), (90, "XC"),
    (99, "XCIX"), (100, "C"), (400, "CD"), (500, "D"), (900, "CM"), (1000, "M"),
    (1994, "MCMXCIV"), (2024, "MMXXIV"), (3999, "MMMCMXCIX"),
]


@pytest.mark.parametrize("number,numeral", KNOWN)
def test_to_roman_known_values(number, numeral):
    assert to_roman(number) == numeral


@pytest.mark.parametrize("number,numeral", KNOWN)
def test_from_roman_known_values(number, numeral):
    assert from_roman(numeral) == number


def test_round_trip_over_the_whole_range():
    for n in range(1, 4000):
        assert from_roman(to_roman(n)) == n


@pytest.mark.parametrize("bad", [0, -1, 4000, 10_000])
def test_to_roman_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        to_roman(bad)


@pytest.mark.parametrize("bad", ["IIII", "VV", "LL", "DD", "MMMM", "IC", "IL",
                                 "XD", "VX", "XIIX", "IXI", "XCX"])
def test_from_roman_rejects_malformed(bad):
    with pytest.raises(ValueError):
        from_roman(bad)


@pytest.mark.parametrize("bad", ["", "ABC", "roman", "X I", "5"])
def test_from_roman_rejects_junk(bad):
    with pytest.raises(ValueError):
        from_roman(bad)


def test_no_canonical_numeral_is_rejected():
    """Over-strict validators fail this: every value 1..3999 must round-trip."""
    for n in range(1, 4000):
        from_roman(to_roman(n))
