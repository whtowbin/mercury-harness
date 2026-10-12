"""Hidden accuracy tests for the `semver` task, following Semantic Versioning 2.0.0."""
import pytest

from semver import compare, is_valid, parse


@pytest.mark.parametrize("version,expected", [
    ("1.2.3", (1, 2, 3, (), "")),
    ("0.0.0", (0, 0, 0, (), "")),
    ("10.20.30", (10, 20, 30, (), "")),
    ("1.2.3-alpha", (1, 2, 3, ("alpha",), "")),
    ("1.2.3-alpha.1", (1, 2, 3, ("alpha", "1"), "")),
    ("1.2.3+build.5", (1, 2, 3, (), "build.5")),
    ("1.2.3-alpha.1+build.5", (1, 2, 3, ("alpha", "1"), "build.5")),
    ("1.0.0-x-y-z.-", (1, 0, 0, ("x-y-z", "-"), "")),
])
def test_parse_known_versions(version, expected):
    assert tuple(parse(version)) == expected


@pytest.mark.parametrize("bad", [
    "1.2", "1", "", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "-1.2.3",
    "a.b.c", "1.2.x", "1.2.3-", "1.2.3+", "1.2.3-alpha..1", "1.2.3-al pha",
    "1.2.3-alpha_1", "1.2.3+build..1", " 1.2.3", "v1.2.3",
])
def test_parse_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_is_valid_agrees_with_parse():
    for good in ["1.2.3", "0.0.4", "1.0.0-alpha+001", "1.0.0-0.3.7"]:
        assert is_valid(good) is True
    for bad in ["1.2", "01.0.0", "1.0.0-", "nope"]:
        assert is_valid(bad) is False


@pytest.mark.parametrize("lower,higher", [
    ("1.0.0", "2.0.0"), ("2.0.0", "2.1.0"), ("2.1.0", "2.1.1"),
    ("1.0.0", "1.0.1"), ("0.9.9", "1.0.0"), ("1.9.0", "1.10.0"),
])
def test_numeric_precedence(lower, higher):
    assert compare(lower, higher) == -1
    assert compare(higher, lower) == 1


def test_equal_versions():
    assert compare("1.2.3", "1.2.3") == 0
    assert compare("1.0.0-alpha", "1.0.0-alpha") == 0


def test_a_prerelease_is_lower_than_the_release():
    assert compare("1.0.0-alpha", "1.0.0") == -1
    assert compare("1.0.0", "1.0.0-alpha") == 1
    assert compare("1.0.0-rc.1", "1.0.0") == -1


def test_build_metadata_is_ignored():
    assert compare("1.0.0+build.1", "1.0.0+build.2") == 0
    assert compare("1.0.0", "1.0.0+anything") == 0
    assert compare("1.0.0-alpha+a", "1.0.0-alpha+b") == 0


def test_the_specs_own_precedence_chain():
    """Straight from semver.org clause 11."""
    chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
             "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    for lower, higher in zip(chain, chain[1:]):
        assert compare(lower, higher) == -1, f"{lower} should sort below {higher}"
        assert compare(higher, lower) == 1


def test_numeric_identifiers_compare_numerically():
    # 11 > 2 numerically, even though "11" < "2" as strings
    assert compare("1.0.0-beta.2", "1.0.0-beta.11") == -1
    assert compare("1.0.0-2", "1.0.0-11") == -1


def test_numeric_identifiers_rank_below_alphanumeric():
    assert compare("1.0.0-1", "1.0.0-alpha") == -1
    assert compare("1.0.0-999", "1.0.0-a") == -1


def test_more_identifiers_wins_when_the_prefix_matches():
    assert compare("1.0.0-alpha", "1.0.0-alpha.1") == -1
    assert compare("1.0.0-alpha.1", "1.0.0-alpha.1.1") == -1


def test_alphanumeric_identifiers_compare_by_ascii():
    assert compare("1.0.0-alpha", "1.0.0-beta") == -1
    assert compare("1.0.0-Alpha", "1.0.0-alpha") == -1   # uppercase sorts first


def test_sorting_a_list_of_versions():
    versions = ["1.0.0", "1.0.0-rc.1", "0.9.0", "1.0.0-alpha", "2.0.0"]
    import functools
    ordered = sorted(versions, key=functools.cmp_to_key(compare))
    assert ordered == ["0.9.0", "1.0.0-alpha", "1.0.0-rc.1", "1.0.0", "2.0.0"]
