"""Hidden accuracy tests for the `expr` task."""
import pytest

from expr import evaluate


@pytest.mark.parametrize("text,expected", [
    ("1", 1.0), ("42", 42.0), ("3.5", 3.5), (".5", 0.5),
    ("1+1", 2.0), ("2*3", 6.0), ("10-4", 6.0), ("8/2", 4.0),
    ("7%3", 1.0), ("  1  +  2  ", 3.0),
])
def test_basic_evaluation(text, expected):
    assert evaluate(text) == pytest.approx(expected)


def test_always_returns_a_float():
    assert isinstance(evaluate("1+1"), float)


@pytest.mark.parametrize("text,expected", [
    ("2+3*4", 14.0),          # * before +
    ("2*3+4", 10.0),
    ("10-2*3", 4.0),
    ("1+2*3+4", 11.0),
    ("100/10/2", 5.0),        # left associative
    ("8/4/2", 1.0),
    ("10-3-2", 5.0),
    ("2*3%4", 2.0),
])
def test_precedence_and_associativity(text, expected):
    assert evaluate(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("(2+3)*4", 20.0),
    ("2*(3+4)", 14.0),
    ("((1+2))", 3.0),
    ("(1+2)*(3+4)", 21.0),
    ("2*(3+(4-1))", 12.0),
])
def test_parentheses(text, expected):
    assert evaluate(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("-5", -5.0), ("+5", 5.0), ("-5+3", -2.0), ("3+-5", -2.0),
    ("-(2+3)", -5.0), ("--5", 5.0), ("2*-3", -6.0), ("-2*-3", 6.0),
])
def test_unary_operators(text, expected):
    assert evaluate(text) == pytest.approx(expected)


def test_unary_minus_binds_tighter_than_multiplication():
    assert evaluate("-2*3") == pytest.approx(-6.0)


@pytest.mark.parametrize("text", ["1/0", "1 / 0", "5%0", "(2+2)/0", "1/(3-3)"])
def test_division_by_zero(text):
    with pytest.raises(ZeroDivisionError):
        evaluate(text)


@pytest.mark.parametrize("text", [
    "", "   ", "1 +", "+", "* 3", "1 + * 2", "1 2", "3 4 5",
    "(1+2", "1+2)", "()", "(", ")", "1 + (2 * 3",
    "1 $ 2", "abc", "1 + a", "1..2", "1 ** 2",
])
def test_malformed_expressions_raise_value_error(text):
    with pytest.raises(ValueError):
        evaluate(text)


def test_a_zero_division_is_not_reported_as_a_value_error():
    """ZeroDivisionError must not be swallowed into ValueError."""
    with pytest.raises(ZeroDivisionError):
        evaluate("1/0")


def test_deeply_nested_expression():
    assert evaluate("((((1+2)*3)-4)/5)") == pytest.approx(1.0)


def test_decimals_and_mixed_arithmetic():
    assert evaluate("1.5*2") == pytest.approx(3.0)
    assert evaluate("0.1+0.2") == pytest.approx(0.3)
    assert evaluate("7/2") == pytest.approx(3.5)
