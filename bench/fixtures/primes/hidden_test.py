"""Hidden accuracy tests for the `primes` task."""
import pytest
from primes import count_primes, is_prime

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 101]
NOT_PRIMES = [-7, -1, 0, 1, 4, 6, 8, 9, 15, 21, 25, 27, 100, 1001]


@pytest.mark.parametrize("n", SMALL_PRIMES)
def test_recognises_primes(n):
    assert is_prime(n) is True


@pytest.mark.parametrize("n", NOT_PRIMES)
def test_rejects_non_primes(n):
    assert is_prime(n) is False


def test_large_prime_and_composite():
    assert is_prime(104729) is True       # the 10000th prime
    assert is_prime(104730) is False


def test_perfect_square_of_a_prime_is_not_prime():
    assert is_prime(9409) is False        # 97 * 97


@pytest.mark.parametrize("limit,expected", [
    (0, 0), (1, 0), (2, 0), (3, 1), (10, 4), (100, 25), (1000, 168),
    (10_000, 1229), (100_000, 9592),
])
def test_count_primes_known_values(limit, expected):
    assert count_primes(limit) == expected


def test_count_primes_is_exclusive_of_the_limit():
    # 7 is prime, so counting below 7 excludes it: 2, 3, 5
    assert count_primes(7) == 3
    assert count_primes(8) == 4


def test_count_primes_rejects_a_negative_limit():
    with pytest.raises(ValueError):
        count_primes(-1)


def test_count_primes_agrees_with_is_prime():
    assert count_primes(500) == sum(1 for n in range(500) if is_prime(n))
