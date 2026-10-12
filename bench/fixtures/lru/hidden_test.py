"""Hidden accuracy tests for the `lru` task."""
import pytest
from lru import LRUCache


def test_rejects_a_nonsense_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)
    with pytest.raises(ValueError):
        LRUCache(-1)


def test_get_and_put_round_trip():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_missing_key_raises():
    cache = LRUCache(2)
    with pytest.raises(KeyError):
        cache.get("nope")
    with pytest.raises(KeyError):
        cache.peek("nope")


def test_len_tracks_contents():
    cache = LRUCache(3)
    assert len(cache) == 0
    cache.put("a", 1)
    cache.put("b", 2)
    assert len(cache) == 2


def test_evicts_the_least_recently_used():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    with pytest.raises(KeyError):
        cache.get("a")
    assert cache.get("b") == 2 and cache.get("c") == 3


def test_get_refreshes_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")          # "b" is now the least recently used
    cache.put("c", 3)
    assert cache.get("a") == 1
    with pytest.raises(KeyError):
        cache.get("b")


def test_updating_an_existing_key_does_not_evict():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 99)      # an update, not an insertion
    assert len(cache) == 2
    assert cache.get("b") == 2
    assert cache.get("a") == 99


def test_updating_refreshes_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)      # "a" is now most recent, so "b" should go
    cache.put("c", 3)
    assert cache.get("a") == 10
    with pytest.raises(KeyError):
        cache.get("b")


def test_peek_does_not_change_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.peek("a") == 1     # must NOT refresh "a"
    cache.put("c", 3)
    with pytest.raises(KeyError):
        cache.get("a")
    assert cache.get("b") == 2


def test_capacity_one_replaces_every_time():
    cache = LRUCache(1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert len(cache) == 1
    assert cache.get("b") == 2


def test_none_is_a_storable_value():
    cache = LRUCache(2)
    cache.put("a", None)
    assert cache.get("a") is None
    assert len(cache) == 1
