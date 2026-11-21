"""Single-threaded functional tests for LRUCache."""

from __future__ import annotations

import random
from collections import OrderedDict

import pytest

from lru_cache import LRUCache


def order(cache):
    """Return keys in current MRU -> LRU order."""
    return [k for k, _ in cache.items()]


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -100])
def test_capacity_must_be_positive(bad):
    with pytest.raises(ValueError):
        LRUCache(bad)


@pytest.mark.parametrize("bad", [True, "3", 2.5, None, [3]])
def test_capacity_must_be_an_int(bad):
    with pytest.raises(TypeError):
        LRUCache(bad)


def test_empty_cache():
    cache = LRUCache[str, int](capacity=3)
    assert len(cache) == 0
    assert "a" not in cache
    assert cache.get("a") is None
    assert list(cache) == []
    assert cache.items() == []
    assert cache.capacity == 3


def test_capacity_is_read_only():
    cache = LRUCache[str, int](capacity=3)
    with pytest.raises(AttributeError):
        cache.capacity = 5


# ----------------------------------------------------------------------
# Basic operations
# ----------------------------------------------------------------------

def test_insert_and_get():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert len(cache) == 2
    assert order(cache) == ["b", "a"]  # most recently inserted first


def test_get_missing_key_returns_default():
    cache = LRUCache[str, int](capacity=2)
    assert cache.get("nope") is None
    assert cache.get("nope", 0) == 0
    assert cache.get("nope", "fallback") == "fallback"


def test_get_present_key_returns_value_not_default():
    cache = LRUCache[str, int](capacity=2)
    cache["k"] = 42
    assert cache.get("k", 0) == 42


def test_get_is_an_access_and_refreshes_order():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert order(cache) == ["c", "b", "a"]
    cache.get("a")
    assert order(cache) == ["a", "c", "b"]


def test_getitem_raises_keyerror_for_missing():
    cache = LRUCache[str, int](capacity=2)
    with pytest.raises(KeyError):
        cache["missing"]


def test_contains_does_not_refresh_order():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert "a" in cache
    assert order(cache) == ["c", "b", "a"]


def test_get_missing_does_not_change_state():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    before = cache.items()
    cache.get("missing")
    assert cache.items() == before
    assert len(cache) == 2


# ----------------------------------------------------------------------
# Eviction
# ----------------------------------------------------------------------

def test_evicts_least_recently_used():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache.get("a")        # order: a, c, b
    cache["d"] = 4        # evicts b (LRU)
    assert "b" not in cache
    assert "a" in cache and "c" in cache and "d" in cache
    assert len(cache) == 3
    assert order(cache) == ["d", "a", "c"]


def test_eviction_keeps_new_item_as_most_recent():
    cache = LRUCache[str, int](capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3        # evicts a
    assert order(cache) == ["c", "b"]


def test_capacity_one():
    cache = LRUCache[str, int](capacity=1)
    cache["a"] = 1
    assert cache["a"] == 1
    cache["b"] = 2        # evicts a
    assert "a" not in cache
    assert cache["b"] == 2
    assert len(cache) == 1


def test_overwrite_existing_key_keeps_size():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 100      # overwrite, not insert
    assert len(cache) == 2
    assert [k for k, _ in cache.items()] == ["a", "b"]
    assert cache.get("a") == 100


def test_overwrite_of_existing_key_refreshes_recency():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3        # order: c, b, a
    cache["b"] = 20       # b becomes most recent
    assert order(cache) == ["b", "c", "a"]


def test_evicted_key_can_be_reinserted():
    cache = LRUCache[str, int](capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3        # evicts a
    cache["a"] = 10       # re-insert
    assert order(cache) == ["a", "c"]
    assert cache["a"] == 10


# ----------------------------------------------------------------------
# Deletion / clearing
# ----------------------------------------------------------------------

def test_delete_removes_entry_and_preserves_order():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    del cache["b"]
    assert "b" not in cache
    assert order(cache) == ["c", "a"]
    assert len(cache) == 2


def test_delete_missing_key_raises_keyerror():
    cache = LRUCache[str, int](capacity=2)
    with pytest.raises(KeyError):
        del cache["missing"]


def test_clear_empties_cache_and_preserves_capacity():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert list(cache) == []
    assert cache.items() == []
    assert cache.capacity == 3
    cache["c"] = 3
    assert cache["c"] == 3


# ----------------------------------------------------------------------
# Iteration / introspection
# ----------------------------------------------------------------------

def test_iter_yields_keys_most_recent_first():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert list(cache) == ["c", "b", "a"]


def test_items_snapshot_yields_pairs_in_order():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.items() == [("b", 2), ("a", 1)]


def test_bool_reflects_size():
    cache = LRUCache[str, int](capacity=2)
    assert not cache
    cache["a"] = 1
    assert cache


def test_repr_mentions_capacity_and_items():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    r = repr(cache)
    assert "LRUCache" in r
    assert "capacity=3" in r
    assert "b" in r


def test_iteration_snapshot_is_taken_under_lock_and_not_mutated():
    cache = LRUCache[str, int](capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    it = iter(cache)
    first = next(it)
    cache["c"] = 3  # mutates the cache between snapshots
    assert first == "b"  # snapshot was taken before mutation


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_stores_none_as_a_value():
    cache = LRUCache[str, int | None](capacity=2)
    cache["k"] = None
    assert "k" in cache                  # stored, even though value is None
    assert len(cache) == 1
    assert cache["k"] is None            # getitem still works
    assert cache.get("k", "fallback") is None   # stored value wins
    assert cache.get("missing", "fallback") == "fallback"  # true miss
    assert ("k", None) in cache.items()


def test_unhashable_key_raises_type_error():
    cache = LRUCache[list, int](capacity=2)
    with pytest.raises(TypeError):
        cache[[1]] = 1


def test_mixed_key_types_in_separate_caches():
    int_cache = LRUCache[int, str](capacity=2)
    int_cache[1] = "one"
    str_cache = LRUCache[str, str](capacity=2)
    str_cache["1"] = "one"
    assert int_cache[1] == "one"
    assert str_cache["1"] == "one"


# ----------------------------------------------------------------------
# Randomized parity check against an independent reference model
# ----------------------------------------------------------------------

class _Oracle:
    """Reference implementation (LRU -> MRU order), used only as a test oracle."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._d: OrderedDict[str, int] = OrderedDict()

    def get(self, k):
        if k not in self._d:
            return None
        self._d.move_to_end(k)
        return self._d[k]

    def put(self, k, v):
        if k in self._d:
            self._d[k] = v
            self._d.move_to_end(k)
            return
        self._d[k] = v
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def delete(self, k):
        del self._d[k]

    def __contains__(self, k):
        return k in self._d

    def __len__(self):
        return len(self._d)

    def items(self):
        return list(reversed(list(self._d.items())))


@pytest.mark.parametrize("seed", [0, 1, 7, 123, 999])
def test_randomized_parity_with_oracle(seed):
    rng = random.Random(seed)
    capacity = 4
    cache = LRUCache[str, int](capacity=capacity)
    oracle = _Oracle(capacity)
    keys = [f"key-{i}" for i in range(12)]

    for _ in range(500):
        r = rng.random()
        if r < 0.40:
            k = rng.choice(keys)
            assert cache.get(k) == oracle.get(k)
        elif r < 0.85:
            k = rng.choice(keys)
            v = rng.randint(0, 100)
            oracle.put(k, v)
            cache[k] = v
        else:
            k = rng.choice(keys)
            if k in oracle:
                oracle.delete(k)
                del cache[k]
            else:
                with pytest.raises(KeyError):
                    del cache[k]

        assert len(cache) == len(oracle)
        assert cache.items() == oracle.items()
        assert list(cache) == [k for k, _ in oracle.items()]
        for k in keys:
            assert (k in cache) == (k in oracle)
