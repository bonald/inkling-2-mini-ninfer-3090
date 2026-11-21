"""Thread-safe O(1) LRU cache, implemented from scratch.

Data structures
---------------
* A doubly linked list, ordered from most recently used (head) to least
  recently used (tail). Two sentinel nodes bracket the list, so the list
  is never empty and no operation needs special-cased "is empty" logic.
* A hash map from key to node for O(1) lookup.

Thread safety
-------------
Every public method is serialized through a single reentrant lock, so all
public operations are amortized O(1) and safe to call from any thread.
Snapshot helpers (``items``, ``keys`` via ``__iter__``, ``__repr__``) are
O(n) and always take a consistent snapshot under the lock.

Access semantics
----------------
``get``, ``__getitem__``, and ``__setitem__`` (including on an existing
key) count as an "access": the entry is moved to most-recently-used.
``__contains__`` is a pure membership check and does *not* reorder.
"""

from __future__ import annotations

import threading
from typing import Any, Generic, Iterator, TypeVar, cast, overload

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type
_D = TypeVar("_D")  # default-value type


class _Node(Generic[K, V]):
    """Doubly linked list node.

    Sentinels are constructed with ``key=None, value=None``; real nodes
    always carry a non-``None`` key and value.
    """

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: K | None, value: V | None) -> None:
        self.key = key
        self.value = value
        self.prev: _Node[K, V] | None = None
        self.next: _Node[K, V] | None = None


class LRUCache(Generic[K, V]):
    """Fixed-capacity least-recently-used cache.

    Stores at most ``capacity`` entries; adding an entry to a full cache
    evicts the least recently used entry.
    """

    __slots__ = ("_capacity", "_map", "_head", "_tail", "_size", "_lock")

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity must be an int, got {type(capacity).__name__}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._capacity = capacity
        self._map: dict[K, _Node[K, V]] = {}

        # Sentinel nodes bracket the list: head.next is the MRU node and
        # tail.prev is the LRU node. They are never stored in the map.
        self._head: _Node[K, V] = _Node(None, None)
        self._tail: _Node[K, V] = _Node(None, None)
        self._head.next = self._tail
        self._tail.prev = self._head

        self._size = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of entries the cache can hold."""
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._map

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys, most recently used first.

        Returns a consistent snapshot; later mutations of the cache are
        not reflected in the iterator.
        """
        with self._lock:
            return iter([cast(K, node.key) for node in self._snapshot_nodes()])

    def __getitem__(self, key: K) -> V:
        """Return ``value`` for ``key`` (raising ``KeyError`` if missing).

        Counts as an access: the entry moves to most recently used.
        """
        with self._lock:
            node = self._map.get(key)
            if node is None:
                raise KeyError(key)
            self._move_to_head(node)
            return cast(V, node.value)

    def __setitem__(self, key: K, value: V) -> None:
        """Insert ``key``/``value``.

        On an existing ``key``, replaces the value and refreshes the
        entry's recency. On a new ``key``, inserts at the MRU position and
        evicts the LRU entry if the cache is full.
        """
        with self._lock:
            node = self._map.get(key)
            if node is not None:
                node.value = value
                self._move_to_head(node)
                return
            node = _Node(key, value)
            self._map[key] = node
            self._insert_head(node)
            self._size += 1
            if self._size > self._capacity:
                self._evict_tail()

    def __delitem__(self, key: K) -> None:
        with self._lock:
            node = self._map.pop(key, None)
            if node is None:
                raise KeyError(key)
            self._unlink(node)
            self._size -= 1

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: _D) -> V | _D: ...

    def get(self, key: K, default: Any = None) -> Any:
        """Return ``value`` for ``key`` or ``default`` if missing.

        Counts as an access: the entry moves to most recently used.
        """
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return default
            self._move_to_head(node)
            return cast(V, node.value)

    def items(self) -> list[tuple[K, V]]:
        """Return a consistent snapshot of ``(key, value)`` pairs, MRU first."""
        with self._lock:
            return [(cast(K, node.key), cast(V, node.value)) for node in self._snapshot_nodes()]

    def clear(self) -> None:
        """Remove all entries. Capacity is preserved."""
        with self._lock:
            self._map.clear()
            self._head.next = self._tail
            self._tail.prev = self._head
            self._size = 0

    def __repr__(self) -> str:
        with self._lock:
            pairs = [(cast(K, node.key), cast(V, node.value)) for node in self._snapshot_nodes()]
            return f"LRUCache(capacity={self._capacity}, items={pairs!r})"

    # ------------------------------------------------------------------
    # Internal helpers (lock must be held by the caller)
    # ------------------------------------------------------------------

    def _unlink(self, node: _Node[K, V]) -> None:
        prev = node.prev
        nxt = node.next
        assert prev is not None and nxt is not None
        prev.next = nxt
        nxt.prev = prev
        node.prev = None
        node.next = None

    def _insert_head(self, node: _Node[K, V]) -> None:
        assert self._head.next is not None  # tail sentinel is always linked
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node[K, V]) -> None:
        if node is self._head.next:
            return  # already most recently used
        self._unlink(node)
        self._insert_head(node)

    def _evict_tail(self) -> None:
        lru = self._tail.prev
        assert lru is not None and lru is not self._head
        self._unlink(lru)
        key = lru.key
        assert key is not None
        del self._map[key]
        self._size -= 1

    def _snapshot_nodes(self) -> list[_Node[K, V]]:
        nodes: list[_Node[K, V]] = []
        node: _Node[K, V] | None = self._head.next
        while node is not None and node is not self._tail:
            nodes.append(node)
            node = node.next
        return nodes
