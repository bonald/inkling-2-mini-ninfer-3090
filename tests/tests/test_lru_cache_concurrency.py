"""Thread-safety tests. Each worker is wrapped so exceptions propagate to the
main thread; joined threads are asserted to have terminated (deadlock check).
"""

from __future__ import annotations

import threading

from lru_cache import LRUCache


def _run_concurrent(worker, *, n_threads, timeout=30.0):
    errors: list[BaseException] = []

    def target(tid):
        try:
            worker(tid)
        except BaseException as exc:  # noqa: BLE001 - re-raised in main thread
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(t,), daemon=True) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    stuck = [t for t in threads if t.is_alive()]
    assert not stuck, f"{len(stuck)} thread(s) did not terminate in time"
    assert not errors, f"worker(s) raised: {errors!r}"


def test_concurrent_writes_and_gets_shared_key_space():
    cache = LRUCache[str, int](capacity=128)
    keys = [f"k{i:02d}" for i in range(60)]

    def worker(t):
        for i in range(200):
            k = keys[(t * 31 + i) % len(keys)]
            cache[k] = t * 1000 + i
            cache.get(k)

    _run_concurrent(worker, n_threads=6)

    # Capacity >= distinct keys -> nothing evicted.
    assert len(cache) == len(keys)
    assert set(k for k, _ in cache.items()) == set(keys)


def test_concurrent_eviction_under_stress():
    cache = LRUCache[int, int](capacity=8)

    def worker(t):
        for i in range(400):
            k = (t * 97 + i) % 32
            cache[k] = t * 100_000 + i
            cache.get(k)

    _run_concurrent(worker, n_threads=6)

    assert len(cache) <= 8
    items = cache.items()
    assert len(items) == len(cache)
    keys = [k for k, _ in items]
    # Duplicated keys in the snapshot would reveal lost writes / list corruption.
    assert len(set(keys)) == len(keys)


def test_concurrent_delete_and_insert():
    cache = LRUCache[int, int](capacity=6)

    def worker(t):
        for i in range(300):
            k = (t * 13 + i) % 6
            if i % 5 == 0:
                try:
                    del cache[k]
                except KeyError:
                    pass
            else:
                cache[k] = t * 10_000 + i

    _run_concurrent(worker, n_threads=4)

    assert len(cache) <= 6
    items = cache.items()
    assert len({k for k, _ in items}) == len(items) == len(cache)


def test_capacity_one_ping_pong_between_two_keys():
    cache = LRUCache[str, int](capacity=1)

    def worker(t):
        key = "A" if t % 2 == 0 else "B"
        for i in range(500):
            cache[key] = t * 1000 + i
            cache.get("A")
            cache.get("B")

    _run_concurrent(worker, n_threads=2)

    assert len(cache) == 1  # capacity one: the cache can never hold more than one entry


def test_snapshot_consistency_during_concurrent_writes():
    cache = LRUCache[int, int](capacity=10)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        try:
            i = 0
            while not stop.is_set():
                cache[i % 10] = i
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(300):
                snap = cache.items()
                # Every call must return an internally consistent snapshot.
                assert len(snap) == len({k for k, _ in snap})
                assert len(snap) <= 10
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer_t = threading.Thread(target=writer, daemon=True)
    reader_t = threading.Thread(target=reader, daemon=True)
    writer_t.start()
    reader_t.start()
    reader_t.join(30)
    stop.set()
    writer_t.join(10)

    assert not reader_t.is_alive() and not writer_t.is_alive()
    assert not errors
