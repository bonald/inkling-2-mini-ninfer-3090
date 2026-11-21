# Demo: one coding request on the RTX 3090

[`inkling-2-mini-3090.mp4`](inkling-2-mini-3090.mp4) is a real-time screen recording of one chat request to this repository's Inkling 2 Mini server on an RTX 3090. The prompt was:

> Write a thread-safe O(1) Python LRU cache from scratch (using a doubly linked list and hash map, no OrderedDict) with clean type annotations and a comprehensive pytest test suite.

The files in this folder are the code from the model's answer.

| File | Contents |
| --- | --- |
| [`lru_cache.py`](lru_cache.py) | The cache: a doubly linked list with sentinel nodes, a dict index and one reentrant lock |
| [`tests/test_lru_cache.py`](tests/test_lru_cache.py) | Behavior tests, including a randomized comparison against an `OrderedDict` reference |
| [`tests/test_lru_cache_concurrency.py`](tests/test_lru_cache_concurrency.py) | Multithreaded tests |
| [`conftest.py`](conftest.py) | Makes `lru_cache` importable from `tests/` |

Run the tests from this folder with pytest installed:

```sh
python -m pytest tests/ -v
```

On 2026-11-21, all 44 tests passed with Python 3.12 and pytest 9.2.0, and `mypy --strict lru_cache.py` reported no issues.

The dashboard in the recording is a separate local viewer, not part of this repository. Its figures are live readings for this single request. The benchmark results in the [main README](../README.md#benchmarks) come from the controlled runs described in [bench/README.md](../bench/README.md).
