"""
Task 1 (Easy) — fix_memory_leak
P3 Alert: cache-service memory at 94%, pod restart imminent.

Bug: LRU cache eviction check uses `>` instead of `>=`.
With `>`, when len == max_size, no eviction happens on insert,
so the cache grows to max_size+1 and keeps growing on each insert thereafter
until len finally exceeds max_size (which it never does stably).
Result: cache exceeds max_size; OOM crash after sustained traffic.

Fix: change `>` to `>=` in the eviction guard.
"""

from typing import Dict
from .base import BaseTask


# ── Buggy source ───────────────────────────────────────────────────────────────

_CACHE_MANAGER_BUGGY = '''\
"""LRU Cache implementation for the session cache service."""
from collections import OrderedDict


class LRUCache:
    """Least-Recently-Used cache with a configurable size limit."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str):
        """Return cached value or None if not present."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value) -> None:
        """Insert or update a key-value pair, evicting LRU entry if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            # Evict least-recently-used entry when at capacity
            if len(self._cache) > self.max_size:   # BUG: should be >=
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache
'''

# ── Tests ──────────────────────────────────────────────────────────────────────

_TEST_CACHE = '''\
"""Tests for cache_manager.LRUCache."""
import pytest
from cache_manager import LRUCache


def test_basic_put_get():
    """Basic put and get operations work."""
    cache = LRUCache(max_size=5)
    cache.put("x", 42)
    assert cache.get("x") == 42


def test_cache_miss_returns_none():
    """get() returns None for keys that were never inserted."""
    cache = LRUCache(max_size=5)
    assert cache.get("nonexistent") is None


def test_eviction_keeps_recently_used():
    """The most recently accessed item should NOT be evicted."""
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")          # mark 'a' as recently used
    cache.put("d", 4)       # should evict LRU — which is now 'b'
    assert cache.get("a") == 1, "Recently accessed 'a' must survive eviction"


def test_eviction_removes_lru_item():
    """The least-recently-used item must be evicted when at capacity."""
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)       # 'a' is LRU and must be evicted
    assert cache.get("a") is None, "LRU item 'a' must be evicted"
    assert cache.get("d") == 4,  "Newly inserted 'd' must be present"


def test_size_never_exceeds_max():
    """Cache size must never exceed max_size, even under sustained inserts."""
    max_size = 5
    cache = LRUCache(max_size=max_size)
    for i in range(50):
        cache.put(f"key_{i}", i)
        assert len(cache) <= max_size, (
            f"After insert {i}: cache size {len(cache)} exceeds max_size {max_size}"
        )
'''

# ── Pre-authored logs / metrics ────────────────────────────────────────────────

_LOGS = """\
=== cache-service logs (last 30 lines) ===
2026-04-04 14:20:01 INFO  cache-service  - Starting cache-service (max_size=10000)
2026-04-04 14:20:01 INFO  cache-service  - Cache initialised, size=0
2026-04-04 14:21:15 INFO  cache-service  - Cache size growing: 9998 items
2026-04-04 14:21:40 INFO  cache-service  - Cache size growing: 9999 items
2026-04-04 14:21:41 INFO  cache-service  - Cache size growing: 10000 items
2026-04-04 14:21:42 INFO  cache-service  - Cache size growing: 10001 items  ← exceeds max!
2026-04-04 14:21:43 INFO  cache-service  - Cache size growing: 10002 items
2026-04-04 14:22:01 WARN  cache-service  - Memory usage: 87% (threshold: 80%)
2026-04-04 14:22:30 WARN  cache-service  - Memory usage: 91%
2026-04-04 14:23:00 ERROR cache-service  - Memory usage: 94% — approaching OOM
2026-04-04 14:23:01 ERROR cache-service  - Cache size: 10847 items (expected max: 10000)
2026-04-04 14:23:01 CRIT  cache-service  - OOM kill imminent; pod will restart
"""

_METRICS = """\
=== cache-service metrics (last 5 min) ===
  memory_usage_pct:  94.0     [CRITICAL — threshold 80%]
  cache_items:       10847    [ANOMALY — expected ≤ 10000]
  evictions_per_sec: 0        [ANOMALY — expected ~100/s under load]
  cache_hit_rate:    0.72
  requests_per_sec:  1240
  error_rate_pct:    12.0
  p99_latency_ms:    850
"""


class MemoryLeakTask(BaseTask):
    task_id = "fix_memory_leak"
    alert = "[P3] cache-service: memory at 94%, pod restart imminent — cache NOT evicting items"
    description = (
        "The LRU cache is not evicting items correctly, causing memory to grow without bound. "
        "Investigate the logs/metrics, read cache_manager.py to find the bug, "
        "fix it so all 5 tests pass, then verify with run_tests."
    )
    services_total = 1
    max_steps = 15

    @property
    def source_files(self) -> Dict[str, str]:
        return {"cache_manager.py": _CACHE_MANAGER_BUGGY}

    @property
    def test_files(self) -> Dict[str, str]:
        return {"test_cache.py": _TEST_CACHE}

    @property
    def logs(self) -> Dict[str, str]:
        return {"cache-service": _LOGS, "cache_service": _LOGS}

    @property
    def metrics(self) -> Dict[str, str]:
        return {"cache-service": _METRICS, "cache_service": _METRICS}

    @property
    def _health_profile(self) -> Dict[str, float]:
        return {"mem": 94.0, "err": 12.0, "lat": 850.0}
