"""
Task 2 (Medium) — fix_api_gateway
P2 Alert: downstream DB connection pool exhausted, 500 error rate at 23%.

Two bugs across two files:

Bug 1 — rate_limiter.py:
  The rate limiter stores request counts using `time.time()` (a float) as the
  dict key rather than the computed integer window bucket.  Because every
  call has a unique float timestamp, counts never accumulate and the limiter
  never fires.  Clients hammer the downstream DB unconstrained.
  Fix: use `full_key` (the pre-computed integer window key) consistently.

Bug 2 — api_gateway.py:
  When the rate limiter rejects a request the gateway returns HTTP 200 with
  an error body instead of HTTP 429.  Clients see 200, do NOT back off, and
  keep retrying at full speed.
  Fix: return status code 429 for rate-limited responses.
"""

from typing import Dict
from .base import BaseTask


# ── Buggy source files ─────────────────────────────────────────────────────────

_RATE_LIMITER_BUGGY = '''\
"""Sliding-window rate limiter for the API gateway."""
import time
from collections import defaultdict


class RateLimiter:
    """Limits requests per client per time window."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._counts: dict = {}

    def is_allowed(self, client_id: str) -> bool:
        """Return True if the request is within the rate limit, False otherwise."""
        current = time.time()
        window_key = int(current / self.window_seconds)
        full_key = f"{client_id}:{window_key}"   # correct compound key

        # BUG: uses `current` (float) instead of `full_key` (int-bucketed string).
        # Every call has a unique float → counts never accumulate → limiter never fires.
        count = self._counts.get(f"{client_id}:{current}", 0)   # BUG
        if count >= self.max_requests:
            return False

        self._counts[f"{client_id}:{current}"] = count + 1      # BUG
        return True

    def reset(self) -> None:
        """Clear all counters (for testing)."""
        self._counts.clear()
'''

_API_GATEWAY_BUGGY = '''\
"""API Gateway — routes requests and enforces rate limits."""
from rate_limiter import RateLimiter


class APIGateway:
    """Routes incoming requests, enforces rate limits, manages DB connection pool."""

    def __init__(self, rate_limiter: RateLimiter | None = None, db_pool_size: int = 10):
        self.rate_limiter = rate_limiter or RateLimiter(max_requests=10, window_seconds=1)
        self.db_pool_size = db_pool_size
        self._active_db_connections = 0

    def handle_request(self, client_id: str, request_data: dict) -> tuple[dict, int]:
        """
        Process a client request.

        Returns:
            (response_body, http_status_code)
        """
        if not self.rate_limiter.is_allowed(client_id):
            # BUG: should return 429 (Too Many Requests), not 200.
            # Clients that receive 200 do not back off and keep retrying,
            # exhausting the downstream DB connection pool.
            return {"error": "rate_limited", "message": "Too many requests"}, 200  # BUG

        if self._active_db_connections >= self.db_pool_size:
            return {"error": "service_unavailable", "message": "DB pool exhausted"}, 503

        self._active_db_connections += 1
        try:
            result = self._process(request_data)
            return {"data": result, "status": "ok"}, 200
        finally:
            self._active_db_connections -= 1

    def _process(self, data: dict) -> dict:
        return {"processed": True, "echo": data}
'''

# ── Tests ──────────────────────────────────────────────────────────────────────

_TEST_GATEWAY = '''\
"""Tests for rate_limiter.RateLimiter and api_gateway.APIGateway."""
import time
import pytest
from rate_limiter import RateLimiter
from api_gateway import APIGateway


# ── RateLimiter tests ──────────────────────────────────────────────────────────

def test_requests_within_limit_are_allowed():
    """Requests within the limit should be allowed."""
    rl = RateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert rl.is_allowed("client_a") is True


def test_requests_exceeding_limit_are_blocked():
    """Requests beyond the limit in the same window must be rejected."""
    rl = RateLimiter(max_requests=3, window_seconds=60)
    results = [rl.is_allowed("client_b") for _ in range(6)]
    allowed = sum(results)
    assert allowed == 3, f"Expected exactly 3 allowed, got {allowed}"


def test_different_clients_have_independent_limits():
    """Each client has its own counter."""
    rl = RateLimiter(max_requests=2, window_seconds=60)
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is False  # alice exhausted
    assert rl.is_allowed("bob")   is True   # bob is unaffected


def test_high_volume_rate_limiting():
    """Under high load, rate limiter must block excess requests."""
    rl = RateLimiter(max_requests=10, window_seconds=60)
    results = [rl.is_allowed("spammer") for _ in range(100)]
    allowed = sum(results)
    assert allowed == 10, f"Expected 10 allowed out of 100, got {allowed}"


# ── APIGateway tests ───────────────────────────────────────────────────────────

def test_normal_request_returns_200():
    """Valid requests within the rate limit return HTTP 200."""
    gw = APIGateway(RateLimiter(max_requests=100, window_seconds=60))
    _, status = gw.handle_request("client", {"key": "value"})
    assert status == 200


def test_rate_limited_request_returns_429():
    """Rate-limited requests must return HTTP 429, not 200."""
    rl = RateLimiter(max_requests=1, window_seconds=60)
    gw = APIGateway(rl)
    gw.handle_request("client", {})          # first: allowed
    _, status = gw.handle_request("client", {})  # second: rate limited
    assert status == 429, f"Expected 429 for rate-limited request, got {status}"


def test_db_pool_not_exhausted_under_rate_limiting():
    """With rate limiting working, the DB pool should never be exhausted."""
    rl = RateLimiter(max_requests=5, window_seconds=60)
    gw = APIGateway(rl, db_pool_size=3)
    statuses = []
    for i in range(20):
        _, code = gw.handle_request(f"client_{i % 4}", {"i": i})
        statuses.append(code)
    # None of the responses should be 503 (pool exhausted)
    assert 503 not in statuses, "DB pool should not be exhausted when rate limiting works"


def test_rate_limit_error_body_contains_message():
    """Rate-limited responses must include an error message."""
    rl = RateLimiter(max_requests=1, window_seconds=60)
    gw = APIGateway(rl)
    gw.handle_request("c", {})
    body, status = gw.handle_request("c", {})
    assert status == 429
    assert "error" in body, "Response body must include 'error' key"


def test_reset_clears_counters():
    """After reset, previously exhausted clients can make requests again."""
    rl = RateLimiter(max_requests=1, window_seconds=60)
    gw = APIGateway(rl)
    gw.handle_request("c", {})
    _, s1 = gw.handle_request("c", {})
    assert s1 == 429
    rl.reset()
    _, s2 = gw.handle_request("c", {})
    assert s2 == 200, "After reset, requests should be allowed again"


def test_multiple_clients_independent_db_connections():
    """Independent clients should not interfere with each other's connections."""
    rl = RateLimiter(max_requests=100, window_seconds=60)
    gw = APIGateway(rl, db_pool_size=5)
    results = []
    for client_id in range(10):
        body, status = gw.handle_request(f"client_{client_id}", {"id": client_id})
        results.append(status)
    assert all(s == 200 for s in results), f"Expected all 200, got: {results}"


def test_large_volume_no_503():
    """High volume with working rate limiting must not exhaust DB pool."""
    rl = RateLimiter(max_requests=5, window_seconds=60)
    gw = APIGateway(rl, db_pool_size=2)
    statuses = [gw.handle_request(f"c{i%10}", {})[1] for i in range(100)]
    assert 503 not in statuses, "No 503s expected when rate limiting is functional"


def test_rate_limiter_is_consistent():
    """Repeatedly checking the same client gives consistent results."""
    rl = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        rl.is_allowed("steady")
    # All subsequent calls within the window must be blocked
    for _ in range(10):
        assert rl.is_allowed("steady") is False, "Must stay blocked within window"


def test_gateway_response_data_on_success():
    """Successful gateway responses must contain 'data' key."""
    rl = RateLimiter(max_requests=100, window_seconds=60)
    gw = APIGateway(rl)
    body, status = gw.handle_request("ok_client", {"payload": "test"})
    assert status == 200
    assert "data" in body
'''

# ── Pre-authored logs / metrics ────────────────────────────────────────────────

_LOGS = """\
=== api-gateway logs (last 40 lines) ===
2026-04-04 15:31:00 INFO  api-gateway  - Gateway started. rate_limit=10/s, db_pool=10
2026-04-04 15:31:45 INFO  api-gateway  - rate limit "triggered" for client_a — returned 200
2026-04-04 15:31:45 INFO  api-gateway  - client_a sent request #11 (limit=10/s)
2026-04-04 15:31:45 INFO  api-gateway  - client_a sent request #50 (limit=10/s)
2026-04-04 15:31:45 INFO  api-gateway  - client_a sent request #847 (limit=10/s)
2026-04-04 15:31:46 WARN  api-gateway  - DB active connections: 8/10
2026-04-04 15:31:46 WARN  api-gateway  - DB active connections: 10/10  [POOL FULL]
2026-04-04 15:31:46 ERROR api-gateway  - DB pool exhausted — returning 503
2026-04-04 15:31:47 ERROR api-gateway  - 503 error rate: 23%
2026-04-04 15:31:47 ERROR api-gateway  - rate_limiter: _counts size = 8,432 (float keys leaking!)
2026-04-04 15:31:48 WARN  rate-limiter - Window key mismatch: stored key has float component
"""

_METRICS = """\
=== api-gateway metrics (last 5 min) ===
  requests_per_sec:      1240
  rate_limited_count:    0      [ANOMALY — expected ~1100/s at this load]
  db_active_conns:       10/10  [CRITICAL — pool exhausted]
  error_rate_pct:        23.0   [CRITICAL — threshold 1%]
  p99_latency_ms:        1200
  memory_usage_pct:      60.0
  _counts_dict_size:     8432   [ANOMALY — rate limiter leaking float keys]
"""


class ApiGatewayTask(BaseTask):
    task_id = "fix_api_gateway"
    alert = (
        "[P2] api-gateway: downstream DB pool exhausted (10/10), "
        "500 error rate at 23% — rate limiter appears non-functional"
    )
    description = (
        "The API gateway's rate limiter is not working, allowing clients to flood "
        "the downstream database. Investigate logs/metrics, read both source files, "
        "identify the two bugs, fix them, and verify all 11 tests pass."
    )
    services_total = 2
    max_steps = 25

    @property
    def source_files(self) -> Dict[str, str]:
        return {
            "rate_limiter.py": _RATE_LIMITER_BUGGY,
            "api_gateway.py": _API_GATEWAY_BUGGY,
        }

    @property
    def test_files(self) -> Dict[str, str]:
        return {"test_gateway.py": _TEST_GATEWAY}

    @property
    def logs(self) -> Dict[str, str]:
        return {"api-gateway": _LOGS, "api_gateway": _LOGS, "rate-limiter": _LOGS}

    @property
    def metrics(self) -> Dict[str, str]:
        return {"api-gateway": _METRICS, "api_gateway": _METRICS}

    @property
    def _health_profile(self) -> Dict[str, float]:
        return {"mem": 60.0, "err": 23.0, "lat": 1200.0}
