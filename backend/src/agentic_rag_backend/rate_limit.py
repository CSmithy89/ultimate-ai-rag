from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import time


@dataclass
class RateLimiter:
    """In-memory rate limiter (per process).

    Note: In multi-worker deployments, each worker maintains its own limiter state.
    Use a shared store (e.g., Redis) for global rate limiting across workers.
    """

    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        """Initialize rate limiter state."""
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = {}
        self._last_seen: dict[str, float] = {}
        self._inactive_ttl = self.window_seconds * 2

    def allow(self, key: str) -> bool:
        now = time()
        cutoff = now - self.window_seconds
        with self._lock:
            self._cleanup(now)
            bucket = self._requests.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if not bucket and key in self._requests:
                self._requests.pop(key, None)
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            self._requests[key] = bucket
            self._last_seen[key] = now
            return True

    def _cleanup(self, now: float) -> None:
        stale_before = now - self._inactive_ttl
        for key, last_seen in list(self._last_seen.items()):
            if last_seen < stale_before:
                self._last_seen.pop(key, None)
                self._requests.pop(key, None)
