from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import time


@dataclass
class RateLimiter:
    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._requests.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True
