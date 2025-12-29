"""Small TTL cache for retrieval results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Simple size-bounded TTL cache."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[Hashable, CacheEntry[T]] = OrderedDict()

    def get(self, key: Hashable) -> T | None:
        now = monotonic()
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at <= now:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: Hashable, value: T) -> None:
        now = monotonic()
        self._store[key] = CacheEntry(value=value, expires_at=now + self.ttl_seconds)
        self._store.move_to_end(key)
        self._prune(now)

    def _prune(self, now: float) -> None:
        expired = [key for key, entry in self._store.items() if entry.expires_at <= now]
        for key in expired:
            self._store.pop(key, None)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
