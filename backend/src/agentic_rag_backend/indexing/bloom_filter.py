"""Simple Bloom filter implementation for large crawl visited sets."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable


class BloomFilter:
    """Space-efficient probabilistic set for membership checks."""

    def __init__(self, capacity: int, error_rate: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if error_rate <= 0 or error_rate >= 1:
            raise ValueError("error_rate must be between 0 and 1")

        self.capacity = capacity
        self.error_rate = error_rate
        self.size = max(1, int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2)))
        self.hash_count = max(1, int((self.size / capacity) * math.log(2)))
        self._bits = bytearray((self.size + 7) // 8)
        self._count = 0

    def _hashes(self, item: str) -> Iterable[int]:
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") or 1
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        for index in self._hashes(item):
            byte_index = index // 8
            bit_index = index % 8
            self._bits[byte_index] |= 1 << bit_index
        self._count += 1

    def __contains__(self, item: str) -> bool:
        for index in self._hashes(item):
            byte_index = index // 8
            bit_index = index % 8
            if not (self._bits[byte_index] & (1 << bit_index)):
                return False
        return True

    def __len__(self) -> int:
        return self._count
