"""Tests for retrieval cache behavior."""

from agentic_rag_backend.retrieval.cache import TTLCache


def test_cache_eviction_by_size() -> None:
    cache = TTLCache[int](max_size=2, ttl_seconds=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_cache_expiration(monkeypatch) -> None:
    now = {"value": 100.0}

    def fake_monotonic() -> float:
        return now["value"]

    monkeypatch.setattr("agentic_rag_backend.retrieval.cache.monotonic", fake_monotonic)

    cache = TTLCache[str](max_size=2, ttl_seconds=5)
    cache.set("key", "value")
    assert cache.get("key") == "value"

    now["value"] = 106.0
    assert cache.get("key") is None
