"""Tests for BloomFilter utility."""

import pytest

from agentic_rag_backend.indexing.bloom_filter import BloomFilter


def test_bloom_filter_add_and_contains():
    bloom = BloomFilter(capacity=100, error_rate=0.01)
    bloom.add("https://example.com")
    assert "https://example.com" in bloom


def test_bloom_filter_invalid_params():
    with pytest.raises(ValueError):
        BloomFilter(capacity=0, error_rate=0.1)
    with pytest.raises(ValueError):
        BloomFilter(capacity=10, error_rate=1.5)
