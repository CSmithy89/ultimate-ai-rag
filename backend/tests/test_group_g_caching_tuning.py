"""Tests for Group G: Caching and Tuning features (Stories 19-G1 through 19-G4).

This test module covers:
- Story 19-G1: Reranking Result Caching
- Story 19-G2: Configurable Contextual Retrieval Prompt
- Story 19-G3: Cross-Encoder Model Preloading
- Story 19-G4: Score Normalization Strategies
"""

import math
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agentic_rag_backend.retrieval.cache import (
    TTLCache,
    CacheStats,
    RerankerCache,
    generate_reranker_cache_key,
    hash_cache_key,
)
from agentic_rag_backend.retrieval.normalization import (
    NormalizationStrategy,
    normalize_min_max,
    normalize_z_score,
    normalize_softmax,
    normalize_percentile,
    normalize_scores,
    get_normalization_strategy,
    aggregate_normalized_scores,
)
from agentic_rag_backend.indexing.contextual import (
    load_contextual_prompt,
    DEFAULT_CONTEXT_GENERATION_PROMPT,
)


# =============================================================================
# Story 19-G1: Reranking Result Caching Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_initial_stats(self) -> None:
        """Test initial statistics are zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.total_requests == 10
        assert stats.hit_rate == 0.7

    def test_reset(self) -> None:
        """Test reset clears all stats."""
        stats = CacheStats(hits=10, misses=5)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0


class TestTTLCache:
    """Tests for TTLCache."""

    def test_basic_set_get(self) -> None:
        """Test basic set and get operations."""
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self) -> None:
        """Test entries expire after TTL."""
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=0.1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_max_size_eviction(self) -> None:
        """Test LRU eviction when max size exceeded."""
        cache: TTLCache[str] = TTLCache(max_size=3, ttl_seconds=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.size == 3

    def test_stats_tracking(self) -> None:
        """Test stats are tracked correctly."""
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")

        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit

        assert cache.stats.hits == 2
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == pytest.approx(2 / 3, rel=0.01)

    def test_clear(self) -> None:
        """Test cache clear."""
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size == 0
        assert cache.get("key1") is None


class TestRerankerCacheKey:
    """Tests for cache key generation."""

    def test_consistent_key(self) -> None:
        """Test cache key is consistent for same inputs."""
        key1 = generate_reranker_cache_key(
            query_text="test query",
            document_ids=["doc1", "doc2"],
            chunk_ids=["chunk1", "chunk2"],
            reranker_model="model-1",
            tenant_id="tenant-1",
            top_k=10,
        )
        key2 = generate_reranker_cache_key(
            query_text="test query",
            document_ids=["doc1", "doc2"],
            chunk_ids=["chunk1", "chunk2"],
            reranker_model="model-1",
            tenant_id="tenant-1",
            top_k=10,
        )
        assert key1 == key2

    def test_document_order_invariant(self) -> None:
        """Test document order doesn't affect key."""
        key1 = generate_reranker_cache_key(
            query_text="test",
            document_ids=["doc1", "doc2", "doc3"],
            chunk_ids=["chunk1", "chunk2", "chunk3"],
            reranker_model="model",
            tenant_id="tenant",
            top_k=10,
        )
        key2 = generate_reranker_cache_key(
            query_text="test",
            document_ids=["doc3", "doc1", "doc2"],
            chunk_ids=["chunk3", "chunk1", "chunk2"],
            reranker_model="model",
            tenant_id="tenant",
            top_k=10,
        )
        assert key1 == key2

    def test_different_query_different_key(self) -> None:
        """Test different queries produce different keys."""
        key1 = generate_reranker_cache_key(
            "query1",
            ["doc1"],
            ["chunk1"],
            "model",
            "tenant",
            10,
        )
        key2 = generate_reranker_cache_key(
            "query2",
            ["doc1"],
            ["chunk1"],
            "model",
            "tenant",
            10,
        )
        assert key1 != key2

    def test_different_tenant_different_key(self) -> None:
        """Test tenant isolation in cache keys."""
        key1 = generate_reranker_cache_key(
            "query",
            ["doc1"],
            ["chunk1"],
            "model",
            "tenant-1",
            10,
        )
        key2 = generate_reranker_cache_key(
            "query",
            ["doc1"],
            ["chunk1"],
            "model",
            "tenant-2",
            10,
        )
        assert key1 != key2

    def test_top_k_affects_key(self) -> None:
        """Test different top_k values produce different keys."""
        key1 = generate_reranker_cache_key(
            "query",
            ["doc1"],
            ["chunk1"],
            "model",
            "tenant",
            5,
        )
        key2 = generate_reranker_cache_key(
            "query",
            ["doc1"],
            ["chunk1"],
            "model",
            "tenant",
            10,
        )
        assert key1 != key2


class TestRerankerCache:
    """Tests for RerankerCache."""

    def test_disabled_cache(self) -> None:
        """Test disabled cache always returns None."""
        cache = RerankerCache(enabled=False, ttl_seconds=60, max_size=100)
        cache.set("query", ["doc1"], ["chunk1"], "model", "tenant", 10, [{"result": 1}])
        assert cache.get("query", ["doc1"], ["chunk1"], "model", "tenant", 10) is None

    def test_enabled_cache(self) -> None:
        """Test enabled cache stores and retrieves results."""
        cache = RerankerCache(enabled=True, ttl_seconds=60, max_size=100)
        results = [{"result": 1}, {"result": 2}]
        cache.set("query", ["doc1"], ["chunk1"], "model", "tenant", 10, results)
        assert cache.get("query", ["doc1"], ["chunk1"], "model", "tenant", 10) == results

    def test_tenant_isolation(self) -> None:
        """Test cache respects tenant isolation."""
        cache = RerankerCache(enabled=True, ttl_seconds=60, max_size=100)
        cache.set("query", ["doc1"], ["chunk1"], "model", "tenant-1", 10, [{"tenant": 1}])
        cache.set("query", ["doc1"], ["chunk1"], "model", "tenant-2", 10, [{"tenant": 2}])

        assert cache.get("query", ["doc1"], ["chunk1"], "model", "tenant-1", 10) == [{"tenant": 1}]
        assert cache.get("query", ["doc1"], ["chunk1"], "model", "tenant-2", 10) == [{"tenant": 2}]

    def test_cache_properties(self) -> None:
        """Test cache properties."""
        cache = RerankerCache(enabled=True, ttl_seconds=60, max_size=100)
        assert cache.enabled is True
        assert cache.size == 0

        cache.set("query", ["doc1"], ["chunk1"], "model", "tenant", 10, [1, 2, 3])
        assert cache.size == 1


# =============================================================================
# Story 19-G2: Configurable Contextual Retrieval Prompt Tests
# =============================================================================


class TestLoadContextualPrompt:
    """Tests for contextual prompt loading."""

    def test_default_prompt_when_no_path(self) -> None:
        """Test default prompt is returned when no path provided."""
        prompt = load_contextual_prompt(None)
        assert prompt == DEFAULT_CONTEXT_GENERATION_PROMPT
        assert "{document_content}" in prompt
        assert "{chunk_content}" in prompt

    def test_load_custom_prompt(self) -> None:
        """Test loading a custom prompt from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Custom prompt with {document} and {chunk} placeholders")
            f.flush()

            prompt = load_contextual_prompt(f.name)
            # Placeholders should be normalized to internal format
            assert "{document_content}" in prompt
            assert "{chunk_content}" in prompt

    def test_fallback_on_missing_file(self) -> None:
        """Test fallback to default when file doesn't exist."""
        prompt = load_contextual_prompt("/nonexistent/path/prompt.txt")
        assert prompt == DEFAULT_CONTEXT_GENERATION_PROMPT

    def test_fallback_on_invalid_template(self) -> None:
        """Test fallback when template is missing required placeholders."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Invalid template without required placeholders")
            f.flush()

            prompt = load_contextual_prompt(f.name)
            assert prompt == DEFAULT_CONTEXT_GENERATION_PROMPT

    def test_accepts_alternative_placeholders(self) -> None:
        """Test both {document}/{chunk} and {document_content}/{chunk_content} work."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Template with {document_content} and {chunk_content}")
            f.flush()

            prompt = load_contextual_prompt(f.name)
            assert "{document_content}" in prompt
            assert "{chunk_content}" in prompt


# =============================================================================
# Story 19-G4: Score Normalization Strategies Tests
# =============================================================================


class TestNormalizationStrategy:
    """Tests for NormalizationStrategy enum."""

    def test_enum_values(self) -> None:
        """Test all expected strategies exist."""
        assert NormalizationStrategy.MIN_MAX.value == "min_max"
        assert NormalizationStrategy.Z_SCORE.value == "z_score"
        assert NormalizationStrategy.SOFTMAX.value == "softmax"
        assert NormalizationStrategy.PERCENTILE.value == "percentile"

    def test_get_strategy_valid(self) -> None:
        """Test getting strategy from valid string."""
        assert get_normalization_strategy("min_max") == NormalizationStrategy.MIN_MAX
        assert get_normalization_strategy("MIN_MAX") == NormalizationStrategy.MIN_MAX
        assert get_normalization_strategy("z_score") == NormalizationStrategy.Z_SCORE

    def test_get_strategy_invalid(self) -> None:
        """Test invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            get_normalization_strategy("invalid_strategy")


class TestNormalizeMinMax:
    """Tests for min-max normalization."""

    def test_basic_normalization(self) -> None:
        """Test basic min-max normalization."""
        scores = [0.0, 0.5, 1.0]
        result = normalize_min_max(scores)
        assert result == [0.0, 0.5, 1.0]

    def test_negative_scores(self) -> None:
        """Test normalization with negative scores."""
        scores = [-10.0, 0.0, 10.0]
        result = normalize_min_max(scores)
        assert result == [0.0, 0.5, 1.0]

    def test_single_value(self) -> None:
        """Test single value returns 0.5."""
        scores = [5.0]
        result = normalize_min_max(scores)
        assert result == [0.5]

    def test_all_same(self) -> None:
        """Test all same values return 0.5."""
        scores = [3.0, 3.0, 3.0]
        result = normalize_min_max(scores)
        assert result == [0.5, 0.5, 0.5]

    def test_empty_list(self) -> None:
        """Test empty list returns empty."""
        assert normalize_min_max([]) == []


class TestNormalizeZScore:
    """Tests for z-score normalization."""

    def test_basic_normalization(self) -> None:
        """Test z-score normalization produces values in [0, 1]."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_z_score(scores)
        for r in result:
            assert 0.0 <= r <= 1.0

    def test_mean_maps_to_half(self) -> None:
        """Test mean value maps close to 0.5."""
        scores = [-10.0, 0.0, 10.0]  # Mean is 0
        result = normalize_z_score(scores)
        # Middle value (mean=0) should map to 0.5
        assert result[1] == pytest.approx(0.5, abs=0.01)

    def test_single_value(self) -> None:
        """Test single value returns 0.5."""
        assert normalize_z_score([5.0]) == [0.5]

    def test_empty_list(self) -> None:
        """Test empty list returns empty."""
        assert normalize_z_score([]) == []


class TestNormalizeSoftmax:
    """Tests for softmax normalization."""

    def test_sums_to_one(self) -> None:
        """Test softmax results sum to 1."""
        scores = [1.0, 2.0, 3.0]
        result = normalize_softmax(scores)
        assert sum(result) == pytest.approx(1.0, abs=0.001)

    def test_preserves_order(self) -> None:
        """Test softmax preserves relative ordering."""
        scores = [1.0, 3.0, 2.0]
        result = normalize_softmax(scores)
        assert result[1] > result[2] > result[0]

    def test_temperature_effect(self) -> None:
        """Test temperature parameter effect."""
        scores = [1.0, 2.0]
        low_temp = normalize_softmax(scores, temperature=0.5)
        high_temp = normalize_softmax(scores, temperature=2.0)
        # Low temperature should produce more extreme differences
        assert abs(low_temp[0] - low_temp[1]) > abs(high_temp[0] - high_temp[1])

    def test_single_value(self) -> None:
        """Test single value returns 1.0."""
        assert normalize_softmax([5.0]) == [1.0]

    def test_empty_list(self) -> None:
        """Test empty list returns empty."""
        assert normalize_softmax([]) == []


class TestNormalizePercentile:
    """Tests for percentile normalization."""

    def test_basic_percentile(self) -> None:
        """Test basic percentile normalization."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = normalize_percentile(scores)
        assert result[0] == 0.0  # Lowest
        assert result[4] == 1.0  # Highest

    def test_handles_ties(self) -> None:
        """Test tied scores get averaged percentile."""
        scores = [1.0, 2.0, 2.0, 3.0]
        result = normalize_percentile(scores)
        # Tied values (indices 1 and 2) should have same percentile
        assert result[1] == result[2]

    def test_single_value(self) -> None:
        """Test single value returns 0.5."""
        assert normalize_percentile([5.0]) == [0.5]

    def test_empty_list(self) -> None:
        """Test empty list returns empty."""
        assert normalize_percentile([]) == []


class TestNormalizeScores:
    """Tests for the unified normalize_scores function."""

    def test_dispatch_min_max(self) -> None:
        """Test MIN_MAX strategy dispatch."""
        scores = [0.0, 0.5, 1.0]
        result = normalize_scores(scores, NormalizationStrategy.MIN_MAX)
        assert result == normalize_min_max(scores)

    def test_dispatch_z_score(self) -> None:
        """Test Z_SCORE strategy dispatch."""
        scores = [1.0, 2.0, 3.0]
        result = normalize_scores(scores, NormalizationStrategy.Z_SCORE)
        assert result == normalize_z_score(scores)

    def test_dispatch_softmax(self) -> None:
        """Test SOFTMAX strategy dispatch."""
        scores = [1.0, 2.0, 3.0]
        result = normalize_scores(scores, NormalizationStrategy.SOFTMAX)
        assert result == normalize_softmax(scores)

    def test_dispatch_percentile(self) -> None:
        """Test PERCENTILE strategy dispatch."""
        scores = [1.0, 2.0, 3.0]
        result = normalize_scores(scores, NormalizationStrategy.PERCENTILE)
        assert result == normalize_percentile(scores)


class TestAggregateNormalizedScores:
    """Tests for score aggregation."""

    def test_mean_aggregation(self) -> None:
        """Test mean aggregation."""
        scores = [0.2, 0.4, 0.6, 0.8]
        assert aggregate_normalized_scores(scores, "mean") == pytest.approx(0.5)

    def test_max_aggregation(self) -> None:
        """Test max aggregation."""
        scores = [0.2, 0.8, 0.5]
        assert aggregate_normalized_scores(scores, "max") == 0.8

    def test_min_aggregation(self) -> None:
        """Test min aggregation."""
        scores = [0.2, 0.8, 0.5]
        assert aggregate_normalized_scores(scores, "min") == 0.2

    def test_median_aggregation_odd(self) -> None:
        """Test median with odd number of elements."""
        scores = [0.2, 0.5, 0.8]
        assert aggregate_normalized_scores(scores, "median") == 0.5

    def test_median_aggregation_even(self) -> None:
        """Test median with even number of elements."""
        scores = [0.2, 0.4, 0.6, 0.8]
        assert aggregate_normalized_scores(scores, "median") == 0.5

    def test_empty_scores(self) -> None:
        """Test empty scores return 0.0."""
        assert aggregate_normalized_scores([], "mean") == 0.0
