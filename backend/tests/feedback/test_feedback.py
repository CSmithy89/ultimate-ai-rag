"""Comprehensive tests for the feedback loop module.

Story 20-E2: Implement Self-Improving Feedback Loop

Tests cover:
- FeedbackType enum values
- UserFeedback model validation
- FeedbackStats aggregation
- QueryBoost calculation
- FeedbackLoop operations
- FeedbackLoopAdapter feature flag
- Tenant isolation
- Feedback decay logic
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from agentic_rag_backend.feedback import (
    DEFAULT_FEEDBACK_BOOST_MAX,
    DEFAULT_FEEDBACK_BOOST_MIN,
    DEFAULT_FEEDBACK_DECAY_DAYS,
    DEFAULT_FEEDBACK_MIN_SAMPLES,
    EmbeddingProvider,
    FeedbackLoop,
    FeedbackLoopAdapter,
    FeedbackRecordResult,
    FeedbackStats,
    FeedbackType,
    QueryBoost,
    UserFeedback,
)


# ============================================================================
# FeedbackType Tests
# ============================================================================


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_type_values(self) -> None:
        """Test that all feedback types have expected values."""
        assert FeedbackType.RELEVANCE.value == "relevance"
        assert FeedbackType.ACCURACY.value == "accuracy"
        assert FeedbackType.COMPLETENESS.value == "completeness"
        assert FeedbackType.PREFERENCE.value == "preference"

    def test_feedback_type_count(self) -> None:
        """Test that we have exactly 4 feedback types."""
        assert len(FeedbackType) == 4

    def test_feedback_type_from_string(self) -> None:
        """Test creating FeedbackType from string value."""
        assert FeedbackType("relevance") == FeedbackType.RELEVANCE
        assert FeedbackType("accuracy") == FeedbackType.ACCURACY

    def test_invalid_feedback_type(self) -> None:
        """Test that invalid feedback type raises ValueError."""
        with pytest.raises(ValueError):
            FeedbackType("invalid")


# ============================================================================
# UserFeedback Tests
# ============================================================================


class TestUserFeedback:
    """Tests for UserFeedback dataclass."""

    def test_create_valid_feedback(self) -> None:
        """Test creating valid user feedback."""
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        assert feedback.query_id == "q-123"
        assert feedback.feedback_type == FeedbackType.RELEVANCE
        assert feedback.score == 0.8
        assert feedback.tenant_id == "tenant-1"
        assert feedback.user_id == "user-1"
        assert feedback.id  # Auto-generated UUID
        assert feedback.result_id is None
        assert feedback.correction is None
        assert isinstance(feedback.created_at, datetime)
        assert feedback.metadata == {}

    def test_create_feedback_with_all_fields(self) -> None:
        """Test creating feedback with all optional fields."""
        created_at = datetime.now(timezone.utc)
        feedback = UserFeedback(
            query_id="q-456",
            feedback_type=FeedbackType.ACCURACY,
            score=-0.5,
            tenant_id="tenant-2",
            user_id="user-2",
            id="custom-id",
            result_id="result-789",
            correction="The correct answer is X",
            created_at=created_at,
            metadata={"source": "api"},
        )

        assert feedback.id == "custom-id"
        assert feedback.result_id == "result-789"
        assert feedback.correction == "The correct answer is X"
        assert feedback.created_at == created_at
        assert feedback.metadata == {"source": "api"}

    def test_feedback_type_string_conversion(self) -> None:
        """Test that string feedback type is converted to enum."""
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type="completeness",  # type: ignore
            score=0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        assert feedback.feedback_type == FeedbackType.COMPLETENESS

    def test_feedback_empty_query_id_raises(self) -> None:
        """Test that empty query_id raises ValueError."""
        with pytest.raises(ValueError, match="query_id cannot be empty"):
            UserFeedback(
                query_id="",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.5,
                tenant_id="tenant-1",
                user_id="user-1",
            )

    def test_feedback_empty_tenant_id_raises(self) -> None:
        """Test that empty tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.5,
                tenant_id="",
                user_id="user-1",
            )

    def test_feedback_empty_user_id_raises(self) -> None:
        """Test that empty user_id raises ValueError."""
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.5,
                tenant_id="tenant-1",
                user_id="",
            )

    def test_feedback_score_out_of_range_raises(self) -> None:
        """Test that score outside -1 to 1 range raises ValueError."""
        with pytest.raises(ValueError, match="score must be between -1.0 and 1.0"):
            UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=1.5,
                tenant_id="tenant-1",
                user_id="user-1",
            )

        with pytest.raises(ValueError, match="score must be between -1.0 and 1.0"):
            UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=-1.5,
                tenant_id="tenant-1",
                user_id="user-1",
            )

    def test_is_positive_property(self) -> None:
        """Test is_positive property."""
        positive = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert positive.is_positive is True

        neutral = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.0,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert neutral.is_positive is False

    def test_is_negative_property(self) -> None:
        """Test is_negative property."""
        negative = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=-0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert negative.is_negative is True

        neutral = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.0,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert neutral.is_negative is False

    def test_has_correction_property(self) -> None:
        """Test has_correction property."""
        without_correction = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert without_correction.has_correction is False

        with_correction = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.ACCURACY,
            score=-0.5,
            tenant_id="tenant-1",
            user_id="user-1",
            correction="This is the corrected answer",
        )
        assert with_correction.has_correction is True

        empty_correction = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.ACCURACY,
            score=-0.5,
            tenant_id="tenant-1",
            user_id="user-1",
            correction="",
        )
        assert empty_correction.has_correction is False


# ============================================================================
# FeedbackStats Tests
# ============================================================================


class TestFeedbackStats:
    """Tests for FeedbackStats dataclass."""

    def test_create_empty_stats(self) -> None:
        """Test creating empty feedback stats."""
        stats = FeedbackStats(query_id="q-123")

        assert stats.query_id == "q-123"
        assert stats.total_count == 0
        assert stats.positive_count == 0
        assert stats.negative_count == 0
        assert stats.average_score == 0.0
        assert stats.correction_count == 0
        assert stats.result_id is None
        assert stats.feedback_types == {}
        assert stats.last_feedback_at is None

    def test_add_positive_feedback(self) -> None:
        """Test adding positive feedback updates stats."""
        stats = FeedbackStats(query_id="q-123")
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        stats.add_feedback(feedback)

        assert stats.total_count == 1
        assert stats.positive_count == 1
        assert stats.negative_count == 0
        assert stats.average_score == 0.8
        assert stats.feedback_types == {"relevance": 1}
        assert stats.last_feedback_at == feedback.created_at

    def test_add_negative_feedback(self) -> None:
        """Test adding negative feedback updates stats."""
        stats = FeedbackStats(query_id="q-123")
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.ACCURACY,
            score=-0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        stats.add_feedback(feedback)

        assert stats.total_count == 1
        assert stats.positive_count == 0
        assert stats.negative_count == 1
        assert stats.average_score == -0.5

    def test_add_feedback_with_correction(self) -> None:
        """Test adding feedback with correction updates correction count."""
        stats = FeedbackStats(query_id="q-123")
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.ACCURACY,
            score=-0.5,
            tenant_id="tenant-1",
            user_id="user-1",
            correction="This is wrong",
        )

        stats.add_feedback(feedback)

        assert stats.correction_count == 1

    def test_average_score_calculation(self) -> None:
        """Test that average score is calculated correctly."""
        stats = FeedbackStats(query_id="q-123")

        # Add multiple feedback items
        for score in [0.5, 0.7, -0.3, 0.9, 0.2]:
            feedback = UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=score,
                tenant_id="tenant-1",
                user_id="user-1",
            )
            stats.add_feedback(feedback)

        expected_avg = (0.5 + 0.7 - 0.3 + 0.9 + 0.2) / 5
        assert abs(stats.average_score - expected_avg) < 0.0001

    def test_feedback_types_aggregation(self) -> None:
        """Test feedback types are aggregated correctly."""
        stats = FeedbackStats(query_id="q-123")

        # Add multiple feedback types
        for fb_type in [
            FeedbackType.RELEVANCE,
            FeedbackType.RELEVANCE,
            FeedbackType.ACCURACY,
            FeedbackType.COMPLETENESS,
        ]:
            feedback = UserFeedback(
                query_id="q-123",
                feedback_type=fb_type,
                score=0.5,
                tenant_id="tenant-1",
                user_id="user-1",
            )
            stats.add_feedback(feedback)

        assert stats.feedback_types == {
            "relevance": 2,
            "accuracy": 1,
            "completeness": 1,
        }


# ============================================================================
# QueryBoost Tests
# ============================================================================


class TestQueryBoost:
    """Tests for QueryBoost dataclass."""

    def test_default_values(self) -> None:
        """Test default QueryBoost values."""
        boost = QueryBoost()

        assert boost.boost == 1.0
        assert boost.based_on_queries == 0
        assert boost.feedback_count == 0
        assert boost.confidence == 0.0
        assert boost.decay_applied is False

    def test_neutral_factory(self) -> None:
        """Test neutral() factory method."""
        boost = QueryBoost.neutral()

        assert boost.boost == 1.0
        assert boost.confidence == 0.0

    def test_boost_clamping(self) -> None:
        """Test that boost values are clamped to valid range."""
        # Below minimum
        boost_low = QueryBoost(boost=0.3)
        assert boost_low.boost == 0.5

        # Above maximum
        boost_high = QueryBoost(boost=2.0)
        assert boost_high.boost == 1.5

    def test_confidence_clamping(self) -> None:
        """Test that confidence values are clamped to valid range."""
        # Below minimum
        boost_low = QueryBoost(confidence=-0.5)
        assert boost_low.confidence == 0.0

        # Above maximum
        boost_high = QueryBoost(confidence=1.5)
        assert boost_high.confidence == 1.0


# ============================================================================
# FeedbackRecordResult Tests
# ============================================================================


class TestFeedbackRecordResult:
    """Tests for FeedbackRecordResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result creation."""
        result = FeedbackRecordResult(
            feedback_id="fb-123",
            success=True,
            stats_updated=True,
        )

        assert result.feedback_id == "fb-123"
        assert result.success is True
        assert result.error is None
        assert result.stats_updated is True

    def test_failure_factory(self) -> None:
        """Test failure() factory method."""
        result = FeedbackRecordResult.failure(error="Something went wrong")

        assert result.feedback_id == ""
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.stats_updated is False


# ============================================================================
# Mock Embedding Provider
# ============================================================================


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension
        self.call_count = 0

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding based on text hash."""
        self.call_count += 1
        # Create a deterministic embedding based on text
        text_hash = hash(text)
        embedding = []
        for i in range(self.dimension):
            # Generate values between -1 and 1
            val = ((text_hash * (i + 1)) % 1000) / 500 - 1
            embedding.append(val)
        return embedding


# ============================================================================
# FeedbackLoop Tests
# ============================================================================


class TestFeedbackLoop:
    """Tests for FeedbackLoop class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        loop = FeedbackLoop()

        assert loop._embeddings is None
        assert loop._min_samples == DEFAULT_FEEDBACK_MIN_SAMPLES
        assert loop._decay_days == DEFAULT_FEEDBACK_DECAY_DAYS
        assert loop._boost_max == DEFAULT_FEEDBACK_BOOST_MAX
        assert loop._boost_min == DEFAULT_FEEDBACK_BOOST_MIN

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(
            embedding_provider=embeddings,
            min_samples=5,
            decay_days=30,
            boost_max=2.0,
            boost_min=0.3,
        )

        assert loop._embeddings is embeddings
        assert loop._min_samples == 5
        assert loop._decay_days == 30
        assert loop._boost_max == 2.0
        assert loop._boost_min == 0.3

    @pytest.mark.asyncio
    async def test_record_feedback_success(self) -> None:
        """Test successful feedback recording."""
        loop = FeedbackLoop()
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        result = await loop.record_feedback(feedback)

        assert result.success is True
        assert result.feedback_id == feedback.id
        assert result.stats_updated is True

    @pytest.mark.asyncio
    async def test_record_feedback_stores_in_tenant(self) -> None:
        """Test that feedback is stored under correct tenant."""
        loop = FeedbackLoop()

        feedback1 = UserFeedback(
            query_id="q-1",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        feedback2 = UserFeedback(
            query_id="q-2",
            feedback_type=FeedbackType.ACCURACY,
            score=0.8,
            tenant_id="tenant-2",
            user_id="user-2",
        )

        await loop.record_feedback(feedback1)
        await loop.record_feedback(feedback2)

        assert loop.get_feedback_count("tenant-1") == 1
        assert loop.get_feedback_count("tenant-2") == 1
        assert loop.get_feedback_count("tenant-3") == 0

    @pytest.mark.asyncio
    async def test_record_feedback_updates_query_stats(self) -> None:
        """Test that recording feedback updates query stats."""
        loop = FeedbackLoop()
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        await loop.record_feedback(feedback)

        stats = loop.get_feedback_stats("q-123")
        assert stats is not None
        assert stats.total_count == 1
        assert stats.average_score == 0.8

    @pytest.mark.asyncio
    async def test_record_feedback_with_correction(self) -> None:
        """Test recording feedback with correction."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(embedding_provider=embeddings)
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.ACCURACY,
            score=-0.5,
            tenant_id="tenant-1",
            user_id="user-1",
            correction="The correct answer is X",
        )

        result = await loop.record_feedback(feedback)

        assert result.success is True
        # Correction embedding should be stored
        assert embeddings.call_count == 1

    @pytest.mark.asyncio
    async def test_get_feedback_for_query(self) -> None:
        """Test getting feedback for a specific query."""
        loop = FeedbackLoop()

        # Add feedback from different tenants
        fb1 = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.5,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        fb2 = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.ACCURACY,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-2",
        )
        fb3 = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.3,
            tenant_id="tenant-2",  # Different tenant
            user_id="user-3",
        )

        await loop.record_feedback(fb1)
        await loop.record_feedback(fb2)
        await loop.record_feedback(fb3)

        # Should only get feedback for tenant-1
        feedback = loop.get_feedback_for_query("q-123", "tenant-1")
        assert len(feedback) == 2

        # Should only get feedback for tenant-2
        feedback = loop.get_feedback_for_query("q-123", "tenant-2")
        assert len(feedback) == 1

    @pytest.mark.asyncio
    async def test_get_query_boost_neutral_without_feedback(self) -> None:
        """Test that neutral boost is returned without feedback."""
        loop = FeedbackLoop()

        boost = await loop.get_query_boost("some query", "tenant-1")

        assert boost.boost == 1.0
        assert boost.confidence == 0.0
        assert boost.based_on_queries == 0

    @pytest.mark.asyncio
    async def test_get_query_boost_not_enough_samples(self) -> None:
        """Test boost with not enough samples for confidence."""
        loop = FeedbackLoop(min_samples=10)

        # Add only 3 feedback items
        for i in range(3):
            feedback = UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.8,
                tenant_id="tenant-1",
                user_id=f"user-{i}",
            )
            await loop.record_feedback(feedback)

        boost = await loop.get_query_boost("similar query", "tenant-1")

        # Should return neutral boost due to insufficient samples
        assert boost.boost == 1.0
        assert boost.confidence < 1.0

    @pytest.mark.asyncio
    async def test_get_query_boost_with_enough_samples(self) -> None:
        """Test boost calculation with enough samples."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(
            embedding_provider=embeddings,
            min_samples=5,
        )

        # Store query embedding
        await loop.store_query_embedding("q-123", "test query")

        # Add feedback for the query
        for i in range(10):
            feedback = UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.8,  # Positive feedback
                tenant_id="tenant-1",
                user_id=f"user-{i}",
            )
            await loop.record_feedback(feedback)

        boost = await loop.get_query_boost("test query", "tenant-1")

        # Should have positive boost
        assert boost.boost > 1.0
        assert boost.feedback_count >= 5
        assert boost.confidence > 0.0

    @pytest.mark.asyncio
    async def test_get_query_boost_tenant_isolation(self) -> None:
        """Test that boost respects tenant isolation."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(
            embedding_provider=embeddings,
            min_samples=3,
        )

        # Store query embedding
        await loop.store_query_embedding("q-123", "test query")

        # Add feedback for tenant-1
        for i in range(5):
            feedback = UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.9,
                tenant_id="tenant-1",
                user_id=f"user-{i}",
            )
            await loop.record_feedback(feedback)

        # Tenant-2 should get neutral boost (no feedback)
        boost = await loop.get_query_boost("test query", "tenant-2")
        assert boost.boost == 1.0 or boost.feedback_count == 0

    @pytest.mark.asyncio
    async def test_clear_tenant_feedback(self) -> None:
        """Test clearing all feedback for a tenant."""
        loop = FeedbackLoop()

        # Add feedback for two tenants
        for tenant_id in ["tenant-1", "tenant-2"]:
            for i in range(3):
                feedback = UserFeedback(
                    query_id=f"q-{tenant_id}-{i}",
                    feedback_type=FeedbackType.RELEVANCE,
                    score=0.5,
                    tenant_id=tenant_id,
                    user_id="user-1",
                )
                await loop.record_feedback(feedback)

        # Clear tenant-1
        count = loop.clear_tenant_feedback("tenant-1")
        assert count == 3

        # Verify tenant-1 is cleared
        assert loop.get_feedback_count("tenant-1") == 0

        # Verify tenant-2 is unchanged
        assert loop.get_feedback_count("tenant-2") == 3

    @pytest.mark.asyncio
    async def test_store_query_embedding(self) -> None:
        """Test storing query embeddings."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(embedding_provider=embeddings)

        result = await loop.store_query_embedding("q-123", "test query")

        assert result is True
        assert "q-123" in loop._query_embeddings

    @pytest.mark.asyncio
    async def test_store_query_embedding_without_provider(self) -> None:
        """Test that storing embedding fails without provider."""
        loop = FeedbackLoop()  # No embedding provider

        result = await loop.store_query_embedding("q-123", "test query")

        assert result is False

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        # Identical vectors
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(FeedbackLoop._cosine_similarity(a, b) - 1.0) < 0.0001

        # Orthogonal vectors
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(FeedbackLoop._cosine_similarity(a, b)) < 0.0001

        # Opposite vectors
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert abs(FeedbackLoop._cosine_similarity(a, b) - (-1.0)) < 0.0001

    def test_cosine_similarity_different_lengths(self) -> None:
        """Test that different length vectors return 0."""
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert FeedbackLoop._cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_zero_vector(self) -> None:
        """Test that zero vector returns 0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert FeedbackLoop._cosine_similarity(a, b) == 0.0


# ============================================================================
# FeedbackLoopAdapter Tests
# ============================================================================


class TestFeedbackLoopAdapter:
    """Tests for FeedbackLoopAdapter class."""

    def test_init_disabled(self) -> None:
        """Test initialization when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        assert adapter.enabled is False
        assert adapter._loop is None

    def test_init_enabled(self) -> None:
        """Test initialization when enabled."""
        adapter = FeedbackLoopAdapter(enabled=True)

        assert adapter.enabled is True
        assert adapter._loop is not None

    @pytest.mark.asyncio
    async def test_record_feedback_when_disabled(self) -> None:
        """Test recording feedback returns success but doesn't store."""
        adapter = FeedbackLoopAdapter(enabled=False)
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        result = await adapter.record_feedback(feedback)

        assert result.success is True
        assert result.stats_updated is False

    @pytest.mark.asyncio
    async def test_record_feedback_when_enabled(self) -> None:
        """Test recording feedback when enabled."""
        adapter = FeedbackLoopAdapter(enabled=True)
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        result = await adapter.record_feedback(feedback)

        assert result.success is True
        assert result.stats_updated is True

    @pytest.mark.asyncio
    async def test_get_query_boost_when_disabled(self) -> None:
        """Test getting boost returns neutral when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        boost = await adapter.get_query_boost("query", "tenant-1")

        assert boost.boost == 1.0
        assert boost.confidence == 0.0

    @pytest.mark.asyncio
    async def test_get_query_boost_when_enabled(self) -> None:
        """Test getting boost when enabled."""
        adapter = FeedbackLoopAdapter(enabled=True)

        boost = await adapter.get_query_boost("query", "tenant-1")

        # Should return neutral (no feedback yet)
        assert boost.boost == 1.0

    def test_get_feedback_stats_when_disabled(self) -> None:
        """Test getting stats returns None when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        stats = adapter.get_feedback_stats("q-123")

        assert stats is None

    def test_get_feedback_for_query_when_disabled(self) -> None:
        """Test getting feedback returns empty list when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        feedback = adapter.get_feedback_for_query("q-123", "tenant-1")

        assert feedback == []

    def test_get_feedback_count_when_disabled(self) -> None:
        """Test getting count returns 0 when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        count = adapter.get_feedback_count("tenant-1")

        assert count == 0

    @pytest.mark.asyncio
    async def test_store_query_embedding_when_disabled(self) -> None:
        """Test storing embedding returns False when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        result = await adapter.store_query_embedding("q-123", "query")

        assert result is False

    def test_clear_tenant_feedback_when_disabled(self) -> None:
        """Test clearing feedback returns 0 when disabled."""
        adapter = FeedbackLoopAdapter(enabled=False)

        count = adapter.clear_tenant_feedback("tenant-1")

        assert count == 0


# ============================================================================
# Feedback Decay Tests
# ============================================================================


class TestFeedbackDecay:
    """Tests for feedback decay functionality."""

    @pytest.mark.asyncio
    async def test_decay_applied_to_old_feedback(self) -> None:
        """Test that decay is applied to feedback older than decay_days."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(
            embedding_provider=embeddings,
            min_samples=2,
            decay_days=30,
        )

        # Store query embedding
        await loop.store_query_embedding("q-123", "test query")

        # Add recent feedback
        recent = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.9,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        await loop.record_feedback(recent)

        # Add old feedback (60 days ago)
        old_feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.9,
            tenant_id="tenant-1",
            user_id="user-2",
            created_at=datetime.now(timezone.utc) - timedelta(days=60),
        )
        await loop.record_feedback(old_feedback)

        boost = await loop.get_query_boost("test query", "tenant-1")

        # Should have decay applied
        assert boost.decay_applied is True

    @pytest.mark.asyncio
    async def test_no_decay_for_recent_feedback(self) -> None:
        """Test that no decay is applied to recent feedback."""
        embeddings = MockEmbeddingProvider()
        loop = FeedbackLoop(
            embedding_provider=embeddings,
            min_samples=2,
            decay_days=30,
        )

        # Store query embedding
        await loop.store_query_embedding("q-123", "test query")

        # Add recent feedback only
        for i in range(3):
            feedback = UserFeedback(
                query_id="q-123",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.9,
                tenant_id="tenant-1",
                user_id=f"user-{i}",
            )
            await loop.record_feedback(feedback)

        boost = await loop.get_query_boost("test query", "tenant-1")

        # Should not have decay applied
        assert boost.decay_applied is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeedbackLoopIntegration:
    """Integration tests for the complete feedback loop."""

    @pytest.mark.asyncio
    async def test_complete_feedback_flow(self) -> None:
        """Test a complete feedback recording and boost retrieval flow."""
        embeddings = MockEmbeddingProvider()
        adapter = FeedbackLoopAdapter(
            enabled=True,
            embedding_provider=embeddings,
            min_samples=3,
        )

        # Store query embedding
        await adapter.store_query_embedding("original-query-id", "What is machine learning?")

        # Record several feedback items
        for i in range(5):
            feedback = UserFeedback(
                query_id="original-query-id",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.7 + (i * 0.05),  # Scores from 0.7 to 0.9
                tenant_id="test-tenant",
                user_id=f"user-{i}",
            )
            result = await adapter.record_feedback(feedback)
            assert result.success is True

        # Get stats
        stats = adapter.get_feedback_stats("original-query-id")
        assert stats is not None
        assert stats.total_count == 5

        # Get boost for a similar query
        boost = await adapter.get_query_boost(
            "What is machine learning?",
            "test-tenant",
        )

        # Should have positive boost due to positive feedback
        assert boost.boost >= 1.0
        assert boost.feedback_count == 5
        assert boost.confidence > 0.0

    @pytest.mark.asyncio
    async def test_multiple_tenants_isolated(self) -> None:
        """Test that feedback is properly isolated between tenants."""
        adapter = FeedbackLoopAdapter(enabled=True, min_samples=2)

        # Add positive feedback for tenant-1
        for i in range(3):
            feedback = UserFeedback(
                query_id="shared-query",
                feedback_type=FeedbackType.RELEVANCE,
                score=0.9,
                tenant_id="tenant-1",
                user_id=f"user-{i}",
            )
            await adapter.record_feedback(feedback)

        # Add negative feedback for tenant-2
        for i in range(3):
            feedback = UserFeedback(
                query_id="shared-query",
                feedback_type=FeedbackType.RELEVANCE,
                score=-0.9,
                tenant_id="tenant-2",
                user_id=f"user-{i}",
            )
            await adapter.record_feedback(feedback)

        # Verify isolation
        assert adapter.get_feedback_count("tenant-1") == 3
        assert adapter.get_feedback_count("tenant-2") == 3

        # Verify feedback retrieval is isolated
        fb1 = adapter.get_feedback_for_query("shared-query", "tenant-1")
        fb2 = adapter.get_feedback_for_query("shared-query", "tenant-2")

        assert all(f.score > 0 for f in fb1)  # All positive
        assert all(f.score < 0 for f in fb2)  # All negative
