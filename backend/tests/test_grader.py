"""Unit tests for the Corrective RAG Grader module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.retrieval.grader import (
    CrossEncoderGrader,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_HEURISTIC_LENGTH_WEIGHT,
    DEFAULT_HEURISTIC_MIN_LENGTH,
    DEFAULT_HEURISTIC_MAX_LENGTH,
    ExpandedQueryFallback,
    FallbackStrategy,
    GraderResult,
    HeuristicGrader,
    RetrievalGrader,
    RetrievalHit,
    SUPPORTED_GRADER_MODELS,
    WebSearchFallback,
    create_grader,
)


class TestRetrievalHit:
    """Tests for RetrievalHit dataclass."""

    def test_retrieval_hit_creation(self):
        """Test creating a retrieval hit with all fields."""
        hit = RetrievalHit(
            content="Test content",
            score=0.85,
            metadata={"source": "test"},
        )
        assert hit.content == "Test content"
        assert hit.score == 0.85
        assert hit.metadata == {"source": "test"}

    def test_retrieval_hit_optional_fields(self):
        """Test creating a retrieval hit with optional fields as None."""
        hit = RetrievalHit(content="Test content")
        assert hit.content == "Test content"
        assert hit.score is None
        assert hit.metadata is None


class TestGraderResult:
    """Tests for GraderResult dataclass."""

    def test_grader_result_passed(self):
        """Test creating a grader result that passed."""
        result = GraderResult(
            score=0.8,
            passed=True,
            threshold=0.5,
            grading_time_ms=10,
        )
        assert result.score == 0.8
        assert result.passed is True
        assert result.threshold == 0.5
        assert result.grading_time_ms == 10
        assert result.fallback_triggered is False
        assert result.fallback_strategy is None

    def test_grader_result_failed_with_fallback(self):
        """Test creating a grader result that failed with fallback."""
        result = GraderResult(
            score=0.3,
            passed=False,
            threshold=0.5,
            grading_time_ms=15,
            fallback_triggered=True,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
        )
        assert result.score == 0.3
        assert result.passed is False
        assert result.fallback_triggered is True
        assert result.fallback_strategy == FallbackStrategy.WEB_SEARCH


class TestFallbackStrategy:
    """Tests for FallbackStrategy enum."""

    def test_web_search_value(self):
        """Test web_search strategy value."""
        assert FallbackStrategy.WEB_SEARCH.value == "web_search"

    def test_expanded_query_value(self):
        """Test expanded_query strategy value."""
        assert FallbackStrategy.EXPANDED_QUERY.value == "expanded_query"

    def test_alternate_index_value(self):
        """Test alternate_index strategy value."""
        assert FallbackStrategy.ALTERNATE_INDEX.value == "alternate_index"


class TestHeuristicGrader:
    """Tests for HeuristicGrader class."""

    @pytest.fixture
    def grader(self):
        """Create a heuristic grader for testing."""
        return HeuristicGrader(top_k=3)

    @pytest.mark.asyncio
    async def test_grade_with_scores(self, grader):
        """Test grading with retrieval scores available."""
        hits = [
            RetrievalHit(content="Hit 1", score=0.9),
            RetrievalHit(content="Hit 2", score=0.8),
            RetrievalHit(content="Hit 3", score=0.7),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        assert result.passed is True
        assert result.score == pytest.approx(0.8, rel=0.01)  # Average of 0.9, 0.8, 0.7
        assert result.threshold == 0.5
        assert result.grading_time_ms >= 0

    @pytest.mark.asyncio
    async def test_grade_below_threshold(self, grader):
        """Test grading that fails threshold."""
        hits = [
            RetrievalHit(content="Hit 1", score=0.3),
            RetrievalHit(content="Hit 2", score=0.2),
            RetrievalHit(content="Hit 3", score=0.1),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        assert result.passed is False
        assert result.score == pytest.approx(0.2, rel=0.01)  # Average of 0.3, 0.2, 0.1
        assert result.fallback_triggered is True

    @pytest.mark.asyncio
    async def test_grade_empty_hits(self, grader):
        """Test grading with no hits."""
        result = await grader.grade("test query", [], threshold=0.5)

        assert result.passed is False
        assert result.score == 0.0
        assert result.fallback_triggered is True

    @pytest.mark.asyncio
    async def test_grade_without_scores(self, grader):
        """Test grading when hits don't have scores (uses content length heuristic)."""
        hits = [
            RetrievalHit(content="A" * 500),
            RetrievalHit(content="B" * 500),
            RetrievalHit(content="C" * 500),
        ]

        result = await grader.grade("test query", hits, threshold=0.3)

        assert result.passed is True
        # With default settings (weight=0.5, min=50, max=2000):
        # length_factor = (500 - 50) / (2000 - 50) = 450 / 1950 ≈ 0.23
        # score = 0.5 * (1 - 0.5) + 0.23 * 0.5 = 0.25 + 0.115 ≈ 0.365
        assert result.score == pytest.approx(0.365, rel=0.05)

    def test_get_model(self, grader):
        """Test get_model returns heuristic."""
        assert grader.get_model() == "heuristic"


class TestHeuristicGraderLengthWeight:
    """Tests for HeuristicGrader configurable length weight (Story 19-F4)."""

    def test_default_constants(self):
        """Test default constants are defined correctly."""
        assert DEFAULT_HEURISTIC_LENGTH_WEIGHT == 0.5
        assert DEFAULT_HEURISTIC_MIN_LENGTH == 50
        assert DEFAULT_HEURISTIC_MAX_LENGTH == 2000

    def test_init_with_custom_length_settings(self):
        """Test initializing grader with custom length settings."""
        grader = HeuristicGrader(
            top_k=5,
            length_weight=0.7,
            min_length=100,
            max_length=3000,
        )
        assert grader.length_weight == 0.7
        assert grader.min_length == 100
        assert grader.max_length == 3000

    def test_init_length_weight_clamped_to_range(self):
        """Test length_weight is clamped to 0.0-1.0 range."""
        grader_low = HeuristicGrader(length_weight=-0.5)
        grader_high = HeuristicGrader(length_weight=1.5)

        assert grader_low.length_weight == 0.0
        assert grader_high.length_weight == 1.0

    def test_init_min_length_minimum_enforced(self):
        """Test min_length is at least 1."""
        grader = HeuristicGrader(min_length=0)
        assert grader.min_length == 1

    def test_init_max_length_greater_than_min(self):
        """Test max_length is always greater than min_length."""
        grader = HeuristicGrader(min_length=100, max_length=50)
        assert grader.max_length > grader.min_length

    @pytest.mark.asyncio
    async def test_weight_zero_returns_base_score(self):
        """Test weight=0 returns base score (0.5) regardless of content length."""
        grader = HeuristicGrader(top_k=3, length_weight=0.0)
        hits = [
            RetrievalHit(content="A" * 2000),  # Very long content
            RetrievalHit(content="B" * 2000),
            RetrievalHit(content="C" * 2000),
        ]

        result = await grader.grade("test query", hits, threshold=0.3)

        # With weight=0, score should always be base_score (0.5)
        assert result.score == pytest.approx(0.5, rel=0.01)
        assert result.passed is True  # 0.5 >= 0.3

    @pytest.mark.asyncio
    async def test_weight_zero_with_short_content(self):
        """Test weight=0 returns base score even with very short content."""
        grader = HeuristicGrader(top_k=3, length_weight=0.0)
        hits = [
            RetrievalHit(content="A"),  # Very short content
            RetrievalHit(content="B"),
            RetrievalHit(content="C"),
        ]

        result = await grader.grade("test query", hits, threshold=0.3)

        # With weight=0, score should always be base_score (0.5)
        assert result.score == pytest.approx(0.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_weight_half_default_behavior(self):
        """Test weight=0.5 (default) provides balanced scoring."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=0.5,
            min_length=50,
            max_length=2000,
        )
        # Content at max_length should give length_factor=1.0
        hits = [
            RetrievalHit(content="A" * 2000),
            RetrievalHit(content="B" * 2000),
            RetrievalHit(content="C" * 2000),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        # base_score=0.5, length_factor=1.0, weight=0.5
        # score = 0.5 * (1 - 0.5) + 1.0 * 0.5 = 0.25 + 0.5 = 0.75
        assert result.score == pytest.approx(0.75, rel=0.01)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_weight_half_with_min_length_content(self):
        """Test weight=0.5 with content at min_length."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=0.5,
            min_length=50,
            max_length=2000,
        )
        # Content at or below min_length gives length_factor=0.0
        hits = [
            RetrievalHit(content="A" * 50),
            RetrievalHit(content="B" * 50),
            RetrievalHit(content="C" * 50),
        ]

        result = await grader.grade("test query", hits, threshold=0.3)

        # base_score=0.5, length_factor=0.0, weight=0.5
        # score = 0.5 * (1 - 0.5) + 0.0 * 0.5 = 0.25 + 0.0 = 0.25
        assert result.score == pytest.approx(0.25, rel=0.01)
        assert result.passed is False  # 0.25 < 0.3

    @pytest.mark.asyncio
    async def test_weight_one_pure_length_scoring(self):
        """Test weight=1.0 provides pure length-based scoring."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=1.0,
            min_length=50,
            max_length=2000,
        )
        hits = [
            RetrievalHit(content="A" * 2000),
            RetrievalHit(content="B" * 2000),
            RetrievalHit(content="C" * 2000),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        # base_score=0.5, length_factor=1.0, weight=1.0
        # score = 0.5 * (1 - 1.0) + 1.0 * 1.0 = 0.0 + 1.0 = 1.0
        assert result.score == pytest.approx(1.0, rel=0.01)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_weight_one_with_short_content(self):
        """Test weight=1.0 gives low score for short content."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=1.0,
            min_length=50,
            max_length=2000,
        )
        # Content below min_length
        hits = [
            RetrievalHit(content="A" * 30),
            RetrievalHit(content="B" * 30),
            RetrievalHit(content="C" * 30),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        # base_score=0.5, length_factor=0.0 (below min_length), weight=1.0
        # score = 0.5 * (1 - 1.0) + 0.0 * 1.0 = 0.0 + 0.0 = 0.0
        assert result.score == pytest.approx(0.0, rel=0.01)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_mid_range_content_length(self):
        """Test scoring with content in the middle of the range."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=0.5,
            min_length=50,
            max_length=2000,
        )
        # Content at ~50% of the range: 50 + (2000-50)*0.5 = 1025
        avg_length = 1025
        hits = [
            RetrievalHit(content="A" * avg_length),
            RetrievalHit(content="B" * avg_length),
            RetrievalHit(content="C" * avg_length),
        ]

        result = await grader.grade("test query", hits, threshold=0.3)

        # length_factor = (1025 - 50) / (2000 - 50) = 975 / 1950 = 0.5
        # score = 0.5 * (1 - 0.5) + 0.5 * 0.5 = 0.25 + 0.25 = 0.5
        assert result.score == pytest.approx(0.5, rel=0.01)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_retrieval_scores_bypass_length_heuristic(self):
        """Test that when retrieval scores are available, length heuristic is not used."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=1.0,  # Would give 0.0 for short content
            min_length=50,
            max_length=2000,
        )
        # Short content but with retrieval scores
        hits = [
            RetrievalHit(content="A" * 10, score=0.9),
            RetrievalHit(content="B" * 10, score=0.8),
            RetrievalHit(content="C" * 10, score=0.7),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        # Should use retrieval scores (avg = 0.8), not length heuristic
        assert result.score == pytest.approx(0.8, rel=0.01)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_min_max_length(self):
        """Test with custom min/max length settings."""
        grader = HeuristicGrader(
            top_k=3,
            length_weight=1.0,
            min_length=100,
            max_length=500,
        )
        # Content at exactly max_length
        hits = [
            RetrievalHit(content="A" * 500),
            RetrievalHit(content="B" * 500),
            RetrievalHit(content="C" * 500),
        ]

        result = await grader.grade("test query", hits, threshold=0.5)

        # length_factor = (500 - 100) / (500 - 100) = 1.0
        # score = 0.5 * (1 - 1.0) + 1.0 * 1.0 = 1.0
        assert result.score == pytest.approx(1.0, rel=0.01)


class TestCrossEncoderGrader:
    """Tests for CrossEncoderGrader class."""

    def test_init(self):
        """Test cross-encoder grader initialization."""
        grader = CrossEncoderGrader(model_name="test-model")
        assert grader.model_name == "test-model"
        assert grader._model is None  # Lazy loaded
        assert grader._loaded_model_name is None
        assert grader.fallback_to_default is True

    def test_init_custom_fallback(self):
        """Test cross-encoder grader with custom fallback setting."""
        grader = CrossEncoderGrader(model_name="test-model", fallback_to_default=False)
        assert grader.fallback_to_default is False

    def test_get_model_before_loading(self):
        """Test get_model returns configured model name before loading."""
        grader = CrossEncoderGrader(model_name="test-model")
        assert grader.get_model() == "test-model"

    def test_get_model_after_loading(self):
        """Test get_model returns loaded model name after loading."""
        grader = CrossEncoderGrader(model_name="test-model")
        # Simulate that a different model was loaded (fallback scenario)
        grader._loaded_model_name = "fallback-model"
        assert grader.get_model() == "fallback-model"

    def test_default_model_constant(self):
        """Test default cross-encoder model constant."""
        assert DEFAULT_CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_supported_models_defined(self):
        """Test that supported models are defined with proper metadata."""
        assert len(SUPPORTED_GRADER_MODELS) >= 4
        for model_name, metadata in SUPPORTED_GRADER_MODELS.items():
            assert "description" in metadata
            assert "size" in metadata
            assert "speed" in metadata
            assert "accuracy" in metadata

    def test_default_model_in_supported_models(self):
        """Test that default model is in supported models list."""
        assert DEFAULT_CROSS_ENCODER_MODEL in SUPPORTED_GRADER_MODELS

    @pytest.mark.asyncio
    async def test_grade_empty_hits(self):
        """Test grading with no hits returns zero score."""
        grader = CrossEncoderGrader()
        result = await grader.grade("test query", [], threshold=0.5)

        assert result.passed is False
        assert result.score == 0.0
        assert result.fallback_triggered is True

    def test_ensure_model_lazy_loading(self):
        """Test that _ensure_model is truly lazy."""
        grader = CrossEncoderGrader(model_name="test-model")
        # Model should not be loaded at initialization
        assert grader._model is None
        assert grader._loaded_model_name is None

    def test_ensure_model_loads_configured_model(self):
        """Test that _ensure_model loads the configured model."""
        mock_model = MagicMock()
        mock_cross_encoder_class = MagicMock(return_value=mock_model)
        mock_sentence_transformers = MagicMock()
        mock_sentence_transformers.CrossEncoder = mock_cross_encoder_class

        with patch.dict(
            "sys.modules", {"sentence_transformers": mock_sentence_transformers}
        ):
            grader = CrossEncoderGrader(model_name="custom-model")
            grader._ensure_model()

            mock_cross_encoder_class.assert_called_once_with("custom-model")
            assert grader._model == mock_model
            assert grader._loaded_model_name == "custom-model"

    def test_ensure_model_fallback_on_failure(self):
        """Test that _ensure_model falls back to default model on failure."""
        mock_model = MagicMock()
        mock_cross_encoder_class = MagicMock(
            side_effect=[Exception("Model not found"), mock_model]
        )
        mock_sentence_transformers = MagicMock()
        mock_sentence_transformers.CrossEncoder = mock_cross_encoder_class

        with patch.dict(
            "sys.modules", {"sentence_transformers": mock_sentence_transformers}
        ):
            grader = CrossEncoderGrader(
                model_name="nonexistent-model", fallback_to_default=True
            )
            grader._ensure_model()

            # Should have called twice: once for configured, once for fallback
            assert mock_cross_encoder_class.call_count == 2
            assert grader._model == mock_model
            assert grader._loaded_model_name == DEFAULT_CROSS_ENCODER_MODEL

    def test_ensure_model_no_fallback_raises_error(self):
        """Test that _ensure_model raises error when fallback is disabled."""
        mock_cross_encoder_class = MagicMock(side_effect=Exception("Model not found"))
        mock_sentence_transformers = MagicMock()
        mock_sentence_transformers.CrossEncoder = mock_cross_encoder_class

        with patch.dict(
            "sys.modules", {"sentence_transformers": mock_sentence_transformers}
        ):
            grader = CrossEncoderGrader(
                model_name="nonexistent-model", fallback_to_default=False
            )
            with pytest.raises(RuntimeError, match="Failed to load cross-encoder model"):
                grader._ensure_model()


class TestWebSearchFallback:
    """Tests for WebSearchFallback class."""

    def test_init(self):
        """Test web search fallback initialization."""
        fallback = WebSearchFallback(api_key="test-key", max_results=10)
        assert fallback.api_key == "test-key"
        assert fallback.max_results == 10
        assert fallback._client is None  # Lazy loaded

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful web search fallback execution."""
        fallback = WebSearchFallback(api_key="test-key")

        # Mock the Tavily client
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "content": "Result 1 content",
                    "score": 0.9,
                    "url": "https://example.com/1",
                    "title": "Result 1",
                },
                {
                    "content": "Result 2 content",
                    "score": 0.8,
                    "url": "https://example.com/2",
                    "title": "Result 2",
                },
            ]
        }

        fallback._client = mock_client

        hits = await fallback.execute("test query")

        assert len(hits) == 2
        assert hits[0].content == "Result 1 content"
        assert hits[0].score == 0.9
        assert hits[0].metadata["source"] == "tavily_web_search"
        assert hits[0].metadata["url"] == "https://example.com/1"

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test web search fallback handles errors gracefully."""
        fallback = WebSearchFallback(api_key="test-key")

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API error")
        fallback._client = mock_client

        hits = await fallback.execute("test query")

        assert len(hits) == 0


class TestExpandedQueryFallback:
    """Tests for ExpandedQueryFallback class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test expanded query fallback execution."""
        fallback = ExpandedQueryFallback()
        hits = await fallback.execute("test query")

        # Placeholder implementation returns empty list
        assert len(hits) == 0


class TestRetrievalGrader:
    """Tests for RetrievalGrader orchestration class."""

    @pytest.fixture
    def grader(self):
        """Create a retrieval grader for testing."""
        base_grader = HeuristicGrader(top_k=3)
        return RetrievalGrader(
            grader=base_grader,
            threshold=0.5,
            fallback_enabled=True,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=None,
        )

    @pytest.mark.asyncio
    async def test_grade_passing(self, grader):
        """Test grading that passes threshold."""
        hits = [
            RetrievalHit(content="Hit 1", score=0.9),
            RetrievalHit(content="Hit 2", score=0.8),
        ]

        result, fallback_hits = await grader.grade_and_fallback("test query", hits)

        assert result.passed is True
        assert len(fallback_hits) == 0

    @pytest.mark.asyncio
    async def test_grade_failing_triggers_fallback(self):
        """Test grading that fails triggers fallback handler."""
        base_grader = HeuristicGrader(top_k=3)
        mock_fallback = AsyncMock()
        mock_fallback.execute.return_value = [
            RetrievalHit(content="Fallback hit", score=0.7)
        ]

        grader = RetrievalGrader(
            grader=base_grader,
            threshold=0.5,
            fallback_enabled=True,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=mock_fallback,
        )

        hits = [
            RetrievalHit(content="Hit 1", score=0.2),
            RetrievalHit(content="Hit 2", score=0.1),
        ]

        result, fallback_hits = await grader.grade_and_fallback("test query", hits)

        assert result.passed is False
        assert result.fallback_triggered is True
        assert result.fallback_strategy == FallbackStrategy.WEB_SEARCH
        assert len(fallback_hits) == 1
        assert fallback_hits[0].content == "Fallback hit"
        mock_fallback.execute.assert_called_once_with("test query", None)

    @pytest.mark.asyncio
    async def test_grade_failing_no_fallback_handler(self, grader):
        """Test grading that fails but no fallback handler configured."""
        hits = [
            RetrievalHit(content="Hit 1", score=0.2),
            RetrievalHit(content="Hit 2", score=0.1),
        ]

        result, fallback_hits = await grader.grade_and_fallback("test query", hits)

        assert result.passed is False
        # No fallback handler, so no fallback hits
        assert len(fallback_hits) == 0

    @pytest.mark.asyncio
    async def test_grade_failing_fallback_disabled(self):
        """Test grading that fails but fallback is disabled."""
        base_grader = HeuristicGrader(top_k=3)
        mock_fallback = AsyncMock()

        grader = RetrievalGrader(
            grader=base_grader,
            threshold=0.5,
            fallback_enabled=False,  # Disabled
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=mock_fallback,
        )

        hits = [
            RetrievalHit(content="Hit 1", score=0.2),
        ]

        result, fallback_hits = await grader.grade_and_fallback("test query", hits)

        assert result.passed is False
        assert len(fallback_hits) == 0
        mock_fallback.execute.assert_not_called()

    def test_get_model(self, grader):
        """Test get_model returns base grader model."""
        assert grader.get_model() == "heuristic"


class TestCreateGrader:
    """Tests for create_grader factory function."""

    def test_grader_disabled(self):
        """Test that grader is None when disabled."""
        settings = MagicMock()
        settings.grader_enabled = False

        grader = create_grader(settings)

        assert grader is None

    def test_grader_enabled_web_search_fallback(self):
        """Test creating grader with web search fallback."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "heuristic"
        settings.grader_threshold = 0.6
        settings.grader_fallback_enabled = True
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = "test-tavily-key"
        settings.grader_heuristic_length_weight = 0.5
        settings.grader_heuristic_min_length = 50
        settings.grader_heuristic_max_length = 2000

        grader = create_grader(settings)

        assert grader is not None
        assert grader.threshold == 0.6
        assert grader.fallback_enabled is True
        assert grader.fallback_strategy == FallbackStrategy.WEB_SEARCH
        assert isinstance(grader.fallback_handler, WebSearchFallback)

    def test_grader_enabled_expanded_query_fallback(self):
        """Test creating grader with expanded query fallback."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "heuristic"
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = True
        settings.grader_fallback_strategy = "expanded_query"
        settings.tavily_api_key = None
        settings.grader_heuristic_length_weight = 0.5
        settings.grader_heuristic_min_length = 50
        settings.grader_heuristic_max_length = 2000

        grader = create_grader(settings)

        assert grader is not None
        assert grader.fallback_strategy == FallbackStrategy.EXPANDED_QUERY
        assert isinstance(grader.fallback_handler, ExpandedQueryFallback)

    def test_grader_enabled_no_tavily_key(self):
        """Test creating grader without Tavily API key."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "heuristic"
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = True
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = None
        settings.grader_heuristic_length_weight = 0.5
        settings.grader_heuristic_min_length = 50
        settings.grader_heuristic_max_length = 2000

        grader = create_grader(settings)

        assert grader is not None
        # No fallback handler because no Tavily key
        assert grader.fallback_handler is None

    def test_grader_enabled_fallback_disabled(self):
        """Test creating grader with fallback disabled."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "heuristic"
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = False
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = "test-key"
        settings.grader_heuristic_length_weight = 0.5
        settings.grader_heuristic_min_length = 50
        settings.grader_heuristic_max_length = 2000

        grader = create_grader(settings)

        assert grader is not None
        assert grader.fallback_enabled is False
        # No fallback handler created when disabled
        assert grader.fallback_handler is None

    def test_grader_with_heuristic_model(self):
        """Test creating grader with heuristic model."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "heuristic"
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = False
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = None
        settings.grader_heuristic_length_weight = 0.5
        settings.grader_heuristic_min_length = 50
        settings.grader_heuristic_max_length = 2000

        grader = create_grader(settings)

        assert grader is not None
        assert isinstance(grader.grader, HeuristicGrader)
        assert grader.get_model() == "heuristic"

    def test_grader_with_heuristic_uses_config_length_settings(self):
        """Test creating grader with heuristic uses config length settings."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "heuristic"
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = False
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = None
        # Custom heuristic settings
        settings.grader_heuristic_length_weight = 0.8
        settings.grader_heuristic_min_length = 100
        settings.grader_heuristic_max_length = 3000

        grader = create_grader(settings)

        assert grader is not None
        assert isinstance(grader.grader, HeuristicGrader)
        # Verify the heuristic grader has the correct settings
        heuristic_grader = grader.grader
        assert heuristic_grader.length_weight == 0.8
        assert heuristic_grader.min_length == 100
        assert heuristic_grader.max_length == 3000

    def test_grader_with_cross_encoder_model(self):
        """Test creating grader with cross-encoder model."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = False
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = None

        grader = create_grader(settings)

        assert grader is not None
        assert isinstance(grader.grader, CrossEncoderGrader)
        assert grader.get_model() == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_grader_with_bge_reranker_model(self):
        """Test creating grader with BGE reranker model."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "BAAI/bge-reranker-base"
        settings.grader_threshold = 0.6
        settings.grader_fallback_enabled = False
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = None

        grader = create_grader(settings)

        assert grader is not None
        assert isinstance(grader.grader, CrossEncoderGrader)
        assert grader.get_model() == "BAAI/bge-reranker-base"

    def test_grader_model_case_insensitive(self):
        """Test that 'Heuristic' (any case) maps to heuristic grader."""
        settings = MagicMock()
        settings.grader_enabled = True
        settings.grader_model = "HEURISTIC"  # uppercase
        settings.grader_threshold = 0.5
        settings.grader_fallback_enabled = False
        settings.grader_fallback_strategy = "web_search"
        settings.tavily_api_key = None
        settings.grader_heuristic_length_weight = 0.5
        settings.grader_heuristic_min_length = 50
        settings.grader_heuristic_max_length = 2000

        grader = create_grader(settings)

        assert grader is not None
        assert isinstance(grader.grader, HeuristicGrader)


class TestConfigIntegration:
    """Tests for configuration integration."""

    def test_grader_disabled_by_default(self):
        """Test that grader is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ["OPENAI_API_KEY"] = "test-key"
            os.environ["DATABASE_URL"] = "postgresql://test"
            os.environ["NEO4J_URI"] = "bolt://localhost"
            os.environ["NEO4J_USER"] = "neo4j"
            os.environ["NEO4J_PASSWORD"] = "password"
            os.environ["REDIS_URL"] = "redis://localhost"

            # Need to clear the LRU cache
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_enabled is False
            assert settings.grader_threshold == 0.5
            assert settings.grader_fallback_enabled is True
            assert settings.grader_fallback_strategy == "web_search"

    def test_grader_enabled_with_config(self):
        """Test grader can be enabled via environment."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_ENABLED": "true",
                "GRADER_THRESHOLD": "0.7",
                "GRADER_FALLBACK_STRATEGY": "expanded_query",
                "TAVILY_API_KEY": "test-tavily-key",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_enabled is True
            assert settings.grader_threshold == 0.7
            assert settings.grader_fallback_strategy == "expanded_query"
            assert settings.tavily_api_key == "test-tavily-key"

    def test_grader_threshold_clamped(self):
        """Test grader threshold is clamped to 0.0-1.0."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_ENABLED": "true",
                "GRADER_THRESHOLD": "1.5",  # Over 1.0
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_threshold == 1.0  # Clamped to max

    def test_grader_invalid_strategy_defaults(self):
        """Test invalid fallback strategy defaults to web_search."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_ENABLED": "true",
                "GRADER_FALLBACK_STRATEGY": "invalid_strategy",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_fallback_strategy == "web_search"

    def test_grader_model_default_is_heuristic(self):
        """Test grader_model defaults to heuristic."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_model == "heuristic"

    def test_grader_model_configurable_cross_encoder(self):
        """Test grader_model can be set to a cross-encoder model."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_grader_model_configurable_bge_reranker(self):
        """Test grader_model can be set to a BGE reranker model."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_MODEL": "BAAI/bge-reranker-large",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_model == "BAAI/bge-reranker-large"

    def test_grader_model_whitespace_stripped(self):
        """Test grader_model strips leading/trailing whitespace."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_MODEL": "  heuristic  ",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_model == "heuristic"

    def test_heuristic_length_weight_defaults(self):
        """Test heuristic length weight has correct defaults."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_heuristic_length_weight == 0.5
            assert settings.grader_heuristic_min_length == 50
            assert settings.grader_heuristic_max_length == 2000

    def test_heuristic_length_weight_configurable(self):
        """Test heuristic length weight can be configured."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_HEURISTIC_LENGTH_WEIGHT": "0.7",
                "GRADER_HEURISTIC_MIN_LENGTH": "100",
                "GRADER_HEURISTIC_MAX_LENGTH": "3000",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_heuristic_length_weight == 0.7
            assert settings.grader_heuristic_min_length == 100
            assert settings.grader_heuristic_max_length == 3000

    def test_heuristic_length_weight_clamped_to_range(self):
        """Test length weight is clamped to 0.0-1.0 range."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_HEURISTIC_LENGTH_WEIGHT": "1.5",  # Over 1.0
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_heuristic_length_weight == 1.0  # Clamped to max

    def test_heuristic_length_weight_zero_disables_length(self):
        """Test length weight of 0 is valid (disables length influence)."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test",
                "NEO4J_URI": "bolt://localhost",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "REDIS_URL": "redis://localhost",
                "GRADER_HEURISTIC_LENGTH_WEIGHT": "0",
            },
            clear=True,
        ):
            from agentic_rag_backend.config import get_settings, load_settings

            get_settings.cache_clear()
            settings = load_settings()

            assert settings.grader_heuristic_length_weight == 0.0
