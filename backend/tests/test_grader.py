"""Unit tests for the Corrective RAG Grader module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.retrieval.grader import (
    CrossEncoderGrader,
    DEFAULT_CROSS_ENCODER_MODEL,
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
        """Test grading when hits don't have scores (uses content length)."""
        hits = [
            RetrievalHit(content="A" * 500),
            RetrievalHit(content="B" * 500),
            RetrievalHit(content="C" * 500),
        ]

        result = await grader.grade("test query", hits, threshold=0.3)

        assert result.passed is True
        # Score should be based on content length: 500/1000 = 0.5
        assert result.score == pytest.approx(0.5, rel=0.01)

    def test_get_model(self, grader):
        """Test get_model returns heuristic."""
        assert grader.get_model() == "heuristic"


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

        grader = create_grader(settings)

        assert grader is not None
        assert isinstance(grader.grader, HeuristicGrader)
        assert grader.get_model() == "heuristic"

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
