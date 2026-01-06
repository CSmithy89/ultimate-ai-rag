"""Tests for Dual-Level Retrieval (Story 20-C2).

This module tests the DualLevelRetriever class and its components:
- Low-level entity/chunk retrieval via Graphiti
- High-level community/theme retrieval via CommunityDetector (20-B1)
- LLM-based synthesis of both levels
- Confidence calculation based on coverage
- Parallel execution and fallback handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from agentic_rag_backend.retrieval.dual_level import DualLevelRetriever
from agentic_rag_backend.retrieval.dual_level_models import (
    LowLevelResult,
    HighLevelResult,
    DualLevelResult,
    SynthesisResult,
    DualLevelRetrieveRequest,
    DualLevelRetrieveResponse,
    LowLevelResultResponse,
    HighLevelResultResponse,
    DualLevelStatusResponse,
)


def _make_mock_node(uuid: str, name: str, summary: str, labels: list, type_str: str = "Entity"):
    """Create a mock Graphiti node."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)
    node.summary = summary
    node.description = summary
    node.labels = labels
    node.type = type_str
    node.source_id = "source-1"
    return node


def _make_mock_settings(
    dual_level_retrieval_enabled: bool = True,
    dual_level_low_weight: float = 0.6,
    dual_level_high_weight: float = 0.4,
    dual_level_low_limit: int = 10,
    dual_level_high_limit: int = 5,
    dual_level_synthesis_model: str = "gpt-4o-mini",
    llm_provider: str = "openai",
    llm_api_key: str = "test-key",
    llm_base_url: str = None,
    community_detection_enabled: bool = True,
):
    """Create mock settings for DualLevelRetriever."""
    settings = MagicMock()
    settings.dual_level_retrieval_enabled = dual_level_retrieval_enabled
    settings.dual_level_low_weight = dual_level_low_weight
    settings.dual_level_high_weight = dual_level_high_weight
    settings.dual_level_low_limit = dual_level_low_limit
    settings.dual_level_high_limit = dual_level_high_limit
    settings.dual_level_synthesis_model = dual_level_synthesis_model
    settings.llm_provider = llm_provider
    settings.llm_api_key = llm_api_key
    settings.llm_base_url = llm_base_url
    settings.community_detection_enabled = community_detection_enabled
    return settings


class TestLowLevelResult:
    """Tests for LowLevelResult dataclass."""

    def test_creation(self):
        """Should create low-level result with all fields."""
        result = LowLevelResult(
            id="entity-1",
            name="FastAPI",
            type="Framework",
            content="Modern Python web framework",
            score=0.95,
            source="docs/fastapi.md",
            labels=["Framework", "Python"],
        )

        assert result.id == "entity-1"
        assert result.name == "FastAPI"
        assert result.type == "Framework"
        assert result.content == "Modern Python web framework"
        assert result.score == 0.95
        assert result.source == "docs/fastapi.md"
        assert result.labels == ["Framework", "Python"]

    def test_to_response(self):
        """Should convert to API response model."""
        result = LowLevelResult(
            id="entity-1",
            name="FastAPI",
            type="Framework",
            content="Web framework",
            score=0.9,
            source="source-1",
        )

        response = result.to_response()

        assert isinstance(response, LowLevelResultResponse)
        assert response.id == "entity-1"
        assert response.name == "FastAPI"
        assert response.type == "Framework"
        assert response.content == "Web framework"
        assert response.score == 0.9

    def test_default_values(self):
        """Should use default values for optional fields."""
        result = LowLevelResult(
            id="entity-1",
            name="Test",
        )

        assert result.type == "Entity"
        assert result.content is None
        assert result.score == 0.0
        assert result.source is None
        assert result.labels == []


class TestHighLevelResult:
    """Tests for HighLevelResult dataclass."""

    def test_creation(self):
        """Should create high-level result with all fields."""
        result = HighLevelResult(
            id="comm-1",
            name="Web Frameworks",
            summary="Collection of Python web frameworks",
            keywords=("web", "framework", "api", "python"),
            level=1,
            entity_count=15,
            score=0.85,
            entity_ids=("e1", "e2", "e3"),
        )

        assert result.id == "comm-1"
        assert result.name == "Web Frameworks"
        assert result.summary == "Collection of Python web frameworks"
        assert "web" in result.keywords
        assert result.level == 1
        assert result.entity_count == 15
        assert result.score == 0.85

    def test_to_response(self):
        """Should convert to API response model."""
        result = HighLevelResult(
            id="comm-1",
            name="Web Frameworks",
            summary="Python web frameworks",
            keywords=("web", "api"),
            level=1,
            entity_count=10,
            score=0.8,
        )

        response = result.to_response()

        assert isinstance(response, HighLevelResultResponse)
        assert response.id == "comm-1"
        assert response.name == "Web Frameworks"
        assert response.summary == "Python web frameworks"
        assert isinstance(response.keywords, list)
        assert "web" in response.keywords
        assert response.level == 1
        assert response.entity_count == 10

    def test_default_values(self):
        """Should use default values for optional fields."""
        result = HighLevelResult(
            id="comm-1",
            name="Test",
        )

        assert result.summary is None
        assert result.keywords == ()
        assert result.level == 0
        assert result.entity_count == 0
        assert result.score == 0.0


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_creation(self):
        """Should create synthesis result with all fields."""
        result = SynthesisResult(
            text="FastAPI is a modern Python web framework built on Starlette.",
            confidence=0.9,
            reasoning="Both levels provide consistent information",
        )

        assert result.text == "FastAPI is a modern Python web framework built on Starlette."
        assert result.confidence == 0.9
        assert result.reasoning == "Both levels provide consistent information"


class TestDualLevelResult:
    """Tests for DualLevelResult dataclass."""

    def test_creation(self):
        """Should create dual-level result with all fields."""
        low_results = [
            LowLevelResult(id="e1", name="FastAPI", score=0.9),
            LowLevelResult(id="e2", name="Starlette", score=0.8),
        ]
        high_results = [
            HighLevelResult(id="c1", name="Web Frameworks", score=0.85),
        ]

        result = DualLevelResult(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            low_level_results=low_results,
            high_level_results=high_results,
            synthesis="FastAPI is a Python web framework.",
            confidence=0.87,
            processing_time_ms=150,
            fallback_used=False,
        )

        assert result.query == "What is FastAPI?"
        assert result.tenant_id == "tenant-1"
        assert len(result.low_level_results) == 2
        assert len(result.high_level_results) == 1
        assert result.synthesis == "FastAPI is a Python web framework."
        assert result.confidence == 0.87
        assert result.processing_time_ms == 150
        assert result.fallback_used is False

    def test_to_response(self):
        """Should convert to API response model."""
        result = DualLevelResult(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            low_level_results=[LowLevelResult(id="e1", name="FastAPI", score=0.9)],
            high_level_results=[HighLevelResult(id="c1", name="Web Frameworks", score=0.85)],
            synthesis="Summary text",
            confidence=0.85,
            processing_time_ms=100,
            fallback_used=False,
        )

        response = result.to_response()

        assert isinstance(response, DualLevelRetrieveResponse)
        assert response.query == "What is FastAPI?"
        assert response.tenant_id == "tenant-1"
        assert response.low_level_count == 1
        assert response.high_level_count == 1
        assert response.synthesis == "Summary text"
        assert response.confidence == 0.85
        assert response.processing_time_ms == 100


class TestDualLevelRetriever:
    """Tests for DualLevelRetriever class."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient."""
        client = MagicMock()
        client.client = MagicMock()
        client.is_connected = True

        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("node-1", "FastAPI", "Python web framework", ["Entity"]),
            _make_mock_node("node-2", "Python", "Programming language", ["Entity"]),
        ]
        search_result.edges = []

        client.client.search = AsyncMock(return_value=search_result)
        return client

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create a mock Neo4j client."""
        client = MagicMock()
        driver = MagicMock()

        # Mock session context manager
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        # Mock run returns for community queries
        community_records = [
            {
                "id": "comm-1",
                "name": "Web Frameworks",
                "summary": "Python web framework community",
                "keywords": ["web", "framework"],
                "level": 1,
                "entity_count": 10,
            }
        ]

        run_result = MagicMock()
        run_result.data = AsyncMock(return_value=community_records)
        session.run = AsyncMock(return_value=run_result)

        driver.session = MagicMock(return_value=session)
        client.driver = driver

        return client

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        return _make_mock_settings()

    @pytest.fixture
    def retriever(self, mock_graphiti_client, mock_neo4j_client, mock_settings):
        """Create DualLevelRetriever with mocked dependencies."""
        return DualLevelRetriever(
            graphiti_client=mock_graphiti_client,
            neo4j_client=mock_neo4j_client,
            settings=mock_settings,
            community_detector=None,
        )

    def test_initialization(self, retriever, mock_settings):
        """Should initialize with correct settings."""
        assert retriever.low_weight == mock_settings.dual_level_low_weight
        assert retriever.high_weight == mock_settings.dual_level_high_weight
        assert retriever.low_limit == mock_settings.dual_level_low_limit
        assert retriever.high_limit == mock_settings.dual_level_high_limit
        assert retriever.synthesis_model == mock_settings.dual_level_synthesis_model

    def test_initialization_with_defaults(self):
        """Should use default settings when not provided."""
        # Create minimal settings without dual-level attributes
        settings = MagicMock(spec=[])

        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=settings,
            community_detector=None,
        )

        # Should use defaults
        assert retriever.low_weight == 0.6
        assert retriever.high_weight == 0.4
        assert retriever.low_limit == 10
        assert retriever.high_limit == 5
        assert retriever.synthesis_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_retrieve_low_level_uses_graphiti(self, retriever, mock_graphiti_client):
        """Should use Graphiti for low-level entity search."""
        results = await retriever._retrieve_low_level(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            limit=5,
        )

        mock_graphiti_client.client.search.assert_called_once()
        call_kwargs = mock_graphiti_client.client.search.call_args[1]
        assert call_kwargs["group_ids"] == ["tenant-1"]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_low_level_returns_correct_type(self, retriever, mock_graphiti_client):
        """Should return LowLevelResult objects."""
        results = await retriever._retrieve_low_level(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            limit=5,
        )

        assert all(isinstance(r, LowLevelResult) for r in results)
        assert results[0].name == "FastAPI"
        assert results[1].name == "Python"

    @pytest.mark.asyncio
    async def test_retrieve_low_level_fallback_when_graphiti_disconnected(self):
        """Should fall back to Neo4j when Graphiti is disconnected."""
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = False

        mock_neo4j = MagicMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        run_result = MagicMock()
        run_result.data = AsyncMock(return_value=[
            {"id": "e1", "name": "Fallback Entity", "type": "Entity", "description": None, "summary": None}
        ])
        session.run = AsyncMock(return_value=run_result)
        mock_neo4j.driver.session = MagicMock(return_value=session)

        settings = _make_mock_settings()
        retriever = DualLevelRetriever(
            graphiti_client=mock_graphiti,
            neo4j_client=mock_neo4j,
            settings=settings,
        )

        results = await retriever._retrieve_low_level(
            query="test",
            tenant_id="tenant-1",
            limit=5,
        )

        assert len(results) == 1
        assert results[0].name == "Fallback Entity"
        # Fallback results have lower score
        assert results[0].score == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_high_level_uses_neo4j(self, retriever, mock_neo4j_client):
        """Should query Neo4j for high-level communities."""
        results = await retriever._retrieve_high_level(
            query="web framework",
            tenant_id="tenant-1",
            limit=5,
        )

        mock_neo4j_client.driver.session.assert_called()
        assert len(results) == 1
        assert results[0].name == "Web Frameworks"
        assert results[0].level == 1

    @pytest.mark.asyncio
    async def test_retrieve_high_level_returns_correct_type(self, retriever, mock_neo4j_client):
        """Should return HighLevelResult objects."""
        results = await retriever._retrieve_high_level(
            query="web framework",
            tenant_id="tenant-1",
            limit=5,
        )

        assert all(isinstance(r, HighLevelResult) for r in results)
        assert results[0].name == "Web Frameworks"
        assert "web" in results[0].keywords

    @pytest.mark.asyncio
    async def test_retrieve_high_level_returns_empty_when_no_neo4j(self):
        """Should return empty list when Neo4j is not available."""
        settings = _make_mock_settings()
        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=settings,
        )

        results = await retriever._retrieve_high_level(
            query="test",
            tenant_id="tenant-1",
            limit=5,
        )

        assert results == []

    def test_calculate_confidence_no_results(self, retriever):
        """Should return 0 confidence for empty results."""
        confidence = retriever._calculate_confidence(
            low_level_results=[],
            high_level_results=[],
            low_weight=0.6,
            high_weight=0.4,
        )

        assert confidence == 0.0

    def test_calculate_confidence_with_both_levels(self, retriever):
        """Should calculate weighted confidence from both levels."""
        low_results = [
            LowLevelResult(id="e1", name="FastAPI", score=0.9),
            LowLevelResult(id="e2", name="Python", score=0.8),
        ]
        high_results = [
            HighLevelResult(id="c1", name="Frameworks", score=0.85),
        ]

        confidence = retriever._calculate_confidence(
            low_level_results=low_results,
            high_level_results=high_results,
            low_weight=0.6,
            high_weight=0.4,
        )

        # Should be between 0 and 1
        assert 0.0 < confidence <= 1.0
        # With good scores on both levels, confidence should be high
        assert confidence >= 0.7

    def test_calculate_confidence_low_level_only(self, retriever):
        """Should calculate confidence from low-level only."""
        low_results = [
            LowLevelResult(id="e1", name="FastAPI", score=0.9),
        ]

        confidence = retriever._calculate_confidence(
            low_level_results=low_results,
            high_level_results=[],
            low_weight=0.6,
            high_weight=0.4,
        )

        # Should have positive confidence from low-level
        assert confidence > 0.0
        assert confidence <= 1.0

    def test_calculate_confidence_high_level_only(self, retriever):
        """Should calculate confidence from high-level only."""
        high_results = [
            HighLevelResult(id="c1", name="Frameworks", score=0.85),
        ]

        confidence = retriever._calculate_confidence(
            low_level_results=[],
            high_level_results=high_results,
            low_weight=0.6,
            high_weight=0.4,
        )

        # Should have positive confidence from high-level
        assert confidence > 0.0
        assert confidence <= 1.0

    def test_calculate_confidence_bonus_for_both_levels(self, retriever):
        """Should give bonus confidence when both levels have results."""
        low_results = [LowLevelResult(id="e1", name="Entity", score=0.5)]
        high_results = [HighLevelResult(id="c1", name="Community", score=0.5)]

        conf_both = retriever._calculate_confidence(
            low_level_results=low_results,
            high_level_results=high_results,
            low_weight=0.5,
            high_weight=0.5,
        )

        conf_low_only = retriever._calculate_confidence(
            low_level_results=low_results,
            high_level_results=[],
            low_weight=0.5,
            high_weight=0.5,
        )

        # Both levels together should have higher confidence
        assert conf_both >= conf_low_only

    @pytest.mark.asyncio
    async def test_retrieve_parallel_execution(self, mock_graphiti_client, mock_neo4j_client, mock_settings):
        """Should execute low and high level retrieval in parallel."""
        retriever = DualLevelRetriever(
            graphiti_client=mock_graphiti_client,
            neo4j_client=mock_neo4j_client,
            settings=mock_settings,
        )

        result = await retriever.retrieve(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            include_synthesis=False,  # Skip synthesis for speed
        )

        assert isinstance(result, DualLevelResult)
        assert len(result.low_level_results) > 0
        assert len(result.high_level_results) > 0

    @pytest.mark.asyncio
    async def test_retrieve_with_weight_overrides(self, mock_graphiti_client, mock_neo4j_client, mock_settings):
        """Should respect weight parameter overrides."""
        retriever = DualLevelRetriever(
            graphiti_client=mock_graphiti_client,
            neo4j_client=mock_neo4j_client,
            settings=mock_settings,
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant-1",
            low_weight=0.8,
            high_weight=0.2,
            include_synthesis=False,
        )

        # Result should be valid
        assert isinstance(result, DualLevelResult)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_retrieve_fallback_detection(self, mock_settings):
        """Should detect fallback when one level has no results."""
        # Graphiti returns results
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = True
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("n1", "FastAPI", "Framework", ["Entity"]),
        ]
        mock_graphiti.client.search = AsyncMock(return_value=search_result)

        # Neo4j returns empty (no communities found)
        mock_neo4j = MagicMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        run_result = MagicMock()
        run_result.data = AsyncMock(return_value=[])
        session.run = AsyncMock(return_value=run_result)
        mock_neo4j.driver.session = MagicMock(return_value=session)

        retriever = DualLevelRetriever(
            graphiti_client=mock_graphiti,
            neo4j_client=mock_neo4j,
            settings=mock_settings,
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant-1",
            include_synthesis=False,
        )

        # Should detect fallback since high-level is empty
        assert result.fallback_used is True
        assert len(result.low_level_results) > 0
        assert len(result.high_level_results) == 0


class TestDualLevelModels:
    """Tests for Pydantic request/response models."""

    def test_retrieve_request_validation(self):
        """Should validate retrieve request fields."""
        from uuid import UUID

        request = DualLevelRetrieveRequest(
            query="What is FastAPI?",
            tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
            low_level_limit=15,
            high_level_limit=8,
            include_synthesis=True,
            low_weight=0.7,
            high_weight=0.3,
        )

        assert request.query == "What is FastAPI?"
        assert request.low_level_limit == 15
        assert request.high_level_limit == 8
        assert request.include_synthesis is True
        assert request.low_weight == 0.7
        assert request.high_weight == 0.3

    def test_retrieve_request_defaults(self):
        """Should use default values for optional fields."""
        from uuid import UUID

        request = DualLevelRetrieveRequest(
            query="Test query",
            tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
        )

        assert request.low_level_limit == 10
        assert request.high_level_limit == 5
        assert request.include_synthesis is True
        assert request.low_weight == 0.6
        assert request.high_weight == 0.4

    def test_retrieve_response_structure(self):
        """Should have all required response fields."""
        response = DualLevelRetrieveResponse(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            low_level_results=[],
            high_level_results=[],
            synthesis="FastAPI is a framework.",
            confidence=0.85,
            low_level_count=5,
            high_level_count=3,
            processing_time_ms=250,
            fallback_used=False,
        )

        assert response.confidence == 0.85
        assert response.processing_time_ms == 250
        assert response.fallback_used is False

    def test_status_response_structure(self):
        """Should have all required status fields."""
        status = DualLevelStatusResponse(
            enabled=True,
            low_weight=0.6,
            high_weight=0.4,
            low_limit=10,
            high_limit=5,
            synthesis_model="gpt-4o-mini",
            synthesis_temperature=0.3,
            graphiti_available=True,
            community_detection_available=True,
        )

        assert status.enabled is True
        assert status.graphiti_available is True
        assert status.community_detection_available is True
        assert status.synthesis_model == "gpt-4o-mini"
        assert status.synthesis_temperature == 0.3


class TestDualLevelSynthesis:
    """Tests for synthesis functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for synthesis tests."""
        return _make_mock_settings()

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self, mock_settings):
        """Should handle empty results gracefully."""
        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=mock_settings,
        )

        result = await retriever._synthesize(
            query="test",
            low_level_results=[],
            high_level_results=[],
        )

        assert result is not None
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_low_level_only(self, mock_settings):
        """Should synthesize with low-level results only."""
        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=mock_settings,
        )

        low_results = [
            LowLevelResult(id="e1", name="FastAPI", content="Python web framework", score=0.9),
        ]

        with patch('agentic_rag_backend.retrieval.dual_level.get_llm_adapter') as mock_adapter:
            # Setup mock adapter to return OpenAI-compatible provider
            adapter = MagicMock()
            adapter.provider = "openai"
            adapter.openai_kwargs.return_value = {"api_key": "test-key"}
            mock_adapter.return_value = adapter

            with patch('agentic_rag_backend.retrieval.dual_level.AsyncOpenAI') as mock_openai:
                # Mock OpenAI response
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "FastAPI is a modern Python web framework."

                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                result = await retriever._synthesize(
                    query="What is FastAPI?",
                    low_level_results=low_results,
                    high_level_results=[],
                )

                assert result is not None
                assert result.text == "FastAPI is a modern Python web framework."
                assert result.confidence > 0.0

    def test_estimate_synthesis_confidence_high(self, mock_settings):
        """Should detect high confidence markers in synthesis."""
        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=mock_settings,
        )

        confidence = retriever._estimate_synthesis_confidence(
            synthesis_text="I have HIGH CONFIDENCE that FastAPI is a Python framework.",
            low_count=5,
            high_count=3,
        )

        assert confidence >= 0.7

    def test_estimate_synthesis_confidence_low(self, mock_settings):
        """Should detect low confidence markers in synthesis."""
        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=mock_settings,
        )

        confidence = retriever._estimate_synthesis_confidence(
            synthesis_text="I am uncertain about this topic. LOW CONFIDENCE.",
            low_count=1,
            high_count=0,
        )

        assert confidence < 0.5

    def test_estimate_synthesis_confidence_boost_both_levels(self, mock_settings):
        """Should boost confidence when both levels have results."""
        retriever = DualLevelRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=mock_settings,
        )

        conf_both = retriever._estimate_synthesis_confidence(
            synthesis_text="The answer is clear.",
            low_count=5,
            high_count=3,
        )

        conf_low_only = retriever._estimate_synthesis_confidence(
            synthesis_text="The answer is clear.",
            low_count=5,
            high_count=0,
        )

        assert conf_both > conf_low_only
