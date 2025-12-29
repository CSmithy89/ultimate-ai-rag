"""Tests for Graphiti episode-based document ingestion."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import hashlib

from agentic_rag_backend.models.documents import UnifiedDocument, SourceType


def _make_content_hash(content: str) -> str:
    """Generate a SHA-256 hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()


class TestGraphitiIngestion:
    """Tests for Graphiti episode ingestion service."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient."""
        client = MagicMock()
        client.client = MagicMock()
        client.client.add_episode = AsyncMock(return_value=MagicMock(
            uuid="episode-123",
            name="Test Document",
            entity_references=[],
            created_at=datetime.now(timezone.utc),
        ))
        client.is_connected = True
        return client

    @pytest.fixture
    def sample_document(self):
        """Create a sample unified document."""
        content = "FastAPI is a modern web framework for Python. It supports async/await."
        return UnifiedDocument(
            id=uuid4(),
            tenant_id=uuid4(),
            source_type=SourceType.URL,
            source_url="https://example.com/docs",
            content=content,
            content_hash=_make_content_hash(content),
        )

    @pytest.mark.asyncio
    async def test_ingest_document_as_episode(self, mock_graphiti_client, sample_document):
        """Should ingest document as Graphiti episode."""
        from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode

        result = await ingest_document_as_episode(
            graphiti_client=mock_graphiti_client,
            document=sample_document,
        )

        assert result is not None
        mock_graphiti_client.client.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_document_uses_tenant_as_group_id(self, mock_graphiti_client, sample_document):
        """Should use tenant_id as group_id for multi-tenancy."""
        from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode

        await ingest_document_as_episode(
            graphiti_client=mock_graphiti_client,
            document=sample_document,
        )

        call_kwargs = mock_graphiti_client.client.add_episode.call_args[1]
        assert call_kwargs["group_id"] == str(sample_document.tenant_id)

    @pytest.mark.asyncio
    async def test_ingest_document_passes_entity_types(self, mock_graphiti_client, sample_document):
        """Should pass custom entity types to Graphiti."""
        from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode

        await ingest_document_as_episode(
            graphiti_client=mock_graphiti_client,
            document=sample_document,
        )

        call_kwargs = mock_graphiti_client.client.add_episode.call_args[1]
        entity_types = call_kwargs.get("entity_types", [])
        # Verify entity types are passed
        assert len(entity_types) == 4

    @pytest.mark.asyncio
    async def test_ingest_document_handles_empty_content(self, mock_graphiti_client):
        """Should handle documents with empty content gracefully."""
        from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode
        from agentic_rag_backend.core.errors import IngestionError

        # Create document with whitespace-only content
        empty_content = "   "
        empty_doc = UnifiedDocument(
            id=uuid4(),
            tenant_id=uuid4(),
            source_type=SourceType.URL,
            source_url="https://example.com",
            content=empty_content,
            content_hash=_make_content_hash(empty_content),
        )

        with pytest.raises(IngestionError):
            await ingest_document_as_episode(
                graphiti_client=mock_graphiti_client,
                document=empty_doc,
            )

    @pytest.mark.asyncio
    async def test_ingest_document_not_connected_raises(self, sample_document):
        """Should raise error if Graphiti client not connected."""
        from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode

        disconnected_client = MagicMock()
        disconnected_client.is_connected = False

        with pytest.raises(RuntimeError, match="not connected"):
            await ingest_document_as_episode(
                graphiti_client=disconnected_client,
                document=sample_document,
            )


class TestIngestionBackendRouting:
    """Tests for ingestion backend feature flag routing."""

    def test_ingestion_backend_default_is_graphiti(self):
        """Default ingestion backend should be graphiti."""
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test",
            "DATABASE_URL": "postgresql://test",
            "NEO4J_URI": "bolt://localhost",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost",
        }, clear=False):
            from agentic_rag_backend.config import load_settings
            # Clear cache
            load_settings.cache_clear() if hasattr(load_settings, 'cache_clear') else None
            settings = load_settings()
            assert settings.ingestion_backend == "graphiti"

    def test_ingestion_backend_can_be_legacy(self):
        """Ingestion backend should support legacy value."""
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test",
            "DATABASE_URL": "postgresql://test",
            "NEO4J_URI": "bolt://localhost",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost",
            "INGESTION_BACKEND": "legacy",
        }, clear=False):
            from agentic_rag_backend.config import load_settings
            load_settings.cache_clear() if hasattr(load_settings, 'cache_clear') else None
            settings = load_settings()
            assert settings.ingestion_backend == "legacy"


class TestEpisodeResult:
    """Tests for episode ingestion result model."""

    def test_episode_result_structure(self):
        """EpisodeIngestionResult should have required fields."""
        from agentic_rag_backend.indexing.graphiti_ingestion import EpisodeIngestionResult

        result = EpisodeIngestionResult(
            document_id="doc-123",
            tenant_id="tenant-456",
            episode_uuid="episode-789",
            entities_extracted=5,
            edges_created=3,
            processing_time_ms=1500,
        )

        assert result.document_id == "doc-123"
        assert result.tenant_id == "tenant-456"
        assert result.episode_uuid == "episode-789"
        assert result.entities_extracted == 5
        assert result.edges_created == 3
        assert result.processing_time_ms == 1500
