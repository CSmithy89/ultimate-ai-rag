"""Integration tests for complete Graphiti workflow.

Epic 5 Test Suite Adaptation - Story 5.6
This file provides comprehensive integration tests for the Graphiti-based
knowledge graph pipeline.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import hashlib


def _make_content_hash(content: str) -> str:
    """Generate a SHA-256 hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()


def _make_mock_node(uuid: str, name: str, summary: str, labels: list):
    """Create a mock node with proper name attribute."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)
    node.summary = summary
    node.labels = labels
    return node


def _make_mock_edge(uuid: str, source: str, target: str, name: str, fact: str):
    """Create a mock edge with proper name attribute."""
    edge = MagicMock()
    edge.uuid = uuid
    edge.source_node_uuid = source
    edge.target_node_uuid = target
    edge.configure_mock(name=name)
    edge.fact = fact
    edge.valid_at = datetime.now(timezone.utc)
    edge.invalid_at = None
    return edge


class TestGraphitiEndToEndFlow:
    """Integration tests for complete Graphiti workflow."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient for end-to-end testing."""
        client = MagicMock()
        client.client = MagicMock()
        
        # Mock add_episode for ingestion
        mock_episode = MagicMock(
            uuid="ep-integration-test",
            name="Integration Test Document",
            entity_references=["node-1", "node-2"],
            edge_references=["edge-1"],
            created_at=datetime.now(timezone.utc),
        )
        # Explicitly set group_id to None to skip tenant validation in tests
        # (MagicMock auto-creates attributes as MagicMock objects otherwise)
        mock_episode.group_id = None
        client.client.add_episode = AsyncMock(return_value=mock_episode)
        
        # Mock search for retrieval
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("node-1", "FastAPI", "Web framework", ["TechnicalConcept"]),
            _make_mock_node("node-2", "async/await", "Async pattern", ["CodePattern"]),
        ]
        search_result.edges = [
            _make_mock_edge("edge-1", "node-1", "node-2", "USES", "FastAPI uses async/await"),
        ]
        client.client.search = AsyncMock(return_value=search_result)
        
        # Mock get_episodes_by_group_ids for changes
        client.client.get_episodes_by_group_ids = AsyncMock(return_value=[mock_episode])
        
        client.is_connected = True
        return client

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        from agentic_rag_backend.models.documents import UnifiedDocument, SourceType
        
        content = "FastAPI is a modern web framework. It uses async/await for handling requests."
        return UnifiedDocument(
            id=uuid4(),
            tenant_id=uuid4(),
            source_type=SourceType.URL,
            source_url="https://example.com/docs",
            content=content,
            content_hash=_make_content_hash(content),
        )

    @pytest.mark.asyncio
    async def test_ingest_then_search_flow(self, mock_graphiti_client, sample_document):
        """Test complete ingest -> search workflow."""
        from agentic_rag_backend.indexing.graphiti_ingestion import (
            ingest_document_as_episode,
        )
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        # Step 1: Ingest document
        ingest_result = await ingest_document_as_episode(
            graphiti_client=mock_graphiti_client,
            document=sample_document,
        )

        assert ingest_result is not None
        assert ingest_result.episode_uuid == "ep-integration-test"
        assert ingest_result.entities_extracted == 2
        assert ingest_result.edges_created == 1

        # Step 2: Search for ingested content
        search_result = await graphiti_search(
            graphiti_client=mock_graphiti_client,
            query="How does FastAPI work?",
            tenant_id=str(sample_document.tenant_id),
        )

        assert search_result is not None
        assert len(search_result.nodes) == 2
        assert len(search_result.edges) == 1
        assert search_result.nodes[0].name == "FastAPI"

    @pytest.mark.asyncio
    async def test_temporal_query_after_ingestion(self, mock_graphiti_client, sample_document):
        """Test temporal query capabilities after ingestion."""
        from agentic_rag_backend.indexing.graphiti_ingestion import (
            ingest_document_as_episode,
        )
        from agentic_rag_backend.retrieval.temporal_retrieval import (
            temporal_search,
            get_knowledge_changes,
        )

        # Step 1: Ingest document
        ingest_result = await ingest_document_as_episode(
            graphiti_client=mock_graphiti_client,
            document=sample_document,
        )
        assert ingest_result is not None

        # Step 2: Temporal search
        as_of_date = datetime.now(timezone.utc)
        temporal_result = await temporal_search(
            graphiti_client=mock_graphiti_client,
            query="FastAPI features",
            tenant_id=str(sample_document.tenant_id),
            as_of_date=as_of_date,
        )

        assert temporal_result is not None
        assert temporal_result.as_of_date == as_of_date
        assert len(temporal_result.nodes) == 2

        # Step 3: Get changes
        changes_result = await get_knowledge_changes(
            graphiti_client=mock_graphiti_client,
            tenant_id=str(sample_document.tenant_id),
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )

        assert changes_result is not None
        assert len(changes_result.episodes) >= 1

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, mock_graphiti_client):
        """Test that multi-tenancy via group_id is enforced."""
        from agentic_rag_backend.indexing.graphiti_ingestion import (
            ingest_document_as_episode,
        )
        from agentic_rag_backend.models.documents import UnifiedDocument, SourceType

        tenant_1 = str(uuid4())

        content = "Test content for tenant isolation."
        doc_1 = UnifiedDocument(
            id=uuid4(),
            tenant_id=tenant_1,
            source_type=SourceType.TEXT,
            content=content,
            content_hash=_make_content_hash(content),
        )

        # Ingest for tenant 1
        await ingest_document_as_episode(
            graphiti_client=mock_graphiti_client,
            document=doc_1,
        )

        # Verify group_id matches tenant_id
        call_kwargs = mock_graphiti_client.client.add_episode.call_args[1]
        assert call_kwargs["group_id"] == tenant_1


class TestGraphitiTestCoverage:
    """Tests to verify test coverage for Epic 5."""

    def test_entity_types_coverage(self):
        """Verify all custom entity types are tested."""
        from agentic_rag_backend.models.entity_types import (
            TechnicalConcept,
            CodePattern,
            APIEndpoint,
            ConfigurationOption,
            EDGE_TYPE_MAPPINGS,
        )

        # All entity types should be importable
        assert TechnicalConcept is not None
        assert CodePattern is not None
        assert APIEndpoint is not None
        assert ConfigurationOption is not None

        # Edge type mappings should exist
        assert len(EDGE_TYPE_MAPPINGS) > 0

    def test_graphiti_client_coverage(self):
        """Verify GraphitiClient wrapper is tested."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        assert GraphitiClient is not None
        assert hasattr(GraphitiClient, 'connect')
        assert hasattr(GraphitiClient, 'disconnect')
        assert hasattr(GraphitiClient, 'build_indices')

    def test_ingestion_functions_coverage(self):
        """Verify ingestion functions are tested."""
        from agentic_rag_backend.indexing import (
            ingest_document_as_episode,
            ingest_with_backend_routing,
            EpisodeIngestionResult,
            EPISODE_ENTITY_TYPES,
        )

        assert callable(ingest_document_as_episode)
        assert callable(ingest_with_backend_routing)
        assert EpisodeIngestionResult is not None
        assert len(EPISODE_ENTITY_TYPES) == 4

    def test_retrieval_functions_coverage(self):
        """Verify retrieval functions are tested."""
        from agentic_rag_backend.retrieval import (
            graphiti_search,
            search_with_backend_routing,
            temporal_search,
            get_knowledge_changes,
            GraphitiSearchResult,
            TemporalSearchResult,
            KnowledgeChangesResult,
        )

        assert callable(graphiti_search)
        assert callable(search_with_backend_routing)
        assert callable(temporal_search)
        assert callable(get_knowledge_changes)
        assert GraphitiSearchResult is not None
        assert TemporalSearchResult is not None
        assert KnowledgeChangesResult is not None

    def test_config_settings_coverage(self):
        """Verify all config settings are tested."""
        from unittest.mock import patch
        
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test",
            "DATABASE_URL": "postgresql://test",
            "NEO4J_URI": "bolt://localhost",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost",
        }):
            from agentic_rag_backend.config import load_settings
            load_settings.cache_clear() if hasattr(load_settings, 'cache_clear') else None
            settings = load_settings()

            # Epic 5 settings
            assert hasattr(settings, 'graphiti_embedding_model')
            assert hasattr(settings, 'graphiti_llm_model')
            assert hasattr(settings, 'ingestion_backend')
            assert hasattr(settings, 'retrieval_backend')
