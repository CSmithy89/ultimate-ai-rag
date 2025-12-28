"""pytest fixtures for Agentic RAG Backend tests."""

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_tenant_id():
    """Provide a sample tenant ID."""
    return uuid4()


@pytest.fixture
def sample_job_id():
    """Provide a sample job ID."""
    return uuid4()


@pytest.fixture
def sample_document_id():
    """Provide a sample document ID."""
    return uuid4()


@pytest.fixture
def sample_crawl_request(sample_tenant_id):
    """Provide a sample crawl request."""
    return {
        "url": "https://docs.example.com",
        "tenant_id": str(sample_tenant_id),
        "max_depth": 3,
        "options": {
            "follow_links": True,
            "respect_robots_txt": True,
            "rate_limit": 1.0,
        },
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.xadd.return_value = b"1234567890-0"
    redis_mock.xreadgroup.return_value = []
    redis_mock.xack.return_value = True
    return redis_mock


@pytest.fixture
def mock_redis_client(mock_redis):
    """Mock RedisClient wrapper."""
    from agentic_rag_backend.db.redis import RedisClient

    client = MagicMock(spec=RedisClient)
    client._client = mock_redis
    client.client = mock_redis
    client.publish_job = AsyncMock(return_value="1234567890-0")
    client.consume_jobs = AsyncMock()
    client.ensure_consumer_group = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


@pytest.fixture
def mock_postgres_client(sample_job_id, sample_tenant_id):
    """Mock PostgresClient wrapper."""
    from agentic_rag_backend.db.postgres import PostgresClient
    from agentic_rag_backend.models.ingest import (
        JobProgress,
        JobStatus,
        JobStatusEnum,
        JobType,
    )
    from datetime import datetime, timezone

    client = MagicMock(spec=PostgresClient)
    client.create_job = AsyncMock(return_value=sample_job_id)
    client.get_job = AsyncMock(
        return_value=JobStatus(
            job_id=sample_job_id,
            tenant_id=sample_tenant_id,
            job_type=JobType.CRAWL,
            status=JobStatusEnum.QUEUED,
            progress=JobProgress(pages_crawled=0, pages_discovered=0, pages_failed=0),
            created_at=datetime.now(timezone.utc),
        )
    )
    client.update_job_status = AsyncMock(return_value=True)
    # list_jobs now returns a tuple of (jobs, total_count)
    client.list_jobs = AsyncMock(return_value=([], 0))
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.create_tables = AsyncMock()
    # Story 4.3 chunk methods
    client.create_chunk = AsyncMock(return_value=uuid4())
    client.get_chunk = AsyncMock(return_value=None)
    client.get_chunks_by_document = AsyncMock(return_value=[])
    client.search_similar_chunks = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4jClient wrapper."""
    from agentic_rag_backend.db.neo4j import Neo4jClient

    client = MagicMock(spec=Neo4jClient)
    client.find_similar_entity = AsyncMock(return_value=None)
    client.create_entity = AsyncMock(return_value={"id": "test-id"})
    client.create_relationship = AsyncMock(return_value=True)
    client.create_document_node = AsyncMock(return_value={})
    client.create_chunk_node = AsyncMock(return_value={})
    client.link_chunk_to_entity = AsyncMock(return_value=True)
    client.get_graph_stats = AsyncMock(return_value={
        "entity_count": 0,
        "document_count": 0,
        "chunk_count": 0,
        "relationship_count": 0,
    })
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.create_indexes = AsyncMock()
    return client


@pytest.fixture
def mock_crawler():
    """Mock Crawl4AI-style crawler for testing."""
    crawler_mock = AsyncMock()
    result = MagicMock()
    result.url = "https://example.com"
    result.markdown = "# Test Content\n\nThis is test content."
    result.title = "Test Page"
    crawler_mock.arun.return_value = result
    return crawler_mock


@pytest.fixture
def client(mock_redis_client, mock_postgres_client, monkeypatch):
    """Create FastAPI test client with mocked dependencies."""
    from agentic_rag_backend.main import app
    from agentic_rag_backend.api.routes.ingest import limiter

    # Disable rate limiting for tests
    limiter.enabled = False

    # Mock the dependency injection functions
    async def mock_get_redis():
        return mock_redis_client

    async def mock_get_postgres():
        return mock_postgres_client

    # Override dependencies
    from agentic_rag_backend.api.routes.ingest import get_redis, get_postgres

    app.dependency_overrides[get_redis] = mock_get_redis
    app.dependency_overrides[get_postgres] = mock_get_postgres

    # Mock environment variables for settings
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    with TestClient(app) as test_client:
        yield test_client

    # Re-enable rate limiting after tests
    limiter.enabled = True
    app.dependency_overrides.clear()
