"""Tests for vector search retrieval service."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agentic_rag_backend.retrieval.types import VectorHit
from agentic_rag_backend.retrieval.vector_search import VectorSearchService


@pytest.mark.asyncio
async def test_vector_search_returns_hits() -> None:
    postgres = MagicMock()
    postgres.search_similar_chunks = AsyncMock(
        return_value=[
            {
                "id": uuid4(),
                "document_id": uuid4(),
                "content": "Relevant chunk",
                "similarity": 0.91,
                "metadata": {"source_url": "https://docs.example.com"},
            }
        ]
    )
    embedding_generator = MagicMock()
    embedding_generator.generate_embedding = AsyncMock(return_value=[0.1] * 1536)

    service = VectorSearchService(
        postgres=postgres,
        embedding_generator=embedding_generator,
        limit=5,
        similarity_threshold=0.6,
    )

    hits = await service.search("How does vector search work?", str(uuid4()))

    assert len(hits) == 1
    assert isinstance(hits[0], VectorHit)
    assert hits[0].similarity == 0.91


@pytest.mark.asyncio
async def test_vector_search_invalid_tenant_returns_empty() -> None:
    postgres = MagicMock()
    embedding_generator = MagicMock()
    service = VectorSearchService(postgres=postgres, embedding_generator=embedding_generator)

    hits = await service.search("Query", "not-a-uuid")

    assert hits == []


@pytest.mark.asyncio
async def test_vector_search_missing_dependencies_returns_empty() -> None:
    service = VectorSearchService(postgres=None, embedding_generator=None)

    hits = await service.search("Query", str(uuid4()))

    assert hits == []
