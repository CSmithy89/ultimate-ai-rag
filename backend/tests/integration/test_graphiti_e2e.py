"""Graphiti end-to-end integration tests (optional)."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from agentic_rag_backend.db.graphiti import GraphitiClient, GRAPHITI_AVAILABLE
from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode
from agentic_rag_backend.models.documents import UnifiedDocument, SourceType
from agentic_rag_backend.retrieval.temporal_retrieval import (
    get_knowledge_changes,
    temporal_search,
)

pytestmark = pytest.mark.integration

if os.getenv("GRAPHITI_E2E") != "1":
    pytest.skip("GRAPHITI_E2E=1 required for Graphiti E2E tests", allow_module_level=True)

if not GRAPHITI_AVAILABLE:
    pytest.skip("graphiti-core not installed", allow_module_level=True)

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY required for Graphiti E2E tests", allow_module_level=True)


@pytest.mark.asyncio
async def test_graphiti_episode_ingestion_and_temporal_queries(
    integration_env: dict[str, str],
    integration_cleanup: str,
) -> None:
    tenant_id = integration_cleanup
    tenant_uuid = UUID(tenant_id)

    graphiti_client = GraphitiClient(
        uri=integration_env["neo4j_uri"],
        user=integration_env["neo4j_user"],
        password=integration_env["neo4j_password"],
        llm_provider="openai",
        llm_api_key=os.getenv("OPENAI_API_KEY", ""),
        embedding_api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    await graphiti_client.connect()
    try:
        content = "Graphiti temporal ingestion test content."
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document = UnifiedDocument(
            id=uuid4(),
            tenant_id=tenant_uuid,
            source_type=SourceType.TEXT,
            content=content,
            content_hash=content_hash,
        )
        ingest_result = await ingest_document_as_episode(
            graphiti_client=graphiti_client,
            document=document,
        )
        assert ingest_result.episode_uuid

        as_of = datetime.now(timezone.utc)
        temporal_result = await temporal_search(
            graphiti_client=graphiti_client,
            query="Graphiti temporal",
            tenant_id=tenant_id,
            as_of_date=as_of,
        )
        assert temporal_result is not None
        assert temporal_result.as_of_date == as_of

        changes = await get_knowledge_changes(
            graphiti_client=graphiti_client,
            tenant_id=tenant_id,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        assert changes is not None
    finally:
        await graphiti_client.disconnect()
