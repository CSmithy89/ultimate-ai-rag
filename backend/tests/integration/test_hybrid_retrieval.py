"""Integration tests for hybrid retrieval (vector + graph)."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

import pytest

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.retrieval.graph_traversal import GraphTraversalService
from agentic_rag_backend.retrieval.hybrid_synthesis import build_hybrid_prompt
from agentic_rag_backend.retrieval.types import VectorHit

pytestmark = pytest.mark.integration


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _embedding(seed: float = 1.0) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = seed
    return vector


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_vector_and_graph_evidence(
    postgres_client: PostgresClient,
    neo4j_client: Neo4jClient,
    integration_cleanup: str,
) -> None:
    tenant_id = integration_cleanup
    tenant_uuid = UUID(tenant_id)

    content = "FastAPI uses Starlette for async web handling."
    doc_id = await postgres_client.create_document(
        tenant_id=tenant_uuid,
        source_type="text",
        content_hash=_content_hash(content),
    )
    embedding = _embedding(0.9)
    chunk_id = await postgres_client.create_chunk(
        tenant_id=tenant_uuid,
        document_id=doc_id,
        content=content,
        chunk_index=0,
        token_count=8,
        embedding=embedding,
    )

    entity_fastapi = str(uuid4())
    entity_starlette = str(uuid4())
    await neo4j_client.create_entity(
        entity_id=entity_fastapi,
        tenant_id=tenant_id,
        name="FastAPI",
        entity_type="Technology",
    )
    await neo4j_client.create_entity(
        entity_id=entity_starlette,
        tenant_id=tenant_id,
        name="Starlette",
        entity_type="Technology",
    )
    await neo4j_client.create_relationship(
        source_id=entity_fastapi,
        target_id=entity_starlette,
        relationship_type="USES",
        tenant_id=tenant_id,
        confidence=0.9,
    )

    rows = await postgres_client.search_similar_chunks(
        tenant_id=tenant_uuid,
        embedding=embedding,
        similarity_threshold=0.1,
        limit=5,
    )
    assert rows

    vector_hits = [
        VectorHit(
            chunk_id=str(row["id"]),
            document_id=str(row["document_id"]),
            content=row["content"],
            similarity=float(row["similarity"]),
            metadata=row.get("metadata"),
        )
        for row in rows
    ]

    traversal = GraphTraversalService(neo4j_client, max_hops=2, path_limit=5)
    graph_result = await traversal.traverse("FastAPI uses Starlette", tenant_id)

    assert graph_result.nodes
    assert graph_result.paths

    prompt = build_hybrid_prompt("FastAPI stack", vector_hits, graph_result)
    assert f"[vector:{chunk_id}]" in prompt
    assert "[graph:" in prompt


@pytest.mark.asyncio
async def test_hybrid_retrieval_is_tenant_scoped(
    postgres_client: PostgresClient,
    neo4j_client: Neo4jClient,
    integration_cleanup: str,
) -> None:
    tenant_a = integration_cleanup
    tenant_b = str(uuid4())

    tenant_a_uuid = UUID(tenant_a)
    tenant_b_uuid = UUID(tenant_b)

    content_a = "Tenant A document"
    content_b = "Tenant B document"

    try:
        doc_a = await postgres_client.create_document(
            tenant_id=tenant_a_uuid,
            source_type="text",
            content_hash=_content_hash(content_a),
        )
        doc_b = await postgres_client.create_document(
            tenant_id=tenant_b_uuid,
            source_type="text",
            content_hash=_content_hash(content_b),
        )

        embedding_a = _embedding(0.5)
        embedding_b = _embedding(0.7)

        await postgres_client.create_chunk(
            tenant_id=tenant_a_uuid,
            document_id=doc_a,
            content=content_a,
            chunk_index=0,
            token_count=3,
            embedding=embedding_a,
        )
        await postgres_client.create_chunk(
            tenant_id=tenant_b_uuid,
            document_id=doc_b,
            content=content_b,
            chunk_index=0,
            token_count=3,
            embedding=embedding_b,
        )

        rows_a = await postgres_client.search_similar_chunks(
            tenant_id=tenant_a_uuid,
            embedding=embedding_a,
            similarity_threshold=0.1,
            limit=5,
        )
        assert all(str(row["tenant_id"]) == tenant_a for row in rows_a)

        entity_a = str(uuid4())
        entity_b = str(uuid4())
        await neo4j_client.create_entity(
            entity_id=entity_a,
            tenant_id=tenant_a,
            name="TenantA",
            entity_type="Concept",
        )
        await neo4j_client.create_entity(
            entity_id=entity_b,
            tenant_id=tenant_b,
            name="TenantB",
            entity_type="Concept",
        )

        traversal = GraphTraversalService(neo4j_client, max_hops=1, path_limit=3)
        graph_a = await traversal.traverse("TenantA", tenant_a)

        assert all(node.id == entity_a for node in graph_a.nodes)
    finally:
        async with postgres_client.pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE tenant_id = $1", tenant_b_uuid)
            await conn.execute("DELETE FROM documents WHERE tenant_id = $1", tenant_b_uuid)
        async with neo4j_client.driver.session() as session:
            await session.run(
                "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n",
                tenant_id=tenant_b,
            )
