"""Benchmarks for query latency and basic scalability (NFR3/NFR5)."""

from __future__ import annotations

import asyncio
import os
import time
from uuid import uuid4

import pytest

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.retrieval.graph_traversal import GraphTraversalService
from tests.benchmarks.utils import record_benchmark

if os.getenv("RUN_BENCHMARKS") != "1":
    pytest.skip("RUN_BENCHMARKS=1 required for benchmark tests", allow_module_level=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    pytest.skip("Neo4j env not configured for benchmarks", allow_module_level=True)


@pytest.mark.asyncio
async def test_query_latency_benchmark() -> None:
    tenant_id = str(uuid4())
    client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    await client.connect()
    await client.create_indexes()

    try:
        entity_a = str(uuid4())
        entity_b = str(uuid4())
        await client.create_entity(entity_a, tenant_id, "LatencyNodeA", "Concept")
        await client.create_entity(entity_b, tenant_id, "LatencyNodeB", "Concept")
        await client.create_relationship(entity_a, entity_b, "RELATED_TO", tenant_id, 0.9)

        traversal = GraphTraversalService(client, max_hops=1, path_limit=5)
        start = time.perf_counter()
        result = await traversal.traverse("LatencyNodeA", tenant_id)
        duration_ms = (time.perf_counter() - start) * 1000

        record_benchmark(
            name="nfr3_query_latency",
            duration_ms=duration_ms,
            metadata={"nodes": len(result.nodes), "paths": len(result.paths)},
        )

        assert duration_ms < 2000
    finally:
        async with client.driver.session() as session:
            await session.run(
                "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n",
                tenant_id=tenant_id,
            )
        await client.disconnect()


@pytest.mark.asyncio
async def test_scalability_benchmark() -> None:
    tenant_id = str(uuid4())
    client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    await client.connect()
    await client.create_indexes()

    try:
        entity_a = str(uuid4())
        entity_b = str(uuid4())
        await client.create_entity(entity_a, tenant_id, "ScaleNodeA", "Concept")
        await client.create_entity(entity_b, tenant_id, "ScaleNodeB", "Concept")
        await client.create_relationship(entity_a, entity_b, "RELATED_TO", tenant_id, 0.9)

        traversal = GraphTraversalService(client, max_hops=1, path_limit=5)
        start = time.perf_counter()
        await asyncio.gather(*[
            traversal.traverse("ScaleNodeA", tenant_id) for _ in range(10)
        ])
        duration_ms = (time.perf_counter() - start) * 1000

        record_benchmark(
            name="nfr5_scalability_smoke",
            duration_ms=duration_ms,
            metadata={"concurrency": 10},
        )

        assert duration_ms < 5000
    finally:
        async with client.driver.session() as session:
            await session.run(
                "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n",
                tenant_id=tenant_id,
            )
        await client.disconnect()
