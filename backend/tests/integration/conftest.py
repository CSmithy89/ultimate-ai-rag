"""Integration test fixtures for real services."""

from __future__ import annotations

import os
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.db.redis import RedisClient

REQUIRED_ENV = [
    "DATABASE_URL",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "REDIS_URL",
]


def _require_integration_env() -> None:
    if os.getenv("INTEGRATION_TESTS") != "1":
        pytest.skip("INTEGRATION_TESTS=1 required for integration tests")
    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing env for integration tests: {', '.join(missing)}")


async def _check_postgres(client: PostgresClient) -> None:
    await client.connect()
    async with client.pool.acquire() as conn:
        await conn.execute("SELECT 1")
    await client.create_tables()


async def _check_neo4j(client: Neo4jClient) -> None:
    await client.connect()
    async with client.driver.session() as session:
        result = await session.run("RETURN 1 AS ok")
        await result.single()


async def _check_redis(client: RedisClient) -> None:
    await client.connect()
    await client.client.ping()


async def _cleanup_postgres(client: PostgresClient, tenant_id: str) -> None:
    tenant_uuid = UUID(tenant_id)
    statements = [
        "DELETE FROM chunks WHERE tenant_id = $1",
        "DELETE FROM ingestion_jobs WHERE tenant_id = $1",
        "DELETE FROM documents WHERE tenant_id = $1",
        "DELETE FROM llm_usage_events WHERE tenant_id = $1",
        "DELETE FROM llm_cost_alerts WHERE tenant_id = $1",
    ]
    async with client.pool.acquire() as conn:
        for statement in statements:
            try:
                await conn.execute(statement, tenant_uuid)
            except asyncpg.PostgresError:
                # Ignore if table is missing in the current schema.
                continue


async def _cleanup_neo4j(client: Neo4jClient, tenant_id: str) -> None:
    async with client.driver.session() as session:
        await session.run(
            "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n",
            tenant_id=tenant_id,
        )


async def _cleanup_redis(client: RedisClient, tenant_id: str) -> None:
    pattern = f"*{tenant_id}*"
    async for key in client.client.scan_iter(match=pattern):
        await client.client.delete(key)


@pytest_asyncio.fixture(scope="session")
async def integration_env() -> dict[str, str]:
    _require_integration_env()
    database_url = os.getenv("DATABASE_URL")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    redis_url = os.getenv("REDIS_URL")

    postgres_client = PostgresClient(database_url)  # type: ignore[arg-type]
    neo4j_client = Neo4jClient(
        neo4j_uri,  # type: ignore[arg-type]
        neo4j_user,  # type: ignore[arg-type]
        neo4j_password,  # type: ignore[arg-type]
    )
    redis_client = RedisClient(redis_url)  # type: ignore[arg-type]

    try:
        await _check_postgres(postgres_client)
        await _check_neo4j(neo4j_client)
        await _check_redis(redis_client)
    except Exception as exc:
        pytest.skip(f"Integration services unavailable: {exc}")
    finally:
        await postgres_client.disconnect()
        await neo4j_client.disconnect()
        await redis_client.disconnect()

    return {
        "database_url": database_url or "",
        "neo4j_uri": neo4j_uri or "",
        "neo4j_user": neo4j_user or "",
        "neo4j_password": neo4j_password or "",
        "redis_url": redis_url or "",
    }


@pytest_asyncio.fixture
async def postgres_client(integration_env: dict[str, str]) -> PostgresClient:
    client = PostgresClient(integration_env["database_url"])
    await client.connect()
    await client.create_tables()
    try:
        yield client
    finally:
        await client.disconnect()


@pytest_asyncio.fixture
async def neo4j_client(integration_env: dict[str, str]) -> Neo4jClient:
    client = Neo4jClient(
        integration_env["neo4j_uri"],
        integration_env["neo4j_user"],
        integration_env["neo4j_password"],
    )
    await client.connect()
    await client.create_indexes()
    try:
        yield client
    finally:
        await client.disconnect()


@pytest_asyncio.fixture
async def redis_client(integration_env: dict[str, str]) -> RedisClient:
    client = RedisClient(integration_env["redis_url"])
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


@pytest.fixture
def integration_tenant_id() -> str:
    return str(uuid4())


@pytest_asyncio.fixture
async def integration_cleanup(
    integration_tenant_id: str,
    postgres_client: PostgresClient,
    neo4j_client: Neo4jClient,
    redis_client: RedisClient,
) -> str:
    try:
        yield integration_tenant_id
    finally:
        await _cleanup_postgres(postgres_client, integration_tenant_id)
        await _cleanup_neo4j(neo4j_client, integration_tenant_id)
        await _cleanup_redis(redis_client, integration_tenant_id)
