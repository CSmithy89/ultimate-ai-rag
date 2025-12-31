#!/usr/bin/env python3
"""Migrate legacy graph data by re-ingesting documents into Graphiti.

This script reconstructs document content from stored chunks and ingests each
as a Graphiti episode. It supports tenant-scoped migrations, optional backups,
and validation checks for legacy vs Graphiti counts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.db.graphiti import (
    GRAPHITI_AVAILABLE,
    GraphitiClient,
    create_graphiti_client,
)
from agentic_rag_backend.db.neo4j import Neo4jClient, get_neo4j_client, close_neo4j_client
from agentic_rag_backend.db.postgres import (
    PostgresClient,
    get_postgres_client,
    close_postgres_client,
)
from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode
from agentic_rag_backend.models.documents import (
    SourceType,
    UnifiedDocument,
    parse_document_metadata,
)

logger = structlog.get_logger(__name__)

CHUNK_BATCH_SIZE = 100


async def _fetch_tenant_ids(postgres: PostgresClient) -> list[str]:
    async with postgres.pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT tenant_id FROM documents")
    return [str(row["tenant_id"]) for row in rows]


async def _fetch_documents(
    postgres: PostgresClient,
    tenant_id: str,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, tenant_id, source_type, source_url, filename, content_hash, metadata "
        "FROM documents WHERE tenant_id = $1 ORDER BY created_at"
    )
    params: list[Any] = [UUID(tenant_id)]
    if limit is not None:
        query = f"{query} LIMIT $2"
        params.append(limit)
    async with postgres.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [dict(row) for row in rows]


async def _fetch_chunks_for_documents(
    postgres: PostgresClient,
    tenant_id: str,
    document_ids: list[UUID],
) -> dict[UUID, list[str]]:
    if not document_ids:
        return {}
    async with postgres.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT document_id, content
            FROM chunks
            WHERE tenant_id = $1 AND document_id = ANY($2)
            ORDER BY document_id, chunk_index
            """,
            UUID(tenant_id),
            document_ids,
        )
    chunks_by_doc: dict[UUID, list[str]] = {}
    for row in rows:
        chunks_by_doc.setdefault(row["document_id"], []).append(row["content"])
    return chunks_by_doc


async def _export_legacy_graph(
    neo4j: Neo4jClient,
    tenant_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with neo4j.driver.session() as session:
        with output_path.open("w", encoding="utf-8") as handle:
            node_result = await session.run(
                "MATCH (e:Entity {tenant_id: $tenant_id}) RETURN e",
                tenant_id=tenant_id,
            )
            nodes = await node_result.data()
            for record in nodes:
                handle.write(
                    json.dumps({"type": "node", "node": dict(record["e"])}) + "\n"
                )

            rel_result = await session.run(
                """
                MATCH (a:Entity {tenant_id: $tenant_id})-[r]->(b:Entity {tenant_id: $tenant_id})
                RETURN a.id AS source_id, b.id AS target_id, type(r) AS rel_type, r AS rel_props
                """,
                tenant_id=tenant_id,
            )
            rels = await rel_result.data()
            for record in rels:
                handle.write(
                    json.dumps(
                        {
                            "type": "relationship",
                            "source_id": record.get("source_id"),
                            "target_id": record.get("target_id"),
                            "relationship_type": record.get("rel_type"),
                            "properties": dict(record.get("rel_props", {})),
                        }
                    )
                    + "\n"
                )


async def _count_legacy_entities(neo4j: Neo4jClient, tenant_id: str) -> int:
    async with neo4j.driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e) AS count",
            tenant_id=tenant_id,
        )
        record = await result.single()
        return int(record["count"]) if record else 0


async def _count_legacy_relationships(neo4j: Neo4jClient, tenant_id: str) -> int:
    async with neo4j.driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Entity {tenant_id: $tenant_id})-[r]->(b:Entity {tenant_id: $tenant_id})
            RETURN count(r) AS count
            """,
            tenant_id=tenant_id,
        )
        record = await result.single()
        return int(record["count"]) if record else 0


async def _count_graphiti_nodes(neo4j: Neo4jClient, tenant_id: str) -> int:
    async with neo4j.driver.session() as session:
        result = await session.run(
            "MATCH (n) WHERE n.group_id = $tenant_id RETURN count(n) AS count",
            tenant_id=tenant_id,
        )
        record = await result.single()
        return int(record["count"]) if record else 0


async def _count_graphiti_relationships(neo4j: Neo4jClient, tenant_id: str) -> int:
    async with neo4j.driver.session() as session:
        result = await session.run(
            "MATCH ()-[r]->() WHERE r.group_id = $tenant_id RETURN count(r) AS count",
            tenant_id=tenant_id,
        )
        record = await result.single()
        return int(record["count"]) if record else 0


async def migrate(
    tenant_id: Optional[str],
    limit: Optional[int],
    dry_run: bool,
    backup_path: Optional[Path],
    validate: bool,
) -> int:
    settings = get_settings()

    if not GRAPHITI_AVAILABLE:
        logger.error("graphiti_not_available", reason="graphiti-core not installed")
        return 1

    postgres = await get_postgres_client(settings.database_url)
    neo4j = await get_neo4j_client(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    graphiti_client: Optional[GraphitiClient] = None

    try:
        graphiti_client = await create_graphiti_client(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.graphiti_embedding_model,
            llm_model=settings.graphiti_llm_model,
        )
        tenant_ids = [tenant_id] if tenant_id else await _fetch_tenant_ids(postgres)
        if not tenant_ids:
            logger.warning("migration_no_tenants_found")
            return 0

        total_migrated = 0
        total_skipped = 0

        for tenant in tenant_ids:
            logger.info("migration_tenant_start", tenant_id=tenant)

            if backup_path:
                tenant_backup = backup_path / f"legacy-graph-{tenant}.jsonl"
                await _export_legacy_graph(neo4j, tenant, tenant_backup)
                logger.info(
                    "legacy_backup_written",
                    tenant_id=tenant,
                    path=str(tenant_backup),
                )

            documents = await _fetch_documents(postgres, tenant, limit)
            if not documents:
                logger.warning("migration_no_documents", tenant_id=tenant)
                continue

            tenant_migrated = 0
            tenant_skipped = 0
            for start in range(0, len(documents), CHUNK_BATCH_SIZE):
                batch = documents[start:start + CHUNK_BATCH_SIZE]
                document_ids = [doc["id"] for doc in batch]
                chunks_by_doc = await _fetch_chunks_for_documents(
                    postgres, tenant, document_ids
                )
                for doc in batch:
                    document_id = doc["id"]
                    chunk_texts = chunks_by_doc.get(document_id, [])
                    content = "\n\n".join(chunk_texts).strip()
                    if not content:
                        total_skipped += 1
                        tenant_skipped += 1
                        logger.warning(
                            "migration_skipped_empty",
                            tenant_id=tenant,
                            document_id=str(document_id),
                        )
                        continue

                    metadata_model = parse_document_metadata(
                        doc.get("metadata"),
                        extra_fields={
                            "legacy_document_id": str(document_id),
                            "migration_source": "legacy",
                        },
                        log=logger,
                        log_context={
                            "tenant_id": tenant,
                            "legacy_document_id": str(document_id),
                        },
                    )

                    try:
                        source_type_enum = SourceType(str(doc.get("source_type")))
                    except ValueError:
                        source_type_enum = SourceType.TEXT

                    unified_doc = UnifiedDocument(
                        id=document_id,
                        tenant_id=UUID(tenant),
                        source_type=source_type_enum,
                        source_url=doc.get("source_url"),
                        filename=doc.get("filename"),
                        content=content,
                        content_hash=doc.get("content_hash")
                        or hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        metadata=metadata_model,
                    )

                    if dry_run:
                        total_migrated += 1
                        tenant_migrated += 1
                        continue

                    await ingest_document_as_episode(
                        graphiti_client=graphiti_client,
                        document=unified_doc,
                    )
                    total_migrated += 1
                    tenant_migrated += 1

            logger.info(
                "migration_tenant_completed",
                tenant_id=tenant,
                migrated=tenant_migrated,
                skipped=tenant_skipped,
            )

            if validate:
                legacy_entities = await _count_legacy_entities(neo4j, tenant)
                legacy_relationships = await _count_legacy_relationships(neo4j, tenant)
                graphiti_nodes = await _count_graphiti_nodes(neo4j, tenant)
                graphiti_relationships = await _count_graphiti_relationships(neo4j, tenant)

                logger.info(
                    "migration_validation",
                    tenant_id=tenant,
                    legacy_entities=legacy_entities,
                    legacy_relationships=legacy_relationships,
                    graphiti_nodes=graphiti_nodes,
                    graphiti_relationships=graphiti_relationships,
                )

                validation_failed = False
                if legacy_entities != graphiti_nodes:
                    logger.error(
                        "migration_validation_failed",
                        tenant_id=tenant,
                        reason="entity_count_mismatch",
                        legacy_entities=legacy_entities,
                        graphiti_nodes=graphiti_nodes,
                    )
                    validation_failed = True
                if legacy_relationships != graphiti_relationships:
                    logger.error(
                        "migration_validation_failed",
                        tenant_id=tenant,
                        reason="relationship_count_mismatch",
                        legacy_relationships=legacy_relationships,
                        graphiti_relationships=graphiti_relationships,
                    )
                    validation_failed = True
                if validation_failed:
                    return 2

        logger.info(
            "migration_complete",
            migrated=total_migrated,
            skipped=total_skipped,
        )
        return 0

    finally:
        if graphiti_client is not None:
            await graphiti_client.disconnect()
        await close_postgres_client()
        await close_neo4j_client()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy graph data to Graphiti")
    parser.add_argument("--tenant-id", help="Limit migration to a single tenant")
    parser.add_argument("--limit", type=int, help="Limit number of documents per tenant")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without ingesting")
    parser.add_argument(
        "--backup-path",
        type=Path,
        help="Write legacy graph backup JSONL per tenant",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate entity/relationship counts after migration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(
        migrate(
            tenant_id=args.tenant_id,
            limit=args.limit,
            dry_run=args.dry_run,
            backup_path=args.backup_path,
            validate=args.validate,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
