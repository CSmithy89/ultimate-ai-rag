"""Graph builder for constructing knowledge graphs in Neo4j.

This module provides utilities for building and managing the knowledge graph
from extracted entities and relationships.
"""

import warnings
from typing import Any, Optional
from uuid import uuid4

import structlog

from agentic_rag_backend.core.errors import GraphBuildError
from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.models.graphs import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphBuildResult,
)

warnings.warn(
    "The graph_builder module is deprecated since v1.0.0 and will be removed in v2.0.0. "
    "Use graphiti_ingestion.ingest_document_as_episode() which handles graph building automatically via Graphiti.",
    DeprecationWarning,
    stacklevel=2,
)

logger = structlog.get_logger(__name__)


class GraphBuilder:
    """
    Builds and manages knowledge graphs in Neo4j.

    Handles entity deduplication, relationship creation, and graph maintenance.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        similarity_threshold: float = 0.95,
    ) -> None:
        """
        Initialize GraphBuilder.

        Args:
            neo4j: Neo4j client for graph operations
            similarity_threshold: Threshold for entity similarity matching
        """
        self.neo4j = neo4j
        self.similarity_threshold = similarity_threshold
        self._entity_cache: dict[str, str] = {}  # normalized_name:type -> entity_id

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        return " ".join(name.lower().strip().split())

    async def find_or_create_entity(
        self,
        entity: ExtractedEntity,
        tenant_id: str,
        source_chunk_id: Optional[str] = None,
    ) -> tuple[str, bool]:
        """
        Find an existing entity or create a new one.

        Uses name normalization and type matching for deduplication.

        Args:
            entity: Entity to find or create
            tenant_id: Tenant identifier
            source_chunk_id: Optional chunk ID where entity was found

        Returns:
            Tuple of (entity_id, is_new)
        """
        normalized = self._normalize_name(entity.name)
        cache_key = f"{normalized}:{entity.type}"

        # Check local cache
        if cache_key in self._entity_cache:
            entity_id = self._entity_cache[cache_key]
            # Update with new chunk reference
            if source_chunk_id:
                await self.neo4j.create_entity(
                    entity_id=entity_id,
                    tenant_id=tenant_id,
                    name=entity.name,
                    entity_type=entity.type,
                    source_chunk_id=source_chunk_id,
                )
            return entity_id, False

        # Check Neo4j
        existing = await self.neo4j.find_similar_entity(
            tenant_id=tenant_id,
            name=entity.name,
            entity_type=entity.type,
        )

        if existing:
            entity_id = existing["id"]
            self._entity_cache[cache_key] = entity_id
            # Update with new chunk reference
            if source_chunk_id:
                await self.neo4j.create_entity(
                    entity_id=entity_id,
                    tenant_id=tenant_id,
                    name=entity.name,
                    entity_type=entity.type,
                    source_chunk_id=source_chunk_id,
                )
            return entity_id, False

        # Create new entity
        entity_id = str(uuid4())
        await self.neo4j.create_entity(
            entity_id=entity_id,
            tenant_id=tenant_id,
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_chunk_id=source_chunk_id,
        )
        self._entity_cache[cache_key] = entity_id
        return entity_id, True

    async def create_relationship(
        self,
        relationship: ExtractedRelationship,
        tenant_id: str,
        entity_name_to_id: dict[str, str],
        source_chunk_id: Optional[str] = None,
    ) -> bool:
        """
        Create a relationship between two entities.

        Args:
            relationship: Relationship to create
            tenant_id: Tenant identifier
            entity_name_to_id: Mapping of normalized entity names to IDs
            source_chunk_id: Optional source chunk ID

        Returns:
            True if relationship was created
        """
        source_norm = self._normalize_name(relationship.source)
        target_norm = self._normalize_name(relationship.target)

        source_id = entity_name_to_id.get(source_norm)
        target_id = entity_name_to_id.get(target_norm)

        if not source_id or not target_id:
            logger.debug(
                "relationship_skipped",
                source=relationship.source,
                target=relationship.target,
                reason="missing entity",
            )
            return False

        return await self.neo4j.create_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship.type,
            tenant_id=tenant_id,
            confidence=relationship.confidence,
            chunk_id=source_chunk_id,
        )

    async def build_graph(
        self,
        entities: list[ExtractedEntity],
        relationships: list[ExtractedRelationship],
        tenant_id: str,
        source_chunk_id: Optional[str] = None,
    ) -> GraphBuildResult:
        """
        Build a graph from extracted entities and relationships.

        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            tenant_id: Tenant identifier
            source_chunk_id: Optional chunk ID for provenance

        Returns:
            GraphBuildResult with creation counts
        """
        try:
            entities_created = 0
            entities_deduplicated = 0
            entity_name_to_id: dict[str, str] = {}

            # Process entities
            for entity in entities:
                normalized = self._normalize_name(entity.name)

                # Skip if already processed
                if normalized in entity_name_to_id:
                    continue

                entity_id, is_new = await self.find_or_create_entity(
                    entity=entity,
                    tenant_id=tenant_id,
                    source_chunk_id=source_chunk_id,
                )

                entity_name_to_id[normalized] = entity_id

                if is_new:
                    entities_created += 1
                else:
                    entities_deduplicated += 1

            # Process relationships
            relationships_created = 0
            relationships_skipped = 0

            for rel in relationships:
                created = await self.create_relationship(
                    relationship=rel,
                    tenant_id=tenant_id,
                    entity_name_to_id=entity_name_to_id,
                    source_chunk_id=source_chunk_id,
                )
                if created:
                    relationships_created += 1
                else:
                    relationships_skipped += 1

            result = GraphBuildResult(
                entities_created=entities_created,
                entities_deduplicated=entities_deduplicated,
                relationships_created=relationships_created,
                relationships_skipped=relationships_skipped,
            )

            logger.info(
                "graph_built",
                tenant_id=tenant_id,
                chunk_id=source_chunk_id,
                **result.model_dump(),
            )

            return result

        except Exception as e:
            raise GraphBuildError("build_graph", str(e)) from e

    def clear_cache(self) -> None:
        """Clear the entity cache for a new document."""
        self._entity_cache.clear()


async def create_graph_from_extractions(
    neo4j: Neo4jClient,
    extractions: list[dict[str, Any]],
    tenant_id: str,
) -> GraphBuildResult:
    """
    Create a graph from a list of extraction results.

    Convenience function for batch graph building.

    Args:
        neo4j: Neo4j client
        extractions: List of extraction results with 'entities', 'relationships', 'chunk_id'
        tenant_id: Tenant identifier

    Returns:
        Combined GraphBuildResult
    """
    builder = GraphBuilder(neo4j)

    total_created = 0
    total_deduplicated = 0
    total_rel_created = 0
    total_rel_skipped = 0

    for extraction in extractions:
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])
        chunk_id = extraction.get("chunk_id")

        result = await builder.build_graph(
            entities=entities,
            relationships=relationships,
            tenant_id=tenant_id,
            source_chunk_id=chunk_id,
        )

        total_created += result.entities_created
        total_deduplicated += result.entities_deduplicated
        total_rel_created += result.relationships_created
        total_rel_skipped += result.relationships_skipped

    return GraphBuildResult(
        entities_created=total_created,
        entities_deduplicated=total_deduplicated,
        relationships_created=total_rel_created,
        relationships_skipped=total_rel_skipped,
    )
