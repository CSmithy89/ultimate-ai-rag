"""Graphiti episode-based document ingestion service.

Provides episode-based document ingestion using Graphiti's temporal
knowledge graph capabilities. Documents are ingested as episodes with
automatic entity extraction and temporal tracking.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

from ..core.errors import IngestionError
from ..db.graphiti import GraphitiClient
from ..models.documents import UnifiedDocument
from ..models.entity_types import (
    TechnicalConcept,
    CodePattern,
    APIEndpoint,
    ConfigurationOption,
)

logger = structlog.get_logger(__name__)

# Minimum content length for meaningful entity extraction.
# Documents shorter than this are too brief to contain extractable
# entities/relationships and would waste LLM tokens without value.
# 10 chars is approximately 2-3 words minimum.
MIN_CONTENT_LENGTH = 10


@dataclass
class EpisodeIngestionResult:
    """Result of episode-based document ingestion."""

    document_id: str
    tenant_id: str
    episode_uuid: str
    entities_extracted: int
    edges_created: int
    processing_time_ms: int
    source_description: Optional[str] = None


# Custom entity types for Graphiti episode ingestion
EPISODE_ENTITY_TYPES = [
    TechnicalConcept,
    CodePattern,
    APIEndpoint,
    ConfigurationOption,
]


async def ingest_document_as_episode(
    graphiti_client: GraphitiClient,
    document: UnifiedDocument,
) -> EpisodeIngestionResult:
    """
    Ingest a document as a Graphiti episode.

    This function:
    1. Validates the document content
    2. Creates an episode in Graphiti with the document content
    3. Uses custom entity types for classification
    4. Tracks temporal information for the episode

    Args:
        graphiti_client: Connected GraphitiClient instance
        document: UnifiedDocument to ingest

    Returns:
        EpisodeIngestionResult with ingestion details

    Raises:
        RuntimeError: If Graphiti client is not connected
        IngestionError: If document content is empty or ingestion fails
    """
    start_time = time.perf_counter()

    # Extract IDs as strings
    document_id = str(document.id)
    tenant_id = str(document.tenant_id)

    # Validate client connection
    if not graphiti_client.is_connected:
        raise RuntimeError("Graphiti client is not connected")

    # Validate document content
    content_stripped = document.content.strip() if document.content else ""
    if len(content_stripped) < MIN_CONTENT_LENGTH:
        raise IngestionError(
            document_id=document_id,
            reason=f"Document content too short (min {MIN_CONTENT_LENGTH} chars)",
        )

    logger.info(
        "graphiti_episode_ingestion_started",
        document_id=document_id,
        tenant_id=tenant_id,
        content_length=len(document.content),
    )

    try:
        # Build source description
        source_description = _build_source_description(document)

        # Get document title from metadata or generate one
        title = None
        if document.metadata and document.metadata.title:
            title = document.metadata.title
        if not title:
            title = f"Document {document_id[:8]}"

        # Ingest document as episode
        episode = await graphiti_client.client.add_episode(
            name=title,
            episode_body=document.content,
            source_description=source_description,
            reference_time=datetime.now(timezone.utc),
            entity_types=EPISODE_ENTITY_TYPES,  # type: ignore[arg-type]
            group_id=tenant_id,  # Multi-tenancy via group_id
        )

        # Calculate processing time
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract result metrics
        entities_extracted = len(getattr(episode, "entity_references", []))
        edges_created = len(getattr(episode, "edge_references", []))

        result = EpisodeIngestionResult(
            document_id=document_id,
            tenant_id=tenant_id,
            episode_uuid=str(getattr(episode, "uuid", "")),
            entities_extracted=entities_extracted,
            edges_created=edges_created,
            processing_time_ms=processing_time_ms,
            source_description=source_description,
        )

        logger.info(
            "graphiti_episode_ingestion_completed",
            document_id=document_id,
            episode_uuid=result.episode_uuid,
            entities_extracted=entities_extracted,
            edges_created=edges_created,
            processing_time_ms=processing_time_ms,
        )

        return result

    except Exception as e:
        logger.error(
            "graphiti_episode_ingestion_failed",
            document_id=document_id,
            error=str(e),
        )
        raise IngestionError(
            document_id=document_id,
            reason=str(e),
        ) from e


def _build_source_description(document: UnifiedDocument) -> str:
    """Build a source description for the episode."""
    parts = []

    if document.source_type:
        parts.append(f"Source type: {document.source_type.value}")

    if document.source_url:
        parts.append(f"URL: {document.source_url}")

    if document.filename:
        parts.append(f"File: {document.filename}")

    if document.metadata:
        if hasattr(document.metadata, "extra") and document.metadata.extra:
            if "section" in document.metadata.extra:
                parts.append(f"Section: {document.metadata.extra['section']}")

    return " | ".join(parts) if parts else f"Document ID: {str(document.id)}"


async def ingest_with_backend_routing(
    document: UnifiedDocument,
    graphiti_client: Optional[GraphitiClient],
    legacy_indexer,  # IndexerAgent type
    ingestion_backend: str,
) -> EpisodeIngestionResult:
    """
    Route document ingestion to appropriate backend based on feature flag.

    Args:
        document: Document to ingest
        graphiti_client: GraphitiClient for Graphiti backend
        legacy_indexer: IndexerAgent for legacy backend
        ingestion_backend: "graphiti" or "legacy"

    Returns:
        EpisodeIngestionResult (or equivalent for legacy)

    Raises:
        ValueError: If invalid backend specified
        RuntimeError: If required client not available
    """
    if ingestion_backend == "graphiti":
        if graphiti_client is None or not graphiti_client.is_connected:
            raise RuntimeError(
                "Graphiti client not available but graphiti backend selected"
            )
        return await ingest_document_as_episode(
            graphiti_client=graphiti_client,
            document=document,
        )

    elif ingestion_backend == "legacy":
        if legacy_indexer is None:
            raise RuntimeError(
                "Legacy indexer not available but legacy backend selected"
            )

        # Use legacy IndexerAgent
        result = await legacy_indexer.index_document(
            document_id=document.id,
            tenant_id=document.tenant_id,
            content=document.content,
            metadata=document.metadata.model_dump() if document.metadata else None,
        )

        # Convert to EpisodeIngestionResult for consistency
        return EpisodeIngestionResult(
            document_id=str(document.id),
            tenant_id=str(document.tenant_id),
            episode_uuid="",  # Legacy doesn't use episodes
            entities_extracted=result.entities_extracted,
            edges_created=result.relationships_extracted,
            processing_time_ms=result.processing_time_ms,
        )

    else:
        raise ValueError(f"Invalid ingestion backend: {ingestion_backend}")
