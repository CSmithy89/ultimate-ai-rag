"""Indexing components for the Knowledge Ingestion Pipeline."""

from .crawler import CrawlerService, crawl_url
from .chunker import chunk_document, chunk_sections, count_tokens, estimate_chunks, ChunkData
from .graphiti_ingestion import (
    ingest_document_as_episode,
    EpisodeIngestionResult,
    EPISODE_ENTITY_TYPES,
)
from .contextual import (
    ContextualChunkEnricher,
    DocumentContext,
    EnrichedChunk,
    create_contextual_enricher,
)

__all__ = [
    # Crawler
    "CrawlerService",
    "crawl_url",
    # Chunker
    "chunk_document",
    "chunk_sections",
    "count_tokens",
    "estimate_chunks",
    "ChunkData",
    # Graphiti ingestion
    "ingest_document_as_episode",
    "EpisodeIngestionResult",
    "EPISODE_ENTITY_TYPES",
    # Epic 12: Contextual Retrieval
    "ContextualChunkEnricher",
    "DocumentContext",
    "EnrichedChunk",
    "create_contextual_enricher",
]
