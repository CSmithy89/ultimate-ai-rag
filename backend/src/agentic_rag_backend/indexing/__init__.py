"""Indexing components for the Knowledge Ingestion Pipeline.

Epic 5 Migration Notes:
----------------------
The following modules are DEPRECATED and will be removed in a future version:
- embeddings.py - Use graphiti_ingestion.ingest_document_as_episode() instead
- entity_extractor.py - Use graphiti_ingestion.ingest_document_as_episode() instead
- graph_builder.py - Use graphiti_ingestion.ingest_document_as_episode() instead

For new code, use the Graphiti-based functions:
- ingest_document_as_episode() - Primary document ingestion
- ingest_with_backend_routing() - With feature flag support
- EpisodeIngestionResult - Ingestion result dataclass
"""

from .crawler import CrawlerService, crawl_url
from .chunker import chunk_document, chunk_sections, count_tokens, estimate_chunks, ChunkData

# Epic 5 - Graphiti-based ingestion (RECOMMENDED)
from .graphiti_ingestion import (
    ingest_document_as_episode,
    ingest_with_backend_routing,
    EpisodeIngestionResult,
    EPISODE_ENTITY_TYPES,
)

# Legacy modules (DEPRECATED - see migration notes above)
# These imports will trigger deprecation warnings
from .embeddings import EmbeddingGenerator, cosine_similarity, get_embedding_generator
from .entity_extractor import EntityExtractor, get_entity_extractor
from .graph_builder import GraphBuilder, create_graph_from_extractions

__all__ = [
    # Crawler (active)
    "CrawlerService",
    "crawl_url",
    # Chunker (active)
    "chunk_document",
    "chunk_sections",
    "count_tokens",
    "estimate_chunks",
    "ChunkData",
    # Graphiti ingestion (RECOMMENDED - Epic 5)
    "ingest_document_as_episode",
    "ingest_with_backend_routing",
    "EpisodeIngestionResult",
    "EPISODE_ENTITY_TYPES",
    # Embeddings (DEPRECATED)
    "EmbeddingGenerator",
    "cosine_similarity",
    "get_embedding_generator",
    # Entity extraction (DEPRECATED)
    "EntityExtractor",
    "get_entity_extractor",
    # Graph building (DEPRECATED)
    "GraphBuilder",
    "create_graph_from_extractions",
]
