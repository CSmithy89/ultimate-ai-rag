"""Indexing components for the Knowledge Ingestion Pipeline.

Epic 5 Migration Notes:
----------------------
The following modules are DEPRECATED as of v1.0.0 and will be removed in v2.0.0:
- embeddings.py - Use graphiti_ingestion.ingest_document_as_episode() instead
- entity_extractor.py - Use graphiti_ingestion.ingest_document_as_episode() instead
- graph_builder.py - Use graphiti_ingestion.ingest_document_as_episode() instead

For new code, use the Graphiti-based functions:
- ingest_document_as_episode() - Primary document ingestion
- ingest_with_backend_routing() - With feature flag support
- EpisodeIngestionResult - Ingestion result dataclass
"""

import warnings

from .crawler import CrawlerService, crawl_url
from .chunker import chunk_document, chunk_sections, count_tokens, estimate_chunks, ChunkData

# Epic 5 - Graphiti-based ingestion (RECOMMENDED)
from .graphiti_ingestion import (
    ingest_document_as_episode,
    ingest_with_backend_routing,
    EpisodeIngestionResult,
    EPISODE_ENTITY_TYPES,
)


def __getattr__(name: str):
    """Lazy import deprecated modules with warnings."""
    deprecated_modules = {
        "EmbeddingGenerator": ("embeddings", "EmbeddingGenerator"),
        "cosine_similarity": ("embeddings", "cosine_similarity"),
        "get_embedding_generator": ("embeddings", "get_embedding_generator"),
        "EntityExtractor": ("entity_extractor", "EntityExtractor"),
        "get_entity_extractor": ("entity_extractor", "get_entity_extractor"),
        "GraphBuilder": ("graph_builder", "GraphBuilder"),
        "create_graph_from_extractions": ("graph_builder", "create_graph_from_extractions"),
    }
    
    if name in deprecated_modules:
        module_name, attr_name = deprecated_modules[name]
        warnings.warn(
            f"{name} is deprecated since v1.0.0 and will be removed in v2.0.0. "
            f"Use graphiti_ingestion.ingest_document_as_episode() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(f".{module_name}", globals(), locals(), [attr_name], 1)
        return getattr(module, attr_name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Embeddings (DEPRECATED - removed in v2.0.0)
    "EmbeddingGenerator",
    "cosine_similarity",
    "get_embedding_generator",
    # Entity extraction (DEPRECATED - removed in v2.0.0)
    "EntityExtractor",
    "get_entity_extractor",
    # Graph building (DEPRECATED - removed in v2.0.0)
    "GraphBuilder",
    "create_graph_from_extractions",
]
