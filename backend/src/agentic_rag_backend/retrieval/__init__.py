"""Retrieval services for knowledge graph queries."""

# Epic 3 - Hybrid Knowledge Retrieval
from .graph_traversal import GraphTraversalService
from .types import GraphEdge, GraphNode, GraphPath, GraphTraversalResult, VectorHit
from .vector_search import VectorSearchService

# Epic 5 - Graphiti Integration
from .graphiti_retrieval import (
    graphiti_search,
    GraphitiSearchResult,
    SearchNode,
    SearchEdge,
)
from .temporal_retrieval import (
    temporal_search,
    get_knowledge_changes,
    TemporalSearchResult,
    TemporalNode,
    TemporalEdge,
    KnowledgeChangesResult,
    EpisodeChange,
)

# Epic 12 - Advanced Retrieval (Reranking)
from .reranking import (
    RerankerClient,
    RerankerProviderAdapter,
    RerankerProviderType,
    RerankedHit,
    CohereRerankerClient,
    FlashRankRerankerClient,
    create_reranker_client,
    get_reranker_adapter,
)

# Epic 12 - Advanced Retrieval (CRAG Grader)
from .grader import (
    BaseGrader,
    HeuristicGrader,
    CrossEncoderGrader,
    RetrievalGrader,
    RetrievalHit,
    GraderResult,
    FallbackStrategy,
    BaseFallbackHandler,
    WebSearchFallback,
    ExpandedQueryFallback,
    create_grader,
    DEFAULT_CROSS_ENCODER_MODEL,
    SUPPORTED_GRADER_MODELS,
)

__all__ = [
    # Epic 3 - Graph traversal and vector search
    "GraphEdge",
    "GraphNode",
    "GraphPath",
    "GraphTraversalResult",
    "GraphTraversalService",
    "VectorHit",
    "VectorSearchService",
    # Epic 5 - Hybrid retrieval
    "graphiti_search",
    "GraphitiSearchResult",
    "SearchNode",
    "SearchEdge",
    # Epic 5 - Temporal retrieval
    "temporal_search",
    "get_knowledge_changes",
    "TemporalSearchResult",
    "TemporalNode",
    "TemporalEdge",
    "KnowledgeChangesResult",
    "EpisodeChange",
    # Epic 12 - Reranking
    "RerankerClient",
    "RerankerProviderAdapter",
    "RerankerProviderType",
    "RerankedHit",
    "CohereRerankerClient",
    "FlashRankRerankerClient",
    "create_reranker_client",
    "get_reranker_adapter",
    # Epic 12 - CRAG Grader
    "BaseGrader",
    "HeuristicGrader",
    "CrossEncoderGrader",
    "RetrievalGrader",
    "RetrievalHit",
    "GraderResult",
    "FallbackStrategy",
    "BaseFallbackHandler",
    "WebSearchFallback",
    "ExpandedQueryFallback",
    "create_grader",
    "DEFAULT_CROSS_ENCODER_MODEL",
    "SUPPORTED_GRADER_MODELS",
]
