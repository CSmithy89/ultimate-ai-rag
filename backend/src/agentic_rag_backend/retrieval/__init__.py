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

# Epic 20 - Query Routing (Story 20-B3)
from .query_router import QueryRouter
from .query_router_models import (
    QueryType,
    RoutingDecision,
    QueryRouteRequest,
    QueryRouteResponse,
    PatternListResponse,
    RouterStatusResponse,
)

# Epic 20 - Graph-Based Rerankers (Story 20-C1)
from .graph_rerankers import (
    GraphRerankerType,
    GraphContext,
    GraphRerankedResult,
    GraphRerankerAdapter,
    GraphReranker,
    EpisodeMentionsReranker,
    NodeDistanceReranker,
    HybridGraphReranker,
    get_graph_reranker_adapter,
    create_graph_reranker,
)

# Epic 20 - Dual-Level Retrieval (Story 20-C2)
from .dual_level import DualLevelRetriever
from .dual_level_models import (
    LowLevelResult,
    HighLevelResult,
    DualLevelResult,
    SynthesisResult,
    DualLevelRetrieveRequest,
    DualLevelRetrieveResponse,
    LowLevelResultResponse,
    HighLevelResultResponse,
    DualLevelStatusResponse,
)

# Epic 20 - Small-to-Big Retrieval (Story 20-C3)
from .small_to_big import (
    SmallToBigResult,
    SmallToBigRetrievalResult,
    SmallToBigRetriever,
    SmallToBigAdapter,
    get_small_to_big_adapter,
)

# Epic 20 - Sparse Vector Search (Story 20-H1)
from .sparse_vectors import (
    SparseVector,
    BM42Encoder,
    HybridVectorSearch,
    HybridSearchResult,
    SparseVectorAdapter,
    DenseSearchProtocol,
    DEFAULT_SPARSE_VECTORS_ENABLED,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_HYBRID_DENSE_WEIGHT,
    DEFAULT_HYBRID_SPARSE_WEIGHT,
)

# Epic 20 - Cross-Language Query (Story 20-H2)
from .cross_language import (
    LanguageDetector,
    LanguageDetectionResult,
    CrossLanguageEmbedding,
    QueryTranslator,
    CrossLanguageAdapter,
    DEFAULT_CROSS_LANGUAGE_ENABLED,
    DEFAULT_CROSS_LANGUAGE_EMBEDDING,
    DEFAULT_CROSS_LANGUAGE_TRANSLATION,
)

# Epic 20 - ColBERT Reranking (Story 20-H5)
from .colbert_reranker import (
    ColBERTResult,
    TokenEmbeddings,
    ColBERTEncoder,
    MaxSimScorer,
    ColBERTReranker,
    create_colbert_reranker,
    DEFAULT_COLBERT_ENABLED,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_COLBERT_MAX_LENGTH,
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
    # Epic 20 - Query Routing
    "QueryRouter",
    "QueryType",
    "RoutingDecision",
    "QueryRouteRequest",
    "QueryRouteResponse",
    "PatternListResponse",
    "RouterStatusResponse",
    # Epic 20 - Graph-Based Rerankers
    "GraphRerankerType",
    "GraphContext",
    "GraphRerankedResult",
    "GraphRerankerAdapter",
    "GraphReranker",
    "EpisodeMentionsReranker",
    "NodeDistanceReranker",
    "HybridGraphReranker",
    "get_graph_reranker_adapter",
    "create_graph_reranker",
    # Epic 20 - Dual-Level Retrieval
    "DualLevelRetriever",
    "LowLevelResult",
    "HighLevelResult",
    "DualLevelResult",
    "SynthesisResult",
    "DualLevelRetrieveRequest",
    "DualLevelRetrieveResponse",
    "LowLevelResultResponse",
    "HighLevelResultResponse",
    "DualLevelStatusResponse",
    # Epic 20 - Small-to-Big Retrieval
    "SmallToBigResult",
    "SmallToBigRetrievalResult",
    "SmallToBigRetriever",
    "SmallToBigAdapter",
    "get_small_to_big_adapter",
    # Epic 20 - Sparse Vector Search
    "SparseVector",
    "BM42Encoder",
    "HybridVectorSearch",
    "HybridSearchResult",
    "SparseVectorAdapter",
    "DenseSearchProtocol",
    "DEFAULT_SPARSE_VECTORS_ENABLED",
    "DEFAULT_SPARSE_MODEL",
    "DEFAULT_HYBRID_DENSE_WEIGHT",
    "DEFAULT_HYBRID_SPARSE_WEIGHT",
    # Epic 20 - Cross-Language Query
    "LanguageDetector",
    "LanguageDetectionResult",
    "CrossLanguageEmbedding",
    "QueryTranslator",
    "CrossLanguageAdapter",
    "DEFAULT_CROSS_LANGUAGE_ENABLED",
    "DEFAULT_CROSS_LANGUAGE_EMBEDDING",
    "DEFAULT_CROSS_LANGUAGE_TRANSLATION",
    # Epic 20 - ColBERT Reranking
    "ColBERTResult",
    "TokenEmbeddings",
    "ColBERTEncoder",
    "MaxSimScorer",
    "ColBERTReranker",
    "create_colbert_reranker",
    "DEFAULT_COLBERT_ENABLED",
    "DEFAULT_COLBERT_MODEL",
    "DEFAULT_COLBERT_MAX_LENGTH",
]
