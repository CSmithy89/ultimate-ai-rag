"""Graph Intelligence module for community detection (Story 20-B1).

This module implements community detection for the knowledge graph,
competing with Microsoft GraphRAG's approach to handling "global" queries.

Key Features:
- Louvain/Leiden community detection algorithms via NetworkX
- Hierarchical community structures with multiple abstraction levels
- LLM-generated community summaries
- Neo4j storage with BELONGS_TO relationships
- Multi-tenancy via tenant_id filtering

Configuration:
- COMMUNITY_DETECTION_ENABLED: Enable/disable feature (default: false)
- COMMUNITY_ALGORITHM: louvain|leiden (default: louvain)
- COMMUNITY_MIN_SIZE: Minimum entities per community (default: 3)
- COMMUNITY_MAX_LEVELS: Maximum hierarchy levels (default: 3)
- COMMUNITY_SUMMARY_MODEL: LLM model for summaries (default: gpt-4o-mini)
- COMMUNITY_REFRESH_SCHEDULE: Cron schedule for refresh (default: "0 3 * * 0")

Usage:
    from agentic_rag_backend.graph import CommunityDetector, CommunityAlgorithm

    detector = CommunityDetector(
        neo4j_client=neo4j_client,
        llm_client=llm_client,
        algorithm=CommunityAlgorithm.LOUVAIN,
    )

    communities = await detector.detect_communities(
        tenant_id="tenant-123",
        generate_summaries=True,
    )

Dependencies:
- networkx>=3.0 (required)
- leidenalg (optional, for Leiden algorithm)
- igraph (optional, required for leidenalg)
"""

from .errors import (
    CommunityDetectionError,
    CommunityNotFoundError,
    GraphTooSmallError,
)
from .models import (
    Community,
    CommunityAlgorithm,
    CommunityDetectionRequest,
    CommunityDetectionResponse,
    CommunityListResponse,
    CommunitySearchRequest,
    CommunitySearchResponse,
    CommunityWithEntities,
)
from .community import (
    CommunityDetector,
    LEIDEN_AVAILABLE,
    NETWORKX_AVAILABLE,
)

__all__ = [
    # Errors
    "CommunityDetectionError",
    "CommunityNotFoundError",
    "GraphTooSmallError",
    # Models
    "Community",
    "CommunityAlgorithm",
    "CommunityDetectionRequest",
    "CommunityDetectionResponse",
    "CommunityListResponse",
    "CommunitySearchRequest",
    "CommunitySearchResponse",
    "CommunityWithEntities",
    # Core
    "CommunityDetector",
    # Availability flags
    "LEIDEN_AVAILABLE",
    "NETWORKX_AVAILABLE",
]
