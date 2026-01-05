"""Retrieval strategy selection based on query analysis.

This module provides intelligent routing of queries to the most appropriate
retrieval strategy (vector, graph, or hybrid) based on query content analysis.
"""

from enum import Enum
import re
from typing import Optional

from .observability.metrics import record_retrieval_request


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


SEMANTIC_HINTS = {
    "semantic",
    "similar",
    "similarity",
    "meaning",
    "summarize",
    "summary",
    "overview",
    "about",
}

RELATIONAL_HINTS = {
    "relationship",
    "related",
    "connected",
    "connection",
    "graph",
    "node",
    "edge",
    "path",
    "traverse",
    "link",
    "network",
}

HYBRID_HINTS = {
    "multi-hop",
    "multi hop",
    "across",
    "combine",
}


def select_retrieval_strategy(
    query: str,
    tenant_id: Optional[str] = None,
    record_metric: bool = True,
) -> RetrievalStrategy:
    """Select a retrieval strategy based on query hints.

    Args:
        query: The user's query string
        tenant_id: Tenant identifier for metrics (optional)
        record_metric: Whether to record the selection as a Prometheus metric

    Returns:
        The selected RetrievalStrategy
    """
    normalized = query.lower()
    semantic = _matches_any(normalized, SEMANTIC_HINTS)
    relational = _matches_any(normalized, RELATIONAL_HINTS)
    hybrid = _matches_any(normalized, HYBRID_HINTS)

    if (semantic and relational) or (hybrid and (semantic or relational)):
        strategy = RetrievalStrategy.HYBRID
    elif relational:
        strategy = RetrievalStrategy.GRAPH
    elif semantic:
        strategy = RetrievalStrategy.VECTOR
    else:
        strategy = RetrievalStrategy.VECTOR

    # Record the strategy selection metric
    if record_metric and tenant_id is not None:
        record_retrieval_request(
            strategy=strategy.value,
            tenant_id=tenant_id,
        )

    return strategy


def _matches_any(text: str, hints: set[str]) -> bool:
    """Return True when any hint is matched as a whole word."""
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in hints)
