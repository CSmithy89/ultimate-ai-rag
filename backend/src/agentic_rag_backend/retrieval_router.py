from enum import Enum
import re


class RetrievalStrategy(str, Enum):
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


def select_retrieval_strategy(query: str) -> RetrievalStrategy:
    """Select a retrieval strategy based on query hints."""
    normalized = query.lower()
    semantic = _matches_any(normalized, SEMANTIC_HINTS)
    relational = _matches_any(normalized, RELATIONAL_HINTS)
    hybrid = _matches_any(normalized, HYBRID_HINTS)

    if (semantic and relational) or (hybrid and (semantic or relational)):
        return RetrievalStrategy.HYBRID
    if relational:
        return RetrievalStrategy.GRAPH
    if semantic:
        return RetrievalStrategy.VECTOR
    return RetrievalStrategy.VECTOR


def _matches_any(text: str, hints: set[str]) -> bool:
    """Return True when any hint is matched as a whole word."""
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in hints)
