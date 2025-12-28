from enum import Enum


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
    "and",
    "combine",
}


def select_retrieval_strategy(query: str) -> RetrievalStrategy:
    normalized = query.lower()
    semantic = any(token in normalized for token in SEMANTIC_HINTS)
    relational = any(token in normalized for token in RELATIONAL_HINTS)
    hybrid = any(token in normalized for token in HYBRID_HINTS)

    if (semantic and relational) or (hybrid and (semantic or relational)):
        return RetrievalStrategy.HYBRID
    if relational:
        return RetrievalStrategy.GRAPH
    if semantic:
        return RetrievalStrategy.VECTOR
    return RetrievalStrategy.VECTOR
