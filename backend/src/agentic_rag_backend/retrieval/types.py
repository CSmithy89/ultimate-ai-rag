from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    document_id: str
    content: str
    similarity: float
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class GraphNode:
    id: str
    name: str
    type: str
    description: Optional[str] = None
    source_chunks: Optional[list[str]] = None


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    type: str
    confidence: Optional[float] = None
    source_chunk: Optional[str] = None


@dataclass(frozen=True)
class GraphPath:
    node_ids: list[str]
    edge_types: list[str]


@dataclass(frozen=True)
class GraphTraversalResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    paths: list[GraphPath]
