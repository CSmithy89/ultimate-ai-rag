from __future__ import annotations

from typing import Iterable

from .constants import MAX_GRAPH_PATHS, MAX_VECTOR_CONTENT_CHARS, MAX_VECTOR_HITS
from .types import GraphTraversalResult, VectorHit


def rank_vector_hits(vector_hits: Iterable[VectorHit]) -> list[VectorHit]:
    """Rank vector hits by similarity (highest first)."""
    return sorted(vector_hits, key=lambda hit: hit.similarity, reverse=True)


def rank_graph_paths(graph_result: GraphTraversalResult) -> list[tuple[list[str], list[str]]]:
    """Rank graph paths by length (shorter is preferred for explainability)."""
    scored_paths: list[tuple[int, list[str], list[str]]] = []
    for path in graph_result.paths:
        # Path length is the only stable score available in the current graph result.
        score = len(path.edge_types)
        scored_paths.append((score, path.node_ids, path.edge_types))
    scored_paths.sort(key=lambda item: item[0])
    return [(node_ids, edge_types) for _, node_ids, edge_types in scored_paths]


def build_hybrid_prompt(
    query: str,
    vector_hits: list[VectorHit],
    graph_result: GraphTraversalResult | None,
) -> str:
    """Build a hybrid prompt combining vector and graph evidence."""
    prompt_parts = [
        "Answer the user question using the evidence below.",
        "Cite sources inline using [vector:chunk_id] or [graph:entity_id].",
        f"Question: {query}",
    ]

    ranked_vectors = rank_vector_hits(vector_hits)[:MAX_VECTOR_HITS]
    if ranked_vectors:
        vector_lines = []
        for hit in ranked_vectors:
            content = hit.content.strip().replace("\n", " ")
            if len(content) > MAX_VECTOR_CONTENT_CHARS:
                content = content[:MAX_VECTOR_CONTENT_CHARS].rstrip() + "..."
            vector_lines.append(f"[vector:{hit.chunk_id}] {content}")
        prompt_parts.append("Vector Evidence:")
        prompt_parts.append("\n".join(vector_lines))

    if graph_result and graph_result.nodes:
        node_lines = []
        for node in graph_result.nodes:
            node_lines.append(
                f"[graph:{node.id}] {node.name} ({node.type})"
            )
        prompt_parts.append("Graph Nodes:")
        prompt_parts.append("\n".join(node_lines))

    if graph_result and graph_result.paths:
        path_lines = []
        for node_ids, edge_types in rank_graph_paths(graph_result)[:MAX_GRAPH_PATHS]:
            segments = []
            for idx, node_id in enumerate(node_ids[:-1]):
                next_id = node_ids[idx + 1]
                edge_type = edge_types[idx] if idx < len(edge_types) else "RELATED_TO"
                segments.append(f"[graph:{node_id}] -[{edge_type}]-> [graph:{next_id}]")
            if segments:
                path_lines.append(" ".join(segments))
        if path_lines:
            prompt_parts.append("Graph Paths:")
            prompt_parts.append("\n".join(path_lines))

    return "\n\n".join(prompt_parts)
