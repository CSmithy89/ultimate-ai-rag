from __future__ import annotations

import re
from typing import Iterable

import structlog

from agentic_rag_backend.db.neo4j import Neo4jClient

from .types import GraphEdge, GraphNode, GraphPath, GraphTraversalResult

logger = structlog.get_logger(__name__)

TERM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "between",
    "can",
    "could",
    "does",
    "for",
    "from",
    "how",
    "into",
    "like",
    "more",
    "show",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "those",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "who",
    "why",
}


def extract_terms(query: str, max_terms: int = 6) -> list[str]:
    tokens = [token.lower() for token in TERM_PATTERN.findall(query)]
    unique: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token not in unique:
            unique.append(token)
    unique.sort(key=len, reverse=True)
    return unique[:max_terms]


class GraphTraversalService:
    """Graph traversal helper for relationship queries."""

    def __init__(
        self,
        neo4j: Neo4jClient,
        max_hops: int = 2,
        path_limit: int = 10,
        entity_limit: int = 12,
        allowed_relationships: Iterable[str] | None = None,
    ) -> None:
        self.neo4j = neo4j
        self.max_hops = max_hops
        self.path_limit = path_limit
        self.entity_limit = entity_limit
        self.allowed_relationships = list(allowed_relationships) if allowed_relationships else None

    async def traverse(self, query: str, tenant_id: str) -> GraphTraversalResult:
        terms = extract_terms(query)
        entities = await self.neo4j.search_entities_by_terms(
            tenant_id=tenant_id,
            terms=terms,
            limit=self.entity_limit,
        )
        start_ids = [entity.get("id") for entity in entities if entity.get("id")]
        paths = await self.neo4j.traverse_paths(
            tenant_id=tenant_id,
            start_entity_ids=start_ids,
            max_hops=self.max_hops,
            limit=self.path_limit,
            allowed_relationships=self.allowed_relationships,
        )
        result = self._build_result(entities, paths)
        logger.info(
            "graph_traversal_completed",
            tenant_id=tenant_id,
            terms=terms,
            entities=len(result.nodes),
            edges=len(result.edges),
            paths=len(result.paths),
        )
        return result

    def _build_result(self, entities: list[dict], paths: list) -> GraphTraversalResult:
        nodes_by_id: dict[str, GraphNode] = {}
        edges_by_key: dict[tuple[str, str, str], GraphEdge] = {}
        path_results: list[GraphPath] = []

        for entity in entities:
            node_id = str(entity.get("id", ""))
            if not node_id:
                continue
            nodes_by_id[node_id] = GraphNode(
                id=node_id,
                name=entity.get("name", ""),
                type=entity.get("type", ""),
                description=entity.get("description"),
                source_chunks=entity.get("source_chunks") or [],
            )

        for path in paths:
            node_ids: list[str] = []
            edge_types: list[str] = []

            for node in getattr(path, "nodes", []):
                props = dict(node)
                node_id = str(props.get("id", ""))
                if not node_id:
                    continue
                if node_id not in nodes_by_id:
                    nodes_by_id[node_id] = GraphNode(
                        id=node_id,
                        name=props.get("name", ""),
                        type=props.get("type", ""),
                        description=props.get("description"),
                        source_chunks=props.get("source_chunks") or [],
                    )
                node_ids.append(node_id)

            for rel in getattr(path, "relationships", []):
                rel_props = dict(rel)
                rel_type = getattr(rel, "type", None) or rel_props.get("type") or "RELATED_TO"
                start_node = getattr(rel, "start_node", None)
                end_node = getattr(rel, "end_node", None)
                source_id = None
                target_id = None
                if start_node is not None:
                    source_id = str(dict(start_node).get("id", ""))
                if end_node is not None:
                    target_id = str(dict(end_node).get("id", ""))
                if not source_id or not target_id:
                    continue
                edge_key = (source_id, target_id, rel_type)
                if edge_key not in edges_by_key:
                    edges_by_key[edge_key] = GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        type=rel_type,
                        confidence=rel_props.get("confidence"),
                        source_chunk=rel_props.get("source_chunk"),
                    )
                edge_types.append(rel_type)

            if node_ids and edge_types:
                path_results.append(GraphPath(node_ids=node_ids, edge_types=edge_types))

        return GraphTraversalResult(
            nodes=list(nodes_by_id.values()),
            edges=list(edges_by_key.values()),
            paths=path_results,
        )
