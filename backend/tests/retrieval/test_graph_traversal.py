"""Tests for graph traversal retrieval."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag_backend.retrieval.graph_traversal import GraphTraversalService, extract_terms
from agentic_rag_backend.retrieval.types import GraphTraversalResult


def test_extract_terms_deduplicates_and_filters() -> None:
    terms = extract_terms("Graph connections between graph and the network", max_terms=5)
    assert "graph" in terms
    assert "the" not in terms
    assert len(terms) == len(set(terms))


@dataclass
class FakeRel:
    type: str
    start_node: dict
    end_node: dict
    confidence: float | None = None

    def __iter__(self):
        return iter({"confidence": self.confidence}.items())


@dataclass
class FakePath:
    nodes: list[dict]
    relationships: list[FakeRel]


@pytest.mark.asyncio
async def test_graph_traversal_builds_result() -> None:
    neo4j = MagicMock()
    neo4j.search_entities_by_terms = AsyncMock(
        return_value=[
            {"id": "n1", "name": "Alpha", "type": "Concept"},
        ]
    )
    node1 = {"id": "n1", "name": "Alpha", "type": "Concept"}
    node2 = {"id": "n2", "name": "Beta", "type": "Technology"}
    path = FakePath(
        nodes=[node1, node2],
        relationships=[FakeRel(type="USES", start_node=node1, end_node=node2, confidence=0.9)],
    )
    neo4j.traverse_paths = AsyncMock(return_value=[path])

    service = GraphTraversalService(neo4j=neo4j)
    result = await service.traverse("Show Alpha relationships", "11111111-1111-1111-1111-111111111111")

    assert isinstance(result, GraphTraversalResult)
    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    assert result.edges[0].type == "USES"
    assert result.paths[0].node_ids == ["n1", "n2"]


@pytest.mark.asyncio
async def test_graph_traversal_skips_invalid_nodes() -> None:
    neo4j = MagicMock()
    neo4j.search_entities_by_terms = AsyncMock(return_value=[])
    node1 = {"id": "n1", "name": "Alpha", "type": "Concept"}
    node2 = {"id": "n2", "name": "Beta", "type": "Technology"}
    path = FakePath(
        nodes=[object(), node1, node2],
        relationships=[FakeRel(type="USES", start_node=node1, end_node=node2)],
    )
    neo4j.traverse_paths = AsyncMock(return_value=[path])

    service = GraphTraversalService(neo4j=neo4j)
    result = await service.traverse("Show Alpha relationships", "11111111-1111-1111-1111-111111111111")

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
