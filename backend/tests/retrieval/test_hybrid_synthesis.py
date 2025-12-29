"""Tests for hybrid synthesis prompt building."""

from agentic_rag_backend.retrieval.hybrid_synthesis import (
    build_hybrid_prompt,
    rank_graph_paths,
    rank_vector_hits,
)
from agentic_rag_backend.retrieval.types import GraphNode, GraphPath, GraphTraversalResult, VectorHit


def test_rank_vector_hits_sorts_by_similarity() -> None:
    hits = [
        VectorHit(chunk_id="c1", document_id="d1", content="A", similarity=0.6),
        VectorHit(chunk_id="c2", document_id="d2", content="B", similarity=0.9),
    ]
    ranked = rank_vector_hits(hits)
    assert ranked[0].chunk_id == "c2"


def test_rank_graph_paths_sorts_by_length() -> None:
    result = GraphTraversalResult(
        nodes=[],
        edges=[],
        paths=[
            GraphPath(node_ids=["n1", "n2"], edge_types=["USES"]),
            GraphPath(node_ids=["n1", "n2", "n3"], edge_types=["USES", "PART_OF"]),
        ],
    )
    ranked = rank_graph_paths(result)
    assert ranked[0][0] == ["n1", "n2"]


def test_build_hybrid_prompt_includes_vector_and_graph() -> None:
    hits = [
        VectorHit(chunk_id="c1", document_id="d1", content="Vector content", similarity=0.9),
    ]
    graph_result = GraphTraversalResult(
        nodes=[GraphNode(id="n1", name="Alpha", type="Concept")],
        edges=[],
        paths=[GraphPath(node_ids=["n1"], edge_types=[])],
    )
    prompt = build_hybrid_prompt("Question?", hits, graph_result)

    assert "[vector:c1]" in prompt
    assert "[graph:n1]" in prompt
