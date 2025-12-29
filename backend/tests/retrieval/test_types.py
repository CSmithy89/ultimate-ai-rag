"""Tests for retrieval type definitions."""

from dataclasses import is_dataclass

from agentic_rag_backend.retrieval.types import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphTraversalResult,
    VectorHit,
)


def test_retrieval_types_are_dataclasses() -> None:
    assert is_dataclass(VectorHit)
    assert is_dataclass(GraphNode)
    assert is_dataclass(GraphEdge)
    assert is_dataclass(GraphPath)
    assert is_dataclass(GraphTraversalResult)
