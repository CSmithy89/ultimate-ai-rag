from __future__ import annotations

from uuid import uuid4

from agentic_rag_backend.agents import orchestrator as orchestrator_module
import pytest

from agentic_rag_backend.agents.orchestrator import OrchestratorAgent
from agentic_rag_backend.retrieval.types import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphTraversalResult,
    VectorHit,
)
from agentic_rag_backend.trajectory import EventType


class StubLogger:
    def __init__(self) -> None:
        self.events: list[tuple[EventType, str]] = []

    async def start_trajectory(
        self, tenant_id: str, session_id: str | None, agent_type: str | None = None
    ):
        return uuid4()

    async def log_events(
        self, tenant_id: str, trajectory_id, events: list[tuple[str, str]]
    ) -> None:
        self.events.extend(events)

    async def log_observation(
        self, tenant_id: str, trajectory_id, observation: str
    ) -> None:
        """Log a single observation event."""
        self.events.append((EventType.OBSERVATION, observation))


@pytest.mark.asyncio
async def test_orchestrator_builds_plan_and_logs_events(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_module, "AgnoAgentImpl", None)
    monkeypatch.setattr(orchestrator_module, "AgnoOpenAIChatImpl", None)

    logger = StubLogger()
    agent = OrchestratorAgent(api_key="test", logger=logger)

    result = await agent.run(
        "compare X versus Y if needed",
        tenant_id="11111111-1111-1111-1111-111111111111",
    )

    assert result.plan
    assert any("Break down" in step.step for step in result.plan)
    assert any("Refine plan" in step.step for step in result.plan)
    assert any(event[0] == EventType.THOUGHT for event in logger.events)
    assert any(event[0] == EventType.ACTION for event in logger.events)
    assert any(event[0] == EventType.OBSERVATION for event in logger.events)


@pytest.mark.asyncio
async def test_orchestrator_hybrid_retrieval_builds_evidence(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_module, "AgnoAgentImpl", None)
    monkeypatch.setattr(orchestrator_module, "AgnoOpenAIChatImpl", None)
    monkeypatch.setattr(
        orchestrator_module,
        "select_retrieval_strategy",
        lambda _: orchestrator_module.RetrievalStrategy.HYBRID,
    )

    logger = StubLogger()
    agent = OrchestratorAgent(api_key="test", logger=logger)

    vector_hits = [
        VectorHit(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Example content",
            similarity=0.8,
            metadata={"source_url": "https://example.com"},
        )
    ]
    graph_result = GraphTraversalResult(
        nodes=[GraphNode(id="n1", name="Alpha", type="Concept")],
        edges=[GraphEdge(source_id="n1", target_id="n2", type="RELATED_TO")],
        paths=[GraphPath(node_ids=["n1", "n2"], edge_types=["RELATED_TO"])],
    )

    async def fake_vector(*_args, **_kwargs):
        return vector_hits

    async def fake_graph(*_args, **_kwargs):
        return graph_result

    monkeypatch.setattr(agent, "_run_vector_search", fake_vector)
    monkeypatch.setattr(agent, "_run_graph_traversal", fake_graph)

    result = await agent.run(
        "hybrid query",
        tenant_id="11111111-1111-1111-1111-111111111111",
    )

    assert result.evidence is not None
    assert len(result.evidence.vector) == 1
    assert result.evidence.graph is not None
    assert len(result.evidence.graph.paths) == 1
