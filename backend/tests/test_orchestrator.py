from __future__ import annotations

from uuid import uuid4

from agentic_rag_backend.agents import orchestrator as orchestrator_module
import pytest

from agentic_rag_backend.agents.orchestrator import OrchestratorAgent
from agentic_rag_backend.trajectory import EventType


class StubLogger:
    def __init__(self) -> None:
        self.events: list[tuple[EventType, str]] = []

    async def start_trajectory(self, tenant_id: str, session_id: str | None):
        return uuid4()

    async def log_events(
        self, tenant_id: str, trajectory_id, events: list[tuple[str, str]]
    ) -> None:
        self.events.extend(events)


@pytest.mark.asyncio
async def test_orchestrator_builds_plan_and_logs_events(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_module, "Agent", None)
    monkeypatch.setattr(orchestrator_module, "OpenAIChat", None)

    logger = StubLogger()
    agent = OrchestratorAgent(api_key="test", logger=logger)

    result = await agent.run("compare X versus Y if needed", tenant_id="tenant")

    assert result.plan
    assert any("Break down" in step.step for step in result.plan)
    assert any("Refine plan" in step.step for step in result.plan)
    assert any(event[0] == EventType.THOUGHT for event in logger.events)
    assert any(event[0] == EventType.ACTION for event in logger.events)
    assert any(event[0] == EventType.OBSERVATION for event in logger.events)
