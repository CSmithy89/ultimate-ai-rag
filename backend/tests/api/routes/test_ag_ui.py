"""Tests for the universal AG-UI endpoint."""

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SKIP_DB_POOL", "1")
os.environ.setdefault("SKIP_GRAPHITI", "1")

import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from agentic_rag_backend.agents.orchestrator import OrchestratorResult, RetrievalStrategy
from agentic_rag_backend.api.routes.ag_ui import AGUIRequest, ag_ui_handler
from agentic_rag_backend.models.copilot import CopilotMessage, MessageRole
from agentic_rag_backend.schemas import PlanStep


class DummyOrchestrator:
    async def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
        return OrchestratorResult(
            answer="ok",
            plan=[PlanStep(step="Step", status="completed")],
            thoughts=["Plan step: Step"],
            retrieval_strategy=RetrievalStrategy.VECTOR,
            trajectory_id=uuid4(),
        )


class AllowLimiter:
    async def allow(self, key: str) -> bool:
        return True


class DenyLimiter:
    async def allow(self, key: str) -> bool:
        return False


async def collect_sse_events(response):
    events = []
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode())
        else:
            chunks.append(chunk)

    body = "".join(chunks)
    for line in body.split("\n"):
        if line.startswith("data: "):
            event_data = json.loads(line[6:])
            events.append(event_data)
    return events


@pytest.mark.asyncio
async def test_ag_ui_streams_events() -> None:
    request = AGUIRequest(
        messages=[CopilotMessage(role=MessageRole.USER, content="Hello")],
        tenant_id="tenant-1",
    )

    response = await ag_ui_handler(
        request=request,
        orchestrator=DummyOrchestrator(),
        limiter=AllowLimiter(),
    )

    events = await collect_sse_events(response)
    event_types = {event["event"] for event in events}

    assert "RUN_STARTED" in event_types
    assert "STATE_SNAPSHOT" in event_types
    assert "TEXT_MESSAGE_START" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert "TEXT_MESSAGE_END" in event_types
    assert "RUN_FINISHED" in event_types


@pytest.mark.asyncio
async def test_ag_ui_rate_limit() -> None:
    request = AGUIRequest(
        messages=[CopilotMessage(role=MessageRole.USER, content="Hello")],
        tenant_id="tenant-1",
    )

    with pytest.raises(HTTPException) as exc_info:
        await ag_ui_handler(
            request=request,
            orchestrator=DummyOrchestrator(),
            limiter=DenyLimiter(),
        )
    assert exc_info.value.status_code == 429
