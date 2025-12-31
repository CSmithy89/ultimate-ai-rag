"""Integration tests for AG-UI streaming bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_rag_backend.models.copilot import (
    AGUIEventType,
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
)
from agentic_rag_backend.protocols.ag_ui_bridge import AGUIBridge
from agentic_rag_backend.schemas import PlanStep

pytestmark = pytest.mark.integration


class DummyOrchestrator:
    async def run(self, query: str, tenant_id: str, session_id: str | None = None):
        return SimpleNamespace(
            answer="hello",
            plan=[PlanStep(step="respond", status="completed")],
            thoughts=["done"],
            retrieval_strategy=SimpleNamespace(value="vector"),
            trajectory_id=None,
            evidence=None,
        )


@pytest.mark.asyncio
async def test_agui_stream_emits_expected_events() -> None:
    bridge = AGUIBridge(orchestrator=DummyOrchestrator())
    request = CopilotRequest(
        messages=[CopilotMessage(role=MessageRole.USER, content="hi")],
        config=CopilotConfig(configurable={"tenant_id": "tenant-1"}),
    )

    events = []
    async for event in bridge.process_request(request):
        events.append(event.event)

    assert events[0] == AGUIEventType.RUN_STARTED
    assert AGUIEventType.STATE_SNAPSHOT in events
    assert AGUIEventType.TEXT_MESSAGE_START in events
    assert AGUIEventType.TEXT_MESSAGE_CONTENT in events
    assert AGUIEventType.TEXT_MESSAGE_END in events
    assert events[-1] == AGUIEventType.RUN_FINISHED
