"""Tests for the CopilotKit API endpoint."""

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
from agentic_rag_backend.api.routes.copilot import copilot_handler
from agentic_rag_backend.models.copilot import (
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
)
from agentic_rag_backend.schemas import PlanStep


class DummyOrchestrator:
    """Stub orchestrator that returns a fixed response."""
    
    def __init__(self, answer: str = "Test answer"):
        self.answer = answer
        self.call_count = 0
        self.last_call_args = None

    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        self.call_count += 1
        self.last_call_args = {"query": query, "tenant_id": tenant_id, "session_id": session_id}
        return OrchestratorResult(
            answer=self.answer,
            plan=[PlanStep(step="Analyze", status="completed")],
            thoughts=["Analyzed query", "Retrieved context"],
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trajectory_id=uuid4(),
        )


class ErrorOrchestrator:
    """Stub orchestrator that raises an error."""
    
    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        raise RuntimeError("Internal database error")


class AllowLimiter:
    """Rate limiter that always allows."""
    
    async def allow(self, key: str) -> bool:
        return True


class DenyLimiter:
    """Rate limiter that always denies."""
    
    async def allow(self, key: str) -> bool:
        return False


async def collect_sse_events(response):
    """Collect all SSE events from a streaming response."""
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


class TestCopilotEndpointDirect:
    """Direct tests for the copilot_handler function."""

    @pytest.mark.asyncio
    async def test_copilot_handler_returns_sse_stream(self):
        """Test that copilot handler returns SSE stream."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="What is RAG?")],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=DummyOrchestrator(),
            limiter=AllowLimiter(),
        )
        
        assert response.media_type == "text/event-stream"
        assert response.headers.get("cache-control") == "no-cache"

    @pytest.mark.asyncio
    async def test_copilot_handler_streams_events(self):
        """Test that copilot handler streams AG-UI events."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="What is RAG?")],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=DummyOrchestrator(),
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        event_types = [e["event"] for e in events]
        
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types
        assert "STATE_SNAPSHOT" in event_types
        assert "TEXT_MESSAGE_START" in event_types
        assert "TEXT_MESSAGE_CONTENT" in event_types
        assert "TEXT_MESSAGE_END" in event_types

    @pytest.mark.asyncio
    async def test_copilot_handler_extracts_tenant_id(self):
        """Test that tenant_id is extracted and passed to orchestrator."""
        tenant_id = str(uuid4())
        orchestrator = DummyOrchestrator()
        
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={"tenant_id": tenant_id}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=orchestrator,
            limiter=AllowLimiter(),
        )
        
        # Consume the response to trigger processing
        await collect_sse_events(response)
        
        assert orchestrator.call_count == 1
        assert orchestrator.last_call_args["tenant_id"] == tenant_id

    @pytest.mark.asyncio
    async def test_copilot_handler_extracts_session_id(self):
        """Test that session_id is extracted and passed to orchestrator."""
        tenant_id = str(uuid4())
        session_id = str(uuid4())
        orchestrator = DummyOrchestrator()
        
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={"tenant_id": tenant_id, "session_id": session_id}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=orchestrator,
            limiter=AllowLimiter(),
        )
        
        # Consume the response to trigger processing
        await collect_sse_events(response)
        
        assert orchestrator.call_count == 1
        assert orchestrator.last_call_args["session_id"] == session_id


class TestCopilotMultiTenancy:
    """Tests for multi-tenancy in copilot endpoint."""

    @pytest.mark.asyncio
    async def test_copilot_missing_tenant_id_returns_error(self):
        """Test that missing tenant_id returns error message in stream."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={}),  # No tenant_id
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=DummyOrchestrator(),
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        
        # Should still have proper event sequence
        event_types = [e["event"] for e in events]
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types
        
        # Error message about tenant_id should be in content
        text_events = [e for e in events if e["event"] == "TEXT_MESSAGE_CONTENT"]
        assert len(text_events) >= 1
        assert "tenant_id" in text_events[0]["data"]["content"].lower()


class TestCopilotRateLimiting:
    """Tests for rate limiting in copilot endpoint."""

    @pytest.mark.asyncio
    async def test_copilot_rate_limit_exceeded(self):
        """Test that rate limiting returns 429."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await copilot_handler(
                request=request,
                orchestrator=DummyOrchestrator(),
                limiter=DenyLimiter(),
            )
        
        assert exc_info.value.status_code == 429
        assert "rate limit" in exc_info.value.detail.lower()


class TestCopilotSSEFormat:
    """Tests for SSE stream format compliance."""

    @pytest.mark.asyncio
    async def test_copilot_sse_event_format(self):
        """Test that SSE events are properly formatted."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="What is RAG?")],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=DummyOrchestrator(),
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        
        # All events should have event and data fields
        for event in events:
            assert "event" in event
            assert "data" in event

    @pytest.mark.asyncio
    async def test_copilot_text_message_sequence(self):
        """Test that text messages follow START -> CONTENT -> END sequence."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="What is RAG?")],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=DummyOrchestrator(answer="RAG is Retrieval Augmented Generation"),
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        event_types = [e["event"] for e in events]
        
        # Find indices
        start_idx = event_types.index("TEXT_MESSAGE_START")
        content_idx = event_types.index("TEXT_MESSAGE_CONTENT")
        end_idx = event_types.index("TEXT_MESSAGE_END")
        
        # Verify order
        assert start_idx < content_idx < end_idx


class TestCopilotErrorHandling:
    """Tests for error handling in copilot endpoint."""

    @pytest.mark.asyncio
    async def test_copilot_handles_orchestrator_error(self):
        """Test that orchestrator errors are handled gracefully."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        response = await copilot_handler(
            request=request,
            orchestrator=ErrorOrchestrator(),
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        
        # Should still emit proper event sequence
        event_types = [e["event"] for e in events]
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types
        
        # Error message should be sanitized (not expose internal details)
        text_events = [e for e in events if e["event"] == "TEXT_MESSAGE_CONTENT"]
        if text_events:
            content = text_events[0]["data"]["content"]
            assert "Internal database error" not in content
            assert "error" in content.lower() or "occurred" in content.lower()

    @pytest.mark.asyncio
    async def test_copilot_empty_messages(self):
        """Test handling of request with no messages."""
        request = CopilotRequest(
            messages=[],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        orchestrator = DummyOrchestrator()
        response = await copilot_handler(
            request=request,
            orchestrator=orchestrator,
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        
        # Should emit RUN_FINISHED without calling orchestrator
        event_types = [e["event"] for e in events]
        assert "RUN_FINISHED" in event_types
        assert orchestrator.call_count == 0

    @pytest.mark.asyncio
    async def test_copilot_no_user_messages(self):
        """Test handling of request with only assistant messages."""
        request = CopilotRequest(
            messages=[
                CopilotMessage(role=MessageRole.ASSISTANT, content="Previous response"),
            ],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )
        
        orchestrator = DummyOrchestrator()
        response = await copilot_handler(
            request=request,
            orchestrator=orchestrator,
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        
        # Should emit RUN_FINISHED without calling orchestrator
        event_types = [e["event"] for e in events]
        assert "RUN_FINISHED" in event_types
        assert orchestrator.call_count == 0
