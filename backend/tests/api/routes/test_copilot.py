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
from pydantic import ValidationError

from agentic_rag_backend.agents.orchestrator import OrchestratorResult, RetrievalStrategy
from agentic_rag_backend.api.routes.copilot import (
    copilot_handler,
    get_hitl_checkpoint,
    list_hitl_checkpoints,
    receive_validation_response,
    ValidationResponseRequest,
)
from agentic_rag_backend.models.copilot import (
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
)
from agentic_rag_backend.protocols.ag_ui_bridge import HITLManager
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
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
                http_request=MockRequest(),
            orchestrator=orchestrator,
            limiter=AllowLimiter(),
        )
        
        events = await collect_sse_events(response)
        
        # Should emit RUN_FINISHED without calling orchestrator
        event_types = [e["event"] for e in events]
        assert "RUN_FINISHED" in event_types
        assert orchestrator.call_count == 0


# Issue 7 Fix: Backend tests for validation-response endpoint
class TestValidationResponseEndpoint:
    """Tests for the HITL validation-response endpoint."""

    def test_validation_request_requires_valid_uuid(self):
        """Test that checkpoint_id must be a valid UUID4."""
        # Valid UUID should work
        valid_request = ValidationResponseRequest(
            checkpoint_id=str(uuid4()),
            approved_source_ids=["source-1"],
        )
        assert valid_request.checkpoint_id is not None

        # Invalid UUID should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ValidationResponseRequest(
                checkpoint_id="not-a-uuid",
                approved_source_ids=["source-1"],
            )
        
        assert "checkpoint_id" in str(exc_info.value)

    def test_validation_request_accepts_empty_approved_ids(self):
        """Test that empty approved_source_ids is valid."""
        request = ValidationResponseRequest(
            checkpoint_id=str(uuid4()),
            approved_source_ids=[],
        )
        assert request.approved_source_ids == []

    def test_validation_request_rejects_invalid_uuid_formats(self):
        """Test various invalid UUID formats are rejected."""
        invalid_uuids = [
            "12345",
            "not-a-uuid-at-all",
            "12345678-1234-1234-1234-123456789012",  # Not a UUID4 (wrong version)
            "",
            "   ",
        ]
        
        for invalid_uuid in invalid_uuids:
            with pytest.raises(ValidationError):
                ValidationResponseRequest(
                    checkpoint_id=invalid_uuid,
                    approved_source_ids=[],
                )


class MockRequest:
    """Mock FastAPI Request for testing."""
    
    def __init__(self, hitl_manager=None):
        self.app = type("App", (), {"state": type("State", (), {"hitl_manager": hitl_manager})()})()


class MockHitlManager:
    """Mock HITL manager for checkpoint queries."""

    def __init__(self, fetch_result=None, list_result=None):
        self._fetch_result = fetch_result
        self._list_result = list_result or []

    async def fetch_checkpoint(self, checkpoint_id):
        return self._fetch_result

    async def list_checkpoints(self, tenant_id, limit=20):
        return list(self._list_result)[:limit]


class TestValidationResponseHandler:
    """Tests for the receive_validation_response handler."""

    @pytest.mark.asyncio
    async def test_validation_response_checkpoint_not_found(self):
        """Test 404 when checkpoint not found."""
        hitl_manager = HITLManager()
        
        request_body = ValidationResponseRequest(
            checkpoint_id=str(uuid4()),
            approved_source_ids=["source-1"],
        )
        mock_request = MockRequest(hitl_manager=hitl_manager)
        
        with pytest.raises(HTTPException) as exc_info:
            await receive_validation_response(
                request_body=request_body,
                request=mock_request,
                tenant_id=None,
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_validation_response_without_hitl_manager(self):
        """Test endpoint works without HITL manager (mock response)."""
        request_body = ValidationResponseRequest(
            checkpoint_id=str(uuid4()),
            approved_source_ids=["source-1", "source-2"],
        )
        mock_request = MockRequest(hitl_manager=None)
        
        result = await receive_validation_response(
            request_body=request_body,
            request=mock_request,
            tenant_id=None,
        )
        
        assert result.checkpoint_id == request_body.checkpoint_id
        assert result.status == "approved"
        assert result.approved_count == 2
        assert result.rejected_count == 0

    @pytest.mark.asyncio
    async def test_validation_response_rejected_when_no_sources_approved(self):
        """Test that empty approved list results in 'rejected' status."""
        request_body = ValidationResponseRequest(
            checkpoint_id=str(uuid4()),
            approved_source_ids=[],
        )
        mock_request = MockRequest(hitl_manager=None)
        
        result = await receive_validation_response(
            request_body=request_body,
            request=mock_request,
            tenant_id=None,
        )
        
        assert result.status == "rejected"
        assert result.approved_count == 0

    @pytest.mark.asyncio
    async def test_validation_response_with_valid_checkpoint(self):
        """Test successful validation response with existing checkpoint."""
        hitl_manager = HITLManager()
        checkpoint_id = str(uuid4())
        
        # Create a checkpoint first
        await hitl_manager.create_checkpoint(
            sources=[
                {"id": "source-1", "title": "Doc 1"},
                {"id": "source-2", "title": "Doc 2"},
            ],
            query="test query",
            checkpoint_id=checkpoint_id,
        )
        
        request_body = ValidationResponseRequest(
            checkpoint_id=checkpoint_id,
            approved_source_ids=["source-1"],
        )
        mock_request = MockRequest(hitl_manager=hitl_manager)
        
        result = await receive_validation_response(
            request_body=request_body,
            request=mock_request,
            tenant_id=None,
        )
        
        assert result.checkpoint_id == checkpoint_id
        assert result.status == "approved"
        assert result.approved_count == 1
        assert result.rejected_count == 1


class TestHitlCheckpointEndpoints:
    """Tests for HITL checkpoint query endpoints."""

    @pytest.mark.asyncio
    async def test_get_hitl_checkpoint_requires_manager(self):
        mock_request = MockRequest(hitl_manager=None)
        with pytest.raises(HTTPException) as exc_info:
            await get_hitl_checkpoint(
                checkpoint_id=str(uuid4()),
                request=mock_request,
                tenant_id=None,
            )
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_get_hitl_checkpoint_tenant_mismatch(self):
        record = {
            "checkpoint_id": str(uuid4()),
            "status": "approved",
            "query": "test query",
            "tenant_id": "tenant-a",
            "sources": [],
            "approved_source_ids": [],
            "rejected_source_ids": [],
        }
        mock_request = MockRequest(hitl_manager=MockHitlManager(fetch_result=record))
        with pytest.raises(HTTPException) as exc_info:
            await get_hitl_checkpoint(
                checkpoint_id=record["checkpoint_id"],
                request=mock_request,
                tenant_id="tenant-b",
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_hitl_checkpoint_success(self):
        record = {
            "checkpoint_id": str(uuid4()),
            "status": "approved",
            "query": "test query",
            "tenant_id": "tenant-a",
            "sources": [{"id": "source-1"}],
            "approved_source_ids": ["source-1"],
            "rejected_source_ids": [],
        }
        mock_request = MockRequest(hitl_manager=MockHitlManager(fetch_result=record))
        result = await get_hitl_checkpoint(
            checkpoint_id=record["checkpoint_id"],
            request=mock_request,
            tenant_id="tenant-a",
        )
        assert result.checkpoint_id == record["checkpoint_id"]
        assert result.status == "approved"

    @pytest.mark.asyncio
    async def test_list_hitl_checkpoints_requires_tenant(self):
        mock_request = MockRequest(hitl_manager=MockHitlManager())
        with pytest.raises(HTTPException) as exc_info:
            await list_hitl_checkpoints(
                request=mock_request,
                tenant_id=None,
                limit=10,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_list_hitl_checkpoints_returns_records(self):
        records = [
            {
                "checkpoint_id": str(uuid4()),
                "status": "approved",
                "query": "query one",
                "tenant_id": "tenant-a",
                "sources": [],
                "approved_source_ids": [],
                "rejected_source_ids": [],
            },
            {
                "checkpoint_id": str(uuid4()),
                "status": "rejected",
                "query": "query two",
                "tenant_id": "tenant-a",
                "sources": [],
                "approved_source_ids": [],
                "rejected_source_ids": [],
            },
        ]
        mock_request = MockRequest(hitl_manager=MockHitlManager(list_result=records))
        result = await list_hitl_checkpoints(
            request=mock_request,
            tenant_id="tenant-a",
            limit=10,
        )
        assert len(result) == 2
        assert result[0].checkpoint_id == records[0]["checkpoint_id"]
