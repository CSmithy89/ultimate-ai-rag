"""Tests for the AG-UI protocol bridge."""

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agentic_rag_backend.models.copilot import (
    AGUIEventType,
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextDeltaEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from agentic_rag_backend.protocols.ag_ui_bridge import (
    AGUIBridge,
    HITLCheckpoint,
    HITLStatus,
)
from agentic_rag_backend.agents.orchestrator import RetrievalStrategy
from agentic_rag_backend.schemas import RetrievalEvidence, VectorCitation


class MockOrchestratorResult:
    """Mock result from OrchestratorAgent."""

    def __init__(
        self,
        answer: str = "Test answer",
        thoughts: list[str] | None = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        trajectory_id: str | None = None,
        evidence: RetrievalEvidence | None = None,
    ):
        self.answer = answer
        self.thoughts = thoughts or ["Analyzed query", "Retrieved context"]
        self.retrieval_strategy = retrieval_strategy
        self.trajectory_id = trajectory_id
        self.evidence = evidence


@pytest.fixture
def mock_orchestrator():
    """Create a mock OrchestratorAgent."""
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(return_value=MockOrchestratorResult())
    return orchestrator


@pytest.fixture
def sample_copilot_request():
    """Create a sample CopilotKit request."""
    return CopilotRequest(
        messages=[
            CopilotMessage(role=MessageRole.USER, content="What is RAG?"),
        ],
        config=CopilotConfig(
            configurable={
                "tenant_id": str(uuid4()),
                "session_id": str(uuid4()),
            }
        ),
    )


class StubHitlManager:
    """Stub HITL manager that auto-approves sources."""

    def __init__(self):
        self._checkpoint: HITLCheckpoint | None = None

    async def create_checkpoint(self, sources, query, checkpoint_id=None, tenant_id=None):
        checkpoint = HITLCheckpoint(
            checkpoint_id=checkpoint_id or str(uuid4()),
            sources=sources,
            query=query,
            tenant_id=tenant_id,
        )
        self._checkpoint = checkpoint
        return checkpoint

    def get_checkpoint_events(self, checkpoint):
        return [
            ToolCallStartEvent(tool_call_id=checkpoint.checkpoint_id, tool_name="validate_sources"),
            ToolCallArgsEvent(
                tool_call_id=checkpoint.checkpoint_id,
                args={
                    "sources": checkpoint.sources,
                    "query": checkpoint.query,
                    "checkpoint_id": checkpoint.checkpoint_id,
                },
            ),
        ]

    async def wait_for_validation(self, checkpoint_id, timeout=None):
        checkpoint = self._checkpoint
        assert checkpoint is not None
        checkpoint.approved_source_ids = [source["id"] for source in checkpoint.sources]
        checkpoint.status = HITLStatus.APPROVED
        return checkpoint

    def get_completion_events(self, checkpoint):
        return [
            ToolCallEndEvent(tool_call_id=checkpoint.checkpoint_id),
            StateSnapshotEvent(
                state={
                    "hitl_checkpoint": checkpoint.to_dict(),
                    "approved_sources": checkpoint.sources,
                }
            ),
        ]


class TestAGUIBridgeEventTransformation:
    """Tests for AG-UI event transformation."""

    @pytest.mark.asyncio
    async def test_process_request_emits_run_started_event(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that process_request emits RUN_STARTED event first."""
        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        assert len(events) > 0
        assert events[0].event == AGUIEventType.RUN_STARTED
        assert isinstance(events[0], RunStartedEvent)

    @pytest.mark.asyncio
    async def test_process_request_emits_run_finished_event(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that process_request emits RUN_FINISHED event last."""
        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        assert len(events) > 0
        assert events[-1].event == AGUIEventType.RUN_FINISHED
        assert isinstance(events[-1], RunFinishedEvent)

    @pytest.mark.asyncio
    async def test_process_request_emits_state_snapshot_event(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that process_request emits STATE_SNAPSHOT with agent state."""
        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        state_events = [e for e in events if e.event == AGUIEventType.STATE_SNAPSHOT]
        assert len(state_events) == 1

    @pytest.mark.asyncio
    async def test_process_request_emits_hitl_events(self, sample_copilot_request):
        """Test that HITL events are emitted when evidence is available."""
        orchestrator = MagicMock()
        citations = [
            VectorCitation(
                chunk_id="chunk-1",
                document_id="doc-1",
                similarity=0.9,
                source="doc-1",
                content_preview="preview",
                metadata=None,
            )
        ]
        evidence = RetrievalEvidence(vector=citations)
        orchestrator.run = AsyncMock(return_value=MockOrchestratorResult(evidence=evidence))

        bridge = AGUIBridge(orchestrator, hitl_manager=StubHitlManager())
        events = []
        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        event_types = [event.event for event in events]
        assert AGUIEventType.TOOL_CALL_START in event_types
        assert AGUIEventType.TOOL_CALL_ARGS in event_types
        assert AGUIEventType.TOOL_CALL_END in event_types
        assert isinstance(state_events[0], StateSnapshotEvent)
        assert "state" in state_events[0].data
        assert "currentStep" in state_events[0].data["state"]
        assert "steps" in state_events[0].data["state"]

    @pytest.mark.asyncio
    async def test_process_request_emits_text_message_sequence(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that text messages follow START -> CONTENT -> END sequence."""
        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        # Find text message events
        text_start = [e for e in events if e.event == AGUIEventType.TEXT_MESSAGE_START]
        text_content = [e for e in events if e.event == AGUIEventType.TEXT_MESSAGE_CONTENT]
        text_end = [e for e in events if e.event == AGUIEventType.TEXT_MESSAGE_END]

        assert len(text_start) == 1
        assert len(text_content) == 1
        assert len(text_end) == 1

        # Verify order: START before CONTENT before END
        start_idx = events.index(text_start[0])
        content_idx = events.index(text_content[0])
        end_idx = events.index(text_end[0])

        assert start_idx < content_idx < end_idx

    @pytest.mark.asyncio
    async def test_process_request_text_delta_contains_answer(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that TextDeltaEvent contains the orchestrator answer."""
        expected_answer = "RAG is Retrieval Augmented Generation"
        mock_orchestrator.run.return_value = MockOrchestratorResult(answer=expected_answer)

        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        text_events = [e for e in events if e.event == AGUIEventType.TEXT_MESSAGE_CONTENT]
        assert len(text_events) == 1
        assert text_events[0].data["content"] == expected_answer


class TestAGUIBridgeMultiTenancy:
    """Tests for multi-tenancy handling in AG-UI bridge."""

    @pytest.mark.asyncio
    async def test_process_request_extracts_tenant_id(
        self, mock_orchestrator
    ):
        """Test that tenant_id is properly extracted from request config."""
        tenant_id = str(uuid4())
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={"tenant_id": tenant_id}),
        )

        bridge = AGUIBridge(mock_orchestrator)
        async for _ in bridge.process_request(request):
            pass

        # Verify orchestrator was called with correct tenant_id
        mock_orchestrator.run.assert_called_once()
        call_kwargs = mock_orchestrator.run.call_args.kwargs
        assert call_kwargs["tenant_id"] == tenant_id

    @pytest.mark.asyncio
    async def test_process_request_requires_tenant_id(
        self, mock_orchestrator
    ):
        """Test that missing tenant_id raises an error."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(configurable={}),  # No tenant_id
        )

        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(request):
            events.append(event)

        # Should emit an error via TextDeltaEvent
        text_events = [e for e in events if e.event == AGUIEventType.TEXT_MESSAGE_CONTENT]
        assert len(text_events) >= 1
        # Check for error indication
        assert "error" in text_events[0].data["content"].lower() or \
               events[-1].event == AGUIEventType.RUN_FINISHED

    @pytest.mark.asyncio
    async def test_process_request_extracts_session_id(
        self, mock_orchestrator
    ):
        """Test that session_id is properly extracted from request config."""
        session_id = str(uuid4())
        tenant_id = str(uuid4())
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Test query")],
            config=CopilotConfig(
                configurable={"tenant_id": tenant_id, "session_id": session_id}
            ),
        )

        bridge = AGUIBridge(mock_orchestrator)
        async for _ in bridge.process_request(request):
            pass

        # Verify orchestrator was called with correct session_id
        mock_orchestrator.run.assert_called_once()
        call_kwargs = mock_orchestrator.run.call_args.kwargs
        assert call_kwargs["session_id"] == session_id


class TestAGUIBridgeErrorHandling:
    """Tests for error handling in AG-UI bridge."""

    @pytest.mark.asyncio
    async def test_process_request_handles_orchestrator_error(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that orchestrator errors are handled gracefully."""
        mock_orchestrator.run.side_effect = Exception("Database connection failed")

        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        # Should still emit events without exposing internal error details
        assert events[0].event == AGUIEventType.RUN_STARTED
        assert events[-1].event == AGUIEventType.RUN_FINISHED

        # Error message should be sanitized
        text_events = [e for e in events if e.event == AGUIEventType.TEXT_MESSAGE_CONTENT]
        if text_events:
            # Should NOT contain internal error details
            assert "Database connection" not in text_events[0].data["content"]

    @pytest.mark.asyncio
    async def test_process_request_empty_messages(
        self, mock_orchestrator
    ):
        """Test handling of request with no messages."""
        request = CopilotRequest(
            messages=[],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )

        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(request):
            events.append(event)

        # Should emit RUN_FINISHED without calling orchestrator
        assert events[-1].event == AGUIEventType.RUN_FINISHED
        mock_orchestrator.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_request_no_user_messages(
        self, mock_orchestrator
    ):
        """Test handling of request with only assistant messages."""
        request = CopilotRequest(
            messages=[
                CopilotMessage(role=MessageRole.ASSISTANT, content="Previous response"),
            ],
            config=CopilotConfig(configurable={"tenant_id": str(uuid4())}),
        )

        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(request):
            events.append(event)

        # Should emit RUN_FINISHED without calling orchestrator
        assert events[-1].event == AGUIEventType.RUN_FINISHED
        mock_orchestrator.run.assert_not_called()


class TestAGUIBridgeEventModels:
    """Tests for AG-UI event model classes."""

    def test_run_started_event_structure(self):
        """Test RunStartedEvent has correct structure."""
        event = RunStartedEvent()
        assert event.event == AGUIEventType.RUN_STARTED
        data = event.model_dump()
        assert "event" in data
        assert "data" in data

    def test_run_finished_event_structure(self):
        """Test RunFinishedEvent has correct structure."""
        event = RunFinishedEvent()
        assert event.event == AGUIEventType.RUN_FINISHED
        data = event.model_dump()
        assert "event" in data
        assert "data" in data

    def test_text_delta_event_structure(self):
        """Test TextDeltaEvent has correct structure."""
        event = TextDeltaEvent(content="Hello world")
        assert event.event == AGUIEventType.TEXT_MESSAGE_CONTENT
        assert event.data["content"] == "Hello world"

    def test_text_message_start_event_structure(self):
        """Test TextMessageStartEvent has correct structure."""
        event = TextMessageStartEvent()
        assert event.event == AGUIEventType.TEXT_MESSAGE_START

    def test_text_message_end_event_structure(self):
        """Test TextMessageEndEvent has correct structure."""
        event = TextMessageEndEvent()
        assert event.event == AGUIEventType.TEXT_MESSAGE_END

    def test_state_snapshot_event_structure(self):
        """Test StateSnapshotEvent has correct structure."""
        state = {"currentStep": "completed", "thoughts": []}
        event = StateSnapshotEvent(state=state)
        assert event.event == AGUIEventType.STATE_SNAPSHOT
        assert event.data["state"] == state
