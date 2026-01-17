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
    StateDeltaEvent,
    TextDeltaEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolCallResultEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingEndEvent,
    ActivitySnapshotEvent,
    ActivityDeltaEvent,
    MessagesSnapshotEvent,
    CustomEvent,
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
        checkpoint.rejected_source_ids = []
        checkpoint.status = HITLStatus.APPROVED
        return checkpoint

    def get_completion_events(self, checkpoint, thread_id=None, run_id=None):
        import json
        validation_result = {
            "status": checkpoint.status.value,
            "approved_count": len(checkpoint.approved_source_ids),
            "rejected_count": len(checkpoint.rejected_source_ids),
            "approved_source_ids": checkpoint.approved_source_ids,
        }
        return [
            # AG-UI Enhancement: Emit TOOL_CALL_RESULT before END
            ToolCallResultEvent(
                tool_call_id=checkpoint.checkpoint_id,
                result=json.dumps(validation_result),
            ),
            ToolCallEndEvent(tool_call_id=checkpoint.checkpoint_id),
            StateSnapshotEvent(
                state={
                    "hitl_checkpoint": checkpoint.to_dict(),
                    "approved_sources": checkpoint.sources,
                },
                threadId=thread_id,
                runId=run_id,
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
        assert events[0].type == AGUIEventType.RUN_STARTED
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
        assert events[-1].type == AGUIEventType.RUN_FINISHED
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

        state_events = [e for e in events if e.type == AGUIEventType.STATE_SNAPSHOT]
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

        event_types = [event.type for event in events]
        assert AGUIEventType.TOOL_CALL_START in event_types
        assert AGUIEventType.TOOL_CALL_ARGS in event_types
        assert AGUIEventType.TOOL_CALL_END in event_types
        state_events = [e for e in events if e.type == AGUIEventType.STATE_SNAPSHOT]
        assert len(state_events) == 2
        assert any(isinstance(event, StateSnapshotEvent) for event in state_events)
        assert any("currentStep" in event.snapshot for event in state_events)
        assert any("hitl_checkpoint" in event.snapshot for event in state_events)

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
        text_start = [e for e in events if e.type == AGUIEventType.TEXT_MESSAGE_START]
        text_content = [e for e in events if e.type == AGUIEventType.TEXT_MESSAGE_CONTENT]
        text_end = [e for e in events if e.type == AGUIEventType.TEXT_MESSAGE_END]

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

        text_events = [e for e in events if e.type == AGUIEventType.TEXT_MESSAGE_CONTENT]
        assert len(text_events) == 1
        assert text_events[0].delta == expected_answer


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

        # Should emit an AGUIErrorEvent with TENANT_REQUIRED code
        error_events = [e for e in events if e.type == AGUIEventType.RUN_ERROR]
        assert len(error_events) >= 1
        # Check for TENANT_REQUIRED error code
        assert error_events[0].code == "TENANT_REQUIRED"
        # AG-UI protocol: RUN_ERROR is terminal, RUN_FINISHED should NOT follow
        assert events[-1].type == AGUIEventType.RUN_ERROR

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
        assert events[0].type == AGUIEventType.RUN_STARTED

        # Should emit error event with sanitized message
        error_events = [e for e in events if e.type == AGUIEventType.RUN_ERROR]
        assert len(error_events) == 1
        # Error message should NOT expose internal details
        assert "Database connection" not in error_events[0].message
        assert error_events[0].code == "AGENT_EXECUTION_ERROR"

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
        assert events[-1].type == AGUIEventType.RUN_FINISHED
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
        assert events[-1].type == AGUIEventType.RUN_FINISHED
        mock_orchestrator.run.assert_not_called()


class TestAGUIBridgeEventModels:
    """Tests for AG-UI event model classes."""

    def test_run_started_event_structure(self):
        """Test RunStartedEvent has correct structure."""
        event = RunStartedEvent()
        assert event.type == AGUIEventType.RUN_STARTED
        data = event.model_dump()
        assert "type" in data
        assert "threadId" in data
        assert "runId" in data

    def test_run_finished_event_structure(self):
        """Test RunFinishedEvent has correct structure."""
        event = RunFinishedEvent()
        assert event.type == AGUIEventType.RUN_FINISHED
        data = event.model_dump()
        assert "type" in data
        assert "threadId" in data
        assert "runId" in data

    def test_text_delta_event_structure(self):
        """Test TextDeltaEvent has correct structure."""
        event = TextDeltaEvent(content="Hello world")
        assert event.type == AGUIEventType.TEXT_MESSAGE_CONTENT
        assert event.delta == "Hello world"

    def test_text_message_start_event_structure(self):
        """Test TextMessageStartEvent has correct structure."""
        event = TextMessageStartEvent()
        assert event.type == AGUIEventType.TEXT_MESSAGE_START

    def test_text_message_end_event_structure(self):
        """Test TextMessageEndEvent has correct structure."""
        event = TextMessageEndEvent()
        assert event.type == AGUIEventType.TEXT_MESSAGE_END

    def test_state_snapshot_event_structure(self):
        """Test StateSnapshotEvent has correct structure."""
        state = {"currentStep": "completed", "thoughts": []}
        event = StateSnapshotEvent(state=state)
        assert event.type == AGUIEventType.STATE_SNAPSHOT
        assert event.snapshot == state


class TestAGUIEnhancedEventModels:
    """Tests for AG-UI Protocol Enhancement event models."""

    def test_state_delta_event_structure(self):
        """Test StateDeltaEvent has correct RFC 6902 structure."""
        delta = [
            {"op": "replace", "path": "/steps/0/status", "value": "completed"},
            {"op": "add", "path": "/steps/-", "value": {"step": "New step"}},
        ]
        event = StateDeltaEvent(delta=delta)
        assert event.type == AGUIEventType.STATE_DELTA
        assert event.delta == delta
        data = event.model_dump()
        assert "threadId" in data
        assert "runId" in data
        assert "delta" in data

    def test_tool_call_result_event_structure(self):
        """Test ToolCallResultEvent has correct structure."""
        event = ToolCallResultEvent(
            tool_call_id="call-123",
            result='{"status": "success", "data": [1, 2, 3]}',
        )
        assert event.type == AGUIEventType.TOOL_CALL_RESULT
        assert event.toolCallId == "call-123"
        assert "success" in event.result

    def test_messages_snapshot_event_structure(self):
        """Test MessagesSnapshotEvent has correct structure."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        event = MessagesSnapshotEvent(messages=messages)
        assert event.type == AGUIEventType.MESSAGES_SNAPSHOT
        assert event.messages == messages

    def test_activity_snapshot_event_structure(self):
        """Test ActivitySnapshotEvent has correct structure."""
        activity = {
            "id": "activity-123",
            "type": "indexing",
            "progress": 0.0,
            "message": "Starting document processing...",
        }
        event = ActivitySnapshotEvent(activity=activity)
        assert event.type == AGUIEventType.ACTIVITY_SNAPSHOT
        assert event.activity == activity

    def test_activity_delta_event_structure(self):
        """Test ActivityDeltaEvent has correct RFC 6902 structure."""
        delta = [
            {"op": "replace", "path": "/progress", "value": 0.45},
            {"op": "replace", "path": "/message", "value": "Processing page 3/7..."},
        ]
        event = ActivityDeltaEvent(delta=delta)
        assert event.type == AGUIEventType.ACTIVITY_DELTA
        assert event.delta == delta

    def test_thinking_start_event_structure(self):
        """Test ThinkingStartEvent has correct structure."""
        event = ThinkingStartEvent(threadId="thread-123", runId="run-456")
        assert event.type == AGUIEventType.THINKING_START
        assert event.threadId == "thread-123"
        assert event.runId == "run-456"

    def test_thinking_text_message_content_event_structure(self):
        """Test ThinkingTextMessageContentEvent has correct structure."""
        event = ThinkingTextMessageContentEvent(
            content="Analyzing the query to determine retrieval strategy...",
            threadId="thread-123",
            runId="run-456",
        )
        assert event.type == AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT
        assert "retrieval strategy" in event.content

    def test_thinking_end_event_structure(self):
        """Test ThinkingEndEvent has correct structure."""
        event = ThinkingEndEvent(threadId="thread-123", runId="run-456")
        assert event.type == AGUIEventType.THINKING_END
        assert event.threadId == "thread-123"

    def test_custom_event_structure(self):
        """Test CustomEvent has correct structure."""
        event = CustomEvent(
            name="render_ui",
            value={
                "type": "approval_dialog",
                "props": {"title": "Approve Sources", "items": []},
            },
        )
        assert event.type == AGUIEventType.CUSTOM
        assert event.name == "render_ui"
        assert event.value["type"] == "approval_dialog"


class TestAGUIThinkingEventsEmission:
    """Tests for THINKING events emission in AG-UI bridge."""

    @pytest.mark.asyncio
    async def test_process_request_emits_thinking_events(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that THINKING events are emitted for orchestrator thoughts."""
        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        # Check for THINKING_START
        thinking_start_events = [e for e in events if e.type == AGUIEventType.THINKING_START]
        assert len(thinking_start_events) == 1

        # Check for THINKING_TEXT_MESSAGE_CONTENT
        thinking_content_events = [e for e in events if e.type == AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT]
        assert len(thinking_content_events) >= 1

        # Check for THINKING_END
        thinking_end_events = [e for e in events if e.type == AGUIEventType.THINKING_END]
        assert len(thinking_end_events) == 1

        # Verify order: START before CONTENT before END
        start_idx = events.index(thinking_start_events[0])
        content_idx = events.index(thinking_content_events[0])
        end_idx = events.index(thinking_end_events[0])
        assert start_idx < content_idx < end_idx

    @pytest.mark.asyncio
    async def test_thinking_events_contain_thought_content(
        self, mock_orchestrator, sample_copilot_request
    ):
        """Test that THINKING_TEXT_MESSAGE_CONTENT contains thought text."""
        expected_thoughts = ["Analyzing query", "Determining retrieval strategy"]
        mock_orchestrator.run.return_value = MockOrchestratorResult(thoughts=expected_thoughts)

        bridge = AGUIBridge(mock_orchestrator)
        events = []

        async for event in bridge.process_request(sample_copilot_request):
            events.append(event)

        thinking_content_events = [
            e for e in events
            if e.type == AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT
        ]

        # Should have one content event per thought
        assert len(thinking_content_events) == len(expected_thoughts)

        # Verify content matches thoughts
        for i, event in enumerate(thinking_content_events):
            assert event.content == expected_thoughts[i]


class TestAGUIToolCallResultEmission:
    """Tests for TOOL_CALL_RESULT events in HITL flow."""

    @pytest.mark.asyncio
    async def test_hitl_emits_tool_call_result(self, sample_copilot_request):
        """Test that HITL completion emits TOOL_CALL_RESULT before TOOL_CALL_END."""
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

        # Check for TOOL_CALL_RESULT
        result_events = [e for e in events if e.type == AGUIEventType.TOOL_CALL_RESULT]
        assert len(result_events) == 1

        # Check for TOOL_CALL_END
        end_events = [e for e in events if e.type == AGUIEventType.TOOL_CALL_END]
        assert len(end_events) == 1

        # Verify order: RESULT before END
        result_idx = events.index(result_events[0])
        end_idx = events.index(end_events[0])
        assert result_idx < end_idx

    @pytest.mark.asyncio
    async def test_tool_call_result_contains_validation_data(self, sample_copilot_request):
        """Test that TOOL_CALL_RESULT contains validation result data."""
        import json

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

        result_events = [e for e in events if e.type == AGUIEventType.TOOL_CALL_RESULT]
        assert len(result_events) == 1

        # Parse and verify result content
        result_data = json.loads(result_events[0].result)
        assert "status" in result_data
        assert "approved_count" in result_data
        assert result_data["status"] == "approved"
