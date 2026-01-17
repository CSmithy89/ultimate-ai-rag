"""AG-UI Protocol Bridge for CopilotKit integration."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Dict, List, Optional, cast

import structlog

from ..agents.orchestrator import OrchestratorAgent
from ..db.redis import RedisClient
from ..models.copilot import (
    AGUIEvent,
    CopilotRequest,
    TextDeltaEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
    RunStartedEvent,
    RunFinishedEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingEndEvent,
    ActivitySnapshotEvent,
    ActivityDeltaEvent,
    CustomEvent,
)
from ..schemas import VectorCitation
from .ag_ui_errors import create_error_event
from .ag_ui_metrics import AGUIMetricsCollector

logger = structlog.get_logger(__name__)

# Generic error message to avoid leaking internal details
GENERIC_ERROR_MESSAGE = "An error occurred while processing your request. Please try again."
HITL_CHECKPOINT_PREFIX = "hitl:checkpoint"
HITL_TENANT_PREFIX = "hitl:tenant"


class ActivityType(str, Enum):
    """Types of tracked activities for AG-UI ACTIVITY events."""
    QUERY_PROCESSING = "query_processing"
    RETRIEVAL = "retrieval"
    HITL_VALIDATION = "hitl_validation"
    RESPONSE_GENERATION = "response_generation"


# ============================================
# SUB-AGENT COMPOSITION - Phase 7.1
# ============================================


@dataclass
class SubAgentContext:
    """
    Context for sub-agent delegation with parent tracing.

    Phase 7.1: Sub-Agent Composition

    This enables nested agent delegation where:
    - Parent run context is preserved
    - Sub-agent runs are tracked separately
    - Results flow back to parent agent
    """
    parent_run_id: str
    parent_thread_id: str
    subagent_name: str
    scope: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parent_run_id": self.parent_run_id,
            "parent_thread_id": self.parent_thread_id,
            "subagent_name": self.subagent_name,
            "scope": self.scope,
            "created_at": self.created_at.isoformat(),
        }


class SubAgentManager:
    """
    Manager for sub-agent delegation and composition.

    Phase 7.1: Sub-Agent Composition

    This class manages:
    - Creating sub-agent contexts from parent runs
    - Tracking sub-agent execution
    - Aggregating results back to parent
    """

    def __init__(self) -> None:
        self._active_subagents: Dict[str, SubAgentContext] = {}
        self._logger = logger.bind(component="subagent_manager")

    def create_subagent_context(
        self,
        parent_run_id: str,
        parent_thread_id: str,
        subagent_name: str,
        scope: Optional[Dict[str, Any]] = None,
    ) -> SubAgentContext:
        """
        Create a context for delegating to a sub-agent.

        Args:
            parent_run_id: The parent run's ID
            parent_thread_id: The parent run's thread ID
            subagent_name: Name of the sub-agent to delegate to
            scope: Scoped data to pass to the sub-agent

        Returns:
            SubAgentContext for the delegation
        """
        context = SubAgentContext(
            parent_run_id=parent_run_id,
            parent_thread_id=parent_thread_id,
            subagent_name=subagent_name,
            scope=scope or {},
        )

        subagent_id = f"sub-{uuid.uuid4().hex[:12]}"
        self._active_subagents[subagent_id] = context

        self._logger.info(
            "subagent_context_created",
            subagent_id=subagent_id,
            parent_run_id=parent_run_id,
            subagent_name=subagent_name,
        )

        return context

    def get_subagent_start_events(
        self,
        context: SubAgentContext,
    ) -> List[AGUIEvent]:
        """
        Get AG-UI events for starting a sub-agent run.

        Args:
            context: The sub-agent context

        Returns:
            List of AG-UI events to emit
        """
        subagent_run_id = f"sub-{uuid.uuid4().hex[:12]}"

        return [
            RunStartedEvent(
                threadId=context.parent_thread_id,
                runId=subagent_run_id,
            ),
            StateSnapshotEvent(
                state={
                    "subagent": {
                        "name": context.subagent_name,
                        "parent_run_id": context.parent_run_id,
                        "scope": context.scope,
                    },
                },
                threadId=context.parent_thread_id,
                runId=subagent_run_id,
            ),
        ]

    def get_subagent_end_events(
        self,
        context: SubAgentContext,
        subagent_run_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> List[AGUIEvent]:
        """
        Get AG-UI events for completing a sub-agent run.

        Args:
            context: The sub-agent context
            subagent_run_id: The sub-agent's run ID
            result: Optional result from the sub-agent

        Returns:
            List of AG-UI events to emit
        """
        events: List[AGUIEvent] = []

        if result:
            events.append(
                StateSnapshotEvent(
                    state={
                        "subagent_result": {
                            "name": context.subagent_name,
                            "result": result,
                        },
                    },
                    threadId=context.parent_thread_id,
                    runId=subagent_run_id,
                )
            )

        events.append(
            RunFinishedEvent(
                threadId=context.parent_thread_id,
                runId=subagent_run_id,
            )
        )

        return events

    def cleanup_subagent(self, subagent_id: str) -> None:
        """Remove a sub-agent from tracking."""
        if subagent_id in self._active_subagents:
            del self._active_subagents[subagent_id]
            self._logger.debug("subagent_cleaned_up", subagent_id=subagent_id)

    def get_active_subagents(
        self, parent_run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active sub-agents, optionally filtered by parent run.

        Args:
            parent_run_id: Optional filter by parent run ID

        Returns:
            List of active sub-agent context dicts
        """
        result = []
        for subagent_id, context in self._active_subagents.items():
            if parent_run_id is None or context.parent_run_id == parent_run_id:
                ctx_dict = context.to_dict()
                ctx_dict["subagent_id"] = subagent_id
                result.append(ctx_dict)
        return result


class RunStatus(str, Enum):
    """Status of an agent run for Cancel/Resume support (Phase 6)."""
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class RunState:
    """
    Tracks state of an agent run for Cancel/Resume support.

    Phase 6.1: Cancel/Resume Agent Runs

    This allows users to:
    - Cancel a running agent mid-execution
    - Resume a paused/cancelled run from its last state
    """
    run_id: str
    thread_id: str
    status: RunStatus
    query: str
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_checkpoint: Optional[Dict[str, Any]] = None
    current_step: int = 0
    total_steps: int = 4
    partial_result: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "status": self.status.value,
            "query": self.query,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "partial_result": self.partial_result,
            "error_message": self.error_message,
        }


class RunManager:
    """
    Manager for cancellable and resumable agent runs.

    Phase 6.1: Cancel/Resume Agent Runs

    This class manages:
    - Tracking active runs and their state
    - Signaling cancellation to running agents
    - Persisting run state for resume capability
    - Resuming runs from checkpoints

    Thread Safety:
    - All operations on _active_runs are protected by _lock
    - This prevents race conditions when multiple requests
      access or modify run state concurrently
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        run_ttl_seconds: int = 3600,
    ):
        """
        Initialize RunManager.

        Args:
            redis_client: Optional Redis client for persistence
            run_ttl_seconds: TTL for persisted run states (default 1 hour)
        """
        self._active_runs: Dict[str, RunState] = {}
        self._redis = redis_client
        self._run_ttl_seconds = run_ttl_seconds
        self._logger = logger.bind(component="run_manager")
        self._lock = asyncio.Lock()  # Thread safety for concurrent access

    def _run_key(self, run_id: str) -> str:
        """Get Redis key for a run."""
        return f"ag_ui:run:{run_id}"

    async def _persist_run(self, run: RunState) -> None:
        """Persist run state to Redis if available."""
        if not self._redis:
            return
        payload = json.dumps(run.to_dict())
        await self._redis.client.set(
            self._run_key(run.run_id),
            payload,
            ex=self._run_ttl_seconds,
        )

    async def _load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load run state from Redis."""
        if not self._redis:
            return None
        raw = await self._redis.client.get(self._run_key(run_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def create_run(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> RunState:
        """
        Create a new run and track it.

        Args:
            query: The user query
            tenant_id: Tenant ID for multi-tenancy
            session_id: Session ID for conversation tracking

        Returns:
            The created RunState
        """
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        thread_id = f"thread-{uuid.uuid4().hex[:12]}"

        run = RunState(
            run_id=run_id,
            thread_id=thread_id,
            status=RunStatus.RUNNING,
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
        )

        async with self._lock:
            self._active_runs[run_id] = run

        self._logger.info(
            "run_created",
            run_id=run_id,
            thread_id=thread_id,
            tenant_id=tenant_id,
        )

        return run

    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get an active run by ID.

        Note: This is synchronous for compatibility, but reads are
        generally safe without the lock. For critical sections,
        use get_run_locked() instead.
        """
        return self._active_runs.get(run_id)

    async def get_run_locked(self, run_id: str) -> Optional[RunState]:
        """Get an active run by ID with lock protection."""
        async with self._lock:
            return self._active_runs.get(run_id)

    async def cancel_run(self, run_id: str) -> bool:
        """
        Cancel a running agent.

        Args:
            run_id: The run ID to cancel

        Returns:
            True if cancelled, False if run not found or already finished
        """
        async with self._lock:
            run = self._active_runs.get(run_id)
            if not run:
                self._logger.warning("cancel_run_not_found", run_id=run_id)
                return False

            if run.status != RunStatus.RUNNING:
                self._logger.info(
                    "cancel_run_not_running",
                    run_id=run_id,
                    status=run.status.value,
                )
                return False

            # Signal cancellation (atomic within lock)
            run.status = RunStatus.CANCELLED
            run.cancel_event.set()

        self._logger.info("run_cancelled", run_id=run_id)
        await self._persist_run(run)

        return True

    def is_cancelled(self, run_id: str) -> bool:
        """Check if a run has been cancelled."""
        run = self._active_runs.get(run_id)
        return run is not None and run.cancel_event.is_set()

    async def update_checkpoint(
        self,
        run_id: str,
        current_step: int,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        partial_result: Optional[str] = None,
    ) -> None:
        """
        Update the checkpoint for a run (allows resume from this point).

        Args:
            run_id: The run ID
            current_step: Current step number
            checkpoint_data: Optional checkpoint state data
            partial_result: Optional partial result generated so far
        """
        run = self._active_runs.get(run_id)
        if not run:
            return

        run.current_step = current_step
        if checkpoint_data:
            run.last_checkpoint = checkpoint_data
        if partial_result:
            run.partial_result = partial_result

        await self._persist_run(run)

    async def complete_run(
        self,
        run_id: str,
        status: RunStatus = RunStatus.COMPLETED,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Mark a run as complete.

        Args:
            run_id: The run ID
            status: Final status (COMPLETED or ERROR)
            error_message: Optional error message if status is ERROR
        """
        run = self._active_runs.get(run_id)
        if not run:
            return

        run.status = status
        if error_message:
            run.error_message = error_message

        await self._persist_run(run)
        self._logger.info(
            "run_completed",
            run_id=run_id,
            status=status.value,
        )

    async def get_resumable_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run state for potential resume.

        Args:
            run_id: The run ID to resume

        Returns:
            Run state dict if resumable, None otherwise
        """
        # Check in-memory first
        run = self._active_runs.get(run_id)
        if run:
            if run.status in (RunStatus.CANCELLED, RunStatus.PAUSED):
                return run.to_dict()
            return None

        # Check Redis for persisted state
        run_data = await self._load_run(run_id)
        if not run_data:
            return None

        # Only allow resume for cancelled or paused runs
        if run_data.get("status") in (RunStatus.CANCELLED.value, RunStatus.PAUSED.value):
            return run_data

        return None

    def cleanup_run(self, run_id: str) -> None:
        """Remove a run from active tracking."""
        if run_id in self._active_runs:
            del self._active_runs[run_id]
            self._logger.debug("run_cleaned_up", run_id=run_id)

    def get_active_runs(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active runs, optionally filtered by tenant.

        Args:
            tenant_id: Optional tenant ID filter

        Returns:
            List of active run dicts
        """
        runs = []
        for run in self._active_runs.values():
            if run.status == RunStatus.RUNNING:
                if tenant_id is None or run.tenant_id == tenant_id:
                    runs.append(run.to_dict())
        return runs


@dataclass
class ActivityState:
    """
    Tracks state of an activity for AG-UI ACTIVITY events.

    Phase 5: ACTIVITY Events support for long-running operations.
    """
    activity_id: str
    activity_type: ActivityType
    message: str
    progress: float = 0.0
    total_steps: int = 0
    current_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AG-UI event payload."""
        return {
            "id": self.activity_id,
            "type": self.activity_type.value,
            "message": self.message,
            "progress": self.progress,
            "totalSteps": self.total_steps,
            "currentStep": self.current_step,
            "metadata": self.metadata,
        }


class AGUIBridge:
    """Bridge between Agno agent responses and AG-UI protocol events."""

    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        hitl_manager: Optional["HITLManager"] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._hitl_manager = hitl_manager
        self._current_activity: Optional[ActivityState] = None

    def _create_activity_snapshot(
        self,
        activity_type: ActivityType,
        message: str,
        total_steps: int = 4,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActivitySnapshotEvent:
        """
        Create an ACTIVITY_SNAPSHOT event for long-running operations.

        Phase 5: ACTIVITY Events for progress tracking.

        Args:
            activity_type: Type of activity being performed
            message: Human-readable description of the activity
            total_steps: Total number of steps in the activity
            thread_id: AG-UI thread ID
            run_id: AG-UI run ID
            metadata: Additional metadata for the activity

        Returns:
            ActivitySnapshotEvent to emit
        """
        activity_id = f"activity-{uuid.uuid4().hex[:12]}"
        self._current_activity = ActivityState(
            activity_id=activity_id,
            activity_type=activity_type,
            message=message,
            progress=0.0,
            total_steps=total_steps,
            current_step=0,
            metadata=metadata or {},
        )
        return ActivitySnapshotEvent(
            activity=self._current_activity.to_dict(),
            threadId=thread_id,
            runId=run_id,
        )

    def _create_activity_delta(
        self,
        current_step: int,
        message: str,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ActivityDeltaEvent]:
        """
        Create an ACTIVITY_DELTA event for progress updates.

        Phase 5: ACTIVITY Events for incremental progress.

        Args:
            current_step: Current step number (1-indexed)
            message: Updated status message
            thread_id: AG-UI thread ID
            run_id: AG-UI run ID
            additional_metadata: Additional metadata to merge

        Returns:
            ActivityDeltaEvent to emit, or None if no activity is tracked
        """
        if not self._current_activity:
            return None

        # Update activity state
        self._current_activity.current_step = current_step
        self._current_activity.message = message
        self._current_activity.progress = (
            current_step / self._current_activity.total_steps
            if self._current_activity.total_steps > 0
            else 0.0
        )
        if additional_metadata:
            self._current_activity.metadata.update(additional_metadata)

        # Create RFC 6902 JSON Patch operations
        delta = [
            {"op": "replace", "path": "/currentStep", "value": current_step},
            {"op": "replace", "path": "/progress", "value": self._current_activity.progress},
            {"op": "replace", "path": "/message", "value": message},
        ]

        return ActivityDeltaEvent(
            delta=delta,
            threadId=thread_id,
            runId=run_id,
        )

    def _format_thought_steps(self, thoughts: list[Any]) -> list[dict[str, Any]]:
        """
        Format thoughts into the steps format expected by the frontend.

        Each step includes:
        - step: The step description
        - status: pending | in_progress | completed
        - timestamp: ISO 8601 formatted timestamp (optional)
        - details: Additional details for expandable view (optional)
        """
        steps = []
        for idx, thought in enumerate(thoughts):
            # Handle both string thoughts and structured thought objects
            if isinstance(thought, str):
                step_data = {
                    "step": thought,
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": None,
                }
            elif hasattr(thought, "content"):
                # Structured thought object
                step_data = {
                    "step": thought.content if hasattr(thought, "content") else str(thought),
                    "status": "completed" if getattr(thought, "completed", True) else "in_progress",
                    "timestamp": (
                        thought.timestamp.isoformat()
                        if hasattr(thought, "timestamp") and thought.timestamp
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "details": getattr(thought, "details", None),
                }
            else:
                # Fallback for unknown thought format
                step_data = {
                    "step": str(thought),
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": None,
                }
            steps.append(step_data)
        return steps

    def _build_hitl_sources(
        self, citations: list[VectorCitation]
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for citation in citations:
            sources.append(
                {
                    "id": citation.chunk_id,
                    "document_id": citation.document_id,
                    "source": citation.source,
                    "content_preview": citation.content_preview,
                    "similarity": citation.similarity,
                    "metadata": citation.metadata or {},
                }
            )
        return sources

    async def process_request(
        self, request: CopilotRequest
    ) -> AsyncIterator[AGUIEvent]:
        """
        Process a CopilotKit request and yield AG-UI events.

        Events emitted:
        - RUN_STARTED at beginning
        - STATE_SNAPSHOT for agent state updates
        - TEXT_MESSAGE_START before text content
        - TEXT_MESSAGE_CONTENT for streaming text
        - TEXT_MESSAGE_END after text content
        - RUN_FINISHED at end

        All events are tracked via Prometheus metrics (Story 22-B1).
        """
        # Extract tenant and session from config
        # Issue #1 Fix: Default to empty dict if config parsing fails
        config: dict[str, Any] = {}
        try:
            if request.config:
                config = request.config.configurable
        except AttributeError:
            # Handle case where config exists but configurable doesn't
            pass

        tenant_id = config.get("tenant_id")
        session_id = config.get("session_id")

        # Initialize metrics collector (collector handles empty/None tenant_id)
        # Issue #7 Fix: Remove redundant "or unknown" - collector handles this
        metrics = AGUIMetricsCollector(tenant_id or "")
        metrics.stream_started()

        # Issue #1 Fix: Default to error=True, only set False on explicit success path
        stream_error = True
        event: AGUIEvent

        try:
            # Validate tenant_id is present (multi-tenancy requirement)
            if not tenant_id:
                logger.warning("copilot_request_missing_tenant_id")
                event = RunStartedEvent()
                metrics.event_emitted(event.type.value)
                yield event
                from ..config import get_settings, is_development_env
                from ..core.errors import TenantRequiredError

                settings = get_settings()
                is_debug = is_development_env(settings.app_env)
                error_event = create_error_event(TenantRequiredError(), is_debug=is_debug)
                metrics.event_emitted(error_event.type.value)
                yield error_event
                # AG-UI protocol: RUN_ERROR is terminal - no events allowed after it
                return

            # Get the latest user message
            user_message = ""
            for msg in reversed(request.messages):
                if msg.role.value == "user":
                    user_message = msg.content
                    break

            if not user_message:
                event = RunStartedEvent()
                metrics.event_emitted(event.type.value)
                yield event

                event = RunFinishedEvent()
                metrics.event_emitted(event.type.value)
                yield event
                return

            # Generate consistent IDs for the entire run
            # AG-UI protocol requires same threadId/runId for RUN_STARTED and RUN_FINISHED
            run_thread_id = f"thread-{uuid.uuid4().hex[:12]}"
            run_id = f"run-{uuid.uuid4().hex[:12]}"

            # Emit run started
            run_started = RunStartedEvent(threadId=run_thread_id, runId=run_id)
            metrics.event_emitted(run_started.type.value)
            yield run_started

            # Phase 5: Emit ACTIVITY_SNAPSHOT to track request progress
            activity_snapshot = self._create_activity_snapshot(
                activity_type=ActivityType.QUERY_PROCESSING,
                message="Processing your query...",
                total_steps=4,
                thread_id=run_thread_id,
                run_id=run_id,
                metadata={"query_preview": user_message[:100] if user_message else ""},
            )
            metrics.event_emitted(activity_snapshot.type.value)
            yield activity_snapshot

            try:
                # Run the orchestrator
                result = await self._orchestrator.run(
                    query=user_message,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )

                # Phase 5: Emit ACTIVITY_DELTA for retrieval completion
                activity_delta = self._create_activity_delta(
                    current_step=1,
                    message="Retrieving relevant information...",
                    thread_id=run_thread_id,
                    run_id=run_id,
                    additional_metadata={
                        "retrieval_strategy": result.retrieval_strategy.value,
                        "evidence_count": len(result.evidence.vector) if result.evidence and result.evidence.vector else 0,
                    },
                )
                if activity_delta:
                    metrics.event_emitted(activity_delta.type.value)
                    yield activity_delta

                # AG-UI Enhancement: Emit THINKING events for agent reasoning
                # This allows the frontend to display the agent's thinking process
                if result.thoughts:
                    thinking_start = ThinkingStartEvent(
                        threadId=run_thread_id,
                        runId=run_id,
                    )
                    metrics.event_emitted(thinking_start.type.value)
                    yield thinking_start

                    # Emit each thought as thinking content
                    for thought in result.thoughts:
                        thought_content = (
                            thought if isinstance(thought, str)
                            else thought.content if hasattr(thought, "content")
                            else str(thought)
                        )
                        thinking_content = ThinkingTextMessageContentEvent(
                            content=thought_content,
                            threadId=run_thread_id,
                            runId=run_id,
                        )
                        metrics.event_emitted(thinking_content.type.value)
                        yield thinking_content

                    thinking_end = ThinkingEndEvent(
                        threadId=run_thread_id,
                        runId=run_id,
                    )
                    metrics.event_emitted(thinking_end.type.value)
                    yield thinking_end

                    # Phase 5: Emit ACTIVITY_DELTA for analysis completion
                    activity_delta = self._create_activity_delta(
                        current_step=2,
                        message="Analyzing retrieved information...",
                        thread_id=run_thread_id,
                        run_id=run_id,
                        additional_metadata={"thought_count": len(result.thoughts)},
                    )
                    if activity_delta:
                        metrics.event_emitted(activity_delta.type.value)
                        yield activity_delta

                # Format thoughts into steps for frontend useCoAgentStateRender
                steps = self._format_thought_steps(result.thoughts)

                # Emit state snapshot with steps (changed from "thoughts" key)
                # AG-UI protocol: threadId and runId required on all events
                state_event = StateSnapshotEvent(
                    state={
                        "currentStep": "completed",
                        "steps": steps,
                        "retrievalStrategy": result.retrieval_strategy.value,
                        "trajectoryId": str(result.trajectory_id) if result.trajectory_id else None,
                    },
                    threadId=run_thread_id,
                    runId=run_id,
                )
                metrics.event_emitted(state_event.type.value)
                yield state_event

                if self._hitl_manager and result.evidence and result.evidence.vector:
                    sources = self._build_hitl_sources(result.evidence.vector)
                    if sources:
                        # Phase 5: Emit ACTIVITY_DELTA for HITL validation start
                        activity_delta = self._create_activity_delta(
                            current_step=3,
                            message="Waiting for source validation...",
                            thread_id=run_thread_id,
                            run_id=run_id,
                            additional_metadata={"source_count": len(sources)},
                        )
                        if activity_delta:
                            metrics.event_emitted(activity_delta.type.value)
                            yield activity_delta

                        checkpoint = await self._hitl_manager.create_checkpoint(
                            sources=sources,
                            query=user_message,
                            tenant_id=tenant_id,
                        )
                        for hitl_event in self._hitl_manager.get_checkpoint_events(checkpoint):
                            metrics.event_emitted(hitl_event.type.value)
                            yield hitl_event
                        checkpoint = await self._hitl_manager.wait_for_validation(
                            checkpoint_id=checkpoint.checkpoint_id,
                        )
                        for hitl_event in self._hitl_manager.get_completion_events(
                            checkpoint,
                            thread_id=run_thread_id,
                            run_id=run_id,
                        ):
                            metrics.event_emitted(hitl_event.type.value)
                            yield hitl_event

                # Phase 5: Emit ACTIVITY_DELTA for response generation
                activity_delta = self._create_activity_delta(
                    current_step=4,
                    message="Generating response...",
                    thread_id=run_thread_id,
                    run_id=run_id,
                )
                if activity_delta:
                    metrics.event_emitted(activity_delta.type.value)
                    yield activity_delta

                # Stream the answer as text with proper envelope events
                # AG-UI protocol requires same messageId for START, CONTENT, and END
                message_id = f"msg-{uuid.uuid4().hex[:12]}"

                text_start = TextMessageStartEvent(messageId=message_id)
                metrics.event_emitted(text_start.type.value)
                yield text_start

                text_content = TextDeltaEvent(content=result.answer, messageId=message_id)
                metrics.event_emitted(text_content.type.value, len(result.answer))
                yield text_content

                text_end = TextMessageEndEvent(messageId=message_id)
                metrics.event_emitted(text_end.type.value)
                yield text_end

            except Exception as e:
                # Log full error server-side but return sanitized message to client
                logger.exception("copilot_request_failed", error=str(e), tenant_id=tenant_id)
                # Issue #1 Fix: stream_error already defaults to True, no need to set again

                # Story 22-B2: Emit structured AG-UI error event
                # Determine if we should include debug details (development mode only)
                from ..config import get_settings, is_development_env
                settings = get_settings()
                is_debug = is_development_env(settings.app_env)

                error_event = create_error_event(e, is_debug=is_debug)
                metrics.event_emitted(error_event.type.value)
                yield error_event
                # AG-UI protocol: RUN_ERROR is terminal - no events allowed after it
            else:
                # Issue #1 Fix: Only mark success if inner try completed without exception
                stream_error = False

                # Emit run finished (only on success path - RUN_ERROR is terminal)
                # AG-UI protocol requires same threadId/runId as RUN_STARTED
                run_finished = RunFinishedEvent(threadId=run_thread_id, runId=run_id)
                metrics.event_emitted(run_finished.type.value)
                yield run_finished

        finally:
            # Record stream completion with appropriate status
            metrics.stream_completed("error" if stream_error else "success")


# ============================================
# HITL SUPPORT - Story 6-4
# ============================================


class HITLStatus(str, Enum):
    """Status of Human-in-the-Loop validation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass
class HITLCheckpoint:
    """Represents a checkpoint waiting for human validation."""

    checkpoint_id: str
    sources: List[Dict[str, Any]]
    query: str
    tenant_id: Optional[str] = None  # Issue 2 Fix: Add tenant_id for authorization
    status: HITLStatus = HITLStatus.PENDING
    approved_source_ids: List[str] = field(default_factory=list)
    rejected_source_ids: List[str] = field(default_factory=list)
    response_event: asyncio.Event = field(default_factory=asyncio.Event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary format."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "sources": self.sources,
            "query": self.query,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "approved_source_ids": self.approved_source_ids,
            "rejected_source_ids": self.rejected_source_ids,
        }


class HITLManager:
    """
    Manager for Human-in-the-Loop validation checkpoints.

    Story 6-4: Human-in-the-Loop Source Validation

    This class manages:
    - Creating HITL checkpoints that pause generation
    - Waiting for human validation decisions
    - Processing validation responses from frontend
    - Resuming generation with approved sources only
    """

    def __init__(
        self,
        timeout: float = 300.0,
        redis_client: Optional[RedisClient] = None,
        checkpoint_ttl_seconds: int = 3600,
        history_limit: int = 100,
    ):
        """
        Initialize HITL manager.

        Args:
            timeout: Default timeout in seconds for validation (default 5 minutes)
        """
        self._pending_checkpoints: Dict[str, HITLCheckpoint] = {}
        self._hitl_timeout = timeout
        self._redis = redis_client
        self._checkpoint_ttl_seconds = checkpoint_ttl_seconds
        self._history_limit = history_limit
        self._logger = logger.bind(component="hitl_manager")

    def _checkpoint_key(self, checkpoint_id: str) -> str:
        return f"{HITL_CHECKPOINT_PREFIX}:{checkpoint_id}"

    def _tenant_key(self, tenant_id: str) -> str:
        return f"{HITL_TENANT_PREFIX}:{tenant_id}"

    async def _persist_checkpoint(
        self,
        checkpoint: HITLCheckpoint,
        record_history: bool = False,
    ) -> None:
        if not self._redis:
            return
        payload = json.dumps(checkpoint.to_dict())
        await self._redis.client.set(
            self._checkpoint_key(checkpoint.checkpoint_id),
            payload,
            ex=self._checkpoint_ttl_seconds,
        )
        if record_history and checkpoint.tenant_id:
            list_key = self._tenant_key(checkpoint.tenant_id)
            await cast(Awaitable[int], self._redis.client.lpush(list_key, checkpoint.checkpoint_id))
            await cast(
                Awaitable[int],
                self._redis.client.ltrim(list_key, 0, self._history_limit - 1),
            )
            await cast(
                Awaitable[int],
                self._redis.client.expire(list_key, self._checkpoint_ttl_seconds),
            )

    async def fetch_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if checkpoint:
            return checkpoint.to_dict()
        if not self._redis:
            return None
        raw = await self._redis.client.get(self._checkpoint_key(checkpoint_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def list_checkpoints(
        self, tenant_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        if not self._redis:
            return []
        list_key = self._tenant_key(tenant_id)
        checkpoint_ids = await cast(
            Awaitable[List[Any]],
            self._redis.client.lrange(list_key, 0, max(limit - 1, 0)),
        )
        results: List[Dict[str, Any]] = []
        for raw_id in checkpoint_ids:
            checkpoint_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            record = await self.fetch_checkpoint(checkpoint_id)
            if record:
                results.append(record)
        return results

    async def create_checkpoint(
        self,
        sources: List[Dict[str, Any]],
        query: str,
        checkpoint_id: Optional[str] = None,
        tenant_id: Optional[str] = None,  # Issue 2 Fix: Accept tenant_id
    ) -> HITLCheckpoint:
        """
        Create a HITL checkpoint for source validation.

        Args:
            sources: List of source dicts to validate
            query: The original user query for context
            checkpoint_id: Optional custom checkpoint ID
            tenant_id: Optional tenant ID for authorization

        Returns:
            The created checkpoint
        """
        checkpoint_id = checkpoint_id or str(uuid.uuid4())

        checkpoint = HITLCheckpoint(
            checkpoint_id=checkpoint_id,
            sources=sources,
            query=query,
            tenant_id=tenant_id,
        )
        self._pending_checkpoints[checkpoint_id] = checkpoint

        self._logger.info(
            "hitl_checkpoint_created",
            checkpoint_id=checkpoint_id,
            source_count=len(sources),
            tenant_id=tenant_id,
        )

        await self._persist_checkpoint(checkpoint, record_history=True)

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[HITLCheckpoint]:
        """
        Get a checkpoint by ID.

        Issue 2 Fix: Added method for authorization check.

        Args:
            checkpoint_id: The checkpoint ID

        Returns:
            The checkpoint if found, None otherwise
        """
        return self._pending_checkpoints.get(checkpoint_id)

    def get_checkpoint_events(
        self,
        checkpoint: HITLCheckpoint,
    ) -> List[AGUIEvent]:
        """
        Get AG-UI events to trigger frontend validation.

        Args:
            checkpoint: The checkpoint to create events for

        Returns:
            List of AG-UI events to emit
        """
        return [
            ToolCallStartEvent(
                tool_call_id=checkpoint.checkpoint_id,
                tool_name="validate_sources",
            ),
            ToolCallArgsEvent(
                tool_call_id=checkpoint.checkpoint_id,
                args={
                    "sources": checkpoint.sources,
                    "query": checkpoint.query,
                    "checkpoint_id": checkpoint.checkpoint_id,
                },
            ),
            # Note: We don't send ToolCallEndEvent until validation completes
        ]

    async def wait_for_validation(
        self,
        checkpoint_id: str,
        timeout: Optional[float] = None,
    ) -> HITLCheckpoint:
        """
        Wait for human validation decision on a checkpoint.

        Args:
            checkpoint_id: The checkpoint to wait for
            timeout: Optional timeout in seconds (default: configured timeout)

        Returns:
            The checkpoint with validation results

        Raises:
            asyncio.TimeoutError: If validation times out
            KeyError: If checkpoint not found
        """
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")

        timeout = timeout or self._hitl_timeout

        try:
            self._logger.info(
                "hitl_waiting_for_validation",
                checkpoint_id=checkpoint_id,
                timeout=timeout,
            )
            await asyncio.wait_for(
                checkpoint.response_event.wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # On timeout, treat as "skip" - approve all sources
            self._logger.warning(
                "hitl_validation_timeout",
                checkpoint_id=checkpoint_id,
            )
            checkpoint.status = HITLStatus.SKIPPED
            checkpoint.approved_source_ids = [s["id"] for s in checkpoint.sources]
        finally:
            await self._persist_checkpoint(checkpoint)
            # Issue 3 Fix: Always cleanup checkpoint on completion or timeout
            self.cleanup_checkpoint(checkpoint_id)

        return checkpoint

    async def receive_validation_response(
        self,
        checkpoint_id: str,
        approved_source_ids: List[str],
    ) -> HITLCheckpoint:
        """
        Receive validation response from frontend.

        Args:
            checkpoint_id: The checkpoint being responded to
            approved_source_ids: List of approved source IDs

        Returns:
            Updated checkpoint

        Raises:
            KeyError: If checkpoint not found
        """
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")

        # Update checkpoint with decisions
        all_source_ids = {s["id"] for s in checkpoint.sources}
        checkpoint.approved_source_ids = approved_source_ids
        checkpoint.rejected_source_ids = list(
            all_source_ids - set(approved_source_ids)
        )
        checkpoint.status = (
            HITLStatus.APPROVED if approved_source_ids else HITLStatus.REJECTED
        )

        self._logger.info(
            "hitl_validation_received",
            checkpoint_id=checkpoint_id,
            approved_count=len(approved_source_ids),
            rejected_count=len(checkpoint.rejected_source_ids),
            status=checkpoint.status.value,
        )

        # Signal waiting coroutine
        checkpoint.response_event.set()

        await self._persist_checkpoint(checkpoint)

        return checkpoint

    def get_completion_events(
        self,
        checkpoint: HITLCheckpoint,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[AGUIEvent]:
        """
        Get AG-UI events to signal validation completion.

        AG-UI Protocol Enhancement: TOOL_CALL_RESULT is now emitted before
        TOOL_CALL_END to provide tool execution results.

        Args:
            checkpoint: The completed checkpoint
            thread_id: Thread ID for consistent event correlation
            run_id: Run ID for consistent event correlation

        Returns:
            List of AG-UI events to emit
        """
        # Prepare the validation result
        approved_sources = [
            s for s in checkpoint.sources
            if s["id"] in checkpoint.approved_source_ids
        ]
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
                    "approved_sources": approved_sources,
                },
                threadId=thread_id,
                runId=run_id,
            ),
        ]

    def get_approved_sources(
        self,
        checkpoint_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get the approved sources from a completed checkpoint.

        Args:
            checkpoint_id: The completed checkpoint

        Returns:
            List of approved source dicts
        """
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            return []

        return [
            s for s in checkpoint.sources
            if s["id"] in checkpoint.approved_source_ids
        ]

    def cleanup_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a checkpoint from memory."""
        if checkpoint_id in self._pending_checkpoints:
            del self._pending_checkpoints[checkpoint_id]
            self._logger.debug(
                "hitl_checkpoint_cleaned",
                checkpoint_id=checkpoint_id,
            )


# Helper function to create HITL events without manager
def create_validate_sources_events(
    sources: List[Dict[str, Any]],
    query: str,
    checkpoint_id: Optional[str] = None,
) -> List[AGUIEvent]:
    """
    Create AG-UI events to trigger source validation on frontend.

    This is a convenience function for triggering HITL validation
    without using the full HITLManager.

    Args:
        sources: List of source dictionaries
        query: The original user query
        checkpoint_id: Optional checkpoint ID

    Returns:
        List of AG-UI events to emit
    """
    checkpoint_id = checkpoint_id or str(uuid.uuid4())

    return [
        ToolCallStartEvent(
            tool_call_id=checkpoint_id,
            tool_name="validate_sources",
        ),
        ToolCallArgsEvent(
            tool_call_id=checkpoint_id,
            args={
                "sources": sources,
                "query": query,
                "checkpoint_id": checkpoint_id,
            },
        ),
    ]
