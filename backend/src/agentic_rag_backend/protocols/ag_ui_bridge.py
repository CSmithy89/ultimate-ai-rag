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
    RunStartedEvent,
    RunFinishedEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
from ..schemas import VectorCitation
from .ag_ui_metrics import AGUIMetricsCollector

logger = structlog.get_logger(__name__)

# Generic error message to avoid leaking internal details
GENERIC_ERROR_MESSAGE = "An error occurred while processing your request. Please try again."
HITL_CHECKPOINT_PREFIX = "hitl:checkpoint"
HITL_TENANT_PREFIX = "hitl:tenant"


class AGUIBridge:
    """Bridge between Agno agent responses and AG-UI protocol events."""

    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        hitl_manager: Optional["HITLManager"] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._hitl_manager = hitl_manager

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

        try:
            # Validate tenant_id is present (multi-tenancy requirement)
            if not tenant_id:
                logger.warning("copilot_request_missing_tenant_id")
                event = RunStartedEvent()
                metrics.event_emitted(event.event.value)
                yield event

                event = TextMessageStartEvent()
                metrics.event_emitted(event.event.value)
                yield event

                error_msg = "Error: tenant_id is required in request configuration."
                event = TextDeltaEvent(content=error_msg)
                metrics.event_emitted(event.event.value, len(error_msg))
                yield event

                event = TextMessageEndEvent()
                metrics.event_emitted(event.event.value)
                yield event

                event = RunFinishedEvent()
                metrics.event_emitted(event.event.value)
                yield event
                return

            # Get the latest user message
            user_message = ""
            for msg in reversed(request.messages):
                if msg.role.value == "user":
                    user_message = msg.content
                    break

            if not user_message:
                event = RunFinishedEvent()
                metrics.event_emitted(event.event.value)
                yield event
                return

            # Emit run started
            event = RunStartedEvent()
            metrics.event_emitted(event.event.value)
            yield event

            try:
                # Run the orchestrator
                result = await self._orchestrator.run(
                    query=user_message,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )

                # Format thoughts into steps for frontend useCoAgentStateRender
                steps = self._format_thought_steps(result.thoughts)

                # Emit state snapshot with steps (changed from "thoughts" key)
                event = StateSnapshotEvent(
                    state={
                        "currentStep": "completed",
                        "steps": steps,
                        "retrievalStrategy": result.retrieval_strategy.value,
                        "trajectoryId": str(result.trajectory_id) if result.trajectory_id else None,
                    }
                )
                metrics.event_emitted(event.event.value)
                yield event

                if self._hitl_manager and result.evidence and result.evidence.vector:
                    sources = self._build_hitl_sources(result.evidence.vector)
                    if sources:
                        checkpoint = await self._hitl_manager.create_checkpoint(
                            sources=sources,
                            query=user_message,
                            tenant_id=tenant_id,
                        )
                        for hitl_event in self._hitl_manager.get_checkpoint_events(checkpoint):
                            metrics.event_emitted(hitl_event.event.value)
                            yield hitl_event
                        checkpoint = await self._hitl_manager.wait_for_validation(
                            checkpoint_id=checkpoint.checkpoint_id,
                        )
                        for hitl_event in self._hitl_manager.get_completion_events(checkpoint):
                            metrics.event_emitted(hitl_event.event.value)
                            yield hitl_event

                # Stream the answer as text with proper envelope events
                event = TextMessageStartEvent()
                metrics.event_emitted(event.event.value)
                yield event

                event = TextDeltaEvent(content=result.answer)
                metrics.event_emitted(event.event.value, len(result.answer))
                yield event

                event = TextMessageEndEvent()
                metrics.event_emitted(event.event.value)
                yield event

            except Exception as e:
                # Log full error server-side but return sanitized message to client
                logger.exception("copilot_request_failed", error=str(e), tenant_id=tenant_id)
                # Issue #1 Fix: stream_error already defaults to True, no need to set again

                event = TextMessageStartEvent()
                metrics.event_emitted(event.event.value)
                yield event

                event = TextDeltaEvent(content=GENERIC_ERROR_MESSAGE)
                metrics.event_emitted(event.event.value, len(GENERIC_ERROR_MESSAGE))
                yield event

                event = TextMessageEndEvent()
                metrics.event_emitted(event.event.value)
                yield event
            else:
                # Issue #1 Fix: Only mark success if inner try completed without exception
                stream_error = False

            # Emit run finished
            event = RunFinishedEvent()
            metrics.event_emitted(event.event.value)
            yield event

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
    ) -> List[AGUIEvent]:
        """
        Get AG-UI events to signal validation completion.

        Args:
            checkpoint: The completed checkpoint

        Returns:
            List of AG-UI events to emit
        """
        return [
            ToolCallEndEvent(tool_call_id=checkpoint.checkpoint_id),
            StateSnapshotEvent(
                state={
                    "hitl_checkpoint": checkpoint.to_dict(),
                    "approved_sources": [
                        s for s in checkpoint.sources
                        if s["id"] in checkpoint.approved_source_ids
                    ],
                }
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
