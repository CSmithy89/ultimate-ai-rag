"""AG-UI Protocol Bridge for CopilotKit integration."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from ..agents.orchestrator import OrchestratorAgent
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

logger = structlog.get_logger(__name__)

# Generic error message to avoid leaking internal details
GENERIC_ERROR_MESSAGE = "An error occurred while processing your request. Please try again."


class AGUIBridge:
    """Bridge between Agno agent responses and AG-UI protocol events."""

    def __init__(self, orchestrator: OrchestratorAgent) -> None:
        self._orchestrator = orchestrator

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
        """
        # Extract tenant and session from config
        config: dict[str, Any] = {}
        if request.config:
            config = request.config.configurable

        tenant_id = config.get("tenant_id")
        session_id = config.get("session_id")

        # Validate tenant_id is present (multi-tenancy requirement)
        if not tenant_id:
            logger.warning("copilot_request_missing_tenant_id")
            yield RunStartedEvent()
            yield TextMessageStartEvent()
            yield TextDeltaEvent(content="Error: tenant_id is required in request configuration.")
            yield TextMessageEndEvent()
            yield RunFinishedEvent()
            return

        # Get the latest user message
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role.value == "user":
                user_message = msg.content
                break

        if not user_message:
            yield RunFinishedEvent()
            return

        # Emit run started
        yield RunStartedEvent()

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
            yield StateSnapshotEvent(
                state={
                    "currentStep": "completed",
                    "steps": steps,
                    "retrievalStrategy": result.retrieval_strategy.value,
                    "trajectoryId": str(result.trajectory_id) if result.trajectory_id else None,
                }
            )

            # Stream the answer as text with proper envelope events
            yield TextMessageStartEvent()
            yield TextDeltaEvent(content=result.answer)
            yield TextMessageEndEvent()

        except Exception as e:
            # Log full error server-side but return sanitized message to client
            logger.exception("copilot_request_failed", error=str(e), tenant_id=tenant_id)
            yield TextMessageStartEvent()
            yield TextDeltaEvent(content=GENERIC_ERROR_MESSAGE)
            yield TextMessageEndEvent()

        # Emit run finished
        yield RunFinishedEvent()


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

    def __init__(self, timeout: float = 300.0):
        """
        Initialize HITL manager.

        Args:
            timeout: Default timeout in seconds for validation (default 5 minutes)
        """
        self._pending_checkpoints: Dict[str, HITLCheckpoint] = {}
        self._hitl_timeout = timeout
        self._logger = logger.bind(component="hitl_manager")

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
            # Issue 3 Fix: Always cleanup checkpoint on completion or timeout
            self.cleanup_checkpoint(checkpoint_id)

        return checkpoint

    def receive_validation_response(
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
