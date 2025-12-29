"""AG-UI Protocol Bridge for CopilotKit integration."""

from datetime import datetime, timezone
from typing import Any, AsyncIterator
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
