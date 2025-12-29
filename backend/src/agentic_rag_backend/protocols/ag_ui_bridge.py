"""AG-UI Protocol Bridge for CopilotKit integration."""

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

            # Emit state snapshot with thoughts
            yield StateSnapshotEvent(
                state={
                    "currentStep": "completed",
                    "thoughts": [
                        {"step": t, "status": "completed"}
                        for t in result.thoughts
                    ],
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
