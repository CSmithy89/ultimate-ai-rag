"""CopilotKit AG-UI protocol endpoint.

Story 21-E1: Voice Input (Speech-to-Text)
Story 21-E2: Voice Output (Text-to-Speech)
"""

import re
from datetime import datetime, timezone
from typing import Any, List, Optional


from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
import structlog

from ...agents.orchestrator import OrchestratorAgent
from ...api.utils import rate_limit_exceeded
from ...models.copilot import CopilotRequest
from ...protocols.ag_ui_bridge import AGUIBridge
from ...rate_limit import RateLimiter
from ...voice import VoiceAdapter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/copilot", tags=["copilot"])


def get_orchestrator(request: Request) -> OrchestratorAgent:
    """Get orchestrator from app state."""
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


def get_voice_adapter(request: Request) -> Optional[VoiceAdapter]:
    """Get voice adapter from app state.

    Story 21-E1, 21-E2: Voice I/O endpoints.
    """
    return getattr(request.app.state, "voice_adapter", None)


@router.post("")
async def copilot_handler(
    request: CopilotRequest,
    http_request: Request,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> StreamingResponse:
    """
    Handle AG-UI protocol requests from CopilotKit.

    Returns SSE stream with AG-UI events:
    - text_delta: Streaming text responses
    - tool_call: Agent tool invocations
    - state_snapshot: Agent state updates
    - action_request: Frontend action requests
    """
    # Extract tenant_id: prefer header, fall back to config
    tenant_id = x_tenant_id or "anonymous"
    if tenant_id == "anonymous" and request.config and request.config.configurable:
        tenant_id = request.config.configurable.get("tenant_id", "anonymous")

    # Inject tenant_id into request config for downstream processing
    if request.config is None:
        from ...models.copilot import CopilotConfig
        request.config = CopilotConfig()
    if not request.config.configurable:
        request.config.configurable = {}
    request.config.configurable["tenant_id"] = tenant_id

    # Check rate limit
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    bridge = AGUIBridge(orchestrator, hitl_manager=get_hitl_manager(http_request))

    async def event_generator():
        try:
            async for event in bridge.process_request(request):
                yield f"data: {event.model_dump_json()}\n\n"
        except Exception as exc:
            # Fallback if bridge fails before emitting error events - ensure terminal event
            logger.exception("copilot_stream_failed", error=str(exc))
            from ...models.copilot import (
                RunFinishedEvent as RunFinished,
                TextDeltaEvent,
                TextMessageEndEvent,
                TextMessageStartEvent,
            )
            yield f"data: {TextMessageStartEvent().model_dump_json()}\n\n"
            yield f"data: {TextDeltaEvent(content='An error occurred while processing your request.').model_dump_json()}\n\n"
            yield f"data: {TextMessageEndEvent().model_dump_json()}\n\n"
            yield f"data: {RunFinished().model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================
# HITL VALIDATION ENDPOINT - Story 6-4
# ============================================

# UUID4 regex pattern for validation
UUID4_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


class ValidationResponseRequest(BaseModel):
    """Request body for HITL validation response."""

    checkpoint_id: str = Field(..., description="ID of the checkpoint being responded to (UUID format)")
    approved_source_ids: List[str] = Field(
        default_factory=list,
        description="List of approved source IDs"
    )

    # Issue 8 Fix: Add UUID validation to checkpoint_id
    @field_validator('checkpoint_id')
    @classmethod
    def validate_checkpoint_id(cls, v: str) -> str:
        """Validate that checkpoint_id is a valid UUID4."""
        if not UUID4_PATTERN.match(v):
            raise ValueError('checkpoint_id must be a valid UUID4')
        return v


class ValidationResponseResult(BaseModel):
    """Response for HITL validation endpoint."""

    checkpoint_id: str
    status: str
    approved_count: int
    rejected_count: int


class HITLCheckpointResponse(BaseModel):
    """Response payload for HITL checkpoint queries."""

    checkpoint_id: str
    status: str
    query: str
    tenant_id: Optional[str] = None
    sources: List[dict[str, Any]]
    approved_source_ids: List[str]
    rejected_source_ids: List[str]


def get_hitl_manager(request: Request):
    """Get HITL manager from app state."""
    return getattr(request.app.state, "hitl_manager", None)


def get_tenant_id_from_header(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> Optional[str]:
    """Extract tenant_id from request header."""
    return x_tenant_id


@router.post("/validation-response", response_model=ValidationResponseResult)
async def receive_validation_response(
    request_body: ValidationResponseRequest,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> ValidationResponseResult:
    """
    Receive Human-in-the-Loop validation response from frontend.

    Story 6-4: Human-in-the-Loop Source Validation

    This endpoint receives the user's approval/rejection decisions
    and signals the waiting agent to continue with approved sources.
    """
    hitl_manager = get_hitl_manager(request)

    if hitl_manager is None:
        # If no HITL manager is configured, return a mock response
        # This allows the endpoint to work for testing even without full setup
        logger.warning(
            "hitl_manager_not_configured",
            checkpoint_id=request_body.checkpoint_id,
        )
        return ValidationResponseResult(
            checkpoint_id=request_body.checkpoint_id,
            status="approved" if request_body.approved_source_ids else "rejected",
            approved_count=len(request_body.approved_source_ids),
            rejected_count=0,
        )

    try:
        # Issue 2 Fix: Verify tenant authorization
        # Get the checkpoint first to check tenant ownership
        checkpoint = hitl_manager.get_checkpoint(request_body.checkpoint_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {request_body.checkpoint_id} not found"
            )

        # Verify tenant_id matches if checkpoint has tenant_id
        checkpoint_tenant = getattr(checkpoint, 'tenant_id', None)
        if checkpoint_tenant and tenant_id and checkpoint_tenant != tenant_id:
            logger.warning(
                "hitl_tenant_mismatch",
                checkpoint_id=request_body.checkpoint_id,
                checkpoint_tenant=checkpoint_tenant,
                request_tenant=tenant_id,
            )
            raise HTTPException(
                status_code=403,
                detail="Not authorized to respond to this checkpoint"
            )

        checkpoint = await hitl_manager.receive_validation_response(
            checkpoint_id=request_body.checkpoint_id,
            approved_source_ids=request_body.approved_source_ids,
        )

        return ValidationResponseResult(
            checkpoint_id=checkpoint.checkpoint_id,
            status=checkpoint.status.value,
            approved_count=len(checkpoint.approved_source_ids),
            rejected_count=len(checkpoint.rejected_source_ids),
        )

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {request_body.checkpoint_id} not found"
        )


@router.get("/hitl/checkpoints/{checkpoint_id}", response_model=HITLCheckpointResponse)
async def get_hitl_checkpoint(
    checkpoint_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> HITLCheckpointResponse:
    """Get a HITL checkpoint by ID."""
    hitl_manager = get_hitl_manager(request)
    if hitl_manager is None:
        raise HTTPException(status_code=503, detail="HITL manager not configured")

    record = await hitl_manager.fetch_checkpoint(checkpoint_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    record_tenant = record.get("tenant_id")
    if record_tenant and tenant_id and record_tenant != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this checkpoint")

    return HITLCheckpointResponse(**record)


@router.get("/hitl/checkpoints", response_model=List[HITLCheckpointResponse])
async def list_hitl_checkpoints(
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
    limit: int = Query(20, ge=1, le=100),
) -> List[HITLCheckpointResponse]:
    """List HITL checkpoints for a tenant."""
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")
    hitl_manager = get_hitl_manager(request)
    if hitl_manager is None:
        raise HTTPException(status_code=503, detail="HITL manager not configured")

    records = await hitl_manager.list_checkpoints(tenant_id, limit=limit)
    return [HITLCheckpointResponse(**record) for record in records]


# ============================================
# RUN CONTROL ENDPOINTS - Phase 6.1
# ============================================


def get_run_manager(request: Request):
    """Get RunManager from app state."""
    return getattr(request.app.state, "run_manager", None)


class CancelRunResponse(BaseModel):
    """Response for cancel run endpoint."""

    run_id: str
    cancelled: bool
    message: str


class RunStateResponse(BaseModel):
    """Response for run state endpoint."""

    run_id: str
    thread_id: str
    status: str
    query: str
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: str
    current_step: int
    total_steps: int
    partial_result: Optional[str] = None
    error_message: Optional[str] = None


class ActiveRunsResponse(BaseModel):
    """Response for list active runs endpoint."""

    runs: List[RunStateResponse]
    count: int


@router.post("/cancel/{run_id}", response_model=CancelRunResponse)
async def cancel_run(
    run_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> CancelRunResponse:
    """
    Cancel a running agent.

    Phase 6.1: Cancel/Resume Agent Runs

    This endpoint signals the running agent to stop execution.
    The agent will emit a RUN_FINISHED event with cancelled status.
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    # Get run to verify tenant authorization
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Verify tenant authorization
    if run.tenant_id and tenant_id and run.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to cancel this run")

    success = await run_manager.cancel_run(run_id)

    return CancelRunResponse(
        run_id=run_id,
        cancelled=success,
        message="Run cancelled successfully" if success else "Run could not be cancelled",
    )


@router.post("/resume/{run_id}", response_model=RunStateResponse)
async def resume_run(
    run_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> RunStateResponse:
    """
    Resume a cancelled or paused run.

    Phase 6.1: Cancel/Resume Agent Runs

    This endpoint resumes a run from its last checkpoint.
    Note: Full resume implementation requires orchestrator support.
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    # Get resumable run state
    run_data = await run_manager.get_resumable_run(run_id)
    if run_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found or not resumable"
        )

    # Verify tenant authorization
    if run_data.get("tenant_id") and tenant_id and run_data["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to resume this run")

    # Return run state (actual resume would need orchestrator integration)
    return RunStateResponse(**run_data)


@router.get("/run/{run_id}", response_model=RunStateResponse)
async def get_run_state(
    run_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> RunStateResponse:
    """
    Get the state of a run.

    Phase 6.1: Cancel/Resume Agent Runs
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    run = run_manager.get_run(run_id)
    if run is None:
        # Try loading from persistence
        run_data = await run_manager._load_run(run_id)
        if run_data is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Verify tenant authorization
        if run_data.get("tenant_id") and tenant_id and run_data["tenant_id"] != tenant_id:
            raise HTTPException(status_code=403, detail="Not authorized to view this run")

        return RunStateResponse(**run_data)

    # Verify tenant authorization
    if run.tenant_id and tenant_id and run.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this run")

    return RunStateResponse(**run.to_dict())


@router.get("/runs", response_model=ActiveRunsResponse)
async def list_active_runs(
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> ActiveRunsResponse:
    """
    List active runs for a tenant.

    Phase 6.1: Cancel/Resume Agent Runs
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    runs = run_manager.get_active_runs(tenant_id=tenant_id)

    return ActiveRunsResponse(
        runs=[RunStateResponse(**run) for run in runs],
        count=len(runs),
    )


# ============================================
# AGENT STEERING ENDPOINTS - Phase 6.2
# ============================================


class SteeringRequest(BaseModel):
    """Request for agent steering."""

    run_id: str = Field(..., description="ID of the run to steer")
    instruction: str = Field(
        ...,
        description="Steering instruction to inject into the agent",
        min_length=1,
        max_length=2000,
    )
    context: Optional[dict[str, Any]] = Field(
        None,
        description="Additional context for the steering instruction",
    )


class SteeringResponse(BaseModel):
    """Response for agent steering."""

    run_id: str
    status: str
    message: str


@router.post("/steer", response_model=SteeringResponse)
async def steer_agent(
    steering: SteeringRequest,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> SteeringResponse:
    """
    Inject steering guidance into a running agent.

    Phase 6.2: Agent Steering API

    This endpoint allows users to redirect agent execution mid-flow
    by injecting additional instructions or guidance.

    Note: Full steering implementation requires orchestrator support
    to incorporate the steering instruction into the agent's decision-making.
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    # Get run to verify it exists and is running
    run = run_manager.get_run(steering.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {steering.run_id} not found")

    # Verify tenant authorization
    if run.tenant_id and tenant_id and run.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to steer this run")

    # Check if run is still active
    from ...protocols.ag_ui_bridge import RunStatus
    if run.status != RunStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot steer run in '{run.status.value}' status"
        )

    # Log the steering instruction
    logger.info(
        "agent_steering_received",
        run_id=steering.run_id,
        instruction_length=len(steering.instruction),
        has_context=steering.context is not None,
    )

    # Store steering instruction in run state for orchestrator to pick up
    # The orchestrator would check for steering instructions at checkpoints
    if run.last_checkpoint is None:
        run.last_checkpoint = {}
    run.last_checkpoint["steering"] = {
        "instruction": steering.instruction,
        "context": steering.context,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await run_manager._persist_run(run)

    return SteeringResponse(
        run_id=steering.run_id,
        status="steering_applied",
        message="Steering instruction received and will be applied at next checkpoint",
    )


@router.get("/steer/{run_id}", response_model=Optional[dict[str, Any]])
async def get_steering_instruction(
    run_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> Optional[dict[str, Any]]:
    """
    Get pending steering instruction for a run.

    Phase 6.2: Agent Steering API

    Used by the orchestrator to check for steering instructions.
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Verify tenant authorization
    if run.tenant_id and tenant_id and run.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this run")

    if run.last_checkpoint and "steering" in run.last_checkpoint:
        return run.last_checkpoint["steering"]

    return None


@router.delete("/steer/{run_id}")
async def clear_steering_instruction(
    run_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> dict[str, str]:
    """
    Clear steering instruction after it has been applied.

    Phase 6.2: Agent Steering API

    Called by the orchestrator after processing a steering instruction.
    """
    run_manager = get_run_manager(request)

    if run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Verify tenant authorization
    if run.tenant_id and tenant_id and run.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this run")

    if run.last_checkpoint and "steering" in run.last_checkpoint:
        del run.last_checkpoint["steering"]
        await run_manager._persist_run(run)

    return {"status": "cleared"}


# ============================================
# VOICE I/O ENDPOINTS - Stories 21-E1, 21-E2
# ============================================

# Allowed audio content types for transcription
ALLOWED_AUDIO_TYPES = frozenset({
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
})

# Maximum audio file size (25MB) - prevents memory exhaustion DoS
MAX_AUDIO_SIZE = 25 * 1024 * 1024


class TranscriptionResponse(BaseModel):
    """Response for audio transcription.

    Story 21-E1: Implement Voice Input (STT).
    """

    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected or specified language")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")


class TTSRequest(BaseModel):
    """Request for text-to-speech synthesis.

    Story 21-E2: Implement Voice Output (TTS).
    """

    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=4096)
    voice: Optional[str] = Field(None, description="Voice to use (alloy, echo, fable, onyx, nova, shimmer)")
    speed: Optional[float] = Field(None, ge=0.25, le=4.0, description="Speech speed multiplier")

    @field_validator("text")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """Sanitize text to prevent injection attacks.

        Removes control characters that could be interpreted as commands.
        Raises ValueError if text becomes empty after sanitization.
        """
        # Remove control characters except newlines and tabs
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", v)
        v = v.strip()
        if not v:
            raise ValueError("Text must not be empty after sanitization")
        return v


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(default="en", description="ISO 639-1 language code hint"),
    voice_adapter: Optional[VoiceAdapter] = Depends(get_voice_adapter),
    limiter: RateLimiter = Depends(get_rate_limiter),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> TranscriptionResponse:
    """Transcribe audio to text using configured transcription service.

    Story 21-E1: Implement Voice Input (Speech-to-Text)

    Accepts audio files (webm, wav, mp3, etc.) and returns transcribed text.
    Uses Whisper for transcription with optional language hints.

    Args:
        audio: Audio file upload
        language: ISO 639-1 language code hint (default: "en")

    Returns:
        TranscriptionResponse with text, language, and confidence

    Raises:
        403: Voice I/O is disabled
        413: File too large
        415: Unsupported media type
        429: Rate limit exceeded
        503: Voice adapter not configured
    """
    # Rate limiting
    tenant_id = x_tenant_id or "anonymous"
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    if voice_adapter is None:
        raise HTTPException(status_code=503, detail="Voice adapter not configured")

    if not voice_adapter.enabled:
        raise HTTPException(status_code=403, detail="Voice I/O is disabled")

    # Validate audio content type
    content_type = audio.content_type or ""
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. Allowed types: {', '.join(sorted(ALLOWED_AUDIO_TYPES))}",
        )

    try:
        # Read audio data with size limit to prevent memory exhaustion DoS
        # Read one byte more than limit to detect oversized files
        audio_data = await audio.read(MAX_AUDIO_SIZE + 1)

        if len(audio_data) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Maximum size: {MAX_AUDIO_SIZE // (1024 * 1024)}MB",
            )

        logger.info(
            "transcribe_audio_request",
            content_type=audio.content_type,
            size_bytes=len(audio_data),
            language=language,
        )

        # Transcribe using voice adapter
        result = await voice_adapter.transcribe(audio_data, language=language)

        logger.info(
            "transcribe_audio_success",
            text_length=len(result.text),
            language=result.language,
            confidence=result.confidence,
        )

        return TranscriptionResponse(
            text=result.text,
            language=result.language or language,
            confidence=result.confidence,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (413, 415, etc.) without converting to 500
        raise
    except RuntimeError as e:
        logger.error("transcribe_audio_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("transcribe_audio_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Transcription failed")


@router.post("/tts")
async def text_to_speech(
    tts_request: TTSRequest,
    voice_adapter: Optional[VoiceAdapter] = Depends(get_voice_adapter),
    limiter: RateLimiter = Depends(get_rate_limiter),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Response:
    """Convert text to speech audio stream.

    Story 21-E2: Implement Voice Output (Text-to-Speech)

    Accepts text and returns audio stream (MP3 format).
    Uses configured TTS provider (OpenAI, ElevenLabs, pyttsx3).

    Args:
        tts_request: Text and optional voice/speed settings

    Returns:
        Audio response (audio/mpeg)

    Raises:
        403: Voice I/O is disabled
        429: Rate limit exceeded
        503: Voice adapter not configured
    """
    # Rate limiting
    tenant_id = x_tenant_id or "anonymous"
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    if voice_adapter is None:
        raise HTTPException(status_code=503, detail="Voice adapter not configured")

    if not voice_adapter.enabled:
        raise HTTPException(status_code=403, detail="Voice I/O is disabled")

    try:
        logger.info(
            "tts_request",
            text_length=len(tts_request.text),
            voice=tts_request.voice,
            speed=tts_request.speed,
        )

        # Synthesize speech using voice adapter
        result = await voice_adapter.synthesize(
            text=tts_request.text,
            voice=tts_request.voice,
            speed=tts_request.speed,
        )

        logger.info(
            "tts_success",
            audio_size=len(result.audio_data),
            format=result.format,
            duration_seconds=result.duration_seconds,
        )

        # Return audio as response with dynamic Content-Type based on actual format
        # Determine file extension from format
        format_to_ext = {"mp3": "mp3", "opus": "opus", "aac": "aac", "flac": "flac", "wav": "wav"}
        ext = format_to_ext.get(result.format, "mp3")
        media_type = f"audio/{result.format}" if result.format else "audio/mpeg"

        return Response(
            content=result.audio_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename=response.{ext}",
                "X-Audio-Duration": str(result.duration_seconds) if result.duration_seconds else "0",
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions without converting to 500
        raise
    except RuntimeError as e:
        logger.error("tts_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("tts_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Text-to-speech synthesis failed")
