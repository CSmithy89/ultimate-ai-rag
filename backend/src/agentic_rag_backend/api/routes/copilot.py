"""CopilotKit AG-UI protocol endpoint.

Story 21-E1: Voice Input (Speech-to-Text)
Story 21-E2: Voice Output (Text-to-Speech)
"""

import re
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
) -> StreamingResponse:
    """
    Handle AG-UI protocol requests from CopilotKit.

    Returns SSE stream with AG-UI events:
    - text_delta: Streaming text responses
    - tool_call: Agent tool invocations
    - state_snapshot: Agent state updates
    - action_request: Frontend action requests
    """
    # Extract tenant_id for rate limiting
    tenant_id = "anonymous"
    if request.config and request.config.configurable:
        tenant_id = request.config.configurable.get("tenant_id", "anonymous")

    # Check rate limit
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    bridge = AGUIBridge(orchestrator, hitl_manager=get_hitl_manager(http_request))

    async def event_generator():
        async for event in bridge.process_request(request):
            yield f"data: {event.model_dump_json()}\n\n"

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
