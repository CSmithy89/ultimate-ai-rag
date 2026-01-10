"""API endpoint tests for voice I/O.

Story 21-E1: Voice Input (Speech-to-Text)
Story 21-E2: Voice Output (Text-to-Speech)

Tests cover:
- /copilot/transcribe endpoint functionality
- /copilot/tts endpoint functionality
- Rate limiting
- Audio type validation
- Error handling
- TTS text sanitization
"""

from unittest.mock import AsyncMock, MagicMock
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag_backend.api.routes.copilot import router
from agentic_rag_backend.voice import TranscriptionResult, TTSResult, VoiceAdapter
from agentic_rag_backend.rate_limit import InMemoryRateLimiter


@pytest.fixture
def mock_voice_adapter():
    """Create a mock voice adapter."""
    adapter = MagicMock(spec=VoiceAdapter)
    adapter.enabled = True
    adapter.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello, world!",
            language="en",
            confidence=0.95,
            duration_seconds=1.5,
        )
    )
    adapter.synthesize = AsyncMock(
        return_value=TTSResult(
            audio_data=b"mock audio data",
            format="mp3",
            duration_seconds=2.0,
        )
    )
    return adapter


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter that always allows."""
    limiter = MagicMock(spec=InMemoryRateLimiter)
    limiter.allow = AsyncMock(return_value=True)
    return limiter


@pytest.fixture
def app(mock_voice_adapter, mock_rate_limiter):
    """Create test FastAPI app with mocked dependencies."""
    test_app = FastAPI()
    test_app.include_router(router)

    # Set up app state
    test_app.state.voice_adapter = mock_voice_adapter
    test_app.state.rate_limiter = mock_rate_limiter
    test_app.state.orchestrator = MagicMock()
    test_app.state.hitl_manager = None

    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestTranscribeEndpoint:
    """Tests for POST /copilot/transcribe."""

    def test_transcribe_success(self, client, mock_voice_adapter):
        """Test successful transcription."""
        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        response = client.post("/copilot/transcribe", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello, world!"
        assert data["language"] == "en"
        assert data["confidence"] == 0.95

    def test_transcribe_with_language_hint(self, client, mock_voice_adapter):
        """Test transcription with language hint."""
        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        response = client.post("/copilot/transcribe?language=es", files=files)

        assert response.status_code == 200
        mock_voice_adapter.transcribe.assert_called_once()
        call_kwargs = mock_voice_adapter.transcribe.call_args.kwargs
        assert call_kwargs.get("language") == "es"

    def test_transcribe_unsupported_audio_type(self, client):
        """Test transcription rejects unsupported audio types."""
        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.txt", io.BytesIO(audio_content), "text/plain")}

        response = client.post("/copilot/transcribe", files=files)

        assert response.status_code == 415
        assert "Unsupported media type" in response.json()["detail"]

    def test_transcribe_allowed_audio_types(self, client, mock_voice_adapter):
        """Test all allowed audio types are accepted."""
        allowed_types = [
            ("audio/webm", "recording.webm"),
            ("audio/wav", "recording.wav"),
            ("audio/x-wav", "recording.wav"),
            ("audio/mp3", "recording.mp3"),
            ("audio/mpeg", "recording.mp3"),
            ("audio/ogg", "recording.ogg"),
            ("audio/flac", "recording.flac"),
            ("audio/m4a", "recording.m4a"),
            ("audio/mp4", "recording.mp4"),
        ]

        for content_type, filename in allowed_types:
            audio_content = b"mock audio bytes"
            files = {"audio": (filename, io.BytesIO(audio_content), content_type)}

            response = client.post("/copilot/transcribe", files=files)

            assert response.status_code == 200, f"Failed for {content_type}"

    def test_transcribe_voice_disabled(self, client, mock_voice_adapter):
        """Test transcription when voice I/O is disabled."""
        mock_voice_adapter.enabled = False

        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        response = client.post("/copilot/transcribe", files=files)

        assert response.status_code == 403
        assert "Voice I/O is disabled" in response.json()["detail"]

    def test_transcribe_no_voice_adapter(self, app, mock_rate_limiter):
        """Test transcription when voice adapter not configured."""
        app.state.voice_adapter = None
        test_client = TestClient(app)

        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        response = test_client.post("/copilot/transcribe", files=files)

        assert response.status_code == 503
        assert "Voice adapter not configured" in response.json()["detail"]

    def test_transcribe_rate_limited(self, client, mock_rate_limiter):
        """Test transcription rate limiting."""
        mock_rate_limiter.allow = AsyncMock(return_value=False)

        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        response = client.post("/copilot/transcribe", files=files)

        assert response.status_code == 429


class TestTTSEndpoint:
    """Tests for POST /copilot/tts."""

    def test_tts_success(self, client, mock_voice_adapter):
        """Test successful TTS synthesis."""
        response = client.post(
            "/copilot/tts",
            json={"text": "Hello, world!"},
        )

        assert response.status_code == 200
        # Content-Type matches the format returned by synthesize (mp3)
        assert response.headers["content-type"] == "audio/mp3"
        assert response.content == b"mock audio data"

    def test_tts_with_voice_option(self, client, mock_voice_adapter):
        """Test TTS with voice option."""
        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!", "voice": "nova"},
        )

        assert response.status_code == 200
        call_kwargs = mock_voice_adapter.synthesize.call_args.kwargs
        assert call_kwargs.get("voice") == "nova"

    def test_tts_with_speed_option(self, client, mock_voice_adapter):
        """Test TTS with speed option."""
        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!", "speed": 1.5},
        )

        assert response.status_code == 200
        call_kwargs = mock_voice_adapter.synthesize.call_args.kwargs
        assert call_kwargs.get("speed") == 1.5

    def test_tts_empty_text_rejected(self, client):
        """Test TTS rejects empty text."""
        response = client.post(
            "/copilot/tts",
            json={"text": ""},
        )

        assert response.status_code == 422  # Validation error

    def test_tts_text_too_long_rejected(self, client):
        """Test TTS rejects text exceeding max length."""
        response = client.post(
            "/copilot/tts",
            json={"text": "a" * 4097},  # Max is 4096
        )

        assert response.status_code == 422

    def test_tts_speed_out_of_range(self, client):
        """Test TTS rejects speed out of range."""
        # Speed too low
        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!", "speed": 0.1},
        )
        assert response.status_code == 422

        # Speed too high
        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!", "speed": 5.0},
        )
        assert response.status_code == 422

    def test_tts_text_sanitization(self, client, mock_voice_adapter):
        """Test TTS sanitizes text by removing control characters."""
        # Text with control characters
        text_with_control = "Hello\x00\x08\x0bWorld"

        response = client.post(
            "/copilot/tts",
            json={"text": text_with_control},
        )

        assert response.status_code == 200
        call_args = mock_voice_adapter.synthesize.call_args
        # The sanitized text should have control chars removed
        assert call_args.kwargs["text"] == "HelloWorld"

    def test_tts_voice_disabled(self, client, mock_voice_adapter):
        """Test TTS when voice I/O is disabled."""
        mock_voice_adapter.enabled = False

        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!"},
        )

        assert response.status_code == 403
        assert "Voice I/O is disabled" in response.json()["detail"]

    def test_tts_no_voice_adapter(self, app, mock_rate_limiter):
        """Test TTS when voice adapter not configured."""
        app.state.voice_adapter = None
        test_client = TestClient(app)

        response = test_client.post(
            "/copilot/tts",
            json={"text": "Hello!"},
        )

        assert response.status_code == 503
        assert "Voice adapter not configured" in response.json()["detail"]

    def test_tts_rate_limited(self, client, mock_rate_limiter):
        """Test TTS rate limiting."""
        mock_rate_limiter.allow = AsyncMock(return_value=False)

        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!"},
        )

        assert response.status_code == 429

    def test_tts_includes_duration_header(self, client, mock_voice_adapter):
        """Test TTS response includes duration header."""
        response = client.post(
            "/copilot/tts",
            json={"text": "Hello!"},
        )

        assert response.status_code == 200
        assert response.headers.get("x-audio-duration") == "2.0"


class TestTenantIsolation:
    """Tests for tenant isolation in voice endpoints."""

    def test_transcribe_uses_tenant_header(self, client, mock_rate_limiter):
        """Test transcribe uses X-Tenant-ID for rate limiting."""
        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        client.post(
            "/copilot/transcribe",
            files=files,
            headers={"X-Tenant-ID": "tenant-123"},
        )

        mock_rate_limiter.allow.assert_called_once_with("tenant-123")

    def test_tts_uses_tenant_header(self, client, mock_rate_limiter):
        """Test TTS uses X-Tenant-ID for rate limiting."""
        client.post(
            "/copilot/tts",
            json={"text": "Hello!"},
            headers={"X-Tenant-ID": "tenant-456"},
        )

        mock_rate_limiter.allow.assert_called_once_with("tenant-456")

    def test_anonymous_tenant_when_no_header(self, client, mock_rate_limiter):
        """Test anonymous tenant used when no header provided."""
        audio_content = b"mock audio bytes"
        files = {"audio": ("recording.webm", io.BytesIO(audio_content), "audio/webm")}

        client.post("/copilot/transcribe", files=files)

        mock_rate_limiter.allow.assert_called_once_with("anonymous")
