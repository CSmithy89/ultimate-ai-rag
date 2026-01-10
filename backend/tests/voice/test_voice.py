"""Unit tests for voice I/O module (Story 20-H4).

Tests cover:
- Voice models (TranscriptionResult, TTSResult, VoiceConfig)
- SpeechToText with mocked Whisper
- TextToSpeech with mocked providers
- VoiceAdapter feature flag behavior
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.voice import (
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_SPEED,
    DEFAULT_TTS_VOICE,
    DEFAULT_VOICE_IO_ENABLED,
    DEFAULT_WHISPER_MODEL,
    OpenAIVoice,
    SpeechToText,
    TextToSpeech,
    TranscriptionResult,
    TranscriptionSegment,
    TTSProvider,
    TTSResult,
    VoiceAdapter,
    VoiceConfig,
    WhisperModel,
    create_voice_adapter,
)


# ============================================================================
# Voice Models Tests
# ============================================================================


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_create_segment(self):
        """Test creating a transcription segment."""
        segment = TranscriptionSegment(
            text="Hello, world!",
            start=0.0,
            end=1.5,
            confidence=0.95,
        )
        assert segment.text == "Hello, world!"
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.confidence == 0.95


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_create_result(self):
        """Test creating a transcription result."""
        result = TranscriptionResult(
            text="Hello, world!",
            language="en",
            confidence=0.9,
            duration_seconds=1.5,
        )
        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.confidence == 0.9
        assert result.duration_seconds == 1.5

    def test_result_with_segments(self):
        """Test result with segments."""
        segments = [
            TranscriptionSegment(text="Hello", start=0.0, end=0.5),
            TranscriptionSegment(text="world", start=0.5, end=1.0),
        ]
        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
        )
        assert len(result.segments) == 2


class TestTTSResult:
    """Tests for TTSResult dataclass."""

    def test_create_result(self):
        """Test creating a TTS result."""
        result = TTSResult(
            audio_data=b"audio bytes",
            format="mp3",
            duration_seconds=2.5,
        )
        assert result.audio_data == b"audio bytes"
        assert result.format == "mp3"
        assert result.duration_seconds == 2.5


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""

    def test_create_config(self):
        """Test creating voice config."""
        config = VoiceConfig(
            enabled=True,
            whisper_model=WhisperModel.BASE,
            tts_provider=TTSProvider.OPENAI,
        )
        assert config.enabled is True
        assert config.whisper_model == WhisperModel.BASE

    def test_config_speed_validation(self):
        """Test speed is clamped to valid range."""
        config = VoiceConfig(tts_speed=10.0)
        assert config.tts_speed == 1.0  # Reset to default

        config = VoiceConfig(tts_speed=0.1)
        assert config.tts_speed == 1.0  # Reset to default


# ============================================================================
# SpeechToText Tests
# ============================================================================


class TestSpeechToText:
    """Tests for SpeechToText class."""

    def test_initialization(self):
        """Test STT initialization."""
        stt = SpeechToText(model=WhisperModel.BASE)
        assert stt._model_name == "base"
        assert stt._model is None

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        stt = SpeechToText()
        languages = stt.get_supported_languages()
        assert "en" in languages
        assert "zh" in languages
        assert "es" in languages
        assert len(languages) > 50

    @pytest.mark.asyncio
    async def test_transcribe_with_mock_faster_whisper(self):
        """Test transcription with mocked faster-whisper."""
        stt = SpeechToText(model=WhisperModel.BASE)

        # Create mock segment
        mock_segment = MagicMock()
        mock_segment.text = " Hello, world!"
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_segment.avg_logprob = -0.1

        # Create mock info
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 1.5

        # Create mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        stt._model = mock_model
        stt._use_faster_whisper = True

        result = await stt.transcribe("test.wav")

        assert "Hello, world!" in result.text
        assert result.language == "en"
        assert result.confidence == 0.95


# ============================================================================
# TextToSpeech Tests
# ============================================================================


class TestTextToSpeech:
    """Tests for TextToSpeech class."""

    def test_initialization(self):
        """Test TTS initialization."""
        tts = TextToSpeech(provider=TTSProvider.OPENAI)
        assert tts._provider == TTSProvider.OPENAI

    def test_get_available_voices_openai(self):
        """Test getting OpenAI voices."""
        tts = TextToSpeech(provider=TTSProvider.OPENAI)
        voices = tts.get_available_voices()
        assert "alloy" in voices
        assert "echo" in voices
        assert "nova" in voices

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """Test synthesizing empty text."""
        tts = TextToSpeech(provider=TTSProvider.OPENAI)
        result = await tts.synthesize("")
        assert result.audio_data == b""
        assert result.duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_openai_with_mock(self):
        """Test OpenAI TTS with mocked client."""
        tts = TextToSpeech(
            provider=TTSProvider.OPENAI,
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.content = b"mock audio data"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tts._client = mock_client

        result = await tts.synthesize("Hello, world!")

        assert result.audio_data == b"mock audio data"
        assert result.format == "mp3"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_without_api_key(self):
        """Test OpenAI TTS raises without API key."""
        tts = TextToSpeech(provider=TTSProvider.OPENAI)

        with pytest.raises(ValueError, match="API key is required"):
            await tts.synthesize("Hello")


# ============================================================================
# VoiceAdapter Tests
# ============================================================================


class TestVoiceAdapter:
    """Tests for VoiceAdapter class."""

    def test_adapter_disabled_by_default(self):
        """Test adapter is disabled by default."""
        adapter = VoiceAdapter()
        assert not adapter.enabled

    def test_adapter_enabled(self):
        """Test adapter can be enabled."""
        adapter = VoiceAdapter(enabled=True)
        assert adapter.enabled

    def test_adapter_config(self):
        """Test adapter configuration."""
        adapter = VoiceAdapter(
            enabled=True,
            whisper_model=WhisperModel.SMALL,
            tts_provider=TTSProvider.PYTTSX3,
        )
        assert adapter.config.whisper_model == WhisperModel.SMALL
        assert adapter.config.tts_provider == TTSProvider.PYTTSX3

    @pytest.mark.asyncio
    async def test_transcribe_when_disabled(self):
        """Test transcribe raises when disabled."""
        adapter = VoiceAdapter(enabled=False)

        with pytest.raises(RuntimeError, match="Voice I/O is disabled"):
            await adapter.transcribe("audio.wav")

    @pytest.mark.asyncio
    async def test_synthesize_when_disabled(self):
        """Test synthesize raises when disabled."""
        adapter = VoiceAdapter(enabled=False)

        with pytest.raises(RuntimeError, match="Voice I/O is disabled"):
            await adapter.synthesize("Hello")

    def test_get_supported_languages_when_disabled(self):
        """Test getting languages returns empty when disabled."""
        adapter = VoiceAdapter(enabled=False)
        languages = adapter.get_supported_stt_languages()
        assert languages == []

    def test_get_available_voices_when_disabled(self):
        """Test getting voices returns empty when disabled."""
        adapter = VoiceAdapter(enabled=False)
        voices = adapter.get_available_tts_voices()
        assert voices == []

    @pytest.mark.asyncio
    async def test_transcribe_with_mock(self):
        """Test transcribe with mocked STT."""
        adapter = VoiceAdapter(enabled=True)

        mock_stt = MagicMock()
        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hello", language="en")
        )

        adapter._stt = mock_stt

        result = await adapter.transcribe("audio.wav")
        assert result.text == "Hello"

    @pytest.mark.asyncio
    async def test_synthesize_with_mock(self):
        """Test synthesize with mocked TTS."""
        adapter = VoiceAdapter(enabled=True, openai_api_key="test")

        mock_tts = MagicMock()
        mock_tts.synthesize = AsyncMock(
            return_value=TTSResult(audio_data=b"audio", format="mp3")
        )

        adapter._tts = mock_tts

        result = await adapter.synthesize("Hello")
        assert result.audio_data == b"audio"


class TestCreateVoiceAdapter:
    """Tests for create_voice_adapter factory function."""

    def test_create_disabled(self):
        """Test creating disabled adapter."""
        adapter = create_voice_adapter(enabled=False)
        assert not adapter.enabled

    def test_create_with_config(self):
        """Test creating adapter with config."""
        adapter = create_voice_adapter(
            enabled=True,
            whisper_model="small",
            tts_provider="pyttsx3",
            tts_voice="nova",
            tts_speed=1.5,
        )
        assert adapter.enabled
        assert adapter.config.whisper_model == WhisperModel.SMALL
        assert adapter.config.tts_provider == TTSProvider.PYTTSX3


# ============================================================================
# Default Constants Tests
# ============================================================================


class TestDefaultConstants:
    """Tests for default configuration constants."""

    def test_default_voice_io_disabled(self):
        """Test voice I/O is disabled by default."""
        assert DEFAULT_VOICE_IO_ENABLED is False

    def test_default_whisper_model(self):
        """Test default Whisper model is base."""
        assert DEFAULT_WHISPER_MODEL == WhisperModel.BASE

    def test_default_tts_provider(self):
        """Test default TTS provider is OpenAI."""
        assert DEFAULT_TTS_PROVIDER == TTSProvider.OPENAI

    def test_default_tts_voice(self):
        """Test default TTS voice is alloy."""
        assert DEFAULT_TTS_VOICE == OpenAIVoice.ALLOY

    def test_default_tts_speed(self):
        """Test default TTS speed is 1.0."""
        assert DEFAULT_TTS_SPEED == 1.0


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for voice-related enums."""

    def test_whisper_models(self):
        """Test Whisper model enum values."""
        assert WhisperModel.TINY.value == "tiny"
        assert WhisperModel.BASE.value == "base"
        assert WhisperModel.SMALL.value == "small"
        assert WhisperModel.MEDIUM.value == "medium"
        assert WhisperModel.LARGE.value == "large"

    def test_tts_providers(self):
        """Test TTS provider enum values."""
        assert TTSProvider.OPENAI.value == "openai"
        assert TTSProvider.PYTTSX3.value == "pyttsx3"

    def test_openai_voices(self):
        """Test OpenAI voice enum values."""
        assert OpenAIVoice.ALLOY.value == "alloy"
        assert OpenAIVoice.ECHO.value == "echo"
        assert OpenAIVoice.FABLE.value == "fable"
        assert OpenAIVoice.ONYX.value == "onyx"
        assert OpenAIVoice.NOVA.value == "nova"
        assert OpenAIVoice.SHIMMER.value == "shimmer"
