"""Data models for voice I/O module.

Story 20-H4: Implement Voice I/O

This module defines data models for speech-to-text and text-to-speech operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TTSProvider(str, Enum):
    """Available TTS providers."""

    OPENAI = "openai"
    PYTTSX3 = "pyttsx3"


class OpenAIVoice(str, Enum):
    """Available OpenAI TTS voices."""

    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing.

    Attributes:
        text: Transcribed text for this segment
        start: Start time in seconds
        end: End time in seconds
        confidence: Confidence score (0.0 to 1.0)
    """

    text: str
    start: float
    end: float
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription.

    Attributes:
        text: Full transcribed text
        language: Detected language code
        confidence: Overall confidence score
        duration_seconds: Audio duration in seconds
        segments: List of transcription segments
        metadata: Additional transcription metadata
    """

    text: str
    language: str = "en"
    confidence: float = 0.0
    duration_seconds: float = 0.0
    segments: list[TranscriptionSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSResult:
    """Result of text-to-speech generation.

    Attributes:
        audio_data: Raw audio bytes
        format: Audio format (mp3, wav, etc.)
        duration_seconds: Audio duration in seconds
        sample_rate: Audio sample rate
        metadata: Additional TTS metadata
    """

    audio_data: bytes
    format: str = "mp3"
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceConfig:
    """Configuration for voice I/O.

    Attributes:
        enabled: Whether voice I/O is enabled
        whisper_model: Whisper model size to use
        tts_provider: TTS provider to use
        tts_voice: Voice to use for TTS
        tts_speed: Speech speed (0.5 to 2.0)
        openai_api_key: OpenAI API key for TTS
    """

    enabled: bool = False
    whisper_model: WhisperModel = WhisperModel.BASE
    tts_provider: TTSProvider = TTSProvider.OPENAI
    tts_voice: OpenAIVoice = OpenAIVoice.ALLOY
    tts_speed: float = 1.0
    openai_api_key: str = ""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.tts_speed < 0.25 or self.tts_speed > 4.0:
            self.tts_speed = 1.0


# Default configuration values
DEFAULT_VOICE_IO_ENABLED = False
DEFAULT_WHISPER_MODEL = WhisperModel.BASE
DEFAULT_TTS_PROVIDER = TTSProvider.OPENAI
DEFAULT_TTS_VOICE = OpenAIVoice.ALLOY
DEFAULT_TTS_SPEED = 1.0
