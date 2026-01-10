"""Voice I/O module for speech-to-text and text-to-speech.

Story 20-H4: Implement Voice I/O

This module provides voice interaction capabilities:
- Speech-to-Text (STT): Convert voice input to text using Whisper
- Text-to-Speech (TTS): Generate spoken responses using OpenAI or pyttsx3

Example:
    from agentic_rag_backend.voice import VoiceAdapter, create_voice_adapter

    # Create adapter
    adapter = create_voice_adapter(
        enabled=True,
        whisper_model="base",
        tts_provider="openai",
        openai_api_key="xxx",
    )

    # Transcribe audio
    result = await adapter.transcribe("audio.wav")
    print(result.text)

    # Generate speech
    audio = await adapter.synthesize("Hello, world!")
"""

from .adapter import VoiceAdapter, create_voice_adapter
from .models import (
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_SPEED,
    DEFAULT_TTS_VOICE,
    DEFAULT_VOICE_IO_ENABLED,
    DEFAULT_WHISPER_MODEL,
    OpenAIVoice,
    TranscriptionResult,
    TranscriptionSegment,
    TTSProvider,
    TTSResult,
    VoiceConfig,
    WhisperModel,
)
from .stt import SpeechToText
from .tts import TextToSpeech

__all__ = [
    # Models
    "WhisperModel",
    "TTSProvider",
    "OpenAIVoice",
    "TranscriptionSegment",
    "TranscriptionResult",
    "TTSResult",
    "VoiceConfig",
    # Components
    "SpeechToText",
    "TextToSpeech",
    "VoiceAdapter",
    "create_voice_adapter",
    # Defaults
    "DEFAULT_VOICE_IO_ENABLED",
    "DEFAULT_WHISPER_MODEL",
    "DEFAULT_TTS_PROVIDER",
    "DEFAULT_TTS_VOICE",
    "DEFAULT_TTS_SPEED",
]
