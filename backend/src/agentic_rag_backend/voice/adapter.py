"""Voice I/O adapter with feature flag support.

Story 20-H4: Implement Voice I/O

This module provides a unified adapter for voice I/O operations
with feature flag support.
"""

from pathlib import Path
from typing import Optional, Union

import structlog

from .models import (
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_SPEED,
    DEFAULT_TTS_VOICE,
    DEFAULT_VOICE_IO_ENABLED,
    DEFAULT_WHISPER_MODEL,
    OpenAIVoice,
    TranscriptionResult,
    TTSProvider,
    TTSResult,
    VoiceConfig,
    WhisperModel,
)
from .stt import SpeechToText
from .tts import TextToSpeech

logger = structlog.get_logger(__name__)


class VoiceAdapter:
    """Unified adapter for voice I/O with feature flag support.

    Provides a single interface for speech-to-text and text-to-speech
    operations, with graceful degradation when voice features are disabled.

    Example:
        adapter = VoiceAdapter(enabled=True, openai_api_key="xxx")

        # Transcribe audio
        result = await adapter.transcribe("audio.wav")
        print(result.text)

        # Generate speech
        audio = await adapter.synthesize("Hello, world!")
        with open("output.mp3", "wb") as f:
            f.write(audio.audio_data)
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_VOICE_IO_ENABLED,
        whisper_model: WhisperModel = DEFAULT_WHISPER_MODEL,
        tts_provider: TTSProvider = DEFAULT_TTS_PROVIDER,
        tts_voice: OpenAIVoice = DEFAULT_TTS_VOICE,
        tts_speed: float = DEFAULT_TTS_SPEED,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Initialize voice adapter.

        Args:
            enabled: Whether voice I/O is enabled
            whisper_model: Whisper model size for STT
            tts_provider: TTS provider to use
            tts_voice: Voice for OpenAI TTS
            tts_speed: Speech speed
            openai_api_key: OpenAI API key for TTS
        """
        self._enabled = enabled
        self._config = VoiceConfig(
            enabled=enabled,
            whisper_model=whisper_model,
            tts_provider=tts_provider,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            openai_api_key=openai_api_key or "",
        )

        self._stt: Optional[SpeechToText] = None
        self._tts: Optional[TextToSpeech] = None

        self._logger = logger.bind(component="VoiceAdapter")

        if enabled:
            self._logger.info(
                "voice_adapter_enabled",
                whisper_model=whisper_model.value if isinstance(whisper_model, WhisperModel) else whisper_model,
                tts_provider=tts_provider.value if isinstance(tts_provider, TTSProvider) else tts_provider,
            )
        else:
            self._logger.info("voice_adapter_disabled")

    @property
    def enabled(self) -> bool:
        """Check if voice I/O is enabled."""
        return self._enabled

    @property
    def config(self) -> VoiceConfig:
        """Get voice configuration."""
        return self._config

    def _ensure_stt(self) -> SpeechToText:
        """Get or create STT instance."""
        if self._stt is None:
            self._stt = SpeechToText(model=self._config.whisper_model)
        return self._stt

    def _ensure_tts(self) -> TextToSpeech:
        """Get or create TTS instance."""
        if self._tts is None:
            self._tts = TextToSpeech(
                provider=self._config.tts_provider,
                voice=self._config.tts_voice,
                speed=self._config.tts_speed,
                api_key=self._config.openai_api_key,
            )
        return self._tts

    async def transcribe(
        self,
        audio: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio file path or bytes
            language: Optional language hint

        Returns:
            TranscriptionResult with text and metadata

        Raises:
            RuntimeError: If voice I/O is disabled
        """
        if not self._enabled:
            raise RuntimeError("Voice I/O is disabled")

        stt = self._ensure_stt()

        if isinstance(audio, bytes):
            return await stt.transcribe_bytes(audio, language)
        else:
            return await stt.transcribe(audio, language)

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSResult:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Override voice (optional)
            speed: Override speed (optional)

        Returns:
            TTSResult with audio data

        Raises:
            RuntimeError: If voice I/O is disabled
        """
        if not self._enabled:
            raise RuntimeError("Voice I/O is disabled")

        tts = self._ensure_tts()
        return await tts.synthesize(text, voice, speed)

    def get_supported_stt_languages(self) -> list[str]:
        """Get list of supported STT languages.

        Returns:
            List of ISO 639-1 language codes
        """
        if not self._enabled:
            return []

        stt = self._ensure_stt()
        return stt.get_supported_languages()

    def get_available_tts_voices(self) -> list[str]:
        """Get list of available TTS voices.

        Returns:
            List of voice names
        """
        if not self._enabled:
            return []

        tts = self._ensure_tts()
        return tts.get_available_voices()

    async def close(self) -> None:
        """Close voice adapter resources."""
        if self._tts:
            await self._tts.close()
            self._tts = None


def create_voice_adapter(
    enabled: bool = DEFAULT_VOICE_IO_ENABLED,
    whisper_model: str = DEFAULT_WHISPER_MODEL.value,
    tts_provider: str = DEFAULT_TTS_PROVIDER.value,
    tts_voice: str = DEFAULT_TTS_VOICE.value,
    tts_speed: float = DEFAULT_TTS_SPEED,
    openai_api_key: Optional[str] = None,
) -> VoiceAdapter:
    """Factory function to create a VoiceAdapter.

    Args:
        enabled: Whether voice I/O is enabled
        whisper_model: Whisper model name
        tts_provider: TTS provider name
        tts_voice: TTS voice name
        tts_speed: Speech speed
        openai_api_key: OpenAI API key

    Returns:
        Configured VoiceAdapter instance
    """
    # Convert string values to enums
    whisper_enum = WhisperModel(whisper_model) if whisper_model else DEFAULT_WHISPER_MODEL
    tts_provider_enum = TTSProvider(tts_provider) if tts_provider else DEFAULT_TTS_PROVIDER
    tts_voice_enum = OpenAIVoice(tts_voice) if tts_voice else DEFAULT_TTS_VOICE

    return VoiceAdapter(
        enabled=enabled,
        whisper_model=whisper_enum,
        tts_provider=tts_provider_enum,
        tts_voice=tts_voice_enum,
        tts_speed=tts_speed,
        openai_api_key=openai_api_key,
    )
