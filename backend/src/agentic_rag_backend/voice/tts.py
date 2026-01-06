"""Text-to-Speech module.

Story 20-H4: Implement Voice I/O

This module provides text-to-speech capabilities using OpenAI TTS or pyttsx3.
"""

import asyncio
import io
from typing import Any, Optional

import httpx
import structlog

from .models import (
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_SPEED,
    DEFAULT_TTS_VOICE,
    OpenAIVoice,
    TTSProvider,
    TTSResult,
)

logger = structlog.get_logger(__name__)

# Average speech rate for duration estimation (words per minute)
DEFAULT_SPEECH_RATE_WPM = 150


class TextToSpeech:
    """Text-to-speech generation.

    Supports OpenAI TTS API for high-quality voices and pyttsx3 for
    offline/local TTS.

    Example:
        async with TextToSpeech(provider=TTSProvider.OPENAI, api_key="xxx") as tts:
            result = await tts.synthesize("Hello, world!")
            with open("output.mp3", "wb") as f:
                f.write(result.audio_data)
    """

    async def __aenter__(self) -> "TextToSpeech":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    def __init__(
        self,
        provider: TTSProvider = DEFAULT_TTS_PROVIDER,
        voice: OpenAIVoice = DEFAULT_TTS_VOICE,
        speed: float = DEFAULT_TTS_SPEED,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize text-to-speech.

        Args:
            provider: TTS provider to use
            voice: Voice to use for OpenAI TTS
            speed: Speech speed (0.25 to 4.0)
            api_key: OpenAI API key (required for OpenAI provider)
        """
        self._provider = provider
        self._voice = voice.value if isinstance(voice, OpenAIVoice) else voice
        self._speed = max(0.25, min(4.0, speed))
        self._api_key = api_key
        self._pyttsx3_engine: Any = None
        self._client: Optional[httpx.AsyncClient] = None

        self._logger = logger.bind(
            component="TextToSpeech",
            provider=provider.value if isinstance(provider, TTSProvider) else provider,
            voice=self._voice,
        )

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Override voice (optional)
            speed: Override speed (optional)

        Returns:
            TTSResult with audio data
        """
        if not text or not text.strip():
            return TTSResult(
                audio_data=b"",
                format="mp3",
                duration_seconds=0.0,
            )

        voice = voice or self._voice
        speed = speed or self._speed

        try:
            if self._provider == TTSProvider.OPENAI:
                return await self._synthesize_openai(text, voice, speed)
            else:
                return await self._synthesize_pyttsx3(text, speed)
        except Exception as e:
            self._logger.error(
                "tts_synthesis_failed",
                error=str(e),
            )
            raise

    async def _synthesize_openai(
        self,
        text: str,
        voice: str,
        speed: float,
    ) -> TTSResult:
        """Synthesize using OpenAI TTS API.

        Args:
            text: Text to synthesize
            voice: Voice to use
            speed: Speech speed

        Returns:
            TTSResult
        """
        if not self._api_key:
            raise ValueError("OpenAI API key is required for OpenAI TTS")

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )

        try:
            response = await self._client.post(
                "/audio/speech",
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": voice,
                    "speed": speed,
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()

            audio_data = response.content

            # Estimate duration based on average speech rate
            word_count = len(text.split())
            duration = (word_count / DEFAULT_SPEECH_RATE_WPM) * 60 / speed

            return TTSResult(
                audio_data=audio_data,
                format="mp3",
                duration_seconds=duration,
                sample_rate=24000,
                metadata={
                    "provider": "openai",
                    "model": "tts-1",
                    "voice": voice,
                    "speed": speed,
                },
            )

        except httpx.HTTPStatusError as e:
            self._logger.error(
                "openai_tts_failed",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise

    async def _synthesize_pyttsx3(
        self,
        text: str,
        speed: float,
    ) -> TTSResult:
        """Synthesize using pyttsx3 (offline TTS).

        Args:
            text: Text to synthesize
            speed: Speech speed

        Returns:
            TTSResult
        """
        loop = asyncio.get_event_loop()

        def _synthesize() -> TTSResult:
            try:
                import pyttsx3

                if self._pyttsx3_engine is None:
                    self._pyttsx3_engine = pyttsx3.init()

                engine = self._pyttsx3_engine

                # Set speech rate (default is ~200 wpm)
                base_rate = engine.getProperty("rate")
                engine.setProperty("rate", int(base_rate * speed))

                # Synthesize to bytes
                audio_buffer = io.BytesIO()

                # Save to temp file (pyttsx3 requires file output)
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                try:
                    engine.save_to_file(text, temp_path)
                    engine.runAndWait()

                    with open(temp_path, "rb") as f:
                        audio_data = f.read()
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

                # Estimate duration based on average speech rate
                word_count = len(text.split())
                duration = (word_count / DEFAULT_SPEECH_RATE_WPM) * 60 / speed

                return TTSResult(
                    audio_data=audio_data,
                    format="wav",
                    duration_seconds=duration,
                    sample_rate=22050,
                    metadata={
                        "provider": "pyttsx3",
                        "speed": speed,
                    },
                )

            except ImportError as e:
                raise ImportError(
                    "pyttsx3 is required for offline TTS. "
                    "Install with: pip install pyttsx3"
                ) from e

        return await loop.run_in_executor(None, _synthesize)

    def get_available_voices(self) -> list[str]:
        """Get list of available voices.

        Returns:
            List of voice names
        """
        if self._provider == TTSProvider.OPENAI:
            return [v.value for v in OpenAIVoice]
        else:
            # pyttsx3 voices vary by system
            try:
                import pyttsx3

                if self._pyttsx3_engine is None:
                    self._pyttsx3_engine = pyttsx3.init()

                voices = self._pyttsx3_engine.getProperty("voices")
                return [v.id for v in voices]
            except ImportError:
                return []

    async def close(self) -> None:
        """Close the TTS client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

        # Cleanup pyttsx3 engine if initialized
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass  # Ignore errors during cleanup
            self._pyttsx3_engine = None
