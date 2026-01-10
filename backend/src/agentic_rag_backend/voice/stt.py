"""Speech-to-Text module using Whisper.

Story 20-H4: Implement Voice I/O

This module provides speech-to-text capabilities using the Whisper model.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any, Optional, Union

import structlog

from .models import (
    DEFAULT_WHISPER_MODEL,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperModel,
)

logger = structlog.get_logger(__name__)

# Maximum audio file size (50MB) to prevent DoS/OOM
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024


class SpeechToText:
    """Speech-to-text transcription using Whisper.

    Uses faster-whisper for efficient local transcription. Falls back
    to openai-whisper if faster-whisper is not available.

    Example:
        stt = SpeechToText(model=WhisperModel.BASE)
        result = await stt.transcribe("audio.wav")
        print(result.text)
    """

    def __init__(
        self,
        model: WhisperModel = DEFAULT_WHISPER_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        """Initialize speech-to-text.

        Args:
            model: Whisper model size to use
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute type (auto, float16, int8, etc.)
        """
        self._model_name = model.value if isinstance(model, WhisperModel) else model
        self._device = device
        self._compute_type = compute_type
        self._model: Any = None
        self._use_faster_whisper = True

        self._model_lock = threading.Lock()

        self._logger = logger.bind(
            component="SpeechToText",
            model=self._model_name,
        )

    def _ensure_model(self) -> None:
        """Lazily load the Whisper model.

        Uses double-check locking pattern for thread safety.
        """
        if self._model is not None:
            return

        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            try:
                from faster_whisper import WhisperModel as FasterWhisperModel

                self._model = FasterWhisperModel(
                    self._model_name,
                    device=self._device if self._device != "auto" else "auto",
                    compute_type=self._compute_type if self._compute_type != "auto" else "default",
                )
                self._use_faster_whisper = True

                self._logger.info(
                    "stt_model_loaded",
                    provider="faster-whisper",
                    model=self._model_name,
                )
            except ImportError:
                self._logger.warning(
                    "faster_whisper_not_available",
                    hint="Install faster-whisper for better performance",
                )
                try:
                    import whisper

                    self._model = whisper.load_model(self._model_name)
                    self._use_faster_whisper = False

                    self._logger.info(
                        "stt_model_loaded",
                        provider="openai-whisper",
                        model=self._model_name,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Neither faster-whisper nor openai-whisper is installed. "
                        "Install with: pip install faster-whisper or pip install openai-whisper"
                    ) from e

    async def transcribe(
        self,
        audio: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio file path or bytes
            language: Optional language hint (ISO 639-1 code)

        Returns:
            TranscriptionResult with text and metadata
        """
        self._ensure_model()

        try:
            if self._use_faster_whisper:
                return await self._transcribe_faster(audio, language)
            else:
                return await self._transcribe_whisper(audio, language)
        except Exception as e:
            self._logger.error(
                "stt_transcription_failed",
                error=str(e),
            )
            raise

    async def _transcribe_faster(
        self,
        audio: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper.

        Args:
            audio: Audio file path or bytes
            language: Optional language hint

        Returns:
            TranscriptionResult
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _transcribe() -> TranscriptionResult:
            segments, info = self._model.transcribe(
                str(audio) if isinstance(audio, Path) else audio,
                language=language,
                beam_size=5,
            )

            result_segments = []
            full_text_parts = []

            for segment in segments:
                result_segments.append(
                    TranscriptionSegment(
                        text=segment.text.strip(),
                        start=segment.start,
                        end=segment.end,
                        confidence=segment.avg_logprob if hasattr(segment, "avg_logprob") else 0.0,
                    )
                )
                full_text_parts.append(segment.text.strip())

            return TranscriptionResult(
                text=" ".join(full_text_parts),
                language=info.language,
                confidence=info.language_probability,
                duration_seconds=info.duration,
                segments=result_segments,
                metadata={
                    "provider": "faster-whisper",
                    "model": self._model_name,
                },
            )

        return await loop.run_in_executor(None, _transcribe)

    async def _transcribe_whisper(
        self,
        audio: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe using openai-whisper.

        Args:
            audio: Audio file path or bytes
            language: Optional language hint

        Returns:
            TranscriptionResult
        """
        loop = asyncio.get_event_loop()

        def _transcribe() -> TranscriptionResult:
            options = {"language": language} if language else {}
            result = self._model.transcribe(
                str(audio) if isinstance(audio, Path) else audio,
                **options,
            )

            segments = []
            for seg in result.get("segments", []):
                segments.append(
                    TranscriptionSegment(
                        text=seg["text"].strip(),
                        start=seg["start"],
                        end=seg["end"],
                        confidence=seg.get("avg_logprob", 0.0),
                    )
                )

            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language", "en"),
                confidence=0.0,  # Not available in openai-whisper
                duration_seconds=segments[-1].end if segments else 0.0,
                segments=segments,
                metadata={
                    "provider": "openai-whisper",
                    "model": self._model_name,
                },
            )

        return await loop.run_in_executor(None, _transcribe)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio from bytes.

        Args:
            audio_bytes: Raw audio bytes
            language: Optional language hint

        Returns:
            TranscriptionResult

        Raises:
            ValueError: If audio size exceeds maximum limit
        """
        import tempfile
        import os

        # Validate audio size to prevent DoS/OOM
        if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
            self._logger.warning(
                "stt_audio_too_large",
                size_bytes=len(audio_bytes),
                max_size=MAX_AUDIO_SIZE_BYTES,
            )
            raise ValueError(
                f"Audio size ({len(audio_bytes)} bytes) exceeds maximum "
                f"allowed ({MAX_AUDIO_SIZE_BYTES} bytes)"
            )

        # Write bytes to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            return await self.transcribe(temp_path, language)
        finally:
            os.unlink(temp_path)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            List of ISO 639-1 language codes
        """
        # Common languages supported by Whisper
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
        ]
