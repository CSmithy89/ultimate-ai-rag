"""YouTube transcript-first video ingestion.

This module provides transcript-based ingestion for YouTube videos,
allowing fast processing without full video download.
"""

import asyncio
import re
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from agentic_rag_backend.config import get_settings

logger = structlog.get_logger(__name__)


# Video ID is always 11 characters, alphanumeric with underscores and hyphens
VIDEO_ID_PATTERNS = [
    # Standard watch URL: youtube.com/watch?v=VIDEO_ID
    r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
    # Short URL: youtu.be/VIDEO_ID
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    # Embed URL: youtube.com/embed/VIDEO_ID
    r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    # Shorts URL: youtube.com/shorts/VIDEO_ID
    r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
]


class YouTubeIngestionError(Exception):
    """Exception raised when YouTube transcript ingestion fails."""

    def __init__(self, video_id: str, reason: str) -> None:
        self.video_id = video_id
        self.reason = reason
        super().__init__(f"YouTube ingestion failed for {video_id}: {reason}")


class TranscriptSegment(BaseModel):
    """A single segment from the YouTube transcript."""

    text: str = Field(..., description="Transcript text for this segment")
    start: float = Field(..., description="Start time in seconds")
    duration: float = Field(..., description="Duration in seconds")


class YouTubeTranscriptResult(BaseModel):
    """Complete transcript result from YouTube."""

    video_id: str = Field(..., description="YouTube video ID")
    title: Optional[str] = Field(None, description="Video title if available")
    language: str = Field(..., description="Transcript language code")
    is_generated: bool = Field(
        ..., description="Whether transcript is auto-generated"
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list, description="Transcript segments"
    )
    full_text: str = Field(..., description="Complete transcript text")
    duration_seconds: float = Field(..., description="Total video duration")


class TranscriptChunk(BaseModel):
    """A chunked portion of transcript for indexing."""

    content: str = Field(..., description="Chunk text content")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    video_id: str = Field(..., description="YouTube video ID")
    chunk_index: int = Field(..., description="Index of this chunk")


class YouTubeIngestionResult(BaseModel):
    """Result of ingesting a YouTube video transcript."""

    video_id: str = Field(..., description="YouTube video ID")
    source_url: str = Field(..., description="Original YouTube URL")
    language: str = Field(..., description="Transcript language")
    is_generated: bool = Field(
        ..., description="Whether transcript is auto-generated"
    )
    chunks: list[TranscriptChunk] = Field(
        default_factory=list, description="Transcript chunks for indexing"
    )
    duration_seconds: float = Field(..., description="Total video duration")
    full_text: str = Field(..., description="Complete transcript text")


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats.

    Supports the following URL formats:
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - youtube.com/embed/VIDEO_ID
    - youtube.com/shorts/VIDEO_ID
    - URLs with additional parameters (t, list, etc.)

    Args:
        url: YouTube URL in any supported format

    Returns:
        11-character video ID

    Raises:
        ValueError: If video ID cannot be extracted
    """
    for pattern in VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.debug(
                "extracted_video_id",
                url=url,
                video_id=video_id,
                pattern=pattern,
            )
            return video_id

    raise ValueError(f"Could not extract video ID from: {url}")


async def fetch_transcript(
    video_id: str,
    languages: Optional[list[str]] = None,
) -> YouTubeTranscriptResult:
    """Fetch transcript for a YouTube video.

    Uses the youtube-transcript-api library to fetch transcripts
    in the preferred language order.

    Args:
        video_id: YouTube video ID
        languages: Preferred languages in order (defaults to settings)

    Returns:
        YouTubeTranscriptResult with transcript data

    Raises:
        YouTubeIngestionError: If transcript cannot be fetched
    """
    # Import here to avoid import errors if package not installed
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    if languages is None:
        settings = get_settings()
        languages = settings.youtube_preferred_languages

    logger.info(
        "fetching_transcript",
        video_id=video_id,
        languages=languages,
    )

    try:
        # Run blocking API call in thread pool for async compatibility
        list_transcripts = getattr(YouTubeTranscriptApi, "list_transcripts", None)
        if list_transcripts is None:
            raise YouTubeIngestionError(
                video_id=video_id,
                reason="youtube-transcript-api is missing list_transcripts",
            )
        transcript_list = await asyncio.to_thread(
            list_transcripts,
            video_id,
        )

        # Find transcript in preferred language order
        transcript = transcript_list.find_transcript(languages)

        # Fetch the actual transcript data
        segments_data = await asyncio.to_thread(transcript.fetch)

        # Convert to Pydantic models
        segments = [
            TranscriptSegment(
                text=seg["text"],
                start=seg["start"],
                duration=seg["duration"],
            )
            for seg in segments_data
        ]

        # Build full text from segments
        full_text = " ".join(seg.text for seg in segments)

        # Calculate total duration from last segment
        duration = (
            max(s.start + s.duration for s in segments) if segments else 0.0
        )

        result = YouTubeTranscriptResult(
            video_id=video_id,
            title=None,
            language=transcript.language_code,
            is_generated=transcript.is_generated,
            segments=segments,
            full_text=full_text,
            duration_seconds=duration,
        )

        logger.info(
            "transcript_fetched",
            video_id=video_id,
            language=transcript.language_code,
            is_generated=transcript.is_generated,
            segment_count=len(segments),
            duration_seconds=duration,
        )

        return result

    except TranscriptsDisabled:
        logger.warning(
            "transcripts_disabled",
            video_id=video_id,
        )
        raise YouTubeIngestionError(
            video_id=video_id,
            reason="Subtitles are disabled for this video",
        )
    except NoTranscriptFound:
        logger.warning(
            "no_transcript_found",
            video_id=video_id,
            languages=languages,
        )
        raise YouTubeIngestionError(
            video_id=video_id,
            reason=f"No transcript found in languages: {languages}",
        )
    except VideoUnavailable:
        logger.warning(
            "video_unavailable",
            video_id=video_id,
        )
        raise YouTubeIngestionError(
            video_id=video_id,
            reason="Video is unavailable",
        )
    except Exception as e:
        logger.exception(
            "transcript_fetch_error",
            video_id=video_id,
            error=str(e),
        )
        raise YouTubeIngestionError(
            video_id=video_id,
            reason=f"Failed to fetch transcript: {str(e)}",
        )


def chunk_transcript(
    result: YouTubeTranscriptResult,
    chunk_duration_seconds: Optional[int] = None,
) -> list[TranscriptChunk]:
    """Chunk transcript by time duration.

    Creates chunks based on time duration rather than character count,
    preserving timestamp metadata for citation and deep linking.

    Args:
        result: YouTube transcript result
        chunk_duration_seconds: Target chunk duration (defaults to settings)

    Returns:
        List of transcript chunks with timestamp metadata
    """
    if chunk_duration_seconds is None:
        settings = get_settings()
        chunk_duration_seconds = settings.youtube_chunk_duration_seconds

    chunks: list[TranscriptChunk] = []
    current_texts: list[str] = []
    current_start: float = 0.0
    chunk_index: int = 0

    for segment in result.segments:
        current_texts.append(segment.text)
        segment_end = segment.start + segment.duration

        # Check if we've exceeded chunk duration
        if segment_end - current_start >= chunk_duration_seconds:
            chunks.append(
                TranscriptChunk(
                    content=" ".join(current_texts),
                    start_time=current_start,
                    end_time=segment_end,
                    video_id=result.video_id,
                    chunk_index=chunk_index,
                )
            )
            current_texts = []
            current_start = segment_end
            chunk_index += 1

    # Don't forget the last chunk
    if current_texts:
        chunks.append(
            TranscriptChunk(
                content=" ".join(current_texts),
                start_time=current_start,
                end_time=result.duration_seconds,
                video_id=result.video_id,
                chunk_index=chunk_index,
            )
        )

    logger.info(
        "transcript_chunked",
        video_id=result.video_id,
        chunk_count=len(chunks),
        chunk_duration_seconds=chunk_duration_seconds,
        total_duration_seconds=result.duration_seconds,
    )

    return chunks


async def ingest_youtube_video(
    url: str,
    languages: Optional[list[str]] = None,
    chunk_duration_seconds: Optional[int] = None,
) -> YouTubeIngestionResult:
    """Ingest a YouTube video by fetching and chunking its transcript.

    This is the main entry point for YouTube ingestion. It extracts
    the video ID, fetches the transcript, and chunks it for indexing.

    Args:
        url: YouTube URL
        languages: Preferred languages (defaults to settings)
        chunk_duration_seconds: Chunk duration (defaults to settings)

    Returns:
        YouTubeIngestionResult with chunked transcript data

    Raises:
        ValueError: If video ID cannot be extracted
        YouTubeIngestionError: If transcript cannot be fetched
    """
    logger.info(
        "ingesting_youtube_video",
        url=url,
    )

    # Extract video ID
    video_id = extract_video_id(url)

    # Fetch transcript
    transcript_result = await fetch_transcript(video_id, languages)

    # Chunk transcript
    chunks = chunk_transcript(transcript_result, chunk_duration_seconds)

    result = YouTubeIngestionResult(
        video_id=video_id,
        source_url=f"https://youtube.com/watch?v={video_id}",
        language=transcript_result.language,
        is_generated=transcript_result.is_generated,
        chunks=chunks,
        duration_seconds=transcript_result.duration_seconds,
        full_text=transcript_result.full_text,
    )

    logger.info(
        "youtube_video_ingested",
        video_id=video_id,
        language=result.language,
        chunk_count=len(result.chunks),
        duration_seconds=result.duration_seconds,
    )

    return result
