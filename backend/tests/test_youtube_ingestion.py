"""Tests for YouTube transcript ingestion module."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_rag_backend.indexing.youtube_ingestion import (
    YouTubeIngestionError,
    YouTubeIngestionResult,
    YouTubeTranscriptResult,
    TranscriptChunk,
    TranscriptSegment,
    chunk_transcript,
    extract_video_id,
    fetch_transcript,
    ingest_youtube_video,
)


# Create a mock module for youtube_transcript_api to allow patching
# The actual import happens inside the function, so we mock at the top level
class MockYouTubeTranscriptApi:
    """Mock class for patching."""
    pass


class TestExtractVideoId:
    """Tests for video ID extraction from various URL formats."""

    def test_extract_video_id_standard_watch_url(self):
        """Extract video ID from standard youtube.com/watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_watch_url_without_www(self):
        """Extract video ID from youtube.com/watch URL without www."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_short_url(self):
        """Extract video ID from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_embed_url(self):
        """Extract video ID from youtube.com/embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_shorts_url(self):
        """Extract video ID from youtube.com/shorts URL."""
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_with_timestamp(self):
        """Extract video ID from URL with timestamp parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_with_playlist(self):
        """Extract video ID from URL with playlist parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_with_multiple_params(self):
        """Extract video ID from URL with multiple parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30&list=PLtest&index=5"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_short_url_with_timestamp(self):
        """Extract video ID from short URL with timestamp."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=30"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_http_url(self):
        """Extract video ID from HTTP (non-HTTPS) URL."""
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid_url_raises_error(self):
        """Raise ValueError for invalid URL."""
        url = "https://example.com/video/12345"
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id(url)

    def test_extract_video_id_empty_url_raises_error(self):
        """Raise ValueError for empty URL."""
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("")

    def test_extract_video_id_malformed_url_raises_error(self):
        """Raise ValueError for malformed YouTube URL."""
        url = "https://www.youtube.com/watch?video=dQw4w9WgXcQ"
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id(url)

    def test_extract_video_id_with_underscore_and_hyphen(self):
        """Extract video ID containing underscores and hyphens."""
        url = "https://www.youtube.com/watch?v=abc_DEF-123"
        assert extract_video_id(url) == "abc_DEF-123"


class TestFetchTranscript:
    """Tests for transcript fetching."""

    @pytest.mark.asyncio
    async def test_fetch_transcript_success(self):
        """Test successful transcript fetching."""
        mock_segments = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "This is a test", "start": 2.0, "duration": 3.0},
            {"text": "of the transcript", "start": 5.0, "duration": 2.5},
        ]

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False
        mock_transcript.fetch.return_value = mock_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.return_value = mock_transcript_list

            result = await fetch_transcript("dQw4w9WgXcQ", ["en"])

            assert result.video_id == "dQw4w9WgXcQ"
            assert result.language == "en"
            assert result.is_generated is False
            assert len(result.segments) == 3
            assert result.full_text == "Hello world This is a test of the transcript"
            assert result.duration_seconds == 7.5

    @pytest.mark.asyncio
    async def test_fetch_transcript_auto_generated(self):
        """Test fetching auto-generated transcript."""
        mock_segments = [
            {"text": "Auto generated", "start": 0.0, "duration": 2.0},
        ]

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en-US"
        mock_transcript.is_generated = True
        mock_transcript.fetch.return_value = mock_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.return_value = mock_transcript_list

            result = await fetch_transcript("test123abc", ["en", "en-US"])

            assert result.is_generated is True
            assert result.language == "en-US"

    @pytest.mark.asyncio
    async def test_fetch_transcript_disabled_error(self):
        """Test error when transcripts are disabled."""
        from youtube_transcript_api._errors import TranscriptsDisabled

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.side_effect = TranscriptsDisabled("test123")

            with pytest.raises(YouTubeIngestionError) as exc_info:
                await fetch_transcript("test123abc", ["en"])

            assert "Subtitles are disabled" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_fetch_transcript_not_found_error(self):
        """Test error when no transcript found in requested languages."""
        from youtube_transcript_api._errors import NoTranscriptFound

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.side_effect = NoTranscriptFound(
                "test123",
                ["en"],
                MagicMock(),
            )

            with pytest.raises(YouTubeIngestionError) as exc_info:
                await fetch_transcript("test123abc", ["en"])

            assert "No transcript found" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_fetch_transcript_unavailable_error(self):
        """Test error when video is unavailable."""
        from youtube_transcript_api._errors import VideoUnavailable

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.side_effect = VideoUnavailable("test123")

            with pytest.raises(YouTubeIngestionError) as exc_info:
                await fetch_transcript("test123abc", ["en"])

            assert "Video is unavailable" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_fetch_transcript_generic_error(self):
        """Test handling of generic errors."""
        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.side_effect = Exception("Network error")

            with pytest.raises(YouTubeIngestionError) as exc_info:
                await fetch_transcript("test123abc", ["en"])

            assert "Failed to fetch transcript" in exc_info.value.reason


class TestChunkTranscript:
    """Tests for transcript chunking."""

    def test_chunk_transcript_by_duration(self):
        """Chunk 6-minute video into 3 chunks (2-min each)."""
        # Create segments totaling ~6 minutes
        segments = [
            TranscriptSegment(text=f"Segment {i}", start=i * 30.0, duration=30.0)
            for i in range(12)  # 12 segments * 30s = 6 minutes
        ]

        result = YouTubeTranscriptResult(
            video_id="test123abc",
            language="en",
            is_generated=False,
            segments=segments,
            full_text=" ".join(s.text for s in segments),
            duration_seconds=360.0,  # 6 minutes
        )

        chunks = chunk_transcript(result, chunk_duration_seconds=120)

        # Should have 3 chunks for 6 minutes at 2-minute intervals
        assert len(chunks) == 3

        # Verify chunk indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.video_id == "test123abc"

    def test_chunk_transcript_metadata(self):
        """Verify start_time and end_time in chunks."""
        segments = [
            TranscriptSegment(text="First segment", start=0.0, duration=60.0),
            TranscriptSegment(text="Second segment", start=60.0, duration=60.0),
            TranscriptSegment(text="Third segment", start=120.0, duration=60.0),
        ]

        result = YouTubeTranscriptResult(
            video_id="test123abc",
            language="en",
            is_generated=False,
            segments=segments,
            full_text="First segment Second segment Third segment",
            duration_seconds=180.0,
        )

        # Chunk at 2 minutes (120 seconds)
        chunks = chunk_transcript(result, chunk_duration_seconds=120)

        assert len(chunks) == 2

        # First chunk: 0-120s
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 120.0

        # Second chunk: 120-180s
        assert chunks[1].start_time == 120.0
        assert chunks[1].end_time == 180.0

    def test_chunk_transcript_short_video(self):
        """Single chunk for video under chunk_duration."""
        segments = [
            TranscriptSegment(text="Short video", start=0.0, duration=30.0),
        ]

        result = YouTubeTranscriptResult(
            video_id="test123abc",
            language="en",
            is_generated=False,
            segments=segments,
            full_text="Short video",
            duration_seconds=30.0,
        )

        chunks = chunk_transcript(result, chunk_duration_seconds=120)

        assert len(chunks) == 1
        assert chunks[0].content == "Short video"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 30.0

    def test_chunk_transcript_includes_video_id(self):
        """Verify video_id in chunk metadata."""
        segments = [
            TranscriptSegment(text="Test", start=0.0, duration=10.0),
        ]

        result = YouTubeTranscriptResult(
            video_id="unique_id123",
            language="en",
            is_generated=False,
            segments=segments,
            full_text="Test",
            duration_seconds=10.0,
        )

        chunks = chunk_transcript(result, chunk_duration_seconds=120)

        assert all(chunk.video_id == "unique_id123" for chunk in chunks)

    def test_chunk_transcript_empty_segments(self):
        """Handle empty segments list."""
        result = YouTubeTranscriptResult(
            video_id="test123abc",
            language="en",
            is_generated=False,
            segments=[],
            full_text="",
            duration_seconds=0.0,
        )

        chunks = chunk_transcript(result, chunk_duration_seconds=120)

        assert len(chunks) == 0

    def test_chunk_transcript_content_concatenation(self):
        """Verify segment text is properly concatenated."""
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=30.0),
            TranscriptSegment(text="world", start=30.0, duration=30.0),
            TranscriptSegment(text="test", start=60.0, duration=30.0),
        ]

        result = YouTubeTranscriptResult(
            video_id="test123abc",
            language="en",
            is_generated=False,
            segments=segments,
            full_text="Hello world test",
            duration_seconds=90.0,
        )

        chunks = chunk_transcript(result, chunk_duration_seconds=120)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello world test"


class TestIngestYoutubeVideo:
    """Tests for the complete ingestion workflow."""

    @pytest.mark.asyncio
    async def test_ingest_youtube_video_success(self):
        """Full workflow with mocked API."""
        mock_segments = [
            {"text": "Hello", "start": 0.0, "duration": 30.0},
            {"text": "world", "start": 30.0, "duration": 30.0},
        ]

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False
        mock_transcript.fetch.return_value = mock_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.return_value = mock_transcript_list

            result = await ingest_youtube_video(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                languages=["en"],
                chunk_duration_seconds=120,
            )

            assert isinstance(result, YouTubeIngestionResult)
            assert result.video_id == "dQw4w9WgXcQ"
            assert result.source_url == "https://youtube.com/watch?v=dQw4w9WgXcQ"
            assert result.language == "en"
            assert result.is_generated is False
            assert len(result.chunks) == 1
            assert result.full_text == "Hello world"
            assert result.duration_seconds == 60.0

    @pytest.mark.asyncio
    async def test_ingest_youtube_video_invalid_url(self):
        """Error propagation for invalid URL."""
        with pytest.raises(ValueError, match="Could not extract video ID"):
            await ingest_youtube_video("https://example.com/video/12345")

    @pytest.mark.asyncio
    async def test_ingest_youtube_video_transcript_error(self):
        """Error propagation when transcript fetch fails."""
        from youtube_transcript_api._errors import TranscriptsDisabled

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.side_effect = TranscriptsDisabled("test123")

            with pytest.raises(YouTubeIngestionError):
                await ingest_youtube_video(
                    "https://www.youtube.com/watch?v=test123abcd",
                    languages=["en"],
                )

    @pytest.mark.asyncio
    async def test_ingest_youtube_video_chunk_metadata(self):
        """Verify chunk metadata is properly set."""
        mock_segments = [
            {"text": "Part 1", "start": 0.0, "duration": 120.0},
            {"text": "Part 2", "start": 120.0, "duration": 120.0},
            {"text": "Part 3", "start": 240.0, "duration": 60.0},
        ]

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = True
        mock_transcript.fetch.return_value = mock_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.list_transcripts.return_value = mock_transcript_list

            result = await ingest_youtube_video(
                "https://youtu.be/dQw4w9WgXcQ",
                languages=["en"],
                chunk_duration_seconds=120,
            )

            assert len(result.chunks) == 3

            # Verify all chunks have video_id
            for chunk in result.chunks:
                assert chunk.video_id == "dQw4w9WgXcQ"

            # Verify chunk indices
            for i, chunk in enumerate(result.chunks):
                assert chunk.chunk_index == i


class TestYouTubeIngestionError:
    """Tests for YouTubeIngestionError."""

    def test_error_message_format(self):
        """Verify error message format."""
        error = YouTubeIngestionError(
            video_id="test123abc",
            reason="Test reason",
        )

        assert error.video_id == "test123abc"
        assert error.reason == "Test reason"
        assert "test123abc" in str(error)
        assert "Test reason" in str(error)


class TestTranscriptModels:
    """Tests for Pydantic model validation."""

    def test_transcript_segment_model(self):
        """Test TranscriptSegment model creation."""
        segment = TranscriptSegment(
            text="Hello world",
            start=0.0,
            duration=5.0,
        )

        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.duration == 5.0

    def test_youtube_transcript_result_model(self):
        """Test YouTubeTranscriptResult model creation."""
        result = YouTubeTranscriptResult(
            video_id="test123abc",
            language="en",
            is_generated=False,
            segments=[],
            full_text="Test",
            duration_seconds=60.0,
        )

        assert result.video_id == "test123abc"
        assert result.title is None  # Optional field
        assert result.language == "en"
        assert result.is_generated is False
        assert result.duration_seconds == 60.0

    def test_transcript_chunk_model(self):
        """Test TranscriptChunk model creation."""
        chunk = TranscriptChunk(
            content="Test content",
            start_time=0.0,
            end_time=120.0,
            video_id="test123abc",
            chunk_index=0,
        )

        assert chunk.content == "Test content"
        assert chunk.start_time == 0.0
        assert chunk.end_time == 120.0
        assert chunk.video_id == "test123abc"
        assert chunk.chunk_index == 0

    def test_youtube_ingestion_result_model(self):
        """Test YouTubeIngestionResult model creation."""
        result = YouTubeIngestionResult(
            video_id="test123abc",
            source_url="https://youtube.com/watch?v=test123abc",
            language="en",
            is_generated=False,
            chunks=[],
            duration_seconds=120.0,
            full_text="Full transcript text",
        )

        assert result.video_id == "test123abc"
        assert result.source_url == "https://youtube.com/watch?v=test123abc"
        assert result.language == "en"
        assert result.is_generated is False
        assert len(result.chunks) == 0
        assert result.duration_seconds == 120.0
        assert result.full_text == "Full transcript text"
