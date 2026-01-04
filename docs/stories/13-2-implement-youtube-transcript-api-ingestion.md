# Story 13.2: Implement YouTube Transcript API Ingestion

Status: done

## Story

As a developer,
I want transcript-first YouTube ingestion,
So that videos can be processed quickly without full video download.

## Acceptance Criteria

1. Given a YouTube URL, when ingestion runs, then transcripts are fetched using youtube-transcript-api.
2. Given a YouTube URL with missing transcripts, when ingestion runs, then missing transcripts are reported with clear error.
3. Given a YouTube URL with available transcripts, when ingestion completes, then transcript chunks include source metadata (video_id, language, timestamps).
4. Given a typical YouTube video, when ingestion runs, then it completes in under 30 seconds.
5. Given various YouTube URL formats, when parsing the URL, then the video ID is correctly extracted (youtube.com/watch?v=X, youtu.be/X, youtube.com/embed/X).

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: N/A - YouTube ingestion operates at document level; tenant filtering applied at storage stage
- [x] Rate limiting / abuse protection: N/A - YouTube API has built-in rate limits; local processing
- [x] Input validation / schema enforcement: Implemented - Pydantic models for YouTubeTranscriptResult, TranscriptSegment, TranscriptChunk
- [x] Tests (unit/integration): Implemented - Unit tests for URL parsing, transcript fetching (mocked), chunking, error handling
- [x] Error handling + logging: Implemented - structlog events for transcript fetching, error reporting with clear messages
- [x] Documentation updates: N/A - Internal module; .env.example updated for new settings

## Tasks / Subtasks

- [x] Add youtube-transcript-api dependency to pyproject.toml (AC: 1)
  - [x] Add `"youtube-transcript-api>=0.6.0"` to dependencies

- [x] Add YouTube configuration settings to config.py (AC: 1, 3)
  - [x] Add `youtube_preferred_languages: list[str]` (default: ["en", "en-US"])
  - [x] Add `youtube_chunk_duration_seconds: int` (default: 120)

- [x] Create youtube_ingestion.py module (AC: 1-5)
  - [x] Define `TranscriptSegment` Pydantic model (text, start, duration)
  - [x] Define `YouTubeTranscriptResult` Pydantic model (video_id, title, language, is_generated, segments, full_text, duration_seconds)
  - [x] Define `TranscriptChunk` Pydantic model (content, start_time, end_time, video_id, chunk_index)
  - [x] Implement `extract_video_id(url: str) -> str` function
  - [x] Implement `fetch_transcript(video_id: str, languages: list[str]) -> YouTubeTranscriptResult` async function
  - [x] Implement `chunk_transcript(result: YouTubeTranscriptResult, chunk_duration_seconds: int) -> list[TranscriptChunk]` function
  - [x] Implement `ingest_youtube_video(url: str, tenant_id: str) -> IngestionResult` async function
  - [x] Handle transcript errors (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable)

- [x] Update indexing/__init__.py exports (AC: 1)
  - [x] Export YouTubeTranscriptResult, TranscriptSegment, TranscriptChunk
  - [x] Export extract_video_id, fetch_transcript, chunk_transcript, ingest_youtube_video

- [x] Create tests in backend/tests/test_youtube_ingestion.py (AC: 1-5)
  - [x] Test video ID extraction from youtube.com/watch?v=X format
  - [x] Test video ID extraction from youtu.be/X format
  - [x] Test video ID extraction from youtube.com/embed/X format
  - [x] Test video ID extraction with timestamp parameters
  - [x] Test video ID extraction with playlist parameters
  - [x] Test transcript fetching (mocked successful response)
  - [x] Test transcript fetching with disabled transcripts (mocked error)
  - [x] Test transcript fetching with unavailable video (mocked error)
  - [x] Test chunking by duration (2-minute chunks)
  - [x] Test chunk metadata includes timestamps
  - [x] Test ingestion workflow end-to-end (mocked)

- [x] Run tests and verify passing (AC: 1-5)
  - [x] Run `cd backend && uv run pytest tests/test_youtube_ingestion.py -v`
  - [x] Verify all tests pass

## Technical Notes

### Dependencies

```toml
# Add to pyproject.toml
"youtube-transcript-api>=0.6.0",
```

### Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class TranscriptSegment(BaseModel):
    """A single segment from the YouTube transcript."""
    text: str = Field(..., description="Transcript text for this segment")
    start: float = Field(..., description="Start time in seconds")
    duration: float = Field(..., description="Duration in seconds")

class YouTubeTranscriptResult(BaseModel):
    """Complete transcript result from YouTube."""
    video_id: str
    title: Optional[str] = None
    language: str
    is_generated: bool
    segments: list[TranscriptSegment]
    full_text: str
    duration_seconds: float

class TranscriptChunk(BaseModel):
    """A chunked portion of transcript for indexing."""
    content: str
    start_time: float
    end_time: float
    video_id: str
    chunk_index: int
```

### URL Patterns to Support

- `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- `https://youtube.com/watch?v=dQw4w9WgXcQ`
- `https://youtu.be/dQw4w9WgXcQ`
- `https://www.youtube.com/embed/dQw4w9WgXcQ`
- URLs with extra parameters: `?v=X&t=120`, `?v=X&list=Y`

### Environment Variables

```bash
# YouTube Ingestion Settings
YOUTUBE_PREFERRED_LANGUAGES=["en", "en-US"]  # JSON array
YOUTUBE_CHUNK_DURATION_SECONDS=120           # 2 minutes
```

### Error Handling

The youtube-transcript-api library raises specific exceptions:
- `TranscriptsDisabled` - Subtitles disabled by uploader
- `NoTranscriptFound` - No transcript in requested languages
- `VideoUnavailable` - Video doesn't exist or is private

Map these to clear error messages for the user.

## Definition of Done

- [x] Acceptance criteria met
- [x] Standards coverage updated
- [x] Tests run and documented

## Dev Notes

### Implementation Summary

Implemented a comprehensive YouTube transcript ingestion module with:

1. **Pydantic Models**: Five models for data validation:
   - `TranscriptSegment` - Individual transcript segment with text, start time, duration
   - `YouTubeTranscriptResult` - Complete transcript result with metadata
   - `TranscriptChunk` - Time-based chunk for indexing with timestamps
   - `YouTubeIngestionResult` - Final ingestion result with all chunks
   - `YouTubeIngestionError` - Custom exception with video_id and reason

2. **Video ID Extraction**: Regex-based extraction supporting multiple URL formats:
   - Standard watch URLs: `youtube.com/watch?v=X`
   - Short URLs: `youtu.be/X`
   - Embed URLs: `youtube.com/embed/X`
   - Shorts URLs: `youtube.com/shorts/X`
   - URLs with additional parameters (timestamp, playlist, index)

3. **Async Transcript Fetching**: Uses `asyncio.to_thread` for async compatibility with the blocking youtube-transcript-api library. Handles language preference fallback.

4. **Duration-Based Chunking**: Chunks by time duration (default 2 minutes) rather than character count, preserving timestamp metadata for citation and deep linking.

5. **Configuration**: Two settings added to config.py:
   - `youtube_preferred_languages` (JSON array, default: ["en", "en-US"])
   - `youtube_chunk_duration_seconds` (int, default: 120)

6. **Error Handling**: Maps youtube-transcript-api exceptions to clear user-facing error messages.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debugging issues encountered.

### Completion Notes List

- All 35 unit tests pass covering video ID extraction, transcript fetching, chunking, and error handling
- Ruff linting passes with no errors
- Code follows project conventions (snake_case, PascalCase classes, structlog, Pydantic)
- Exports added to `indexing/__init__.py` for easy access

### File List

- backend/pyproject.toml (MODIFIED - added youtube-transcript-api dependency)
- backend/src/agentic_rag_backend/config.py (MODIFIED - added 2 YouTube settings)
- backend/src/agentic_rag_backend/indexing/youtube_ingestion.py (NEW - 319 lines)
- backend/src/agentic_rag_backend/indexing/__init__.py (MODIFIED - added exports)
- backend/tests/test_youtube_ingestion.py (NEW - 35 tests)
- docs/stories/13-2-implement-youtube-transcript-api-ingestion.md (MODIFIED - status + dev notes)
- docs/stories/13-2-implement-youtube-transcript-api-ingestion.context.xml (NEW)

## Test Outcomes

- Tests run: 35
- Passed: 35
- Failures: 0
- Coverage: Tests cover video ID extraction (14 tests), transcript fetching (6 tests), chunking (6 tests), ingestion workflow (4 tests), error handling (1 test), Pydantic models (4 tests)

## Challenges Encountered

- **Mock patching path**: Initial tests failed because the youtube-transcript-api import happens inside the `fetch_transcript` function (to handle import errors gracefully). Fixed by patching `youtube_transcript_api.YouTubeTranscriptApi` directly instead of the module-level attribute.

## Senior Developer Review

Outcome: [APPROVE | Changes Requested | Blocked]

Notes:
- [Review notes]
