# Story 20-H4: Implement Voice I/O

Status: done

## Story

As a developer building accessible AI applications,
I want to support voice input and output,
so that users can interact with the RAG system through speech.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group H: Competitive Features. It implements voice interaction capabilities enabling:

- **Speech-to-Text (STT)**: Convert voice input to text using Whisper
- **Text-to-Speech (TTS)**: Generate spoken responses using OpenAI or local TTS
- **Voice Adapter**: Feature flag wrapper for voice capabilities

**Competitive Positioning**: Voice interaction makes RAG systems more accessible and enables hands-free operation in various use cases.

**Dependencies**:
- openai-whisper or faster-whisper for STT
- OpenAI TTS API or pyttsx3 for TTS
- sounddevice/soundfile for audio handling

## Acceptance Criteria

1. Given VOICE_IO_ENABLED=true, when the system starts, then voice capabilities are available.
2. Given voice input, when STT runs, then accurate text transcription is returned.
3. Given text response, when TTS runs, then audio output is generated.
4. Given VOICE_IO_ENABLED=false (default), when the system starts, then voice features are not active.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/voice/
├── __init__.py                # Exports
├── stt.py                     # SpeechToText with Whisper
├── tts.py                     # TextToSpeech with OpenAI/pyttsx3
├── adapter.py                 # VoiceAdapter feature flag wrapper
└── models.py                  # VoiceConfig, TranscriptionResult models
```

### Core Components

1. **SpeechToText** - Whisper-based transcription:
   - Uses faster-whisper for efficient local transcription
   - Supports multiple Whisper models (tiny, base, small, medium, large)
   - Returns transcription with confidence and timing

2. **TextToSpeech** - Audio generation:
   - OpenAI TTS API for high-quality voices
   - pyttsx3 fallback for offline/local TTS
   - Configurable voice and speed

3. **VoiceAdapter** - Feature flag wrapper:
   - Wraps STT and TTS behind feature flag
   - Graceful fallback when disabled

### Configuration

```bash
VOICE_IO_ENABLED=true|false              # Default: false
WHISPER_MODEL=tiny|base|small|medium     # Default: base
TTS_PROVIDER=openai|pyttsx3              # Default: openai
TTS_VOICE=alloy|echo|fable|onyx|nova     # Default: alloy
TTS_SPEED=0.5-2.0                        # Default: 1.0
```

## Tasks / Subtasks

- [x] Create voice/ module directory
- [x] Create models.py with VoiceConfig, TranscriptionResult
- [x] Implement SpeechToText with Whisper
- [x] Implement TextToSpeech with OpenAI/pyttsx3
- [x] Implement VoiceAdapter with feature flag
- [x] Create __init__.py with exports
- [x] Add configuration variables to settings
- [x] Write unit tests for all components

## Testing Requirements

### Unit Tests
- STT transcription with mocked Whisper
- TTS generation with mocked API
- Feature flag behavior
- Adapter fallback when disabled

## Definition of Done

- [x] All acceptance criteria pass
- [x] All tasks completed
- [x] Unit test coverage >= 80%
- [x] Feature flag (VOICE_IO_ENABLED) works correctly
- [x] Configuration documented
- [x] Code review approved
- [x] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-H4 section)
- faster-whisper is more efficient than openai-whisper
- OpenAI TTS is higher quality but requires API key
- pyttsx3 works offline but lower quality

---

## Dev Agent Record

### File List

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/voice/__init__.py` | NEW | Module exports |
| `backend/src/agentic_rag_backend/voice/models.py` | NEW | VoiceConfig, TranscriptionResult |
| `backend/src/agentic_rag_backend/voice/stt.py` | NEW | SpeechToText with Whisper |
| `backend/src/agentic_rag_backend/voice/tts.py` | NEW | TextToSpeech with OpenAI/pyttsx3 |
| `backend/src/agentic_rag_backend/voice/adapter.py` | NEW | VoiceAdapter feature flag wrapper |
| `backend/src/agentic_rag_backend/config.py` | MODIFIED | Add voice settings |
| `backend/tests/voice/test_voice.py` | NEW | Unit tests |

### Change Log

| Date | Change | Details |
|------|--------|---------|
| 2026-01-06 | Initial implementation | Created story file |
| 2026-01-06 | Full implementation | Created voice/ module with SpeechToText (Whisper/faster-whisper), TextToSpeech (OpenAI/pyttsx3), VoiceAdapter feature flag wrapper. Added 5 config settings. 33 unit tests passing. |
