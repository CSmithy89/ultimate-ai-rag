# Story 22-TD3: Document Voice I/O Configuration

Status: backlog

Epic: 22 - Advanced Protocol Integration
Priority: P0 - HIGH
Story Points: 1
Owner: Documentation
Origin: Epic 21 Retrospective (Action Item 3)

## Story

As a **developer deploying the platform**,
I want **comprehensive documentation for Voice I/O configuration**,
So that **I can correctly enable and configure speech-to-text and text-to-speech features**.

## Background

Epic 21 Groups E-F added Voice I/O capabilities:
- Speech-to-text via Whisper (OpenAI or local faster-whisper)
- Text-to-speech via OpenAI, ElevenLabs, or pyttsx3
- New environment variables for configuration
- New API endpoints (`/copilot/transcribe`, `/copilot/tts`)

This new feature needs user-facing documentation.

## Acceptance Criteria

1. **Given** a developer reads the documentation, **when** they want to enable voice features, **then** they know which environment variables to set.

2. **Given** multiple TTS providers exist, **when** the developer reviews options, **then** they understand trade-offs (cost, quality, latency, offline capability).

3. **Given** the documentation covers STT, **when** reviewed, **then** Whisper model selection and language hints are explained.

4. **Given** the documentation covers frontend components, **when** reviewed, **then** VoiceInput and SpeakButton usage is documented.

5. **Given** troubleshooting is included, **when** common issues occur, **then** solutions are documented (microphone permissions, audio format errors, rate limits).

## Documentation Outline

### 1. Overview
- What Voice I/O provides
- Feature flag: `VOICE_IO_ENABLED`
- Browser compatibility requirements

### 2. Backend Configuration

#### Speech-to-Text (STT)
| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_IO_ENABLED` | `false` | Master enable/disable |
| `WHISPER_MODEL` | `base` | Model size (tiny, base, small, medium, large) |
| `WHISPER_PROVIDER` | `openai` | Provider (openai, faster-whisper) |

#### Text-to-Speech (TTS)
| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PROVIDER` | `openai` | Provider (openai, elevenlabs, pyttsx3) |
| `TTS_VOICE` | `alloy` | Voice selection |
| `TTS_SPEED` | `1.0` | Speech speed (0.25-4.0) |
| `ELEVENLABS_API_KEY` | - | Required if using ElevenLabs |

### 3. Frontend Components

#### VoiceInput Component
```tsx
import { VoiceInput } from "@/components/copilot/VoiceInput";

<VoiceInput
  onTranscription={(text) => handleTranscription(text)}
  disabled={!voiceEnabled}
/>
```

#### SpeakButton Component
```tsx
import { SpeakButton } from "@/components/copilot/SpeakButton";

<SpeakButton
  text={responseText}
  voice="nova"
  speed={1.2}
/>
```

### 4. API Endpoints

#### POST /copilot/transcribe
- Accepts: audio file (webm, wav, mp3, etc.)
- Returns: `{ text, language, confidence }`
- Max file size: 25MB

#### POST /copilot/tts
- Accepts: `{ text, voice?, speed? }`
- Returns: audio/mpeg stream
- Max text length: 4096 characters

### 5. Provider Comparison

| Provider | Cost | Quality | Latency | Offline |
|----------|------|---------|---------|---------|
| OpenAI Whisper | $$$ | Excellent | Medium | No |
| faster-whisper | Free | Good | Fast | Yes |
| OpenAI TTS | $$$ | Excellent | Medium | No |
| ElevenLabs | $$$$ | Premium | Medium | No |
| pyttsx3 | Free | Basic | Fast | Yes |

### 6. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Microphone permission denied" | Browser blocked access | Check browser permissions |
| "Voice adapter not configured" | Missing env vars | Set VOICE_IO_ENABLED=true |
| "Unsupported media type" | Wrong audio format | Use webm, wav, or mp3 |
| 413 error | File too large | Keep under 25MB |

## Tasks

- [ ] **Task 1: Create Voice I/O Guide**
  - [ ] Create `docs/guides/voice-io-configuration.md`
  - [ ] Include all sections from outline
  - [ ] Add code examples

- [ ] **Task 2: Update .env.example**
  - [ ] Ensure all voice-related variables are documented
  - [ ] Add comments explaining each variable

- [ ] **Task 3: Add to README**
  - [ ] Add Voice I/O to feature list
  - [ ] Link to detailed guide

## Definition of Done

- [ ] Voice I/O guide created in `docs/guides/`
- [ ] `.env.example` includes all voice variables with comments
- [ ] README references voice features
- [ ] Guide reviewed for accuracy

## Files to Create/Modify

1. **Create:** `docs/guides/voice-io-configuration.md`
2. **Modify:** `.env.example`
3. **Modify:** `README.md`
