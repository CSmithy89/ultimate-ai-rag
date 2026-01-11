# Voice I/O Configuration Guide

This guide covers configuring the Voice I/O features for speech-to-text (STT) and text-to-speech (TTS) capabilities.

Origin: Epic 21 Groups E-F

## Overview

Voice I/O provides:
- **Speech-to-Text (STT)**: Transcribe audio input via OpenAI Whisper
- **Text-to-Speech (TTS)**: Generate spoken audio from text via OpenAI, ElevenLabs, or pyttsx3

## Quick Start

Enable voice features by setting:

```bash
VOICE_IO_ENABLED=true
OPENAI_API_KEY=your-openai-key  # Required for STT and OpenAI TTS
```

## Backend Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_IO_ENABLED` | `false` | Master enable/disable for voice features |
| `WHISPER_MODEL` | `base` | Whisper model size for STT |
| `TTS_PROVIDER` | `openai` | TTS provider (openai, elevenlabs, pyttsx3) |
| `TTS_VOICE` | `alloy` | Voice selection for TTS |
| `TTS_SPEED` | `1.0` | Speech speed (0.25-4.0) |
| `ELEVENLABS_API_KEY` | - | Required if using ElevenLabs TTS |

### Speech-to-Text (STT) Settings

The STT feature uses OpenAI Whisper for transcription.

**Whisper Model Options:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | 39M | Fastest | Lower | Quick prototyping |
| `base` | 74M | Fast | Good | Default, balanced |
| `small` | 244M | Medium | Better | Production |
| `medium` | 769M | Slow | High | High accuracy needed |
| `large` | 1550M | Slowest | Best | Maximum accuracy |

```bash
# Example: Use small model for better accuracy
WHISPER_MODEL=small
```

### Text-to-Speech (TTS) Settings

**Provider Options:**

| Provider | Cost | Quality | Latency | Offline |
|----------|------|---------|---------|---------|
| `openai` | $$$ | Excellent | Medium | No |
| `elevenlabs` | $$$$ | Premium | Medium | No |
| `pyttsx3` | Free | Basic | Fast | Yes |

**OpenAI Voice Options:**

- `alloy` - Neutral, balanced
- `echo` - Deep, resonant
- `fable` - Expressive, warm
- `onyx` - Deep, authoritative
- `nova` - Bright, energetic
- `shimmer` - Soft, pleasant

```bash
# Example: Use Nova voice at 1.2x speed
TTS_PROVIDER=openai
TTS_VOICE=nova
TTS_SPEED=1.2
```

**ElevenLabs Configuration:**

```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your-elevenlabs-key
TTS_VOICE=rachel  # ElevenLabs voice ID
```

**Offline Mode (pyttsx3):**

```bash
TTS_PROVIDER=pyttsx3
# No API key needed - uses local system TTS
```

## API Endpoints

### POST /copilot/transcribe

Transcribe audio to text.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/copilot/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "file=@recording.webm"
```

**Accepted Formats:** webm, wav, mp3, m4a, ogg, flac
**Max File Size:** 25MB

**Response:**
```json
{
  "text": "Hello, this is the transcribed text.",
  "language": "en",
  "confidence": 0.95,
  "duration_seconds": 3.2
}
```

### POST /copilot/tts

Generate speech from text.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/copilot/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice": "nova", "speed": 1.0}' \
  --output speech.mp3
```

**Max Text Length:** 4096 characters

**Response:** `audio/mpeg` stream (MP3 data)

## Frontend Components

### VoiceInput Component

Captures audio from the user's microphone and sends it for transcription.

```tsx
import { VoiceInput } from "@/components/copilot/VoiceInput";

function ChatInput() {
  const handleTranscription = (text: string) => {
    console.log("Transcribed:", text);
    // Send text to chat
  };

  return (
    <VoiceInput
      onTranscription={handleTranscription}
      disabled={!voiceEnabled}
    />
  );
}
```

**Props:**

| Prop | Type | Description |
|------|------|-------------|
| `onTranscription` | `(text: string) => void` | Callback when transcription completes |
| `disabled` | `boolean` | Disable the voice input button |
| `className` | `string` | Additional CSS classes |

### SpeakButton Component

Plays text-to-speech for a given text.

```tsx
import { SpeakButton } from "@/components/copilot/SpeakButton";

function MessageBubble({ message }) {
  return (
    <div>
      <p>{message.text}</p>
      <SpeakButton
        text={message.text}
        voice="nova"
        speed={1.0}
      />
    </div>
  );
}
```

**Props:**

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `string` | required | Text to speak |
| `voice` | `OpenAIVoice` | `"alloy"` | Voice to use |
| `speed` | `number` | `1.0` | Speech speed (0.25-4.0) |
| `className` | `string` | - | Additional CSS classes |

## Browser Compatibility

Voice I/O requires browser support for:
- MediaRecorder API (for audio capture)
- Web Audio API (for audio playback)

**Supported Browsers:**
- Chrome 49+
- Firefox 29+
- Safari 14.1+
- Edge 79+

**Note:** Users must grant microphone permission for voice input.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Microphone permission denied" | Browser blocked access | Check browser permissions for the site |
| "Voice adapter not configured" | VOICE_IO_ENABLED=false | Set `VOICE_IO_ENABLED=true` in .env |
| "Unsupported media type" | Wrong audio format | Use webm, wav, or mp3 |
| 413 error | File too large | Keep audio under 25MB |
| "Rate limit exceeded" | Too many requests | Wait and retry |
| No audio output | TTS provider issue | Check API key and provider settings |

## Cost Considerations

**OpenAI Whisper (STT):**
- $0.006 per minute of audio

**OpenAI TTS:**
- Standard: $0.015 per 1K characters
- HD: $0.030 per 1K characters

**ElevenLabs:**
- Varies by plan ($5-$330/month)

**pyttsx3:**
- Free (uses local system TTS)

## Security Considerations

- Audio is processed server-side; sensitive audio should use secure connections (HTTPS)
- OpenAI API key is required for STT and OpenAI TTS
- ElevenLabs API key is stored server-side only
- Audio files are not persisted after processing
