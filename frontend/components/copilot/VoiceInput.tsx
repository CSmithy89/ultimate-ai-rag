"use client";

/**
 * VoiceInput component for speech-to-text recording.
 *
 * Story 21-E1: Implement Voice Input (Speech-to-Text)
 *
 * Features:
 * - Audio recording via MediaRecorder API
 * - Transcription via backend /copilot/transcribe endpoint
 * - Visual feedback during recording/transcription
 * - Error handling for microphone permissions
 * - Accessibility labels for screen readers
 */

import { useState, useRef, useCallback, memo } from "react";
import { Mic, MicOff, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface VoiceInputProps {
  /** Callback when transcription completes */
  onTranscription: (text: string) => void;
  /** Whether the button is disabled */
  disabled?: boolean;
  /** Additional class names */
  className?: string;
  /** Language hint for transcription (ISO 639-1 code) */
  language?: string;
}

/**
 * VoiceInput provides a microphone button for speech-to-text input.
 *
 * Usage:
 * ```tsx
 * <VoiceInput
 *   onTranscription={(text) => setMessage(prev => prev + text)}
 *   disabled={isProcessing}
 *   language="en"
 * />
 * ```
 */
export const VoiceInput = memo(function VoiceInput({
  onTranscription,
  disabled = false,
  className,
  language = "en",
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    setError(null);

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Create MediaRecorder with webm format (widely supported)
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release microphone
        streamRef.current?.getTracks().forEach((track) => track.stop());

        if (chunksRef.current.length > 0) {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          await transcribeAudio(blob);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to start recording:", err);
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError("Microphone permission denied");
      } else if (err instanceof DOMException && err.name === "NotFoundError") {
        setError("No microphone found");
      } else {
        setError("Failed to start recording");
      }
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const transcribeAudio = useCallback(
    async (blob: Blob) => {
      setIsTranscribing(true);
      setError(null);

      try {
        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");

        const response = await fetch(`/api/copilot/transcribe?language=${language}`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Transcription failed: ${response.status}`);
        }

        const data = await response.json();

        if (data.text) {
          onTranscription(data.text);
        }
      } catch (err) {
        console.error("Transcription error:", err);
        setError(err instanceof Error ? err.message : "Transcription failed");
      } finally {
        setIsTranscribing(false);
      }
    },
    [language, onTranscription]
  );

  const handleClick = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const isDisabled = disabled || isTranscribing;

  return (
    <div className={cn("relative inline-flex", className)}>
      <button
        type="button"
        onClick={handleClick}
        disabled={isDisabled}
        className={cn(
          "inline-flex items-center justify-center rounded-md p-2",
          "transition-colors duration-200",
          "focus:outline-none focus:ring-2 focus:ring-offset-2",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          isRecording
            ? "text-red-500 hover:text-red-600 hover:bg-red-50 focus:ring-red-500"
            : "text-slate-600 hover:text-slate-700 hover:bg-slate-100 focus:ring-slate-500"
        )}
        aria-label={
          isTranscribing
            ? "Transcribing audio..."
            : isRecording
            ? "Stop recording"
            : "Start voice input"
        }
        title={
          isTranscribing
            ? "Transcribing..."
            : isRecording
            ? "Stop recording"
            : "Voice input"
        }
      >
        {isTranscribing ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : isRecording ? (
          <MicOff className="h-4 w-4" />
        ) : (
          <Mic className="h-4 w-4" />
        )}
      </button>

      {/* Recording indicator */}
      {isRecording && (
        <span className="absolute -top-1 -right-1 flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
        </span>
      )}

      {/* Error tooltip */}
      {error && (
        <div
          className={cn(
            "absolute bottom-full left-1/2 -translate-x-1/2 mb-2",
            "px-2 py-1 text-xs text-white bg-red-600 rounded shadow-lg",
            "whitespace-nowrap"
          )}
          role="alert"
        >
          {error}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-red-600" />
        </div>
      )}
    </div>
  );
});

export default VoiceInput;
