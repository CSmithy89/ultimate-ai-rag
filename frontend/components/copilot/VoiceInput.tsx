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
 * - Browser-compatible MIME type detection
 * - Keyboard support (Escape to cancel recording)
 */

import { useState, useRef, useCallback, memo, useEffect } from "react";
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
 * Get a supported MIME type for MediaRecorder.
 * Checks browser support and returns the first available option.
 */
function getSupportedMimeType(): string {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
  ];

  for (const type of types) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }

  // Fallback - let browser choose
  return "";
}

/**
 * Get file extension from MIME type.
 */
function getFileExtension(mimeType: string): string {
  if (mimeType.includes("webm")) return "webm";
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("mpeg")) return "mp3";
  return "webm"; // fallback
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
  const mimeTypeRef = useRef<string>("");

  // Auto-dismiss error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Cleanup stream helper
  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Detect supported MIME type for browser compatibility
      const mimeType = getSupportedMimeType();
      mimeTypeRef.current = mimeType;

      // Create MediaRecorder with detected MIME type
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {};
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release microphone
        cleanupStream();

        if (chunksRef.current.length > 0) {
          const actualMimeType = mimeTypeRef.current || "audio/webm";
          const blob = new Blob(chunksRef.current, { type: actualMimeType });
          await transcribeAudio(blob, actualMimeType);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      // Clean up any acquired stream on error
      cleanupStream();

      console.error("Failed to start recording:", err);
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError("Microphone permission denied");
      } else if (err instanceof DOMException && err.name === "NotFoundError") {
        setError("No microphone found");
      } else if (err instanceof DOMException && err.name === "NotSupportedError") {
        setError("Audio recording not supported");
      } else {
        setError("Failed to start recording");
      }
    }
  }, [cleanupStream]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const cancelRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      // Discard chunks before stopping
      chunksRef.current = [];
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const transcribeAudio = useCallback(
    async (blob: Blob, mimeType: string) => {
      setIsTranscribing(true);
      setError(null);

      try {
        const extension = getFileExtension(mimeType);
        const formData = new FormData();
        formData.append("audio", blob, `recording.${extension}`);

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

  // Keyboard support: Escape to cancel recording
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape" && isRecording) {
        e.preventDefault();
        cancelRecording();
      }
    },
    [isRecording, cancelRecording]
  );

  const isDisabled = disabled || isTranscribing;

  return (
    <div className={cn("relative inline-flex", className)}>
      <button
        type="button"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
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
            ? "Stop recording (or press Escape to cancel)"
            : "Start voice input"
        }
        title={
          isTranscribing
            ? "Transcribing..."
            : isRecording
            ? "Stop recording (Escape to cancel)"
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

      {/* ARIA live region for screen readers */}
      <span className="sr-only" aria-live="polite" role="status">
        {isRecording ? "Recording audio" : isTranscribing ? "Transcribing audio" : ""}
      </span>

      {/* Error tooltip (click to dismiss) */}
      {error && (
        <button
          type="button"
          onClick={() => setError(null)}
          className={cn(
            "absolute bottom-full left-1/2 -translate-x-1/2 mb-2",
            "px-2 py-1 text-xs text-white bg-red-600 rounded shadow-lg",
            "whitespace-nowrap cursor-pointer hover:bg-red-700"
          )}
          role="alert"
          aria-label={`Error: ${error}. Click to dismiss.`}
        >
          {error}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-red-600" />
        </button>
      )}
    </div>
  );
});

export default VoiceInput;
