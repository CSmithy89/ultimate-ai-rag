"use client";

/**
 * SpeakButton component for text-to-speech playback.
 *
 * Story 21-E2: Implement Voice Output (Text-to-Speech)
 *
 * Features:
 * - TTS synthesis via backend /copilot/tts endpoint
 * - Audio playback with stop control
 * - Voice/speed configuration options
 * - Error handling for synthesis failures
 * - Accessibility: screen reader compatible
 */

import { useState, useRef, useCallback, memo, useEffect } from "react";
import { Volume2, VolumeX, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SpeakButtonProps {
  /** Text to synthesize and speak */
  text: string;
  /** Whether the button is disabled */
  disabled?: boolean;
  /** Additional class names */
  className?: string;
  /** Voice to use (alloy, echo, fable, onyx, nova, shimmer) */
  voice?: string;
  /** Speech speed multiplier (0.25 to 4.0) */
  speed?: number;
}

/**
 * SpeakButton provides a button to play text-to-speech audio.
 *
 * Usage:
 * ```tsx
 * <SpeakButton
 *   text="Hello, this is the AI response."
 *   voice="alloy"
 *   speed={1.0}
 * />
 * ```
 */
export const SpeakButton = memo(function SpeakButton({
  text,
  disabled = false,
  className,
  voice,
  speed,
}: SpeakButtonProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  // Auto-dismiss error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const speak = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const requestBody: { text: string; voice?: string; speed?: number } = { text };
      if (voice) requestBody.voice = voice;
      if (speed) requestBody.speed = speed;

      const response = await fetch("/api/copilot/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `TTS failed: ${response.status}`);
      }

      // Cleanup previous audio if any
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      objectUrlRef.current = url;

      audioRef.current = new Audio(url);
      audioRef.current.onended = () => {
        setIsPlaying(false);
      };
      audioRef.current.onerror = () => {
        // Clean up on error to prevent memory leak
        if (objectUrlRef.current) {
          URL.revokeObjectURL(objectUrlRef.current);
          objectUrlRef.current = null;
        }
        setError("Audio playback failed");
        setIsPlaying(false);
      };

      await audioRef.current.play();
      setIsPlaying(true);
    } catch (err) {
      console.error("TTS error:", err);
      setError(err instanceof Error ? err.message : "TTS failed");
    } finally {
      setIsLoading(false);
    }
  }, [text, voice, speed]);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  }, []);

  const handleClick = useCallback(() => {
    if (isPlaying) {
      stop();
    } else {
      speak();
    }
  }, [isPlaying, speak, stop]);

  const isDisabled = disabled || isLoading || !text;

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
          isPlaying
            ? "text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 focus:ring-indigo-500"
            : "text-slate-600 hover:text-slate-700 hover:bg-slate-100 focus:ring-slate-500"
        )}
        aria-label={
          isLoading
            ? "Loading audio..."
            : isPlaying
            ? "Stop speaking"
            : "Read aloud"
        }
        title={
          isLoading
            ? "Loading..."
            : isPlaying
            ? "Stop"
            : "Read aloud"
        }
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : isPlaying ? (
          <VolumeX className="h-4 w-4" />
        ) : (
          <Volume2 className="h-4 w-4" />
        )}
      </button>

      {/* Playing indicator */}
      {isPlaying && (
        <span className="absolute -top-1 -right-1 flex h-3 w-3">
          <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500" />
        </span>
      )}

      {/* ARIA live region for screen readers */}
      <span className="sr-only" aria-live="polite" role="status">
        {isLoading ? "Loading audio" : isPlaying ? "Playing audio" : ""}
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

export default SpeakButton;
