"use client";

/**
 * VoiceChatInput component - Voice input integrated with chat.
 *
 * Story 21-E1: Implement Voice Input (Speech-to-Text)
 * AC4: Transcribed text populates chat input
 *
 * This component wraps VoiceInput and uses useProgrammaticChat
 * to send transcribed text directly to the chat.
 *
 * Features:
 * - Records audio and transcribes to text
 * - Automatically sends transcribed text as chat message
 * - OR appends to input (controlled via sendMode prop)
 * - Visual feedback during recording/transcription
 */

import { memo, useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { VoiceInput } from "./VoiceInput";
import { useProgrammaticChat } from "@/hooks/use-programmatic-chat";

export type VoiceSendMode = "send" | "append";

export interface VoiceChatInputProps {
  /** How to handle transcribed text */
  sendMode?: VoiceSendMode;
  /** Callback when text is appended (for append mode) */
  onAppend?: (text: string) => void;
  /** Whether the button is disabled */
  disabled?: boolean;
  /** Additional class names */
  className?: string;
  /** Language hint for transcription */
  language?: string;
}

/**
 * VoiceChatInput integrates voice recording with the chat system.
 *
 * Usage:
 * ```tsx
 * // Auto-send mode (default) - transcribed text is sent as message
 * <VoiceChatInput sendMode="send" />
 *
 * // Append mode - transcribed text is passed to callback
 * const [input, setInput] = useState("");
 * <VoiceChatInput
 *   sendMode="append"
 *   onAppend={(text) => setInput(prev => prev + text)}
 * />
 * ```
 */
export const VoiceChatInput = memo(function VoiceChatInput({
  sendMode = "send",
  onAppend,
  disabled = false,
  className,
  language = "en",
}: VoiceChatInputProps) {
  const { sendMessage, isLoading } = useProgrammaticChat();
  const [lastTranscription, setLastTranscription] = useState<string | null>(null);

  const handleTranscription = useCallback(
    (text: string) => {
      if (!text.trim()) return;

      setLastTranscription(text);

      if (sendMode === "send") {
        sendMessage(text);
      } else if (sendMode === "append" && onAppend) {
        onAppend(text);
      }
    },
    [sendMode, sendMessage, onAppend]
  );

  return (
    <div className={cn("voice-chat-input inline-flex items-center", className)}>
      <VoiceInput
        onTranscription={handleTranscription}
        disabled={disabled || isLoading}
        language={language}
      />

      {/* Show last transcription for feedback (optional visual confirmation) */}
      {lastTranscription && sendMode === "send" && (
        <span className="sr-only">Sent: {lastTranscription}</span>
      )}
    </div>
  );
});

export default VoiceChatInput;
