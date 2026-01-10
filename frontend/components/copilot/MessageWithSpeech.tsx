"use client";

/**
 * MessageWithSpeech component - Assistant message with TTS button.
 *
 * Story 21-E2: Implement Voice Output (Text-to-Speech)
 * AC3: Speak button on assistant messages
 *
 * This component wraps assistant message content and adds a
 * SpeakButton for text-to-speech playback.
 *
 * Features:
 * - Speak button appears on hover or focus
 * - Configurable voice and speed
 * - Works with any text content
 */

import { memo, ReactNode, useState } from "react";
import { cn } from "@/lib/utils";
import { SpeakButton } from "./SpeakButton";

export interface MessageWithSpeechProps {
  /** The text content to speak */
  text: string;
  /** Children to render (the message UI) */
  children: ReactNode;
  /** Voice to use for TTS */
  voice?: string;
  /** Speech speed multiplier */
  speed?: number;
  /** Whether to always show the speak button (vs hover only) */
  alwaysShowButton?: boolean;
  /** Additional class names */
  className?: string;
}

/**
 * MessageWithSpeech wraps assistant messages with a TTS button.
 *
 * Usage:
 * ```tsx
 * <MessageWithSpeech text={message.content}>
 *   <div className="message-bubble">
 *     {message.content}
 *   </div>
 * </MessageWithSpeech>
 * ```
 *
 * Integration with CopilotKit:
 * This component can be used in custom message renderers or
 * alongside the GenerativeUIRenderer for assistant messages.
 */
export const MessageWithSpeech = memo(function MessageWithSpeech({
  text,
  children,
  voice,
  speed,
  alwaysShowButton = false,
  className,
}: MessageWithSpeechProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  const showButton = alwaysShowButton || isHovered || isFocused;

  return (
    <div
      className={cn("message-with-speech group relative", className)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onFocus={() => setIsFocused(true)}
      onBlur={() => setIsFocused(false)}
    >
      {children}

      {/* Speak button positioned at top-right of message */}
      <div
        className={cn(
          "absolute -top-2 -right-2 transition-opacity duration-200",
          showButton ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
      >
        <SpeakButton
          text={text}
          voice={voice}
          speed={speed}
          className="bg-white dark:bg-slate-800 shadow-md rounded-full"
        />
      </div>
    </div>
  );
});

export default MessageWithSpeech;
