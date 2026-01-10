"use client";

import { useCallback, useMemo } from "react";
import { useCopilotChat } from "@copilotkit/react-core";
import { TextMessage, MessageRole } from "@copilotkit/runtime-client-gql";
import type { ChatMessage, ProgrammaticChatReturn } from "@/types/copilot";

/**
 * Validates that message content is non-empty.
 *
 * @param content - The message content to validate
 * @returns True if content is valid, false otherwise
 */
export function isValidMessageContent(content: string): boolean {
  return typeof content === "string" && content.trim().length > 0;
}

/**
 * Converts CopilotKit message to simplified ChatMessage type.
 *
 * @param msg - CopilotKit message object
 * @returns Simplified ChatMessage object
 */
export function toChatMessage(msg: {
  id: string;
  role?: string;
  content?: string;
}): ChatMessage {
  return {
    id: msg.id,
    role: (msg.role as ChatMessage["role"]) || "assistant",
    content: msg.content || "",
  };
}

/**
 * useProgrammaticChat provides programmatic control over the CopilotKit chat.
 *
 * Story 21-A6: Implement useCopilotChat for Headless Control
 *
 * This hook wraps CopilotKit's useCopilotChat with convenient methods for:
 * - Sending messages programmatically
 * - Regenerating responses
 * - Stopping generation
 * - Clearing chat history
 *
 * Use cases:
 * - Quick action buttons that send preset messages
 * - Test automation for chat interactions
 * - Building custom chat UIs
 * - Onboarding flows with scripted messages
 * - Keyboard shortcuts for power users
 *
 * @example
 * ```tsx
 * // Basic usage with quick actions
 * function ChatWithActions() {
 *   const { sendMessage, isLoading } = useProgrammaticChat();
 *
 *   return (
 *     <Button
 *       disabled={isLoading}
 *       onClick={() => sendMessage("Summarize the document")}
 *     >
 *       Summarize
 *     </Button>
 *   );
 * }
 * ```
 *
 * @example
 * ```tsx
 * // Custom chat UI with full control
 * function CustomChat() {
 *   const {
 *     messages,
 *     sendMessage,
 *     regenerateLastResponse,
 *     clearHistory,
 *     isLoading,
 *   } = useProgrammaticChat();
 *
 *   return (
 *     <div>
 *       {messages.map((msg) => (
 *         <div key={msg.id}>{msg.content}</div>
 *       ))}
 *       <Button onClick={regenerateLastResponse}>Regenerate</Button>
 *       <Button onClick={clearHistory}>Clear</Button>
 *     </div>
 *   );
 * }
 * ```
 *
 * @returns Object with messages, control functions, and loading state
 */
export function useProgrammaticChat(): ProgrammaticChatReturn {
  const {
    visibleMessages,
    appendMessage,
    reloadMessages,
    stopGeneration: copilotStopGeneration,
    reset,
    isLoading,
  } = useCopilotChat();

  /**
   * Send a user message programmatically.
   * Validates content before sending.
   *
   * @param content - The message content to send
   * @throws Logs error to console if send fails
   */
  const sendMessage = useCallback(
    async (content: string): Promise<void> => {
      if (!isValidMessageContent(content)) {
        console.warn("useProgrammaticChat: Cannot send empty message");
        return;
      }

      try {
        await appendMessage(
          new TextMessage({
            role: MessageRole.User,
            content: content.trim(),
          })
        );
      } catch (error) {
        console.error("useProgrammaticChat: Failed to send message:", error);
      }
    },
    [appendMessage]
  );

  /**
   * Regenerate the last assistant response.
   * Finds the last message and requests regeneration.
   *
   * @throws Logs error to console if regeneration fails
   */
  const regenerateLastResponse = useCallback(async (): Promise<void> => {
    if (visibleMessages.length === 0) {
      console.warn("useProgrammaticChat: No messages to regenerate");
      return;
    }

    // Find the last message with an ID
    const lastMessage = visibleMessages[visibleMessages.length - 1];

    if (!lastMessage?.id) {
      console.warn("useProgrammaticChat: Last message has no ID");
      return;
    }

    try {
      await reloadMessages(lastMessage.id);
    } catch (error) {
      console.error(
        "useProgrammaticChat: Failed to regenerate response:",
        error
      );
    }
  }, [visibleMessages, reloadMessages]);

  /**
   * Stop the current generation.
   * Wraps CopilotKit's stopGeneration with error handling.
   */
  const stopGeneration = useCallback((): void => {
    try {
      copilotStopGeneration();
    } catch (error) {
      console.error("useProgrammaticChat: Failed to stop generation:", error);
    }
  }, [copilotStopGeneration]);

  /**
   * Clear all messages and reset the chat state.
   * Wraps CopilotKit's reset with error handling.
   */
  const clearHistory = useCallback((): void => {
    try {
      reset();
    } catch (error) {
      console.error("useProgrammaticChat: Failed to clear history:", error);
    }
  }, [reset]);

  /**
   * Convert CopilotKit messages to simplified ChatMessage format.
   */
  const messages = useMemo<ChatMessage[]>(
    () =>
      visibleMessages.map((msg) =>
        toChatMessage(msg as { id: string; role?: string; content?: string })
      ),
    [visibleMessages]
  );

  /**
   * Get the count of visible messages.
   */
  const messageCount = useMemo(() => messages.length, [messages]);

  return {
    messages,
    messageCount,
    isLoading,
    sendMessage,
    regenerateLastResponse,
    stopGeneration,
    clearHistory,
  };
}

export default useProgrammaticChat;
