"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import { GenerativeUIRenderer } from "./GenerativeUIRenderer";
import { useAnalytics } from "@/hooks/use-analytics";

/**
 * ChatSidebar component wrapping CopilotKit's CopilotSidebar
 * with custom styling following the project's design system.
 *
 * Story 6-2: Chat Sidebar Interface
 * Story 6-3: Generative UI Components
 * Story 21-B1: Configure Observability Hooks and Dev Console
 *
 * Design System:
 * - Primary (Indigo-600): #4F46E5
 * - Secondary (Emerald-500): #10B981
 * - Neutral: Slate colors
 *
 * Observability Hooks (21-B1):
 * - onMessageSent: Track message length (not content) for analytics
 * - onChatExpanded/onChatMinimized: Track UI state changes
 * - onMessageRegenerated: Track regeneration requests
 * - onMessageCopied: Track content copy events (length only)
 * - onFeedbackGiven: Track user feedback (thumbs up/down)
 * - onChatStarted/onChatStopped: Track generation lifecycle
 */
export function ChatSidebar() {
  const { track } = useAnalytics();

  /**
   * Observability hooks for tracking user interactions.
   *
   * Story 21-B1: These hooks fire on user actions and emit telemetry
   * events to our analytics pipeline. Only non-sensitive data is tracked
   * (lengths, IDs, timestamps - never raw content).
   *
   * Note: observabilityHooks require publicApiKey or publicLicenseKey
   * on the CopilotKit provider to function. If not configured, hooks
   * are silently ignored - analytics still works via the useAnalytics
   * fallback in error handler.
   */
  const observabilityHooks = {
    onMessageSent: (message: string) => {
      // Track message length only - never send raw content
      track("copilot_message_sent", {
        messageLength: message?.length ?? 0,
      });
    },
    onChatExpanded: () => {
      track("copilot_chat_expanded");
    },
    onChatMinimized: () => {
      track("copilot_chat_minimized");
    },
    onMessageRegenerated: (messageId: string) => {
      track("copilot_message_regenerated", { messageId });
    },
    onMessageCopied: (content: string) => {
      // Track content length only - never send raw content
      track("copilot_message_copied", {
        contentLength: content?.length ?? 0,
      });
    },
    onFeedbackGiven: (messageId: string, type: string) => {
      track("copilot_feedback", { messageId, type });
    },
    onChatStarted: () => {
      track("copilot_generation_started");
    },
    onChatStopped: () => {
      track("copilot_generation_stopped");
    },
  };

  return (
    <CopilotErrorBoundary>
      <CopilotSidebar
        defaultOpen={true}
        labels={{
          title: "AI Copilot",
          initial: "How can I help you today?",
        }}
        className="copilot-sidebar"
        observabilityHooks={observabilityHooks}
      >
        <ThoughtTraceStepper />
        <GenerativeUIRenderer />
      </CopilotSidebar>
    </CopilotErrorBoundary>
  );
}
