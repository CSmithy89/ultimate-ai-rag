"use client";

/**
 * EmbeddedChat component - inline embedded chat panel.
 *
 * Story 21-F2: Implement CopilotChat Embedded Component
 *
 * Features:
 * - Embedded chat panel within page content
 * - No sidebar or popup - inline experience
 * - Ideal for dedicated chat pages or sections
 * - Full-width or contained layouts
 * - Responsive height/width
 */

import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { cn } from "@/lib/utils";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import { GenerativeUIRenderer } from "./GenerativeUIRenderer";

export interface EmbeddedChatProps {
  /** Additional class names */
  className?: string;
  /** Welcome message displayed initially */
  welcomeMessage?: string;
  /** Chat title/header */
  title?: string;
}

/**
 * EmbeddedChat provides an inline chat panel for page integration.
 *
 * Usage:
 * ```tsx
 * <EmbeddedChat
 *   className="h-[calc(100vh-8rem)] border rounded-lg"
 *   welcomeMessage="Ask me anything about your documents."
 * />
 * ```
 *
 * Dedicated chat page example:
 * ```tsx
 * export default function ChatPage() {
 *   return (
 *     <div className="container mx-auto h-screen py-4">
 *       <h1 className="text-2xl font-bold mb-4">AI Assistant</h1>
 *       <EmbeddedChat className="h-[calc(100vh-8rem)]" />
 *     </div>
 *   );
 * }
 * ```
 */
export function EmbeddedChat({
  className,
  welcomeMessage = "Welcome! Ask me anything about your documents.",
  title = "AI Assistant",
}: EmbeddedChatProps) {
  return (
    <CopilotErrorBoundary>
      <div className={cn("embedded-chat-container", className)}>
        <CopilotChat
          labels={{
            title,
            initial: welcomeMessage,
          }}
          className="h-full"
        >
          <ThoughtTraceStepper />
          <GenerativeUIRenderer />
        </CopilotChat>
      </div>
    </CopilotErrorBoundary>
  );
}

export default EmbeddedChat;
