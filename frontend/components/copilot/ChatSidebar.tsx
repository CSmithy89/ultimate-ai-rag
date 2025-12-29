"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import { GenerativeUIRenderer } from "./GenerativeUIRenderer";

/**
 * ChatSidebar component wrapping CopilotKit's CopilotSidebar
 * with custom styling following the project's design system.
 *
 * Story 6-2: Chat Sidebar Interface
 * Story 6-3: Generative UI Components
 *
 * Design System:
 * - Primary (Indigo-600): #4F46E5
 * - Secondary (Emerald-500): #10B981
 * - Neutral: Slate colors
 */
export function ChatSidebar() {
  return (
    <CopilotErrorBoundary>
      <CopilotSidebar
        defaultOpen={true}
        labels={{
          title: "AI Copilot",
          initial: "How can I help you today?",
        }}
        className="copilot-sidebar"
      >
        <ThoughtTraceStepper />
        <GenerativeUIRenderer />
      </CopilotSidebar>
    </CopilotErrorBoundary>
  );
}
