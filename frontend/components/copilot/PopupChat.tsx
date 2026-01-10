"use client";

/**
 * PopupChat component - floating popup chat option.
 *
 * Story 21-F1: Implement CopilotPopup Component
 *
 * Features:
 * - Floating button with expandable chat window
 * - Less intrusive than full sidebar
 * - Ideal for secondary/contextual assistance
 * - Customizable position and styling
 * - Click-outside-to-close behavior
 */

import { CopilotPopup } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { cn } from "@/lib/utils";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import { GenerativeUIRenderer } from "./GenerativeUIRenderer";

export type PopupPosition = "bottom-right" | "bottom-left" | "top-right" | "top-left";

export interface PopupChatProps {
  /** Position of the popup button and chat window */
  position?: PopupPosition;
  /** Button label text */
  buttonLabel?: string;
  /** Chat title */
  title?: string;
  /** Initial message displayed */
  initialMessage?: string;
  /** Whether the popup is open by default */
  defaultOpen?: boolean;
  /** Whether clicking outside closes the popup */
  clickOutsideToClose?: boolean;
  /** Additional class names */
  className?: string;
}

/**
 * Get CSS class for popup position.
 */
function getPositionClass(position: PopupPosition): string {
  switch (position) {
    case "bottom-left":
      return "!left-4 !right-auto";
    case "top-right":
      return "!bottom-auto !top-4";
    case "top-left":
      return "!left-4 !right-auto !bottom-auto !top-4";
    case "bottom-right":
    default:
      return "";
  }
}

/**
 * PopupChat provides a floating chat popup as an alternative to sidebar.
 *
 * Usage:
 * ```tsx
 * <PopupChat
 *   position="bottom-right"
 *   title="AI Assistant"
 *   initialMessage="How can I help you today?"
 * />
 * ```
 *
 * Configuration:
 * Set NEXT_PUBLIC_COPILOT_UI_MODE=popup in .env to use popup by default.
 */
export function PopupChat({
  position = "bottom-right",
  buttonLabel: _buttonLabel = "AI Assistant", // Reserved for future customization
  title = "RAG Assistant",
  initialMessage = "How can I help you today?",
  defaultOpen = false,
  clickOutsideToClose = true,
  className,
}: PopupChatProps) {
  // Note: buttonLabel is reserved for potential CopilotPopup button customization
  void _buttonLabel;
  return (
    <CopilotErrorBoundary>
      <CopilotPopup
        labels={{
          title,
          initial: initialMessage,
        }}
        defaultOpen={defaultOpen}
        clickOutsideToClose={clickOutsideToClose}
        className={cn("copilot-popup", getPositionClass(position), className)}
      >
        <ThoughtTraceStepper />
        <GenerativeUIRenderer />
      </CopilotPopup>
    </CopilotErrorBoundary>
  );
}

export default PopupChat;
