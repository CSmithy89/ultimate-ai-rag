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
 * Includes mobile-responsive adjustments.
 */
function getPositionClass(position: PopupPosition): string {
  // Base position classes
  const positionClasses: Record<PopupPosition, string> = {
    "bottom-right": "",
    "bottom-left": "!left-4 !right-auto",
    "top-right": "!bottom-auto !top-4",
    "top-left": "!left-4 !right-auto !bottom-auto !top-4",
  };

  return positionClasses[position] || "";
}

/**
 * Mobile-responsive CSS that gets applied to the popup.
 * On small screens, the popup expands to full width with minimal margins.
 */
const RESPONSIVE_CLASSES = [
  // Mobile: full-width with small margins
  "max-sm:!left-2 max-sm:!right-2 max-sm:!bottom-2",
  "max-sm:!w-[calc(100%-1rem)]",
  "max-sm:!max-h-[80vh]",
  // Tablet: slightly smaller margins
  "sm:max-md:!max-w-[90vw]",
].join(" ");

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
  title = "RAG Assistant",
  initialMessage = "How can I help you today?",
  defaultOpen = false,
  clickOutsideToClose = true,
  className,
}: PopupChatProps) {
  return (
    <CopilotErrorBoundary>
      <CopilotPopup
        labels={{
          title,
          initial: initialMessage,
        }}
        defaultOpen={defaultOpen}
        clickOutsideToClose={clickOutsideToClose}
        className={cn("copilot-popup", getPositionClass(position), RESPONSIVE_CLASSES, className)}
      >
        <ThoughtTraceStepper />
        <GenerativeUIRenderer />
      </CopilotPopup>
    </CopilotErrorBoundary>
  );
}

export default PopupChat;
