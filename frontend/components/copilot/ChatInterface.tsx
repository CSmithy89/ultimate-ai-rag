"use client";

/**
 * ChatInterface component - conditionally renders sidebar or popup.
 *
 * Story 21-F1, 21-F2: Alternative UI Components
 *
 * Features:
 * - Environment variable controlled UI mode
 * - Supports sidebar (default), popup, and embedded modes
 * - Same internal components across all modes
 */

import { ChatSidebar } from "./ChatSidebar";
import { PopupChat, type PopupPosition } from "./PopupChat";
import { EmbeddedChat } from "./EmbeddedChat";

export type CopilotUIMode = "sidebar" | "popup" | "embedded";

export interface ChatInterfaceProps {
  /** Override the UI mode from environment */
  mode?: CopilotUIMode;
  /** Popup position (only used in popup mode) */
  popupPosition?: PopupPosition;
  /** Embedded container class (only used in embedded mode) */
  embeddedClassName?: string;
}

/**
 * ChatInterface provides flexible chat UI based on configuration.
 *
 * Configuration via environment variable:
 * - NEXT_PUBLIC_COPILOT_UI_MODE=sidebar (default)
 * - NEXT_PUBLIC_COPILOT_UI_MODE=popup
 * - NEXT_PUBLIC_COPILOT_UI_MODE=embedded
 *
 * Usage:
 * ```tsx
 * // Uses environment variable
 * <ChatInterface />
 *
 * // Override mode
 * <ChatInterface mode="popup" popupPosition="bottom-right" />
 * ```
 */
export function ChatInterface({
  mode,
  popupPosition = "bottom-right",
  embeddedClassName = "h-full",
}: ChatInterfaceProps) {
  // Determine UI mode from prop or environment variable
  const uiMode: CopilotUIMode =
    mode ||
    (process.env.NEXT_PUBLIC_COPILOT_UI_MODE as CopilotUIMode) ||
    "sidebar";

  switch (uiMode) {
    case "popup":
      return <PopupChat position={popupPosition} />;

    case "embedded":
      return <EmbeddedChat className={embeddedClassName} />;

    case "sidebar":
    default:
      return <ChatSidebar />;
  }
}

export default ChatInterface;
