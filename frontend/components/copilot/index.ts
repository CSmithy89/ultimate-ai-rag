/**
 * Copilot Components Barrel Export
 *
 * Central export point for all copilot-related components.
 * Organized by feature group for easier imports.
 */

// =============================================================================
// Core Chat Components
// =============================================================================
export { ChatSidebar } from "./ChatSidebar";
export { ChatInterface, type ChatInterfaceProps, type CopilotUIMode } from "./ChatInterface";
export { PopupChat, type PopupChatProps, type PopupPosition } from "./PopupChat";
export { EmbeddedChat, type EmbeddedChatProps } from "./EmbeddedChat";

// =============================================================================
// Voice I/O Components (Story 21-E1, 21-E2)
// =============================================================================
export { VoiceInput, type VoiceInputProps } from "./VoiceInput";
export { SpeakButton, type SpeakButtonProps } from "./SpeakButton";
export { VoiceChatInput, type VoiceChatInputProps, type VoiceSendMode } from "./VoiceChatInput";
export { MessageWithSpeech, type MessageWithSpeechProps } from "./MessageWithSpeech";

// =============================================================================
// AI-Powered Input (Story 21-F3)
// =============================================================================
export { AITextarea, type AITextareaProps, type AutosuggestionsConfig } from "./AITextarea";

// =============================================================================
// UI Feedback & Actions
// =============================================================================
export { QuickActions, type QuickActionsProps, DEFAULT_QUICK_ACTIONS } from "./QuickActions";
export { ThoughtTraceStepper } from "./ThoughtTraceStepper";
export { GenerativeUIRenderer } from "./GenerativeUIRenderer";
export { StatusBadge } from "./StatusBadge";
export { MCPToolCallCard } from "./MCPToolCallCard";

// =============================================================================
// Error Handling
// =============================================================================
export { CopilotErrorBoundary } from "./CopilotErrorBoundary";
export {
  useAGUIErrorHandler,
  parseAGUIError,
  isErrorCode,
  type UseAGUIErrorHandlerOptions,
  type UseAGUIErrorHandlerReturn,
  AGUIErrorCode,
  type AGUIErrorData,
} from "./ErrorHandler";

// =============================================================================
// HITL / Validation
// =============================================================================
export { SourceValidationDialog } from "./SourceValidationDialog";
export { SourceValidationPanel } from "./SourceValidationPanel";

// =============================================================================
// Action Components
// =============================================================================
export { ActionButtons } from "./ActionButtons";
export { ActionPanel } from "./ActionPanel";
