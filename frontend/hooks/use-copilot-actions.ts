"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useFrontendTool } from "@copilotkit/react-core";
import { useToast } from "@/hooks/use-toast";
import {
  saveToWorkspaceToolParams,
  exportContentToolParams,
  shareContentToolParams,
  bookmarkContentToolParams,
  suggestFollowUpToolParams,
  type SaveToWorkspaceParams,
  type ExportContentParams,
  type ShareContentParams,
  type BookmarkContentParams,
  type SuggestFollowUpParams,
} from "@/lib/schemas/tools";

/**
 * Clipboard write with fallback for older browsers.
 */
async function writeToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  // Fallback for older browsers
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.left = "-9999px";
  textArea.style.top = "-9999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(textArea);
  }
}

/**
 * Action types supported by the system.
 */
export type ActionType = "save" | "export" | "share" | "bookmark" | "followUp";

/**
 * State of an action.
 */
export type ActionState = "idle" | "loading" | "success" | "error";

/**
 * Export format options.
 */
export type ExportFormat = "markdown" | "pdf" | "json";

/**
 * Content that can be actioned upon.
 */
export interface ActionableContent {
  /** Unique ID for this content */
  id: string;
  /** The response text/content */
  content: string;
  /** Optional title for saved content */
  title?: string;
  /** Original query that generated this response */
  query?: string;
  /** Sources used in generating the response */
  sources?: Array<{
    id: string;
    title: string;
    url?: string;
  }>;
  /** Timestamp of the response */
  timestamp?: string;
  /** Session/conversation ID */
  sessionId?: string;
  /** Trajectory ID for this response */
  trajectoryId?: string;
}

/**
 * Custom Event: copilot:follow-up
 *
 * This event is dispatched when a user triggers a follow-up question action.
 * Components can listen for this event to populate the chat input with a suggested query.
 *
 * @event copilot:follow-up
 *
 * @example
 * // Listen for follow-up events
 * useEffect(() => {
 *   const handler = (e: CustomEvent<CopilotFollowUpEventDetail>) => {
 *     const { suggestedQuery, context } = e.detail;
 *     setChatInput(suggestedQuery);
 *   };
 *   document.addEventListener("copilot:follow-up", handler as EventListener);
 *   return () => document.removeEventListener("copilot:follow-up", handler as EventListener);
 * }, []);
 *
 * @property {string} suggestedQuery - The suggested follow-up question text
 * @property {ActionableContent} context - The original content context for the follow-up
 */
export interface CopilotFollowUpEventDetail {
  suggestedQuery: string;
  context: ActionableContent;
}

/**
 * Result of an action.
 */
export interface ActionResult {
  success: boolean;
  message?: string;
  data?: unknown;
  error?: string;
}

/**
 * State for all actions.
 */
export type ActionStates = Record<ActionType, ActionState>;

/**
 * Options for the useCopilotActions hook.
 */
export interface UseCopilotActionsOptions {
  /** Tenant ID for multi-tenant operations */
  tenantId?: string;
  /** Callback when save completes */
  onSaveComplete?: (result: ActionResult) => void;
  /** Callback when export completes */
  onExportComplete?: (result: ActionResult) => void;
  /** Callback when share completes */
  onShareComplete?: (result: ActionResult, shareUrl: string) => void;
  /** Callback when bookmark completes */
  onBookmarkComplete?: (result: ActionResult) => void;
  /** Callback to handle follow-up query */
  onFollowUp?: (query: string, context: ActionableContent) => void;
  /** State reset delay after success (ms) */
  successResetDelay?: number;
  /** State reset delay after error (ms) */
  errorResetDelay?: number;
}

/**
 * Return type for the useCopilotActions hook.
 */
export interface UseCopilotActionsReturn {
  /** Current state of all actions */
  actionStates: ActionStates;
  /** Save content to workspace */
  saveToWorkspace: (content: ActionableContent) => Promise<void>;
  /** Export content in specified format */
  exportContent: (content: ActionableContent, format: ExportFormat) => Promise<void>;
  /** Share content and get shareable link */
  shareContent: (content: ActionableContent) => Promise<string>;
  /** Bookmark content */
  bookmarkContent: (content: ActionableContent) => Promise<void>;
  /** Trigger follow-up query */
  triggerFollowUp: (content: ActionableContent) => void;
  /** Reset all action states */
  resetStates: () => void;
  /** Whether any action is in progress */
  isLoading: boolean;
}

/**
 * Initial state for all actions.
 */
const initialActionStates: ActionStates = {
  save: "idle",
  export: "idle",
  share: "idle",
  bookmark: "idle",
  followUp: "idle",
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

/**
 * useCopilotActions hook manages frontend actions for AI responses
 * and integrates with CopilotKit for agent-triggered actions.
 *
 * Story 6-5: Frontend Actions
 *
 * Supported actions:
 * - save_to_workspace: Save content to user's workspace
 * - export_content: Export as markdown/PDF/JSON
 * - share_content: Generate shareable link
 * - bookmark_content: Save for quick access
 * - follow_up_query: Pre-fill follow-up input
 */
export function useCopilotActions(
  options: UseCopilotActionsOptions = {}
): UseCopilotActionsReturn {
  const {
    tenantId,
    onSaveComplete,
    onExportComplete,
    onShareComplete,
    onBookmarkComplete,
    onFollowUp,
    successResetDelay = 2000,
    errorResetDelay = 3000,
  } = options;

  const { toast } = useToast();
  const [actionStates, setActionStates] = useState<ActionStates>(initialActionStates);
  const resetTimersRef = useRef<Map<ActionType, NodeJS.Timeout>>(new Map());

  // Cleanup timers on unmount to prevent memory leaks
  useEffect(() => {
    const timersRef = resetTimersRef.current;
    return () => {
      timersRef.forEach((timer) => clearTimeout(timer));
      timersRef.clear();
    };
  }, []);

  /**
   * Update state for a specific action with auto-reset.
   */
  const setActionState = useCallback(
    (action: ActionType, state: ActionState) => {
      setActionStates((prev) => ({ ...prev, [action]: state }));

      // Clear existing timer for this action
      const existingTimer = resetTimersRef.current.get(action);
      if (existingTimer) {
        clearTimeout(existingTimer);
      }

      // Auto-reset after success or error
      if (state === "success" || state === "error") {
        const delay = state === "success" ? successResetDelay : errorResetDelay;
        const timer = setTimeout(() => {
          setActionStates((prev) => ({ ...prev, [action]: "idle" }));
        }, delay);
        resetTimersRef.current.set(action, timer);
      }
    },
    [successResetDelay, errorResetDelay]
  );

  /**
   * Save content to workspace.
   */
  const saveToWorkspace = useCallback(
    async (content: ActionableContent): Promise<void> => {
      setActionState("save", "loading");

      try {
        const response = await fetch(API_BASE_URL + "/workspace/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content_id: content.id,
            content: content.content,
            title: content.title || "Response - " + new Date().toLocaleString(),
            query: content.query,
            sources: content.sources,
            session_id: content.sessionId,
            trajectory_id: content.trajectoryId,
            tenant_id: tenantId,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Failed to save content");
        }

        const data = await response.json();

        setActionState("save", "success");
        toast({
          variant: "default",
          title: "Saved to workspace",
          description: "Content saved successfully",
        });

        const result: ActionResult = {
          success: true,
          message: "Content saved to workspace",
          data: data.data,
        };
        onSaveComplete?.(result);
      } catch (error) {
        setActionState("save", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to save content";
        toast({
          variant: "destructive",
          title: "Save failed",
          description: errorMessage,
        });

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onSaveComplete?.(result);
      }
    },
    [tenantId, setActionState, toast, onSaveComplete]
  );

  /**
   * Export content in specified format.
   */
  const exportContent = useCallback(
    async (content: ActionableContent, format: ExportFormat): Promise<void> => {
      setActionState("export", "loading");

      try {
        const response = await fetch(API_BASE_URL + "/workspace/export", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content_id: content.id,
            content: content.content,
            title: content.title || "AI Response",
            query: content.query,
            sources: content.sources,
            format,
            tenant_id: tenantId,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Failed to export content");
        }

        // Create download
        const dateStr = new Date().toISOString().split("T")[0];
        let blob: Blob;
        let filename: string;

        if (format === "json") {
          const data = await response.json();
          blob = new Blob([JSON.stringify(data, null, 2)], {
            type: "application/json",
          });
          filename = "response-" + dateStr + ".json";
        } else if (format === "markdown") {
          const text = await response.text();
          blob = new Blob([text], { type: "text/markdown" });
          filename = "response-" + dateStr + ".md";
        } else {
          blob = await response.blob();
          filename = "response-" + dateStr + ".pdf";
        }

        // Trigger download
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        setActionState("export", "success");
        toast({
          variant: "default",
          title: "Export complete",
          description: "Downloaded as " + format.toUpperCase(),
        });

        const result: ActionResult = {
          success: true,
          message: "Exported as " + format,
          data: { filename },
        };
        onExportComplete?.(result);
      } catch (error) {
        setActionState("export", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to export content";
        toast({
          variant: "destructive",
          title: "Export failed",
          description: errorMessage,
        });

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onExportComplete?.(result);
      }
    },
    [tenantId, setActionState, toast, onExportComplete]
  );

  /**
   * Share content and get shareable link.
   */
  const shareContent = useCallback(
    async (content: ActionableContent): Promise<string> => {
      setActionState("share", "loading");

      try {
        const response = await fetch(API_BASE_URL + "/workspace/share", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content_id: content.id,
            content: content.content,
            title: content.title || "Shared AI Response",
            query: content.query,
            sources: content.sources,
            session_id: content.sessionId,
            tenant_id: tenantId,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Failed to create share link");
        }

        const data = await response.json();
        const shareUrl = data.data?.share_url || data.share_url;

        // Copy to clipboard (with fallback for older browsers)
        await writeToClipboard(shareUrl);

        setActionState("share", "success");
        toast({
          variant: "default",
          title: "Link copied!",
          description: "Shareable link copied to clipboard",
        });

        const result: ActionResult = {
          success: true,
          message: "Share link created and copied",
          data: { shareUrl },
        };
        onShareComplete?.(result, shareUrl);

        return shareUrl;
      } catch (error) {
        setActionState("share", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to create share link";
        toast({
          variant: "destructive",
          title: "Share failed",
          description: errorMessage,
        });

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onShareComplete?.(result, "");

        throw error;
      }
    },
    [tenantId, setActionState, toast, onShareComplete]
  );

  /**
   * Bookmark content.
   */
  const bookmarkContent = useCallback(
    async (content: ActionableContent): Promise<void> => {
      setActionState("bookmark", "loading");

      try {
        const response = await fetch(API_BASE_URL + "/workspace/bookmark", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content_id: content.id,
            content: content.content,
            title: content.title || "Bookmarked Response",
            query: content.query,
            session_id: content.sessionId,
            tenant_id: tenantId,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Failed to bookmark content");
        }

        setActionState("bookmark", "success");
        toast({
          variant: "default",
          title: "Bookmarked",
          description: "Response added to bookmarks",
        });

        const result: ActionResult = {
          success: true,
          message: "Content bookmarked",
        };
        onBookmarkComplete?.(result);
      } catch (error) {
        setActionState("bookmark", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to bookmark content";
        toast({
          variant: "destructive",
          title: "Bookmark failed",
          description: errorMessage,
        });

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onBookmarkComplete?.(result);
      }
    },
    [tenantId, setActionState, toast, onBookmarkComplete]
  );

  /**
   * Trigger follow-up query.
   */
  const triggerFollowUp = useCallback(
    (content: ActionableContent): void => {
      // Generate suggested follow-up based on content
      const suggestedQuery = 'Following up on "' + (content.query || "the previous response") + '": ';

      if (onFollowUp) {
        onFollowUp(suggestedQuery, content);
      } else {
        // Default behavior: dispatch event for chat input to capture
        document.dispatchEvent(
          new CustomEvent("copilot:follow-up", {
            detail: {
              suggestedQuery,
              context: content,
            },
          })
        );
      }
    },
    [onFollowUp]
  );

  /**
   * Reset all action states.
   */
  const resetStates = useCallback(() => {
    // Clear all timers
    resetTimersRef.current.forEach((timer) => clearTimeout(timer));
    resetTimersRef.current.clear();

    // Reset states
    setActionStates(initialActionStates);
  }, []);

  // Register CopilotKit frontend tools for agent-triggered operations
  // Story 21-A1: Migrated from useCopilotAction to useFrontendTool
  // NOTE: Using Parameter[] format for CopilotKit 1.x compatibility.
  // Typed parameters are available via imported types for handler inference.

  // Save to Workspace Tool
  useFrontendTool<typeof saveToWorkspaceToolParams>({
    name: "save_to_workspace",
    description:
      "Save the AI response to the user's workspace for later reference",
    parameters: saveToWorkspaceToolParams,
    handler: async (params) => {
      const { content_id, content_text, title, query } =
        params as unknown as SaveToWorkspaceParams;
      const content: ActionableContent = {
        id: content_id,
        content: content_text,
        title,
        query,
      };
      await saveToWorkspace(content);
      return { success: true, action: "save_to_workspace" };
    },
  });

  // Export Content Tool
  useFrontendTool<typeof exportContentToolParams>({
    name: "export_content",
    description:
      "Export the AI response in a specified format (markdown, pdf, json)",
    parameters: exportContentToolParams,
    handler: async (params) => {
      const { content_id, content_text, format, title } =
        params as unknown as ExportContentParams;
      const content: ActionableContent = {
        id: content_id,
        content: content_text,
        title,
      };
      await exportContent(content, format);
      return { success: true, action: "export_content", format };
    },
  });

  // Share Content Tool
  useFrontendTool<typeof shareContentToolParams>({
    name: "share_content",
    description: "Generate a shareable link for the AI response",
    parameters: shareContentToolParams,
    handler: async (params) => {
      const { content_id, content_text, title } =
        params as unknown as ShareContentParams;
      const content: ActionableContent = {
        id: content_id,
        content: content_text,
        title,
      };
      const shareUrl = await shareContent(content);
      return { success: true, action: "share_content", shareUrl };
    },
  });

  // Bookmark Content Tool
  useFrontendTool<typeof bookmarkContentToolParams>({
    name: "bookmark_content",
    description: "Bookmark the AI response for quick access later",
    parameters: bookmarkContentToolParams,
    handler: async (params) => {
      const { content_id, content_text, title } =
        params as unknown as BookmarkContentParams;
      const content: ActionableContent = {
        id: content_id,
        content: content_text,
        title,
      };
      await bookmarkContent(content);
      return { success: true, action: "bookmark_content" };
    },
  });

  // Follow-up Query Tool
  useFrontendTool<typeof suggestFollowUpToolParams>({
    name: "suggest_follow_up",
    description: "Suggest a follow-up query based on the current response",
    parameters: suggestFollowUpToolParams,
    handler: async (params) => {
      const { suggested_query, context } =
        params as unknown as SuggestFollowUpParams;
      document.dispatchEvent(
        new CustomEvent("copilot:follow-up", {
          detail: {
            suggestedQuery: suggested_query,
            context: { id: "suggested", content: context || "" },
          },
        })
      );
      return { success: true, action: "suggest_follow_up" };
    },
  });

  // Compute loading state
  const isLoading = Object.values(actionStates).some(
    (state) => state === "loading"
  );

  return {
    actionStates,
    saveToWorkspace,
    exportContent,
    shareContent,
    bookmarkContent,
    triggerFollowUp,
    resetStates,
    isLoading,
  };
}

export default useCopilotActions;
