# Story 6-5: Frontend Actions

Status: done
Epic: 6 - Interactive Copilot Experience
Priority: High
Depends on: Story 6-4 (Human-in-the-Loop Source Validation)

## User Story

As an **end-user**,
I want **the AI to perform common actions like saving to workspace, exporting, sharing, bookmarking, and triggering follow-up queries**,
So that **I can efficiently manage AI-generated content and seamlessly integrate results into my workflow without leaving the chat interface**.

## Acceptance Criteria

- Given the AI generates a response with actionable content
- When action buttons are displayed below the response
- Then users see a row of action buttons: Save, Export, Share, Bookmark, Follow-up
- And clicking Save triggers backend API to save content to user's workspace
- And clicking Export offers download in markdown/PDF/JSON formats
- And clicking Share generates a shareable link and copies to clipboard
- And clicking Bookmark saves the response for quick access later
- And clicking Follow-up opens a pre-filled input for related queries
- And each action displays loading state during API call
- And each action shows success toast notification on completion
- And each action shows error toast notification on failure
- And actions are registered with useCopilotAction hooks
- And the AI can programmatically trigger actions via AG-UI protocol
- And action completion is logged in agent trajectory

## Technical Approach

### 1. Create ActionButtons Component

**File:** `frontend/components/copilot/ActionButtons.tsx`

Create a row of action buttons that appear below AI responses:

```typescript
"use client";

import { memo, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Save,
  Download,
  Share2,
  Bookmark,
  MessageSquarePlus,
  Loader2,
  Check,
  X,
  FileText,
  FileJson,
  FileType,
  ChevronDown,
} from "lucide-react";
import type { ActionState, ActionType } from "@/hooks/use-copilot-actions";

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

interface ActionButtonsProps {
  /** Content to perform actions on */
  content: ActionableContent;
  /** Current state of each action */
  actionStates: Record<ActionType, ActionState>;
  /** Callback to save content to workspace */
  onSave: (content: ActionableContent) => Promise<void>;
  /** Callback to export content */
  onExport: (content: ActionableContent, format: ExportFormat) => Promise<void>;
  /** Callback to share content */
  onShare: (content: ActionableContent) => Promise<string>;
  /** Callback to bookmark content */
  onBookmark: (content: ActionableContent) => Promise<void>;
  /** Callback to trigger follow-up query */
  onFollowUp: (content: ActionableContent) => void;
  /** Whether actions are disabled */
  disabled?: boolean;
  /** Optional class name */
  className?: string;
  /** Compact mode for smaller button size */
  compact?: boolean;
}

/**
 * Get icon for action state.
 */
function getStateIcon(
  state: ActionState,
  defaultIcon: React.ReactNode
): React.ReactNode {
  switch (state) {
    case "loading":
      return <Loader2 className="h-4 w-4 animate-spin" />;
    case "success":
      return <Check className="h-4 w-4 text-emerald-500" />;
    case "error":
      return <X className="h-4 w-4 text-red-500" />;
    default:
      return defaultIcon;
  }
}

/**
 * ActionButtons provides a row of action buttons for AI responses.
 *
 * Story 6-5: Frontend Actions
 *
 * Actions:
 * - Save: Save content to user's workspace
 * - Export: Download as markdown/PDF/JSON
 * - Share: Generate shareable link
 * - Bookmark: Quick-access bookmark
 * - Follow-up: Pre-fill follow-up query
 *
 * Design System:
 * - Indigo-600 (#4F46E5) for primary actions
 * - Emerald-500 (#10B981) for success state
 * - Red-500 for error state
 * - Slate for neutral/secondary
 *
 * @example
 * ```tsx
 * <ActionButtons
 *   content={{ id: "1", content: "AI response...", query: "What is..." }}
 *   actionStates={actionStates}
 *   onSave={handleSave}
 *   onExport={handleExport}
 *   onShare={handleShare}
 *   onBookmark={handleBookmark}
 *   onFollowUp={handleFollowUp}
 * />
 * ```
 */
export const ActionButtons = memo(function ActionButtons({
  content,
  actionStates,
  onSave,
  onExport,
  onShare,
  onBookmark,
  onFollowUp,
  disabled = false,
  className,
  compact = false,
}: ActionButtonsProps) {
  const [isBookmarked, setIsBookmarked] = useState(false);

  // Handle save action
  const handleSave = useCallback(async () => {
    await onSave(content);
  }, [content, onSave]);

  // Handle export action
  const handleExport = useCallback(
    async (format: ExportFormat) => {
      await onExport(content, format);
    },
    [content, onExport]
  );

  // Handle share action
  const handleShare = useCallback(async () => {
    await onShare(content);
  }, [content, onShare]);

  // Handle bookmark action
  const handleBookmark = useCallback(async () => {
    await onBookmark(content);
    setIsBookmarked((prev) => !prev);
  }, [content, onBookmark]);

  // Handle follow-up action
  const handleFollowUp = useCallback(() => {
    onFollowUp(content);
  }, [content, onFollowUp]);

  const buttonSize = compact ? "sm" : "default";
  const iconSize = compact ? "h-3.5 w-3.5" : "h-4 w-4";

  return (
    <TooltipProvider delayDuration={300}>
      <div
        className={cn(
          "flex items-center gap-1 flex-wrap",
          compact ? "gap-0.5" : "gap-1",
          className
        )}
        role="toolbar"
        aria-label="Response actions"
      >
        {/* Save to Workspace */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size={buttonSize}
              onClick={handleSave}
              disabled={disabled || actionStates.save === "loading"}
              className={cn(
                "text-slate-600 hover:text-indigo-600 hover:bg-indigo-50",
                actionStates.save === "success" && "text-emerald-600",
                actionStates.save === "error" && "text-red-600"
              )}
              aria-label="Save to workspace"
            >
              {getStateIcon(
                actionStates.save,
                <Save className={iconSize} />
              )}
              {!compact && <span className="ml-1.5">Save</span>}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Save to workspace</p>
          </TooltipContent>
        </Tooltip>

        {/* Export Dropdown */}
        <DropdownMenu>
          <Tooltip>
            <TooltipTrigger asChild>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size={buttonSize}
                  disabled={disabled || actionStates.export === "loading"}
                  className={cn(
                    "text-slate-600 hover:text-indigo-600 hover:bg-indigo-50",
                    actionStates.export === "success" && "text-emerald-600",
                    actionStates.export === "error" && "text-red-600"
                  )}
                  aria-label="Export content"
                >
                  {getStateIcon(
                    actionStates.export,
                    <Download className={iconSize} />
                  )}
                  {!compact && (
                    <>
                      <span className="ml-1.5">Export</span>
                      <ChevronDown className="h-3 w-3 ml-0.5" />
                    </>
                  )}
                </Button>
              </DropdownMenuTrigger>
            </TooltipTrigger>
            <TooltipContent>
              <p>Export content</p>
            </TooltipContent>
          </Tooltip>
          <DropdownMenuContent align="start">
            <DropdownMenuItem onClick={() => handleExport("markdown")}>
              <FileText className="h-4 w-4 mr-2" />
              Markdown (.md)
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleExport("pdf")}>
              <FileType className="h-4 w-4 mr-2" />
              PDF Document
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleExport("json")}>
              <FileJson className="h-4 w-4 mr-2" />
              JSON Data
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Share */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size={buttonSize}
              onClick={handleShare}
              disabled={disabled || actionStates.share === "loading"}
              className={cn(
                "text-slate-600 hover:text-indigo-600 hover:bg-indigo-50",
                actionStates.share === "success" && "text-emerald-600",
                actionStates.share === "error" && "text-red-600"
              )}
              aria-label="Share response"
            >
              {getStateIcon(
                actionStates.share,
                <Share2 className={iconSize} />
              )}
              {!compact && <span className="ml-1.5">Share</span>}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Copy shareable link</p>
          </TooltipContent>
        </Tooltip>

        {/* Bookmark */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size={buttonSize}
              onClick={handleBookmark}
              disabled={disabled || actionStates.bookmark === "loading"}
              className={cn(
                "text-slate-600 hover:text-indigo-600 hover:bg-indigo-50",
                (actionStates.bookmark === "success" || isBookmarked) &&
                  "text-amber-500",
                actionStates.bookmark === "error" && "text-red-600"
              )}
              aria-label={isBookmarked ? "Remove bookmark" : "Bookmark response"}
              aria-pressed={isBookmarked}
            >
              {getStateIcon(
                actionStates.bookmark,
                <Bookmark
                  className={cn(iconSize, isBookmarked && "fill-current")}
                />
              )}
              {!compact && <span className="ml-1.5">Bookmark</span>}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{isBookmarked ? "Remove bookmark" : "Bookmark for later"}</p>
          </TooltipContent>
        </Tooltip>

        {/* Follow-up Query */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size={buttonSize}
              onClick={handleFollowUp}
              disabled={disabled}
              className="text-slate-600 hover:text-indigo-600 hover:bg-indigo-50"
              aria-label="Ask follow-up question"
            >
              <MessageSquarePlus className={iconSize} />
              {!compact && <span className="ml-1.5">Follow-up</span>}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Ask a follow-up question</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
});

export default ActionButtons;
```

Key implementation details:
- Row of action buttons with consistent styling
- Dropdown menu for export format selection
- Loading, success, and error states for each action
- Tooltips for accessibility
- Compact mode for space-constrained layouts
- Bookmark toggle state visual feedback
- Design system colors: Indigo-600 primary, Emerald-500 success, Red-500 error

### 2. Create use-copilot-actions Hook

**File:** `frontend/hooks/use-copilot-actions.ts`

Create a custom hook for managing action state and CopilotKit integration:

```typescript
"use client";

import { useState, useCallback, useRef } from "react";
import { useCopilotAction } from "@copilotkit/react-core";
import { useToast } from "@/hooks/use-toast";
import { apiClient } from "@/lib/api-client";
import type { ActionableContent, ExportFormat } from "@/components/copilot/ActionButtons";

/**
 * Action types supported by the system.
 */
export type ActionType = "save" | "export" | "share" | "bookmark" | "followUp";

/**
 * State of an action.
 */
export type ActionState = "idle" | "loading" | "success" | "error";

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
  /** Base URL for API calls */
  baseUrl?: string;
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
 *
 * @example
 * ```tsx
 * function ChatResponse({ content }: { content: ActionableContent }) {
 *   const {
 *     actionStates,
 *     saveToWorkspace,
 *     exportContent,
 *     shareContent,
 *     bookmarkContent,
 *     triggerFollowUp,
 *   } = useCopilotActions({
 *     onSaveComplete: (result) => console.log("Saved:", result),
 *   });
 *
 *   return (
 *     <ActionButtons
 *       content={content}
 *       actionStates={actionStates}
 *       onSave={saveToWorkspace}
 *       onExport={exportContent}
 *       onShare={shareContent}
 *       onBookmark={bookmarkContent}
 *       onFollowUp={triggerFollowUp}
 *     />
 *   );
 * }
 * ```
 */
export function useCopilotActions(
  options: UseCopilotActionsOptions = {}
): UseCopilotActionsReturn {
  const {
    baseUrl = "/api/v1",
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
   * Show toast notification.
   */
  const showToast = useCallback(
    (
      variant: "default" | "destructive",
      title: string,
      description?: string
    ) => {
      toast({
        variant,
        title,
        description,
      });
    },
    [toast]
  );

  /**
   * Save content to workspace.
   */
  const saveToWorkspace = useCallback(
    async (content: ActionableContent): Promise<void> => {
      setActionState("save", "loading");

      try {
        const response = await apiClient.post(`${baseUrl}/workspace/save`, {
          content_id: content.id,
          content: content.content,
          title: content.title || `Response - ${new Date().toLocaleString()}`,
          query: content.query,
          sources: content.sources,
          session_id: content.sessionId,
          trajectory_id: content.trajectoryId,
          tenant_id: tenantId,
        });

        setActionState("save", "success");
        showToast("default", "Saved to workspace", "Content saved successfully");

        const result: ActionResult = {
          success: true,
          message: "Content saved to workspace",
          data: response.data,
        };
        onSaveComplete?.(result);
      } catch (error) {
        setActionState("save", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to save content";
        showToast("destructive", "Save failed", errorMessage);

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onSaveComplete?.(result);
      }
    },
    [baseUrl, tenantId, setActionState, showToast, onSaveComplete]
  );

  /**
   * Export content in specified format.
   */
  const exportContent = useCallback(
    async (content: ActionableContent, format: ExportFormat): Promise<void> => {
      setActionState("export", "loading");

      try {
        const response = await apiClient.post(
          `${baseUrl}/workspace/export`,
          {
            content_id: content.id,
            content: content.content,
            title: content.title || "AI Response",
            query: content.query,
            sources: content.sources,
            format,
            tenant_id: tenantId,
          },
          {
            responseType: format === "json" ? "json" : "blob",
          }
        );

        // Create download link
        let blob: Blob;
        let filename: string;
        const timestamp = new Date().toISOString().split("T")[0];

        switch (format) {
          case "markdown":
            blob = new Blob([response.data], { type: "text/markdown" });
            filename = `response-${timestamp}.md`;
            break;
          case "pdf":
            blob = response.data;
            filename = `response-${timestamp}.pdf`;
            break;
          case "json":
            blob = new Blob([JSON.stringify(response.data, null, 2)], {
              type: "application/json",
            });
            filename = `response-${timestamp}.json`;
            break;
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
        showToast("default", "Export complete", `Downloaded as ${format.toUpperCase()}`);

        const result: ActionResult = {
          success: true,
          message: `Exported as ${format}`,
          data: { filename },
        };
        onExportComplete?.(result);
      } catch (error) {
        setActionState("export", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to export content";
        showToast("destructive", "Export failed", errorMessage);

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onExportComplete?.(result);
      }
    },
    [baseUrl, tenantId, setActionState, showToast, onExportComplete]
  );

  /**
   * Share content and get shareable link.
   */
  const shareContent = useCallback(
    async (content: ActionableContent): Promise<string> => {
      setActionState("share", "loading");

      try {
        const response = await apiClient.post(`${baseUrl}/workspace/share`, {
          content_id: content.id,
          content: content.content,
          title: content.title || "Shared AI Response",
          query: content.query,
          sources: content.sources,
          session_id: content.sessionId,
          tenant_id: tenantId,
        });

        const shareUrl = response.data.share_url as string;

        // Copy to clipboard
        await navigator.clipboard.writeText(shareUrl);

        setActionState("share", "success");
        showToast("default", "Link copied!", "Shareable link copied to clipboard");

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
        showToast("destructive", "Share failed", errorMessage);

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onShareComplete?.(result, "");

        throw error;
      }
    },
    [baseUrl, tenantId, setActionState, showToast, onShareComplete]
  );

  /**
   * Bookmark content.
   */
  const bookmarkContent = useCallback(
    async (content: ActionableContent): Promise<void> => {
      setActionState("bookmark", "loading");

      try {
        await apiClient.post(`${baseUrl}/workspace/bookmark`, {
          content_id: content.id,
          content: content.content,
          title: content.title || "Bookmarked Response",
          query: content.query,
          session_id: content.sessionId,
          tenant_id: tenantId,
        });

        setActionState("bookmark", "success");
        showToast("default", "Bookmarked", "Response added to bookmarks");

        const result: ActionResult = {
          success: true,
          message: "Content bookmarked",
        };
        onBookmarkComplete?.(result);
      } catch (error) {
        setActionState("bookmark", "error");
        const errorMessage =
          error instanceof Error ? error.message : "Failed to bookmark content";
        showToast("destructive", "Bookmark failed", errorMessage);

        const result: ActionResult = {
          success: false,
          error: errorMessage,
        };
        onBookmarkComplete?.(result);
      }
    },
    [baseUrl, tenantId, setActionState, showToast, onBookmarkComplete]
  );

  /**
   * Trigger follow-up query.
   */
  const triggerFollowUp = useCallback(
    (content: ActionableContent): void => {
      // Generate suggested follow-up based on content
      const suggestedQuery = `Following up on "${content.query || "the previous response"}": `;

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
    for (const timer of resetTimersRef.current.values()) {
      clearTimeout(timer);
    }
    resetTimersRef.current.clear();

    // Reset states
    setActionStates(initialActionStates);
  }, []);

  // Register CopilotKit actions for agent-triggered operations
  // Save to Workspace Action
  useCopilotAction({
    name: "save_to_workspace",
    description: "Save the AI response to the user's workspace for later reference",
    parameters: [
      {
        name: "content_id",
        type: "string",
        description: "Unique ID of the content to save",
        required: true,
      },
      {
        name: "title",
        type: "string",
        description: "Optional title for the saved content",
        required: false,
      },
    ],
    handler: async ({ content_id, title }) => {
      const content: ActionableContent = {
        id: content_id,
        content: "", // Will be populated by backend
        title,
      };
      await saveToWorkspace(content);
      return { success: true, action: "save_to_workspace" };
    },
  });

  // Export Content Action
  useCopilotAction({
    name: "export_content",
    description: "Export the AI response in a specified format (markdown, pdf, json)",
    parameters: [
      {
        name: "content_id",
        type: "string",
        description: "Unique ID of the content to export",
        required: true,
      },
      {
        name: "format",
        type: "string",
        description: "Export format: markdown, pdf, or json",
        required: true,
      },
    ],
    handler: async ({ content_id, format }) => {
      const content: ActionableContent = {
        id: content_id,
        content: "",
      };
      await exportContent(content, format as ExportFormat);
      return { success: true, action: "export_content", format };
    },
  });

  // Share Content Action
  useCopilotAction({
    name: "share_content",
    description: "Generate a shareable link for the AI response",
    parameters: [
      {
        name: "content_id",
        type: "string",
        description: "Unique ID of the content to share",
        required: true,
      },
    ],
    handler: async ({ content_id }) => {
      const content: ActionableContent = {
        id: content_id,
        content: "",
      };
      const shareUrl = await shareContent(content);
      return { success: true, action: "share_content", shareUrl };
    },
  });

  // Bookmark Content Action
  useCopilotAction({
    name: "bookmark_content",
    description: "Bookmark the AI response for quick access later",
    parameters: [
      {
        name: "content_id",
        type: "string",
        description: "Unique ID of the content to bookmark",
        required: true,
      },
    ],
    handler: async ({ content_id }) => {
      const content: ActionableContent = {
        id: content_id,
        content: "",
      };
      await bookmarkContent(content);
      return { success: true, action: "bookmark_content" };
    },
  });

  // Follow-up Query Action
  useCopilotAction({
    name: "suggest_follow_up",
    description: "Suggest a follow-up query based on the current response",
    parameters: [
      {
        name: "suggested_query",
        type: "string",
        description: "The suggested follow-up query",
        required: true,
      },
      {
        name: "context",
        type: "string",
        description: "Context from the current response",
        required: false,
      },
    ],
    handler: async ({ suggested_query, context }) => {
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
```

Key implementation details:
- Registers 5 CopilotKit actions: save, export, share, bookmark, follow-up
- Each action has loading/success/error states with auto-reset
- Toast notifications for user feedback
- API calls to backend endpoints
- Export triggers file download
- Share copies link to clipboard
- Follow-up dispatches custom event or calls callback
- Agent can trigger actions programmatically

### 3. Create ActionPanel Component

**File:** `frontend/components/copilot/ActionPanel.tsx`

Create a panel showing action history and status:

```typescript
"use client";

import { memo, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  History,
  Save,
  Download,
  Share2,
  Bookmark,
  MessageSquarePlus,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronRight,
} from "lucide-react";
import type { ActionType, ActionState } from "@/hooks/use-copilot-actions";

/**
 * Action history entry.
 */
export interface ActionHistoryEntry {
  id: string;
  type: ActionType;
  timestamp: Date;
  status: "success" | "error";
  contentTitle?: string;
  details?: string;
  errorMessage?: string;
}

/**
 * Get icon for action type.
 */
function getActionIcon(type: ActionType): React.ReactNode {
  const iconClass = "h-4 w-4";
  switch (type) {
    case "save":
      return <Save className={iconClass} />;
    case "export":
      return <Download className={iconClass} />;
    case "share":
      return <Share2 className={iconClass} />;
    case "bookmark":
      return <Bookmark className={iconClass} />;
    case "followUp":
      return <MessageSquarePlus className={iconClass} />;
    default:
      return <Clock className={iconClass} />;
  }
}

/**
 * Get label for action type.
 */
function getActionLabel(type: ActionType): string {
  switch (type) {
    case "save":
      return "Saved to Workspace";
    case "export":
      return "Exported Content";
    case "share":
      return "Shared Link";
    case "bookmark":
      return "Bookmarked";
    case "followUp":
      return "Follow-up Query";
    default:
      return "Action";
  }
}

/**
 * Format relative time.
 */
function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);

  if (diffSec < 60) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHour < 24) return `${diffHour}h ago`;
  return date.toLocaleDateString();
}

interface ActionPanelProps {
  /** Action history entries */
  history: ActionHistoryEntry[];
  /** Callback to clear history */
  onClearHistory?: () => void;
  /** Callback when clicking an entry */
  onEntryClick?: (entry: ActionHistoryEntry) => void;
  /** Optional class name */
  className?: string;
}

/**
 * ActionPanel shows action history and status in a slide-out sheet.
 *
 * Story 6-5: Frontend Actions
 *
 * Features:
 * - Action history with timestamps
 * - Success/error status for each action
 * - Clear history option
 * - Click to view action details
 *
 * @example
 * ```tsx
 * <ActionPanel
 *   history={actionHistory}
 *   onClearHistory={() => setActionHistory([])}
 * />
 * ```
 */
export const ActionPanel = memo(function ActionPanel({
  history,
  onClearHistory,
  onEntryClick,
  className,
}: ActionPanelProps) {
  const successCount = history.filter((e) => e.status === "success").length;
  const errorCount = history.filter((e) => e.status === "error").length;

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            "gap-2 text-slate-600 hover:text-indigo-600",
            className
          )}
          aria-label="View action history"
        >
          <History className="h-4 w-4" />
          <span>Actions</span>
          {history.length > 0 && (
            <Badge variant="secondary" className="ml-1 h-5 min-w-[20px] px-1">
              {history.length}
            </Badge>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[540px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Action History
          </SheetTitle>
          <SheetDescription>
            Recent actions performed on AI responses
          </SheetDescription>
        </SheetHeader>

        {/* Statistics */}
        <div className="flex items-center gap-4 py-4 border-b border-slate-100">
          <Badge
            variant="outline"
            className="bg-emerald-50 text-emerald-700 border-emerald-200"
          >
            <CheckCircle2 className="h-3 w-3 mr-1" />
            {successCount} successful
          </Badge>
          {errorCount > 0 && (
            <Badge
              variant="outline"
              className="bg-red-50 text-red-700 border-red-200"
            >
              <XCircle className="h-3 w-3 mr-1" />
              {errorCount} failed
            </Badge>
          )}
          {onClearHistory && history.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClearHistory}
              className="ml-auto text-slate-500 hover:text-slate-700"
            >
              Clear
            </Button>
          )}
        </div>

        {/* History list */}
        <ScrollArea className="h-[calc(100vh-200px)] pr-4">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-slate-500">
              <Clock className="h-12 w-12 mb-4 text-slate-300" />
              <p className="text-sm">No actions yet</p>
              <p className="text-xs text-slate-400 mt-1">
                Actions will appear here as you interact with responses
              </p>
            </div>
          ) : (
            <div className="space-y-2 py-4">
              {history.map((entry) => (
                <button
                  key={entry.id}
                  onClick={() => onEntryClick?.(entry)}
                  className={cn(
                    "w-full flex items-start gap-3 p-3 rounded-lg border transition-colors text-left",
                    entry.status === "success"
                      ? "border-slate-200 hover:border-emerald-200 hover:bg-emerald-50/30"
                      : "border-red-200 bg-red-50/30 hover:bg-red-50/50"
                  )}
                >
                  {/* Status icon */}
                  <div
                    className={cn(
                      "flex-shrink-0 p-2 rounded-full",
                      entry.status === "success"
                        ? "bg-emerald-100 text-emerald-600"
                        : "bg-red-100 text-red-600"
                    )}
                  >
                    {getActionIcon(entry.type)}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm text-slate-900">
                        {getActionLabel(entry.type)}
                      </span>
                      <span className="text-xs text-slate-500">
                        {formatRelativeTime(entry.timestamp)}
                      </span>
                    </div>
                    {entry.contentTitle && (
                      <p className="text-sm text-slate-600 truncate mt-0.5">
                        {entry.contentTitle}
                      </p>
                    )}
                    {entry.details && (
                      <p className="text-xs text-slate-500 mt-1">
                        {entry.details}
                      </p>
                    )}
                    {entry.status === "error" && entry.errorMessage && (
                      <p className="text-xs text-red-600 mt-1">
                        {entry.errorMessage}
                      </p>
                    )}
                  </div>

                  {/* Arrow */}
                  <ChevronRight className="h-4 w-4 text-slate-400 flex-shrink-0 mt-2" />
                </button>
              ))}
            </div>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
});

export default ActionPanel;
```

### 4. Create Backend Workspace Endpoints

**File:** `backend/src/agentic_rag_backend/api/routes/workspace.py`

Create backend endpoints for save/export/share/bookmark operations:

```python
"""
Workspace API routes for frontend actions.

Story 6-5: Frontend Actions

Endpoints:
- POST /workspace/save - Save content to workspace
- POST /workspace/export - Export content in various formats
- POST /workspace/share - Generate shareable link
- POST /workspace/bookmark - Bookmark content
- GET /workspace/bookmarks - List bookmarks
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io
import json

router = APIRouter(prefix="/api/v1/workspace", tags=["workspace"])


# ============================================================================
# Request/Response Models
# ============================================================================


class SourceReference(BaseModel):
    """Reference to a source used in content."""

    id: str
    title: str
    url: Optional[str] = None


class SaveContentRequest(BaseModel):
    """Request to save content to workspace."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to save")
    title: str = Field(..., description="Title for the saved content")
    query: Optional[str] = Field(None, description="Original query")
    sources: Optional[List[SourceReference]] = Field(None, description="Source references")
    session_id: Optional[str] = Field(None, description="Session ID")
    trajectory_id: Optional[str] = Field(None, description="Trajectory ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenancy")


class SaveContentResponse(BaseModel):
    """Response after saving content."""

    id: str
    content_id: str
    title: str
    saved_at: datetime
    workspace_path: str


class ExportContentRequest(BaseModel):
    """Request to export content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to export")
    title: str = Field(..., description="Title for export")
    query: Optional[str] = Field(None, description="Original query")
    sources: Optional[List[SourceReference]] = Field(None, description="Source references")
    format: str = Field(..., description="Export format: markdown, pdf, json")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class ShareContentRequest(BaseModel):
    """Request to share content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to share")
    title: str = Field(..., description="Title for shared content")
    query: Optional[str] = Field(None, description="Original query")
    sources: Optional[List[SourceReference]] = Field(None, description="Source references")
    session_id: Optional[str] = Field(None, description="Session ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class ShareContentResponse(BaseModel):
    """Response with shareable link."""

    share_id: str
    share_url: str
    expires_at: Optional[datetime] = None


class BookmarkRequest(BaseModel):
    """Request to bookmark content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="Content preview")
    title: str = Field(..., description="Bookmark title")
    query: Optional[str] = Field(None, description="Original query")
    session_id: Optional[str] = Field(None, description="Session ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class BookmarkResponse(BaseModel):
    """Response after bookmarking."""

    bookmark_id: str
    content_id: str
    title: str
    created_at: datetime


class BookmarkListItem(BaseModel):
    """Item in bookmark list."""

    bookmark_id: str
    content_id: str
    title: str
    preview: str
    query: Optional[str] = None
    created_at: datetime


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/save", response_model=SaveContentResponse)
async def save_to_workspace(request: SaveContentRequest) -> SaveContentResponse:
    """
    Save content to user's workspace.

    Story 6-5: Frontend Actions

    This endpoint saves AI-generated content to the user's workspace
    for later reference. Content is stored with metadata including
    the original query, sources, and trajectory ID.
    """
    # TODO: Implement actual persistence to database
    # For now, return mock response

    saved_id = str(uuid4())
    workspace_path = f"/workspace/{request.tenant_id or 'default'}/{saved_id}"

    # Log action to trajectory if trajectory_id provided
    if request.trajectory_id:
        # TODO: Log to trajectory system
        pass

    return SaveContentResponse(
        id=saved_id,
        content_id=request.content_id,
        title=request.title,
        saved_at=datetime.utcnow(),
        workspace_path=workspace_path,
    )


@router.post("/export")
async def export_content(request: ExportContentRequest):
    """
    Export content in specified format.

    Story 6-5: Frontend Actions

    Supported formats:
    - markdown: Plain text with markdown formatting
    - pdf: PDF document
    - json: Structured JSON data
    """
    if request.format == "markdown":
        # Generate markdown content
        md_content = f"# {request.title}\n\n"
        if request.query:
            md_content += f"**Query:** {request.query}\n\n"
        md_content += f"{request.content}\n\n"

        if request.sources:
            md_content += "## Sources\n\n"
            for idx, source in enumerate(request.sources, 1):
                url_part = f" - [{source.url}]({source.url})" if source.url else ""
                md_content += f"{idx}. {source.title}{url_part}\n"

        md_content += f"\n---\n*Exported on {datetime.utcnow().isoformat()}*\n"

        return StreamingResponse(
            io.BytesIO(md_content.encode("utf-8")),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{request.title}.md"'
            },
        )

    elif request.format == "pdf":
        # TODO: Implement PDF generation (requires reportlab or similar)
        # For now, return markdown as fallback
        raise HTTPException(
            status_code=501,
            detail="PDF export not yet implemented. Please use markdown format.",
        )

    elif request.format == "json":
        # Generate JSON export
        export_data = {
            "id": request.content_id,
            "title": request.title,
            "content": request.content,
            "query": request.query,
            "sources": [s.model_dump() for s in (request.sources or [])],
            "exported_at": datetime.utcnow().isoformat(),
            "tenant_id": request.tenant_id,
        }

        return export_data

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format: {request.format}. Supported: markdown, pdf, json",
        )


@router.post("/share", response_model=ShareContentResponse)
async def share_content(request: ShareContentRequest) -> ShareContentResponse:
    """
    Generate a shareable link for content.

    Story 6-5: Frontend Actions

    Creates a shareable link that allows others to view the AI response.
    Links can optionally have expiration dates.
    """
    # TODO: Implement actual share link persistence
    # For now, return mock response

    share_id = str(uuid4())[:8]  # Short ID for URL

    # Generate shareable URL
    # In production, this would be a proper domain
    share_url = f"https://app.example.com/share/{share_id}"

    return ShareContentResponse(
        share_id=share_id,
        share_url=share_url,
        expires_at=None,  # No expiration by default
    )


@router.post("/bookmark", response_model=BookmarkResponse)
async def bookmark_content(request: BookmarkRequest) -> BookmarkResponse:
    """
    Bookmark content for quick access.

    Story 6-5: Frontend Actions

    Bookmarks are stored per-user and can be retrieved via GET /bookmarks.
    """
    # TODO: Implement actual bookmark persistence
    # For now, return mock response

    bookmark_id = str(uuid4())

    return BookmarkResponse(
        bookmark_id=bookmark_id,
        content_id=request.content_id,
        title=request.title,
        created_at=datetime.utcnow(),
    )


@router.get("/bookmarks", response_model=List[BookmarkListItem])
async def list_bookmarks(
    tenant_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[BookmarkListItem]:
    """
    List user's bookmarks.

    Story 6-5: Frontend Actions
    """
    # TODO: Implement actual bookmark retrieval
    # For now, return empty list

    return []


@router.delete("/bookmark/{bookmark_id}")
async def delete_bookmark(bookmark_id: str, tenant_id: Optional[str] = None):
    """
    Delete a bookmark.

    Story 6-5: Frontend Actions
    """
    # TODO: Implement actual bookmark deletion
    # For now, return success

    return {"success": True, "bookmark_id": bookmark_id}
```

### 5. Update Types

**File:** `frontend/types/copilot.ts` (modify)

Add types for frontend actions:

```typescript
// Add to existing types file

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
  sources?: Source[];
  /** Timestamp of the response */
  timestamp?: string;
  /** Session/conversation ID */
  sessionId?: string;
  /** Trajectory ID for this response */
  trajectoryId?: string;
}

/**
 * Result of an action operation.
 */
export interface ActionResult {
  success: boolean;
  message?: string;
  data?: unknown;
  error?: string;
}

/**
 * Response from save endpoint.
 */
export interface SaveContentResponse {
  id: string;
  contentId: string;
  title: string;
  savedAt: string;
  workspacePath: string;
}

/**
 * Response from share endpoint.
 */
export interface ShareContentResponse {
  shareId: string;
  shareUrl: string;
  expiresAt?: string;
}

/**
 * Response from bookmark endpoint.
 */
export interface BookmarkResponse {
  bookmarkId: string;
  contentId: string;
  title: string;
  createdAt: string;
}

/**
 * Bookmark list item.
 */
export interface BookmarkListItem {
  bookmarkId: string;
  contentId: string;
  title: string;
  preview: string;
  query?: string;
  createdAt: string;
}

// Zod schemas for validation
import { z } from "zod";

export const ActionTypeSchema = z.enum([
  "save",
  "export",
  "share",
  "bookmark",
  "followUp",
]);

export const ActionStateSchema = z.enum([
  "idle",
  "loading",
  "success",
  "error",
]);

export const ExportFormatSchema = z.enum(["markdown", "pdf", "json"]);

export const ActionableContentSchema = z.object({
  id: z.string(),
  content: z.string(),
  title: z.string().optional(),
  query: z.string().optional(),
  sources: z.array(SourceSchema).optional(),
  timestamp: z.string().optional(),
  sessionId: z.string().optional(),
  trajectoryId: z.string().optional(),
});

export const ActionResultSchema = z.object({
  success: z.boolean(),
  message: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
});

export const SaveContentResponseSchema = z.object({
  id: z.string(),
  contentId: z.string(),
  title: z.string(),
  savedAt: z.string(),
  workspacePath: z.string(),
});

export const ShareContentResponseSchema = z.object({
  shareId: z.string(),
  shareUrl: z.string(),
  expiresAt: z.string().optional(),
});

export const BookmarkResponseSchema = z.object({
  bookmarkId: z.string(),
  contentId: z.string(),
  title: z.string(),
  createdAt: z.string(),
});
```

### 6. Integrate ActionButtons into Chat Messages

**File:** `frontend/components/copilot/ChatMessage.tsx` (modify or create)

Show ActionButtons below AI responses:

```typescript
"use client";

import { memo, useMemo } from "react";
import { cn } from "@/lib/utils";
import { ActionButtons } from "./ActionButtons";
import { useCopilotActions } from "@/hooks/use-copilot-actions";
import type { Message } from "@copilotkit/react-core";
import type { ActionableContent } from "@/types/copilot";

interface ChatMessageProps {
  /** The message to display */
  message: Message;
  /** Whether this is the latest message */
  isLatest?: boolean;
  /** Session ID for context */
  sessionId?: string;
  /** Trajectory ID for context */
  trajectoryId?: string;
  /** Callback when follow-up is triggered */
  onFollowUp?: (query: string) => void;
}

/**
 * ChatMessage displays a single message with action buttons for AI responses.
 *
 * Story 6-5: Frontend Actions
 */
export const ChatMessage = memo(function ChatMessage({
  message,
  isLatest = false,
  sessionId,
  trajectoryId,
  onFollowUp,
}: ChatMessageProps) {
  const isAssistant = message.role === "assistant";

  // Initialize actions hook for this message
  const {
    actionStates,
    saveToWorkspace,
    exportContent,
    shareContent,
    bookmarkContent,
    triggerFollowUp,
  } = useCopilotActions({
    onFollowUp: onFollowUp
      ? (query) => onFollowUp(query)
      : undefined,
  });

  // Create actionable content from message
  const actionableContent: ActionableContent = useMemo(
    () => ({
      id: message.id,
      content: typeof message.content === "string" ? message.content : "",
      title: `Response - ${new Date().toLocaleDateString()}`,
      sessionId,
      trajectoryId,
    }),
    [message.id, message.content, sessionId, trajectoryId]
  );

  return (
    <div
      className={cn(
        "flex flex-col gap-2 p-4 rounded-lg",
        isAssistant
          ? "bg-slate-50 border border-slate-200"
          : "bg-indigo-50 border border-indigo-200 ml-8"
      )}
    >
      {/* Message content */}
      <div className="prose prose-sm max-w-none">
        {typeof message.content === "string" ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          message.content
        )}
      </div>

      {/* Action buttons for AI responses */}
      {isAssistant && (
        <div className="pt-2 border-t border-slate-100">
          <ActionButtons
            content={actionableContent}
            actionStates={actionStates}
            onSave={saveToWorkspace}
            onExport={exportContent}
            onShare={shareContent}
            onBookmark={bookmarkContent}
            onFollowUp={triggerFollowUp}
            compact={!isLatest}
          />
        </div>
      )}
    </div>
  );
});

export default ChatMessage;
```

### 7. Register Actions in CopilotProvider

**File:** `frontend/components/copilot/CopilotProvider.tsx` (modify)

Ensure actions are registered at the provider level:

```typescript
"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { ReactNode } from "react";
import { useFrontendActions } from "@/hooks/use-copilot-actions";

interface CopilotProviderProps {
  children: ReactNode;
}

/**
 * ActionRegistrar registers global CopilotKit actions.
 */
function ActionRegistrar() {
  // This hook registers the actions with CopilotKit
  useFrontendActions();
  return null;
}

/**
 * CopilotProvider wraps the application with CopilotKit and registers actions.
 *
 * Story 6-1: CopilotKit React Integration
 * Story 6-5: Frontend Actions
 */
export function CopilotProvider({ children }: CopilotProviderProps) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <ActionRegistrar />
      {children}
    </CopilotKit>
  );
}

export default CopilotProvider;
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `frontend/components/copilot/ActionButtons.tsx` | Row of action buttons (save, export, share, bookmark, follow-up) |
| `frontend/components/copilot/ActionPanel.tsx` | Slide-out panel showing action history |
| `frontend/hooks/use-copilot-actions.ts` | Custom hook for action state and CopilotKit integration |
| `backend/src/agentic_rag_backend/api/routes/workspace.py` | Backend endpoints for save/export/share/bookmark |

### Modified Files

| File | Change |
|------|--------|
| `frontend/types/copilot.ts` | Add ActionType, ActionState, ActionableContent, export response types |
| `frontend/components/copilot/CopilotProvider.tsx` | Register global actions |
| `frontend/components/copilot/ChatMessage.tsx` | Integrate ActionButtons below AI responses |
| `backend/src/agentic_rag_backend/main.py` | Add workspace router |

## Dependencies

### Frontend Dependencies (npm)

Already installed from previous stories:
```json
{
  "@copilotkit/react-core": "^1.50.1",
  "@copilotkit/react-ui": "^1.50.1",
  "lucide-react": "^0.x.x"
}
```

shadcn/ui components required (already installed or to add):
- Button
- DropdownMenu
- Tooltip
- Sheet
- Badge
- ScrollArea

Toast hook (shadcn/ui toast or similar):
```bash
npx shadcn-ui@latest add toast
```

### Backend Dependencies (pip)

No new dependencies required. Uses existing:
- FastAPI
- Pydantic
- Standard library (io, json, datetime)

### Environment Variables

No new environment variables required.

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| ActionButtons renders all action buttons | `frontend/__tests__/components/copilot/ActionButtons.test.tsx` |
| ActionButtons shows loading state during action | `frontend/__tests__/components/copilot/ActionButtons.test.tsx` |
| ActionButtons shows success/error states | `frontend/__tests__/components/copilot/ActionButtons.test.tsx` |
| ActionButtons export dropdown shows format options | `frontend/__tests__/components/copilot/ActionButtons.test.tsx` |
| ActionPanel shows action history | `frontend/__tests__/components/copilot/ActionPanel.test.tsx` |
| useCopilotActions saves to workspace | `frontend/__tests__/hooks/use-copilot-actions.test.ts` |
| useCopilotActions exports in all formats | `frontend/__tests__/hooks/use-copilot-actions.test.ts` |
| useCopilotActions copies share link to clipboard | `frontend/__tests__/hooks/use-copilot-actions.test.ts` |
| useCopilotActions bookmarks content | `frontend/__tests__/hooks/use-copilot-actions.test.ts` |
| useCopilotActions triggers follow-up | `frontend/__tests__/hooks/use-copilot-actions.test.ts` |

### Integration Tests

| Test | Location |
|------|----------|
| Save action calls backend API | `frontend/__tests__/integration/actions.test.tsx` |
| Export action downloads file | `frontend/__tests__/integration/actions.test.tsx` |
| Share action copies URL to clipboard | `frontend/__tests__/integration/actions.test.tsx` |
| Agent can trigger actions via CopilotKit | `frontend/__tests__/integration/actions.test.tsx` |
| Toast notifications appear for all actions | `frontend/__tests__/integration/actions.test.tsx` |

### E2E Tests

| Test | Location |
|------|----------|
| Full save flow: click save -> API call -> success toast | `frontend/tests/e2e/actions.spec.ts` |
| Full export flow: select format -> download file | `frontend/tests/e2e/actions.spec.ts` |
| Full share flow: click share -> URL copied | `frontend/tests/e2e/actions.spec.ts` |
| Bookmark toggle works correctly | `frontend/tests/e2e/actions.spec.ts` |
| Follow-up pre-fills chat input | `frontend/tests/e2e/actions.spec.ts` |
| Action panel shows history | `frontend/tests/e2e/actions.spec.ts` |

### Backend Tests

| Test | Location |
|------|----------|
| POST /workspace/save returns correct response | `backend/tests/api/routes/test_workspace.py` |
| POST /workspace/export returns markdown | `backend/tests/api/routes/test_workspace.py` |
| POST /workspace/export returns JSON | `backend/tests/api/routes/test_workspace.py` |
| POST /workspace/share generates URL | `backend/tests/api/routes/test_workspace.py` |
| POST /workspace/bookmark creates bookmark | `backend/tests/api/routes/test_workspace.py` |
| GET /workspace/bookmarks returns list | `backend/tests/api/routes/test_workspace.py` |

### Manual Verification Steps

1. Start backend with `cd backend && uv run uvicorn agentic_rag_backend.main:app --reload`
2. Start frontend with `cd frontend && pnpm dev`
3. Open browser to `http://localhost:3000`
4. Submit a query and wait for AI response
5. Verify action buttons appear below response:
   - Save button visible
   - Export dropdown with 3 format options
   - Share button visible
   - Bookmark button visible (toggles fill on click)
   - Follow-up button visible
6. Test each action:
   - **Save:** Click save -> loading state -> success toast
   - **Export Markdown:** Click export -> markdown -> file downloads
   - **Export JSON:** Click export -> JSON -> file downloads
   - **Share:** Click share -> URL copied toast -> verify clipboard
   - **Bookmark:** Click bookmark -> fills yellow -> success toast
   - **Follow-up:** Click follow-up -> chat input pre-filled
7. Test error states:
   - Disconnect network -> click save -> error toast
   - Verify error state styling (red color)
8. Verify toast notifications:
   - Success toasts appear bottom-right
   - Error toasts appear with destructive styling
9. Verify action panel:
   - Click "Actions" button in toolbar
   - Panel slides out with history
   - Success/error entries styled differently
   - Clear button removes history
10. Verify accessibility:
    - Tab through all buttons
    - Tooltips appear on focus
    - Screen reader announces button purposes

## Definition of Done

- [ ] All acceptance criteria met
- [ ] ActionButtons component created with all 5 actions
- [ ] Export dropdown offers markdown/PDF/JSON formats
- [ ] use-copilot-actions hook manages state and API calls
- [ ] useCopilotAction hooks registered for each action type
- [ ] Backend workspace endpoints implemented
- [ ] Loading states shown during API calls
- [ ] Success toasts displayed on completion
- [ ] Error toasts displayed on failure
- [ ] Action history tracked in ActionPanel
- [ ] Agent can trigger actions programmatically via CopilotKit
- [ ] TypeScript types added for all data structures
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Backend tests passing
- [ ] Manual verification completed
- [ ] No TypeScript errors
- [ ] Code follows project naming conventions
- [ ] Design system colors applied correctly

## Technical Notes

### CopilotKit Action Registration

Actions are registered using `useCopilotAction` hook:

```typescript
useCopilotAction({
  name: "save_to_workspace",
  description: "Save content to user's workspace",
  parameters: [
    { name: "content_id", type: "string", required: true },
  ],
  handler: async ({ content_id }) => {
    // Execute action
    return { success: true };
  },
});
```

The agent can trigger these actions by name via the AG-UI protocol.

### Toast Notifications

Using shadcn/ui toast for notifications:

```typescript
const { toast } = useToast();

toast({
  variant: "default", // or "destructive" for errors
  title: "Success!",
  description: "Content saved to workspace",
});
```

### Export File Download

Export creates a blob and triggers download:

```typescript
const blob = new Blob([content], { type: "text/markdown" });
const url = URL.createObjectURL(blob);
const link = document.createElement("a");
link.href = url;
link.download = "filename.md";
link.click();
URL.revokeObjectURL(url);
```

### Follow-up Query Pattern

Follow-up uses custom event to communicate with chat input:

```typescript
document.dispatchEvent(
  new CustomEvent("copilot:follow-up", {
    detail: { suggestedQuery, context },
  })
);
```

Chat input listens for this event to pre-fill.

### State Auto-Reset

Action states auto-reset after success/error:

```typescript
// Success resets after 2 seconds
// Error resets after 3 seconds
const timer = setTimeout(() => {
  setActionState(action, "idle");
}, state === "success" ? 2000 : 3000);
```

## Design System Colors

Per UX Design Specification:
- **Indigo-600 (#4F46E5):** Primary action buttons (hover state)
- **Emerald-500 (#10B981):** Success state
- **Red-500 (#EF4444):** Error state
- **Amber-500 (#F59E0B):** Bookmark active state
- **Slate:** Neutral button text and borders

## Accessibility Considerations

- All buttons have `aria-label` attributes
- Tooltips provide additional context
- Keyboard navigation: Tab through buttons
- Focus indicators visible
- Screen reader announces button states
- Bookmark uses `aria-pressed` for toggle state

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API errors during action | Toast notification with error message, retry option |
| Clipboard API not available | Fallback to select-and-copy UI |
| Large content export | Streaming response, progress indicator |
| PDF generation complexity | Start with markdown only, add PDF in future iteration |
| Action spam | Debounce rapid clicks, disable during loading |

## References

- [CopilotKit useCopilotAction](https://docs.copilotkit.ai/reference/hooks/useCopilotAction)
- [shadcn/ui Toast](https://ui.shadcn.com/docs/components/toast)
- [shadcn/ui DropdownMenu](https://ui.shadcn.com/docs/components/dropdown-menu)
- [Epic 6 Tech Spec](_bmad-output/implementation-artifacts/epic-6-tech-spec.md)
- [Story 6-4: Human-in-the-Loop Source Validation](_bmad-output/implementation-artifacts/stories/6-4-human-in-the-loop-source-validation.md)
