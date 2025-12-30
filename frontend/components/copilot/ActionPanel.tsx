"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  Check,
  AlertCircle,
  Loader2,
  Save,
  Download,
  Share2,
  Bookmark,
  MessageSquare,
  Trash2,
  Clock,
} from "lucide-react";
import type { ActionType } from "@/hooks/use-copilot-actions";

/**
 * Single action history item.
 */
export interface ActionHistoryItem {
  id: string;
  type: ActionType;
  status: "pending" | "success" | "error";
  timestamp: string;
  title: string;
  error?: string;
  data?: {
    shareUrl?: string;
    filename?: string;
    [key: string]: unknown;
  };
}

/**
 * Props for the ActionPanel component.
 */
export interface ActionPanelProps {
  /** List of action history items */
  actions: ActionHistoryItem[];
  /** Whether the panel is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Clear history handler */
  onClearHistory: () => void;
  /** Additional class names */
  className?: string;
}

/**
 * Get icon for action type.
 */
function getActionTypeIcon(type: ActionType): React.ComponentType<{ className?: string }> {
  switch (type) {
    case "save":
      return Save;
    case "export":
      return Download;
    case "share":
      return Share2;
    case "bookmark":
      return Bookmark;
    case "followUp":
      return MessageSquare;
    default:
      return Save;
  }
}

/**
 * Get status styles.
 */
function getStatusStyles(status: "pending" | "success" | "error"): {
  border: string;
  bg: string;
  icon: React.ReactNode;
} {
  switch (status) {
    case "success":
      return {
        border: "border-emerald-200",
        bg: "bg-emerald-50",
        icon: <Check className="h-4 w-4 text-emerald-500" />,
      };
    case "error":
      return {
        border: "border-red-200",
        bg: "bg-red-50",
        icon: <AlertCircle className="h-4 w-4 text-red-500" />,
      };
    case "pending":
    default:
      return {
        border: "border-blue-200",
        bg: "bg-blue-50",
        icon: <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />,
      };
  }
}

/**
 * Format timestamp for display.
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) {
    return "Just now";
  } else if (diffMins < 60) {
    return diffMins + " min ago";
  } else if (diffMins < 1440) {
    const hours = Math.floor(diffMins / 60);
    return hours + " hour" + (hours > 1 ? "s" : "") + " ago";
  } else {
    return date.toLocaleDateString();
  }
}

/**
 * ActionItem displays a single action history item.
 */
const ActionItem = memo(function ActionItem({
  action,
}: {
  action: ActionHistoryItem;
}) {
  const ActionIcon = getActionTypeIcon(action.type);
  const { border, bg, icon } = getStatusStyles(action.status);

  return (
    <div
      className={cn(
        "flex items-start gap-3 p-3 rounded-lg border",
        border,
        bg
      )}
    >
      {/* Action type icon */}
      <div className="flex-shrink-0 mt-0.5">
        <ActionIcon className="h-4 w-4 text-slate-500" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-800 truncate">
            {action.title}
          </span>
          {icon}
        </div>

        {/* Error message */}
        {action.status === "error" && action.error && (
          <p className="mt-1 text-xs text-red-600">{action.error}</p>
        )}

        {/* Share URL */}
        {action.type === "share" && action.data?.shareUrl && (
          <p className="mt-1 text-xs text-slate-500 truncate">
            {action.data.shareUrl}
          </p>
        )}

        {/* Timestamp */}
        <div className="flex items-center gap-1 mt-1.5">
          <Clock className="h-3 w-3 text-slate-400" />
          <span className="text-xs text-slate-400">
            {formatTimestamp(action.timestamp)}
          </span>
        </div>
      </div>
    </div>
  );
});

/**
 * ActionPanel displays action history in a slide-out panel.
 *
 * Story 6-5: Frontend Actions
 *
 * Features:
 * - Lists recent actions with status icons
 * - Shows action details (share URL, errors)
 * - Clear history button
 * - Timestamps
 */
export const ActionPanel = memo(function ActionPanel({
  actions,
  isOpen,
  onClose,
  onClearHistory,
  className,
}: ActionPanelProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <aside
      className={cn(
        "fixed right-0 top-0 h-full w-80 bg-white shadow-xl z-40",
        "border-l border-slate-200",
        "transform transition-transform duration-300 ease-in-out",
        className
      )}
      role="complementary"
      aria-label="Action history panel"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">Action History</h2>
        <button
          type="button"
          onClick={onClose}
          className="p-1.5 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
          aria-label="Close action history"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Actions toolbar */}
      <div className="flex items-center justify-end px-4 py-2 border-b border-slate-100">
        <button
          type="button"
          onClick={onClearHistory}
          className={cn(
            "flex items-center gap-1.5 px-2 py-1 text-xs",
            "text-slate-500 hover:text-red-600",
            "rounded hover:bg-red-50",
            "transition-colors duration-200"
          )}
          disabled={actions.length === 0}
        >
          <Trash2 className="h-3.5 w-3.5" />
          Clear
        </button>
      </div>

      {/* Action list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {actions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Clock className="h-8 w-8 mb-2" />
            <p className="text-sm">No actions yet</p>
          </div>
        ) : (
          actions.map((action) => (
            <ActionItem key={action.id} action={action} />
          ))
        )}
      </div>
    </aside>
  );
});

export default ActionPanel;
