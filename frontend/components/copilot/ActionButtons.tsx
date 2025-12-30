"use client";

import { memo, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Save,
  Download,
  Share2,
  Bookmark,
  MessageSquare,
  ChevronDown,
  Loader2,
  Check,
  X,
  FileText,
  FileJson,
} from "lucide-react";
import type { ActionStates, ActionableContent, ExportFormat } from "@/hooks/use-copilot-actions";

/**
 * Props for the ActionButtons component.
 */
export interface ActionButtonsProps {
  /** Content to perform actions on */
  content: ActionableContent;
  /** Current state of all actions */
  actionStates: ActionStates;
  /** Save to workspace handler */
  onSave: (content: ActionableContent) => void;
  /** Export handler */
  onExport: (content: ActionableContent, format: ExportFormat) => void;
  /** Share handler */
  onShare: (content: ActionableContent) => void;
  /** Bookmark handler */
  onBookmark: (content: ActionableContent) => void;
  /** Follow-up handler */
  onFollowUp: (content: ActionableContent) => void;
  /** Compact mode for smaller layouts */
  compact?: boolean;
  /** Additional class names */
  className?: string;
}

/**
 * Get icon based on action state.
 */
function getStateIcon(
  state: "idle" | "loading" | "success" | "error",
  DefaultIcon: React.ComponentType<{ className?: string }>
): React.ReactNode {
  switch (state) {
    case "loading":
      return <Loader2 className="h-4 w-4 animate-spin" />;
    case "success":
      return <Check className="h-4 w-4" />;
    case "error":
      return <X className="h-4 w-4" />;
    default:
      return <DefaultIcon className="h-4 w-4" />;
  }
}

/**
 * Get button color based on action state.
 */
function getStateColor(state: "idle" | "loading" | "success" | "error", defaultColor: string): string {
  switch (state) {
    case "success":
      return "text-emerald-600 hover:text-emerald-700";
    case "error":
      return "text-red-600 hover:text-red-700";
    default:
      return defaultColor;
  }
}

/**
 * ActionButtons component displays action buttons for AI responses.
 *
 * Story 6-5: Frontend Actions
 *
 * Actions:
 * - Save: Indigo-600
 * - Export: Slate-600 (dropdown)
 * - Share: Indigo-600
 * - Bookmark: Amber-500
 * - Follow-up: Slate-600
 */
export const ActionButtons = memo(function ActionButtons({
  content,
  actionStates,
  onSave,
  onExport,
  onShare,
  onBookmark,
  onFollowUp,
  compact = false,
  className,
}: ActionButtonsProps) {
  const [exportDropdownOpen, setExportDropdownOpen] = useState(false);

  const handleExport = useCallback(
    (format: ExportFormat) => {
      onExport(content, format);
      setExportDropdownOpen(false);
    },
    [content, onExport]
  );

  const buttonBase = cn(
    "inline-flex items-center justify-center rounded-md",
    "transition-colors duration-200",
    "focus:outline-none focus:ring-2 focus:ring-offset-2",
    "disabled:opacity-50 disabled:cursor-not-allowed",
    compact ? "p-1.5" : "p-2"
  );

  return (
    <div
      className={cn(
        "flex items-center gap-1",
        compact ? "gap-0.5" : "gap-2",
        className
      )}
    >
      {/* Save Button */}
      <button
        type="button"
        onClick={() => onSave(content)}
        disabled={actionStates.save === "loading"}
        className={cn(
          buttonBase,
          getStateColor(actionStates.save, "text-indigo-600 hover:text-indigo-700"),
          "hover:bg-indigo-50 focus:ring-indigo-500"
        )}
        aria-label="Save to workspace"
        title="Save to workspace"
      >
        {getStateIcon(actionStates.save, Save)}
      </button>

      {/* Export Dropdown */}
      <div className="relative">
        <button
          type="button"
          onClick={() => setExportDropdownOpen(!exportDropdownOpen)}
          disabled={actionStates.export === "loading"}
          className={cn(
            buttonBase,
            getStateColor(actionStates.export, "text-slate-600 hover:text-slate-700"),
            "hover:bg-slate-100 focus:ring-slate-500"
          )}
          aria-label="Export"
          aria-expanded={exportDropdownOpen}
          aria-haspopup="true"
          title="Export"
        >
          {getStateIcon(actionStates.export, Download)}
          <ChevronDown className={cn("h-3 w-3 ml-0.5", exportDropdownOpen && "rotate-180")} />
        </button>

        {exportDropdownOpen && (
          <>
            {/* Backdrop to close dropdown */}
            <div
              className="fixed inset-0 z-10"
              onClick={() => setExportDropdownOpen(false)}
            />

            {/* Dropdown menu */}
            <div
              className={cn(
                "absolute right-0 mt-1 w-40 rounded-md shadow-lg z-20",
                "bg-white border border-slate-200",
                "py-1"
              )}
              role="menu"
            >
              <button
                type="button"
                onClick={() => handleExport("markdown")}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-slate-700 hover:bg-slate-100"
                role="menuitem"
              >
                <FileText className="h-4 w-4" />
                Markdown
              </button>
              <button
                type="button"
                onClick={() => handleExport("json")}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-slate-700 hover:bg-slate-100"
                role="menuitem"
              >
                <FileJson className="h-4 w-4" />
                JSON
              </button>
              {/* PDF export hidden until backend implementation is ready (returns 501) */}
            </div>
          </>
        )}
      </div>

      {/* Share Button */}
      <button
        type="button"
        onClick={() => onShare(content)}
        disabled={actionStates.share === "loading"}
        className={cn(
          buttonBase,
          getStateColor(actionStates.share, "text-indigo-600 hover:text-indigo-700"),
          "hover:bg-indigo-50 focus:ring-indigo-500"
        )}
        aria-label="Share"
        title="Share"
      >
        {getStateIcon(actionStates.share, Share2)}
      </button>

      {/* Bookmark Button */}
      <button
        type="button"
        onClick={() => onBookmark(content)}
        disabled={actionStates.bookmark === "loading"}
        className={cn(
          buttonBase,
          getStateColor(actionStates.bookmark, "text-amber-500 hover:text-amber-600"),
          "hover:bg-amber-50 focus:ring-amber-500"
        )}
        aria-label="Bookmark"
        title="Bookmark"
      >
        {getStateIcon(actionStates.bookmark, Bookmark)}
      </button>

      {/* Follow-up Button */}
      <button
        type="button"
        onClick={() => onFollowUp(content)}
        disabled={actionStates.followUp === "loading"}
        className={cn(
          buttonBase,
          getStateColor(actionStates.followUp, "text-slate-600 hover:text-slate-700"),
          "hover:bg-slate-100 focus:ring-slate-500"
        )}
        aria-label="Follow up"
        title="Follow up"
      >
        {getStateIcon(actionStates.followUp, MessageSquare)}
      </button>
    </div>
  );
});

export default ActionButtons;
