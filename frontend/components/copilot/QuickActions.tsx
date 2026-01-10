"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import {
  FileText,
  Lightbulb,
  Search,
  RefreshCw,
  Trash2,
  Loader2,
} from "lucide-react";
import { useProgrammaticChat } from "@/hooks/use-programmatic-chat";
import type { QuickActionConfig } from "@/types/copilot";

/**
 * Default quick actions for RAG-specific operations.
 */
export const DEFAULT_QUICK_ACTIONS: QuickActionConfig[] = [
  {
    label: "Summarize",
    message: "Summarize the current document or conversation context",
    icon: "FileText",
    description: "Get a concise summary",
  },
  {
    label: "Key Insights",
    message: "Extract the key insights and important points",
    icon: "Lightbulb",
    description: "Identify main takeaways",
  },
  {
    label: "Related Topics",
    message: "Find related topics and suggest further exploration areas",
    icon: "Search",
    description: "Discover related content",
  },
];

/**
 * Props for the QuickActions component.
 */
export interface QuickActionsProps {
  /** Custom actions to display instead of defaults */
  actions?: QuickActionConfig[];
  /** Whether to show regenerate button */
  showRegenerate?: boolean;
  /** Whether to show clear button */
  showClear?: boolean;
  /** Orientation of the button layout */
  orientation?: "horizontal" | "vertical";
  /** Size variant */
  size?: "sm" | "md" | "lg";
  /** Additional class names */
  className?: string;
}

/**
 * Map of icon names to Lucide components.
 */
const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  FileText,
  Lightbulb,
  Search,
};

/**
 * Get icon component by name.
 */
function getIcon(iconName?: string): React.ComponentType<{ className?: string }> | null {
  if (!iconName) return null;
  return ICON_MAP[iconName] || null;
}

/**
 * QuickActions component provides preset action buttons for common chat operations.
 *
 * Story 21-A6: Implement useCopilotChat for Headless Control
 *
 * Features:
 * - Preset buttons for common RAG operations (Summarize, Key Insights, Related Topics)
 * - Optional regenerate and clear buttons
 * - Customizable actions via props
 * - Loading state management (buttons disabled during generation)
 * - Flexible layout options (horizontal/vertical)
 *
 * @example
 * ```tsx
 * // Basic usage with default actions
 * <QuickActions />
 * ```
 *
 * @example
 * ```tsx
 * // Custom actions
 * <QuickActions
 *   actions={[
 *     { label: "Explain", message: "Explain this in detail" },
 *     { label: "Compare", message: "Compare these concepts" },
 *   ]}
 *   showRegenerate
 *   showClear
 * />
 * ```
 *
 * @example
 * ```tsx
 * // Vertical layout for sidebar
 * <QuickActions
 *   orientation="vertical"
 *   size="sm"
 *   className="border-b pb-2"
 * />
 * ```
 */
export const QuickActions = memo(function QuickActions({
  actions = DEFAULT_QUICK_ACTIONS,
  showRegenerate = false,
  showClear = false,
  orientation = "horizontal",
  size = "md",
  className,
}: QuickActionsProps) {
  const {
    sendMessage,
    regenerateLastResponse,
    clearHistory,
    isLoading,
    messageCount,
  } = useProgrammaticChat();

  // Size-based styles
  const sizeStyles = {
    sm: {
      button: "px-2 py-1 text-xs",
      icon: "h-3 w-3",
      gap: "gap-1",
    },
    md: {
      button: "px-3 py-1.5 text-sm",
      icon: "h-4 w-4",
      gap: "gap-2",
    },
    lg: {
      button: "px-4 py-2 text-base",
      icon: "h-5 w-5",
      gap: "gap-3",
    },
  };

  const currentSize = sizeStyles[size];

  const buttonBase = cn(
    "inline-flex items-center justify-center rounded-md",
    "border border-slate-200 bg-white",
    "text-slate-700 hover:bg-slate-50",
    "transition-colors duration-200",
    "focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-indigo-500",
    "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-white",
    currentSize.button
  );

  const containerClasses = cn(
    "flex",
    orientation === "vertical" ? "flex-col" : "flex-row flex-wrap",
    currentSize.gap,
    className
  );

  return (
    <div className={containerClasses}>
      {/* Action Buttons */}
      {actions.map((action) => {
        const Icon = getIcon(action.icon);

        return (
          <button
            key={action.label}
            type="button"
            onClick={() => sendMessage(action.message)}
            disabled={isLoading}
            className={buttonBase}
            title={action.description || action.label}
            aria-label={action.label}
          >
            {isLoading ? (
              <Loader2 className={cn(currentSize.icon, "animate-spin mr-1.5")} />
            ) : Icon ? (
              <Icon className={cn(currentSize.icon, "mr-1.5")} />
            ) : null}
            {action.label}
          </button>
        );
      })}

      {/* Regenerate Button */}
      {showRegenerate && messageCount > 0 && (
        <button
          type="button"
          onClick={regenerateLastResponse}
          disabled={isLoading}
          className={cn(
            buttonBase,
            "text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50",
            "border-indigo-200"
          )}
          title="Regenerate last response"
          aria-label="Regenerate"
        >
          {isLoading ? (
            <Loader2 className={cn(currentSize.icon, "animate-spin mr-1.5")} />
          ) : (
            <RefreshCw className={cn(currentSize.icon, "mr-1.5")} />
          )}
          Regenerate
        </button>
      )}

      {/* Clear Button */}
      {showClear && messageCount > 0 && (
        <button
          type="button"
          onClick={clearHistory}
          disabled={isLoading}
          className={cn(
            buttonBase,
            "text-red-600 hover:text-red-700 hover:bg-red-50",
            "border-red-200"
          )}
          title="Clear chat history"
          aria-label="Clear history"
        >
          <Trash2 className={cn(currentSize.icon, "mr-1.5")} />
          Clear
        </button>
      )}
    </div>
  );
});

export default QuickActions;
