"use client";

import { memo } from "react";
import { Loader2, Play, CheckCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Tool call status type supporting both CopilotKit 1.x and 2.x
 *
 * CopilotKit 1.x uses lowercase: "inProgress", "executing", "complete"
 * CopilotKit 2.x uses PascalCase: "InProgress", "Executing", "Complete"
 *
 * Added error/failed states for failed tool calls (Issue 3.2)
 */
export type ToolStatus =
  | "inProgress"
  | "executing"
  | "complete" // 1.x lowercase
  | "InProgress"
  | "Executing"
  | "Complete" // 2.x PascalCase
  | "error"
  | "failed"
  | "Error"
  | "Failed"; // Error states (Issue 3.2)

/**
 * Check if status indicates tool is preparing/in progress.
 */
export function isInProgress(status: ToolStatus): boolean {
  return status === "inProgress" || status === "InProgress";
}

/**
 * Check if status indicates tool is currently executing.
 */
export function isExecuting(status: ToolStatus): boolean {
  return status === "executing" || status === "Executing";
}

/**
 * Check if status indicates tool execution is complete.
 */
export function isComplete(status: ToolStatus): boolean {
  return status === "complete" || status === "Complete";
}

/**
 * Check if status indicates tool execution failed.
 * (Issue 3.2: StatusBadge Missing Error State)
 */
export function isError(status: ToolStatus): boolean {
  return status === "error" || status === "Error" ||
         status === "failed" || status === "Failed";
}

interface StatusBadgeProps {
  /** Current tool execution status */
  status: ToolStatus;
  /** Additional CSS classes */
  className?: string;
}

/**
 * StatusBadge displays the current state of a tool call.
 *
 * Story 21-A3: Implement Tool Call Visualization
 *
 * States:
 * - inProgress/InProgress: Blue background, Loader2 spinning icon, "Preparing" text
 * - executing/Executing: Yellow background, Play icon, "Running" text
 * - complete/Complete: Green background, CheckCircle icon, "Complete" text
 *
 * Design follows the project's badge pattern from SourceCard.tsx:
 * - bg-{color}-100 text-{color}-800 border border-{color}-200
 */
export const StatusBadge = memo(function StatusBadge({
  status,
  className,
}: StatusBadgeProps) {
  if (isInProgress(status)) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
          "bg-blue-100 text-blue-800 border border-blue-200",
          className
        )}
        data-testid="status-badge-inprogress"
      >
        <Loader2
          className="h-3 w-3 animate-spin"
          aria-hidden="true"
          data-testid="icon-loader"
        />
        Preparing
      </span>
    );
  }

  if (isExecuting(status)) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
          "bg-yellow-100 text-yellow-800 border border-yellow-200",
          className
        )}
        data-testid="status-badge-executing"
      >
        <Play
          className="h-3 w-3"
          aria-hidden="true"
          data-testid="icon-play"
        />
        Running
      </span>
    );
  }

  if (isComplete(status)) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
          "bg-emerald-100 text-emerald-800 border border-emerald-200",
          className
        )}
        data-testid="status-badge-complete"
      >
        <CheckCircle
          className="h-3 w-3"
          aria-hidden="true"
          data-testid="icon-check"
        />
        Complete
      </span>
    );
  }

  // Error state (Issue 3.2)
  if (isError(status)) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
          "bg-red-100 text-red-800 border border-red-200",
          className
        )}
        data-testid="status-badge-error"
      >
        <XCircle
          className="h-3 w-3"
          aria-hidden="true"
          data-testid="icon-error"
        />
        Failed
      </span>
    );
  }

  // Fallback for unknown status
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium",
        "bg-slate-100 text-slate-600 border border-slate-200",
        className
      )}
      data-testid="status-badge-unknown"
    >
      {status}
    </span>
  );
});

export default StatusBadge;
