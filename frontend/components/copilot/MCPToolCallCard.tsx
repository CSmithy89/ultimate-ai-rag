"use client";

import { memo, useState, useCallback } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusBadge, type ToolStatus, isComplete } from "./StatusBadge";
import { redactSensitiveKeys } from "@/lib/utils/redact";

/** Maximum characters for result display before truncation */
const MAX_RESULT_LENGTH = 500;

export interface MCPToolCallCardProps {
  /** Tool name (e.g., "vector_search", "ingest_url") */
  name: string;
  /** Tool arguments */
  args: Record<string, unknown>;
  /** Current execution status */
  status: ToolStatus;
  /** Tool result (when complete) */
  result?: unknown;
  /** Additional CSS classes */
  className?: string;
}

/**
 * MCPToolCallCard displays a collapsible card for MCP tool calls.
 *
 * Story 21-A3: Implement Tool Call Visualization
 *
 * Features:
 * - Collapsible header with tool name and status badge
 * - Redacted display of tool arguments (sensitive keys hidden)
 * - Truncated display of results (max 500 chars)
 * - Expand/collapse animation with ChevronDown rotation
 * - Keyboard accessible (Enter/Space to toggle)
 *
 * Design follows project patterns from SourceCard.tsx and AnswerPanel.tsx:
 * - Uses cn() for class merging
 * - Consistent slate color palette for neutral UI
 * - Proper focus states for accessibility
 */
export const MCPToolCallCard = memo(function MCPToolCallCard({
  name,
  args,
  status,
  result,
  className,
}: MCPToolCallCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Redact sensitive keys from args and result
  const redactedArgs = redactSensitiveKeys(args);
  const redactedResult =
    result && typeof result === "object"
      ? redactSensitiveKeys(result as Record<string, unknown>)
      : result;

  // Truncate result for display
  const resultString = redactedResult
    ? JSON.stringify(redactedResult, null, 2)
    : null;
  const truncatedResult = resultString
    ? resultString.slice(0, MAX_RESULT_LENGTH)
    : null;
  const isResultTruncated =
    resultString && resultString.length > MAX_RESULT_LENGTH;

  const handleToggle = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleToggle();
      }
    },
    [handleToggle]
  );

  return (
    <div
      className={cn(
        "my-2 border border-slate-200 rounded-lg bg-white overflow-hidden",
        className
      )}
      data-testid="mcp-tool-call-card"
    >
      {/* Header */}
      <button
        type="button"
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        className={cn(
          "w-full p-3 flex items-center justify-between",
          "hover:bg-slate-50 transition-colors cursor-pointer",
          "focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
        )}
        aria-expanded={isExpanded}
        aria-controls={`tool-content-${name}`}
      >
        <div className="flex items-center gap-2">
          <StatusBadge status={status} />
          <span className="font-mono text-sm text-slate-700">{name}</span>
        </div>
        <ChevronDown
          className={cn(
            "h-4 w-4 text-slate-500 transition-transform duration-200",
            isExpanded && "rotate-180"
          )}
          aria-hidden="true"
          data-testid="icon-chevron"
        />
      </button>

      {/* Expandable content */}
      {isExpanded && (
        <div
          id={`tool-content-${name}`}
          className="p-3 pt-0 space-y-3 border-t border-slate-100"
        >
          {/* Arguments */}
          <div>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
              Arguments
            </span>
            <pre
              className="mt-1 text-xs bg-slate-50 p-2 rounded overflow-auto max-h-40 text-slate-700"
              data-testid="tool-args"
            >
              {JSON.stringify(redactedArgs, null, 2)}
            </pre>
          </div>

          {/* Result (when complete) */}
          {isComplete(status) && truncatedResult && (
            <div>
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Result
              </span>
              <pre
                className="mt-1 text-xs bg-slate-50 p-2 rounded overflow-auto max-h-40 text-slate-700"
                data-testid="tool-result"
              >
                {truncatedResult}
                {isResultTruncated && (
                  <span className="text-slate-400">...</span>
                )}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default MCPToolCallCard;
