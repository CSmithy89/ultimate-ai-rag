"use client";

/**
 * useSourceValidationV2 - Simplified HITL with renderAndWaitForResponse
 *
 * AG-UI Enhancement: This hook provides a cleaner implementation of Human-in-the-Loop
 * source validation using CopilotKit's `renderAndWaitForResponse` pattern.
 *
 * Benefits over v1:
 * - Simpler code with less state management
 * - Automatic blocking until user responds
 * - Cleaner separation of concerns
 * - No manual respond callback handling
 *
 * @example
 * ```tsx
 * function ChatWithHITL() {
 *   useSourceValidationV2({
 *     onApprove: (sources) => console.log("Approved:", sources),
 *     onReject: () => console.log("Rejected all"),
 *   });
 *
 *   // Dialog automatically renders and blocks agent until user responds
 *   return <ChatSidebar />;
 * }
 * ```
 */

import React, { useState, useRef } from "react";
import { useCopilotAction } from "@copilotkit/react-core";
import type { Source } from "@/types/copilot";

// ============================================
// TYPES
// ============================================

/**
 * Options for the useSourceValidationV2 hook.
 */
export interface UseSourceValidationV2Options {
  /** Callback when sources are approved */
  onApprove?: (approvedSources: Source[]) => void;
  /** Callback when validation is rejected/cancelled */
  onReject?: () => void;
  /** Callback when user skips validation (approve all) */
  onSkip?: (sources: Source[]) => void;
  /** Auto-approve threshold (skip dialog if all sources meet threshold) */
  autoApproveThreshold?: number;
  /** Custom title for the dialog */
  dialogTitle?: string;
  /** Whether to show similarity scores */
  showSimilarityScores?: boolean;
  /** Whether to allow partial approval */
  allowPartialApproval?: boolean;
}

/**
 * Internal state for the validation card.
 */
interface ValidationCardState {
  selectedIds: Set<string>;
}


// ============================================
// HOOK IMPLEMENTATION
// ============================================

/**
 * useSourceValidationV2 provides simplified HITL source validation
 * using CopilotKit's renderAndWaitForResponse pattern.
 *
 * AG-UI Enhancement: This cleaner implementation:
 * - Eliminates manual respond callback handling
 * - Automatically blocks agent until user responds
 * - Provides inline validation UI without modal
 *
 * @param options - Configuration options
 */
export function useSourceValidationV2(
  options: UseSourceValidationV2Options = {}
): void {
  const {
    onApprove,
    onReject,
    onSkip,
    autoApproveThreshold,
    dialogTitle = "Review Retrieved Sources",
    showSimilarityScores = true,
    allowPartialApproval = true,
  } = options;

  // Track which calls we've already responded to
  const respondedRef = useRef<Set<string>>(new Set());

  // State for selected sources in the validation card
  const [cardState, setCardState] = useState<ValidationCardState>({
    selectedIds: new Set(),
  });

  useCopilotAction({
    name: "validate_sources",
    description:
      "Request human approval for retrieved sources before generating answer",
    parameters: [
      {
        name: "sources",
        type: "object[]",
        description: "Array of sources to validate",
        required: true,
      },
      {
        name: "query",
        type: "string",
        description: "The original query for context",
        required: true,
      },
      {
        name: "checkpoint_id",
        type: "string",
        description: "Unique identifier for this validation checkpoint",
        required: true,
      },
    ],
    // renderAndWaitForResponse automatically blocks until respond is called
    renderAndWaitForResponse: ({ args, respond, status }) => {
      // Extract sources safely
      const sources: Source[] = Array.isArray(args?.sources)
        ? (args.sources as Source[])
        : [];
      const query = (args?.query as string) || "";
      const checkpointId = (args?.checkpoint_id as string) || "";

      // Generate unique call ID
      const callId = checkpointId || sources.map((s) => s.id).sort().join(",");

      // Handle completion state
      if (status === "complete") {
        return React.createElement(
          "div",
          { className: "flex items-center gap-2 text-sm text-green-600 p-2" },
          React.createElement("span", null, "✓"),
          "Sources validated"
        );
      }

      // Check if already responded
      if (respondedRef.current.has(callId)) {
        return React.createElement(React.Fragment);
      }

      // Auto-approve if all sources meet threshold
      if (autoApproveThreshold !== undefined && sources.length > 0) {
        const allAboveThreshold = sources.every(
          (s) => s.similarity >= autoApproveThreshold
        );
        if (allAboveThreshold) {
          // Mark as responded
          respondedRef.current.add(callId);

          // Auto-approve all sources
          const approvedIds = sources.map((s) => s.id);
          onApprove?.(sources);
          respond?.({ approved: approvedIds });
          return React.createElement(
            "div",
            { className: "text-sm text-muted-foreground p-2" },
            `Auto-approved ${sources.length} sources (all above ${(autoApproveThreshold * 100).toFixed(0)}% threshold)`
          );
        }
      }

      // Initialize selection state if not set
      const selectedIds = cardState.selectedIds.size > 0
        ? cardState.selectedIds
        : new Set(sources.map((s) => s.id));

      // Create handlers
      const handleToggle = (sourceId: string) => {
        setCardState((prev) => {
          const next = new Set(prev.selectedIds);
          if (next.has(sourceId)) {
            next.delete(sourceId);
          } else {
            next.add(sourceId);
          }
          return { selectedIds: next };
        });
      };

      const handleSelectAll = () => {
        setCardState({ selectedIds: new Set(sources.map((s) => s.id)) });
      };

      const handleSelectNone = () => {
        setCardState({ selectedIds: new Set() });
      };

      const handleApprove = () => {
        respondedRef.current.add(callId);
        const approvedIds = Array.from(selectedIds);
        const approvedSources = sources.filter((s) => approvedIds.includes(s.id));
        onApprove?.(approvedSources);
        respond?.({ approved: approvedIds });
      };

      const handleReject = () => {
        respondedRef.current.add(callId);
        onReject?.();
        respond?.({ approved: [] });
      };

      const handleSkip = () => {
        respondedRef.current.add(callId);
        const allIds = sources.map((s) => s.id);
        onSkip?.(sources);
        respond?.({ approved: allIds });
      };

      // Helper to get similarity color class
      const getSimilarityColor = (similarity: number): string => {
        if (similarity >= 0.8) return "text-green-600 bg-green-50";
        if (similarity >= 0.6) return "text-yellow-600 bg-yellow-50";
        return "text-red-600 bg-red-50";
      };

      // Build the source list items
      const sourceItems = sources.map((source) => {
        const isSelected = selectedIds.has(source.id);

        return React.createElement(
          "div",
          {
            key: source.id,
            className: `flex items-start gap-3 p-3 rounded-lg border transition-colors cursor-pointer ${
              isSelected
                ? "bg-blue-50 border-blue-200"
                : "bg-gray-50 border-transparent hover:border-gray-200"
            }`,
            onClick: () => allowPartialApproval && handleToggle(source.id),
          },
          // Checkbox
          allowPartialApproval &&
            React.createElement("input", {
              type: "checkbox",
              checked: isSelected,
              onChange: () => handleToggle(source.id),
              className: "mt-1 h-4 w-4 rounded border-gray-300",
            }),
          // Source content
          React.createElement(
            "div",
            { className: "flex-1 min-w-0" },
            // Title row
            React.createElement(
              "div",
              { className: "flex items-center gap-2 mb-1" },
              React.createElement(
                "span",
                { className: "font-medium text-sm truncate" },
                source.title
              ),
              showSimilarityScores &&
                React.createElement(
                  "span",
                  {
                    className: `text-xs px-1.5 py-0.5 rounded ${getSimilarityColor(source.similarity)}`,
                  },
                  `${(source.similarity * 100).toFixed(0)}%`
                )
            ),
            // Preview
            React.createElement(
              "p",
              { className: "text-xs text-gray-500 line-clamp-2" },
              source.preview
            )
          )
        );
      });

      // Build the full card
      return React.createElement(
        "div",
        { className: "w-full max-w-2xl mx-auto border rounded-lg shadow-lg bg-white" },
        // Header
        React.createElement(
          "div",
          { className: "p-4 border-b" },
          React.createElement(
            "h3",
            { className: "text-lg font-semibold flex items-center gap-2" },
            "📄 ",
            dialogTitle
          ),
          query &&
            React.createElement(
              "p",
              { className: "text-sm text-gray-500 mt-1" },
              `Query: "${query.length > 100 ? `${query.slice(0, 100)}...` : query}"`
            )
        ),
        // Content
        React.createElement(
          "div",
          { className: "p-4" },
          // Selection controls
          React.createElement(
            "div",
            { className: "flex items-center justify-between mb-3" },
            React.createElement(
              "span",
              { className: "text-sm text-gray-500" },
              `${selectedIds.size} of ${sources.length} selected`
            ),
            allowPartialApproval &&
              React.createElement(
                "div",
                { className: "flex gap-2" },
                React.createElement(
                  "button",
                  {
                    type: "button",
                    onClick: handleSelectAll,
                    className: "text-sm text-blue-600 hover:underline",
                  },
                  "Select All"
                ),
                React.createElement(
                  "button",
                  {
                    type: "button",
                    onClick: handleSelectNone,
                    className: "text-sm text-blue-600 hover:underline",
                  },
                  "Select None"
                )
              )
          ),
          // Source list
          React.createElement(
            "div",
            { className: "max-h-64 overflow-y-auto space-y-2 rounded border p-2" },
            ...sourceItems
          )
        ),
        // Footer
        React.createElement(
          "div",
          { className: "p-4 border-t flex justify-between gap-2" },
          React.createElement(
            "button",
            {
              type: "button",
              onClick: handleReject,
              className: "px-4 py-2 text-sm border rounded-md hover:bg-gray-50",
            },
            "✕ Reject All"
          ),
          React.createElement(
            "div",
            { className: "flex gap-2" },
            React.createElement(
              "button",
              {
                type: "button",
                onClick: handleSkip,
                className: "px-4 py-2 text-sm border rounded-md hover:bg-gray-50",
              },
              "Skip (Approve All)"
            ),
            React.createElement(
              "button",
              {
                type: "button",
                onClick: handleApprove,
                disabled: selectedIds.size === 0,
                className: `px-4 py-2 text-sm rounded-md text-white ${
                  selectedIds.size === 0
                    ? "bg-gray-300 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700"
                }`,
              },
              `✓ Approve (${selectedIds.size})`
            )
          )
        )
      );
    },
  });
}

// ============================================
// COMPLETION STATUS COMPONENT
// ============================================

/**
 * SourceValidationComplete - Rendered when validation is complete.
 * Can be used independently to show validation results.
 */
export function SourceValidationComplete({
  approvedCount,
  totalCount,
  className,
}: {
  approvedCount: number;
  totalCount: number;
  className?: string;
}): React.ReactElement {
  const allApproved = approvedCount === totalCount;
  const noneApproved = approvedCount === 0;

  let statusClass = "text-yellow-600 bg-yellow-50";
  let icon = "✓";
  let message = `${approvedCount} of ${totalCount} sources approved`;

  if (allApproved) {
    statusClass = "text-green-600 bg-green-50";
    icon = "✓";
  } else if (noneApproved) {
    statusClass = "text-red-600 bg-red-50";
    icon = "✕";
    message = "All sources rejected";
  }

  return React.createElement(
    "div",
    {
      className: `flex items-center gap-2 text-sm p-2 rounded-md ${statusClass} ${className || ""}`,
    },
    React.createElement("span", null, icon),
    React.createElement("span", null, message)
  );
}

export default useSourceValidationV2;
