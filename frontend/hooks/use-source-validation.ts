"use client";

import React, { useState, useCallback } from "react";
import { useHumanInTheLoop } from "@copilotkit/react-core";
import { SourceValidationDialog } from "@/components/copilot/SourceValidationDialog";
import { validateSourcesToolParams } from "@/lib/schemas/tools";
import type { Source, ValidationDecision, SourceValidationState } from "@/types/copilot";

/**
 * Options for the useSourceValidation hook.
 */
export interface UseSourceValidationOptions {
  /** Callback when validation completes */
  onValidationComplete?: (approvedIds: string[]) => void;
  /** Callback when validation is cancelled */
  onValidationCancelled?: () => void;
  /** Auto-approve sources at or above this confidence threshold */
  autoApproveThreshold?: number;
  /** Auto-reject sources below this confidence threshold */
  autoRejectThreshold?: number;
}

/**
 * Return type for the useSourceValidation hook.
 *
 * Story 21-A2: Migrated to useHumanInTheLoop pattern.
 * The dialog is now rendered inside the hook via useHumanInTheLoop's render function.
 * Deprecated functions are kept for backward compatibility with existing consumers.
 */
export interface UseSourceValidationReturn {
  /** Current validation state */
  state: SourceValidationState;
  /**
   * @deprecated No longer needed - hook manages lifecycle via useHumanInTheLoop.
   * Kept for backward compatibility.
   */
  startValidation: (sources: Source[]) => void;
  /**
   * @deprecated Use respond callback from useHumanInTheLoop instead.
   * Kept for backward compatibility.
   */
  submitValidation: (approvedIds: string[]) => void;
  /**
   * @deprecated Use respond callback from useHumanInTheLoop instead.
   * Kept for backward compatibility.
   */
  cancelValidation: () => void;
  /** Reset validation state */
  resetValidation: () => void;
  /**
   * @deprecated Dialog is rendered inside hook via useHumanInTheLoop.
   * Kept for backward compatibility - always returns false.
   */
  isDialogOpen: boolean;
}

/**
 * Initial state for source validation.
 */
const initialState: SourceValidationState = {
  isValidating: false,
  pendingSources: [],
  decisions: new Map(),
  approvedIds: [],
  rejectedIds: [],
  isSubmitting: false,
  error: null,
};

/**
 * Apply auto-approve/reject thresholds to sources.
 * Returns a Map of source IDs to their initial validation decisions.
 */
function applyThresholds(
  sources: Source[],
  autoApproveThreshold?: number,
  autoRejectThreshold?: number
): Map<string, ValidationDecision> {
  const decisions = new Map<string, ValidationDecision>();

  for (const source of sources) {
    if (autoApproveThreshold && source.similarity >= autoApproveThreshold) {
      decisions.set(source.id, "approved");
    } else if (autoRejectThreshold && source.similarity < autoRejectThreshold) {
      decisions.set(source.id, "rejected");
    } else {
      decisions.set(source.id, "pending");
    }
  }

  return decisions;
}

/**
 * useSourceValidation hook manages Human-in-the-Loop source validation
 * state and integrates with CopilotKit's useHumanInTheLoop hook.
 *
 * Story 6-4: Human-in-the-Loop Source Validation
 * Story 21-A2: Migrate to useHumanInTheLoop Pattern
 *
 * Migration Notes (21-A2):
 * - Replaced useCopilotAction with useHumanInTheLoop
 * - Removed setTimeout workaround - respond callback is lifecycle-safe
 * - Dialog is now rendered inside the hook's render function
 * - Removed validationTriggeredRef and respondRef - no longer needed
 *
 * @example
 * ```tsx
 * function ChatWithHITL() {
 *   const { state, resetValidation } = useSourceValidation({
 *     onValidationComplete: (ids) => console.log("Approved:", ids),
 *   });
 *
 *   // Dialog is auto-rendered by useHumanInTheLoop when executing
 *   return <ChatSidebar />;
 * }
 * ```
 */
export function useSourceValidation(
  options: UseSourceValidationOptions = {}
): UseSourceValidationReturn {
  const {
    onValidationComplete,
    onValidationCancelled,
    autoApproveThreshold,
    autoRejectThreshold,
  } = options;

  // Validation state
  const [state, setState] = useState<SourceValidationState>(initialState);

  // Reset validation state
  const resetValidation = useCallback(() => {
    setState(initialState);
  }, []);

  // Deprecated: Start validation (kept for backward compatibility)
  // In the new pattern, useHumanInTheLoop manages the lifecycle
  const startValidation = useCallback(
    (sources: Source[]) => {
      const decisions = applyThresholds(sources, autoApproveThreshold, autoRejectThreshold);
      setState({
        isValidating: true,
        pendingSources: sources,
        decisions,
        approvedIds: [],
        rejectedIds: [],
        isSubmitting: false,
        error: null,
      });
    },
    [autoApproveThreshold, autoRejectThreshold]
  );

  // Deprecated: Submit validation (kept for backward compatibility)
  const submitValidation = useCallback(
    (approvedIds: string[]) => {
      setState((prev) => {
        const rejectedIds = prev.pendingSources
          .map((s) => s.id)
          .filter((id) => !approvedIds.includes(id));

        onValidationComplete?.(approvedIds);

        return {
          ...prev,
          isValidating: false,
          isSubmitting: false,
          approvedIds,
          rejectedIds,
        };
      });
    },
    [onValidationComplete]
  );

  // Deprecated: Cancel validation (kept for backward compatibility)
  const cancelValidation = useCallback(() => {
    setState(initialState);
    onValidationCancelled?.();
  }, [onValidationCancelled]);

  // Register CopilotKit useHumanInTheLoop hook
  // Story 21-A2: Replaces deprecated useCopilotAction with render prop
  useHumanInTheLoop({
    name: "validate_sources",
    description:
      "Request human approval for retrieved sources before answer generation",
    parameters: validateSourcesToolParams,
    render: ({ status, args, respond, result }) => {
      // Guard: respond is only available during "executing" status
      // CopilotKit 1.x uses lowercase status values
      if (status === "executing" && respond) {
        // Safely extract sources from args - may be undefined or wrong type
        const rawSources = args?.sources;
        const sources: Source[] = Array.isArray(rawSources)
          ? (rawSources as unknown as Source[])
          : [];

        // Apply auto-thresholds if no sources require manual review
        const decisions = applyThresholds(sources, autoApproveThreshold, autoRejectThreshold);
        const pendingCount = Array.from(decisions.values()).filter(d => d === "pending").length;

        // If all sources are auto-approved/rejected, auto-respond
        if (pendingCount === 0 && sources.length > 0) {
          const autoApprovedIds = sources
            .filter(s => decisions.get(s.id) === "approved")
            .map(s => s.id);

          // Update state and respond
          setState({
            isValidating: false,
            pendingSources: sources,
            decisions,
            approvedIds: autoApprovedIds,
            rejectedIds: sources.filter(s => !autoApprovedIds.includes(s.id)).map(s => s.id),
            isSubmitting: false,
            error: null,
          });
          onValidationComplete?.(autoApprovedIds);
          respond({ approved: autoApprovedIds });
          // Return empty fragment (CopilotKit requires a ReactElement)
          return React.createElement(React.Fragment);
        }

        // Render the validation dialog
        return React.createElement(SourceValidationDialog, {
          open: true,
          sources: sources,
          onSubmit: (approvedIds: string[]) => {
            // Update state
            const rejectedIds = sources
              .map((s) => s.id)
              .filter((id) => !approvedIds.includes(id));

            setState({
              isValidating: false,
              pendingSources: sources,
              decisions: new Map(sources.map(s => [s.id, approvedIds.includes(s.id) ? "approved" : "rejected"] as const)),
              approvedIds,
              rejectedIds,
              isSubmitting: false,
              error: null,
            });

            // Call completion callback
            onValidationComplete?.(approvedIds);

            // Respond to the agent
            respond({ approved: approvedIds });
          },
          onCancel: () => {
            // Update state
            setState(initialState);

            // Call cancellation callback
            onValidationCancelled?.();

            // Respond with empty approval (cancellation)
            respond({ approved: [] });
          },
          isSubmitting: false,
        });
      }

      // Show completion state
      // CopilotKit 1.x uses lowercase status values
      if (status === "complete" && result) {
        const approvedCount = (result as { approved?: string[] }).approved?.length ?? 0;
        return React.createElement(
          "div",
          { className: "text-sm text-muted-foreground p-2" },
          approvedCount > 0
            ? `Approved ${approvedCount} source(s)`
            : "Validation cancelled"
        );
      }

      // Return empty fragment for other states (idle, inProgress, etc.)
      // CopilotKit requires a ReactElement, not null
      return React.createElement(React.Fragment);
    },
  });

  return {
    state,
    startValidation,
    submitValidation,
    cancelValidation,
    resetValidation,
    // Dialog is rendered inside useHumanInTheLoop - always return false
    isDialogOpen: false,
  };
}

export default useSourceValidation;
