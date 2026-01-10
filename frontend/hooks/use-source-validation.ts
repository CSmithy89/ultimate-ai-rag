"use client";

import React, { useState, useCallback, useRef } from "react";
import { useCopilotAction } from "@copilotkit/react-core";
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
 */
export interface UseSourceValidationReturn {
  /** Current validation state */
  state: SourceValidationState;
  /** Open the validation dialog with sources */
  startValidation: (sources: Source[]) => void;
  /** Submit validation decisions */
  submitValidation: (approvedIds: string[]) => void;
  /** Cancel validation */
  cancelValidation: () => void;
  /** Reset validation state */
  resetValidation: () => void;
  /** Whether the validation dialog should be open */
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
 * useSourceValidation hook manages Human-in-the-Loop source validation
 * state and integrates with CopilotKit actions.
 *
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * @example
 * ```tsx
 * function ChatWithHITL() {
 *   const {
 *     state,
 *     isDialogOpen,
 *     submitValidation,
 *     cancelValidation,
 *   } = useSourceValidation({
 *     onValidationComplete: (ids) => console.log("Approved:", ids),
 *   });
 *
 *   return (
 *     <>
 *       <ChatSidebar />
 *       <SourceValidationDialog
 *         open={isDialogOpen}
 *         sources={state.pendingSources}
 *         onSubmit={submitValidation}
 *         onCancel={cancelValidation}
 *       />
 *     </>
 *   );
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

  // Reference to store the respond function from renderAndWait
  // Using any type to avoid CopilotKit type issues
  // eslint-disable-next-line
  const respondRef = useRef<((response: any) => void) | null>(null);

  // Track if we've already triggered validation for current action
  const validationTriggeredRef = useRef<boolean>(false);

  // Start validation with a set of sources
  const startValidation = useCallback(
    (sources: Source[]) => {
      // Apply auto-approve/reject thresholds if configured
      const decisions = new Map<string, ValidationDecision>();

      for (const source of sources) {
        if (autoApproveThreshold && source.similarity >= autoApproveThreshold) {
          decisions.set(source.id, "approved");
        } else if (
          autoRejectThreshold &&
          source.similarity < autoRejectThreshold
        ) {
          decisions.set(source.id, "rejected");
        } else {
          decisions.set(source.id, "pending");
        }
      }

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

  // Submit validation decisions
  const submitValidation = useCallback(
    (approvedIds: string[]) => {
      setState((prev) => ({
        ...prev,
        isSubmitting: true,
      }));

      // Compute rejected IDs from current state
      setState((prev) => {
        const rejectedIds = prev.pendingSources
          .map((s) => s.id)
          .filter((id) => !approvedIds.includes(id));

        // Call the respond function if available (renderAndWait pattern)
        if (respondRef.current) {
          respondRef.current({ approved: approvedIds });
          respondRef.current = null;
        }

        // Reset validation triggered flag
        validationTriggeredRef.current = false;

        // Call completion callback
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

  // Cancel validation
  const cancelValidation = useCallback(() => {
    // If renderAndWait is active, respond with empty approved list
    if (respondRef.current) {
      respondRef.current({ approved: [] });
      respondRef.current = null;
    }

    // Reset validation triggered flag
    validationTriggeredRef.current = false;

    setState(initialState);

    onValidationCancelled?.();
  }, [onValidationCancelled]);

  // Reset validation state
  const resetValidation = useCallback(() => {
    validationTriggeredRef.current = false;
    setState(initialState);
  }, []);

  // Register CopilotKit action for HITL
  // Using render instead of renderAndWait to avoid type issues
  // The validation UI is managed externally by the parent component
  useCopilotAction({
    name: "validate_sources",
    description:
      "Request human approval for retrieved sources before answer generation",
    parameters: [
      {
        name: "sources",
        type: "object[]",
        description: "Array of sources requiring validation",
        required: true,
      },
      {
        name: "query",
        type: "string",
        description: "The original user query for context",
        required: false,
      },
    ],
    render: ({ status, args }) => {
      // Extract sources from args when action is executing
      if (status === "executing" && args.sources && !validationTriggeredRef.current) {
        const sources = args.sources as Source[];
        if (sources.length > 0) {
          // Mark as triggered to prevent multiple calls
          validationTriggeredRef.current = true;
          // Issue 4 Fix: Known CopilotKit pattern limitation
          // We use setTimeout(fn, 0) to defer the state update to the next event loop tick.
          // This is necessary because CopilotKit's render callback is called during React's
          // render phase, and calling setState directly would cause a "Cannot update a component
          // while rendering a different component" warning. This is a documented pattern for
          // CopilotKit actions that need to trigger React state updates from render callbacks.
          // See: https://react.dev/reference/react/useState#im-getting-an-error-too-many-re-renders
          setTimeout(() => startValidation(sources), 0);
        }
      }
      // Return empty fragment - actual UI is rendered externally
      return React.createElement(React.Fragment);
    },
  });

  return {
    state,
    startValidation,
    submitValidation,
    cancelValidation,
    resetValidation,
    isDialogOpen: state.isValidating,
  };
}

export default useSourceValidation;
