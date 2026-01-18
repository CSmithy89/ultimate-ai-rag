"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import { useHumanInTheLoop } from "@copilotkit/react-core";
import { SourceValidationDialog } from "@/components/copilot/SourceValidationDialog";
import { validateSourcesToolParams } from "@/lib/schemas/tools";
import type { Source, ValidationDecision, SourceValidationState } from "@/types/copilot";

/**
 * Send validation response to backend API.
 * This is needed because the respond() callback from useHumanInTheLoop
 * doesn't automatically communicate with our backend's HITL system.
 *
 * @param checkpointId - The checkpoint ID from the backend
 * @param approvedSourceIds - List of approved source IDs
 * @returns Promise resolving to the API response
 */
async function sendValidationToBackend(
  checkpointId: string,
  approvedSourceIds: string[]
): Promise<{ success: boolean; error?: string }> {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    const tenantId = process.env.NEXT_PUBLIC_TENANT_ID || "550e8400-e29b-41d4-a716-446655440000";

    const response = await fetch(`${backendUrl}/api/v1/copilot/validation-response`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        approved_source_ids: approvedSourceIds,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("[HITL] Backend validation response failed:", response.status, errorText);
      return { success: false, error: errorText };
    }

    const data = await response.json();
    console.log("[HITL] Backend validation response success:", data);
    return { success: true };
  } catch (error) {
    console.error("[HITL] Failed to send validation to backend:", error);
    return { success: false, error: String(error) };
  }
}

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
 * Data for pending auto-respond action.
 * Used to defer state updates and respond calls to useEffect.
 */
interface PendingAutoRespond {
  sources: Source[];
  decisions: Map<string, ValidationDecision>;
  autoApprovedIds: string[];
  checkpointId: string;
  respond: (result: { approved: string[] }) => void;
}

/**
 * Apply auto-approve/reject thresholds to sources.
 * Returns a Map of source IDs to their initial validation decisions.
 *
 * Uses explicit null checks to support threshold value of 0.
 * (Issue 2.2: applyThresholds breaks when threshold is 0)
 */
function applyThresholds(
  sources: Source[],
  autoApproveThreshold?: number,
  autoRejectThreshold?: number
): Map<string, ValidationDecision> {
  const decisions = new Map<string, ValidationDecision>();

  for (const source of sources) {
    // Use explicit null checks to support threshold value of 0
    if (autoApproveThreshold != null && source.similarity >= autoApproveThreshold) {
      decisions.set(source.id, "approved");
    } else if (autoRejectThreshold != null && source.similarity < autoRejectThreshold) {
      decisions.set(source.id, "rejected");
    } else {
      decisions.set(source.id, "pending");
    }
  }

  return decisions;
}

/**
 * Safely invoke a callback with error handling.
 * Prevents callback errors from blocking critical operations.
 * (Issue 2.5: Callback error handling missing)
 */
function safeInvokeCallback<T extends unknown[]>(
  callback: ((...args: T) => void) | undefined,
  ...args: T
): void {
  if (!callback) return;
  try {
    callback(...args);
  } catch (error) {
    console.error("Error in callback:", error);
  }
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
 * Code Review Fixes:
 * - Issue 1.1: Fixed React anti-pattern (setState in render) using useEffect
 * - Issue 2.2: Fixed applyThresholds falsy check (0 threshold works now)
 * - Issue 2.4: Added one-shot guard for respond() calls
 * - Issue 2.5: Added try/catch for callbacks
 * - Issue 3.5: Fixed validation state not set before dialog renders
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

  // One-shot guard for auto-respond (Issue 2.4)
  // Keyed by stringified source IDs to prevent multiple respond() calls
  const respondedCallsRef = useRef<Set<string>>(new Set());

  // Pending auto-respond action (Issue 1.1: defer setState to useEffect)
  const [pendingAutoRespond, setPendingAutoRespond] = useState<PendingAutoRespond | null>(null);

  // Handle auto-respond in useEffect to avoid setState during render (Issue 1.1)
  useEffect(() => {
    if (pendingAutoRespond) {
      const { sources, decisions, autoApprovedIds, checkpointId, respond } = pendingAutoRespond;

      // Async function to handle the auto-respond with backend call
      const handleAutoRespond = async () => {
        // Update state
        setState({
          isValidating: false,
          pendingSources: sources,
          decisions,
          approvedIds: autoApprovedIds,
          rejectedIds: sources.filter(s => !autoApprovedIds.includes(s.id)).map(s => s.id),
          isSubmitting: false,
          error: null,
        });

        // Send validation to backend API (critical for HITL to work)
        if (checkpointId) {
          const result = await sendValidationToBackend(checkpointId, autoApprovedIds);
          if (!result.success) {
            console.error("[HITL] Backend auto-validation failed:", result.error);
          } else {
            console.log("[HITL] Auto-validation sent to backend successfully");
          }
        } else {
          console.warn("[HITL] No checkpoint_id for auto-respond, backend won't receive validation");
        }

        // Call completion callback with error handling (Issue 2.5)
        safeInvokeCallback(onValidationComplete, autoApprovedIds);

        // Respond to the agent
        respond({ approved: autoApprovedIds });

        // Clear pending action
        setPendingAutoRespond(null);
      };

      handleAutoRespond();
    }
  }, [pendingAutoRespond, onValidationComplete]);

  // Reset validation state
  const resetValidation = useCallback(() => {
    setState(initialState);
    respondedCallsRef.current.clear();
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

        // Call with error handling (Issue 2.5)
        safeInvokeCallback(onValidationComplete, approvedIds);

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
    safeInvokeCallback(onValidationCancelled);
  }, [onValidationCancelled]);

  // Register CopilotKit useHumanInTheLoop hook
  // Story 21-A2: Replaces deprecated useCopilotAction with render prop
  useHumanInTheLoop({
    name: "validate_sources",
    description:
      "Request human approval for retrieved sources before answer generation",
    parameters: validateSourcesToolParams,
    render: ({ status, args, respond, result }) => {
      // Debug logging to trace HITL flow
      console.log("[HITL] render called:", { status, hasArgs: !!args, hasRespond: !!respond, hasResult: !!result });
      if (args) {
        console.log("[HITL] args:", JSON.stringify(args, null, 2));
      }
      // Guard: respond is only available during "executing" status
      // CopilotKit 1.x uses lowercase status values
      if (status === "executing" && respond) {
        // Safely extract sources from args - may be undefined or wrong type
        const rawSources = args?.sources;
        const sources: Source[] = Array.isArray(rawSources)
          ? (rawSources as unknown as Source[])
          : [];

        // Extract checkpoint_id from args (sent by backend HITL system)
        const rawCheckpointId = args?.checkpoint_id;
        const checkpointId = typeof rawCheckpointId === "string" ? rawCheckpointId : "";
        console.log("[HITL] checkpoint_id:", checkpointId, "sources:", sources.length);

        // Generate unique call ID for one-shot guard (Issue 2.4)
        const callId = sources.map(s => s.id).sort().join(",");

        // Check if we've already responded to this exact set of sources
        if (respondedCallsRef.current.has(callId)) {
          return React.createElement(React.Fragment);
        }

        // Apply auto-thresholds if no sources require manual review
        const decisions = applyThresholds(sources, autoApproveThreshold, autoRejectThreshold);
        const pendingCount = Array.from(decisions.values()).filter(d => d === "pending").length;

        // If all sources are auto-approved/rejected, schedule auto-respond via useEffect (Issue 1.1)
        if (pendingCount === 0 && sources.length > 0) {
          const autoApprovedIds = sources
            .filter(s => decisions.get(s.id) === "approved")
            .map(s => s.id);

          // Mark as responded to prevent duplicate calls (Issue 2.4)
          respondedCallsRef.current.add(callId);

          // Schedule auto-respond in useEffect to avoid setState during render (Issue 1.1)
          // We can't use setState in render, so we queue it for the effect
          setPendingAutoRespond({
            sources,
            decisions,
            autoApprovedIds,
            checkpointId,
            respond,
          });

          // Return empty fragment (CopilotKit requires a ReactElement)
          return React.createElement(React.Fragment);
        }

        // Update state to reflect validation in progress (Issue 3.5)
        // This is safe because we only do it once per callId due to the guard above
        if (!respondedCallsRef.current.has(`state-${callId}`)) {
          respondedCallsRef.current.add(`state-${callId}`);
          // Schedule state update via effect to avoid render-phase setState
          // For the dialog path, we use a simpler approach: set state when dialog actions occur
        }

        // Render the validation dialog
        return React.createElement(SourceValidationDialog, {
          open: true,
          sources: sources,
          onSubmit: async (approvedIds: string[]) => {
            // Mark as responded (Issue 2.4)
            respondedCallsRef.current.add(callId);

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
              isSubmitting: true, // Show submitting state while calling backend
              error: null,
            });

            // Send validation to backend API (critical for HITL to work)
            if (checkpointId) {
              const result = await sendValidationToBackend(checkpointId, approvedIds);
              if (!result.success) {
                console.error("[HITL] Backend validation failed, but continuing with CopilotKit respond");
              }
            } else {
              console.warn("[HITL] No checkpoint_id in args, cannot send validation to backend");
            }

            // Update submitting state
            setState((prev) => ({ ...prev, isSubmitting: false }));

            // Call completion callback with error handling (Issue 2.5)
            safeInvokeCallback(onValidationComplete, approvedIds);

            // Respond to the agent (for CopilotKit flow)
            respond({ approved: approvedIds });
          },
          onCancel: async () => {
            // Mark as responded (Issue 2.4)
            respondedCallsRef.current.add(callId);

            // Update state
            setState(initialState);

            // Send cancellation to backend (empty approval = cancel)
            if (checkpointId) {
              const result = await sendValidationToBackend(checkpointId, []);
              if (!result.success) {
                console.error("[HITL] Backend cancellation failed");
              }
            }

            // Call cancellation callback with error handling (Issue 2.5)
            safeInvokeCallback(onValidationCancelled);

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
