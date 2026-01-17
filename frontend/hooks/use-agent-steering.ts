"use client";

/**
 * useAgentSteering - AG-UI Enhancement (Phase 6.2)
 *
 * Provides ability to inject steering guidance into running agents.
 * This allows users to redirect agent execution mid-flow.
 *
 * IMPORTANT: Steering can only be applied to runs with "running" status.
 * The hook provides `canSteer` to check if a run can be steered.
 *
 * @example
 * ```tsx
 * function SteeringInput({ runId }: { runId: string }) {
 *   const { steerAgent, isSteering, canSteer, checkRunStatus } = useAgentSteering();
 *
 *   // Check if run can be steered before showing UI
 *   const isSteerableRun = await checkRunStatus(runId);
 *
 *   const handleSubmit = async (instruction: string) => {
 *     const success = await steerAgent(runId, instruction);
 *     if (!success) {
 *       // Handle failure (run may have completed)
 *     }
 *   };
 *
 *   if (!isSteerableRun) return null;
 *
 *   return (
 *     <form onSubmit={e => handleSubmit(e.target.instruction.value)}>
 *       <input name="instruction" placeholder="Redirect the agent..." />
 *       <button disabled={isSteering}>Steer</button>
 *     </form>
 *   );
 * }
 * ```
 */

import { useState, useCallback, useMemo } from "react";
import type { SteeringContext, SteeringResult } from "@/types/ag-ui";

// Re-export types for convenience
export type { SteeringContext, SteeringResult } from "@/types/ag-ui";

/**
 * Return type for useAgentSteering hook.
 */
export interface UseAgentSteeringResult {
  /** Send a steering instruction to an agent */
  steerAgent: (
    runId: string,
    instruction: string,
    context?: SteeringContext
  ) => Promise<boolean>;
  /** Whether a steering request is in progress */
  isSteering: boolean;
  /** Last steering result */
  lastResult: SteeringResult | null;
  /** Error message if steering failed */
  error: string | null;
  /** Clear the error */
  clearError: () => void;
  /** Check if a run can be steered (is in running state) */
  checkRunStatus: (runId: string) => Promise<boolean>;
}

/**
 * Configuration for useAgentSteering.
 */
export interface UseAgentSteeringConfig {
  /** Base URL for the API (defaults to /api/v1) */
  baseUrl?: string;
  /** Tenant ID for multi-tenant operations */
  tenantId?: string;
  /** Callback when steering is successful */
  onSuccess?: (result: SteeringResult) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/**
 * Hook to steer agent execution by injecting guidance.
 */
export function useAgentSteering(
  config: UseAgentSteeringConfig = {}
): UseAgentSteeringResult {
  const { baseUrl = "/api/v1", tenantId, onSuccess, onError } = config;

  const [isSteering, setIsSteering] = useState(false);
  const [lastResult, setLastResult] = useState<SteeringResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Check if a run is in a state that can be steered
  const checkRunStatus = useCallback(
    async (runId: string): Promise<boolean> => {
      try {
        const response = await fetch(`${baseUrl}/copilot/run/${runId}`, {
          headers: {
            ...(tenantId ? { "X-Tenant-ID": tenantId } : {}),
          },
        });

        if (!response.ok) {
          return false; // Can't steer if we can't get status
        }

        const runState = await response.json();
        // Only "running" status can be steered
        return runState.status === "running";
      } catch {
        return false;
      }
    },
    [baseUrl, tenantId]
  );

  // Send steering instruction
  const steerAgent = useCallback(
    async (
      runId: string,
      instruction: string,
      context?: SteeringContext
    ): Promise<boolean> => {
      if (!runId || !instruction) {
        setError("Run ID and instruction are required");
        return false;
      }

      // Pre-check: verify run is still active before attempting to steer
      const canSteer = await checkRunStatus(runId);
      if (!canSteer) {
        setError("Cannot steer: run is not in 'running' status");
        return false;
      }

      setIsSteering(true);
      setError(null);

      try {
        const response = await fetch(`${baseUrl}/copilot/steer`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(tenantId ? { "X-Tenant-ID": tenantId } : {}),
          },
          body: JSON.stringify({
            run_id: runId,
            instruction,
            context,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.detail || `Failed to steer agent: ${response.status}`
          );
        }

        const result: SteeringResult = await response.json();
        setLastResult(result);
        onSuccess?.(result);
        return true;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error.message);
        onError?.(error);
        return false;
      } finally {
        setIsSteering(false);
      }
    },
    [baseUrl, tenantId, onSuccess, onError, checkRunStatus]
  );

  return useMemo(
    () => ({
      steerAgent,
      isSteering,
      lastResult,
      error,
      clearError,
      checkRunStatus,
    }),
    [steerAgent, isSteering, lastResult, error, clearError, checkRunStatus]
  );
}

export default useAgentSteering;
