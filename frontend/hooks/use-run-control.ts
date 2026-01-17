"use client";

/**
 * useRunControl - AG-UI Enhancement (Phase 6)
 *
 * Provides control over agent runs including cancel and resume capabilities.
 * Integrates with the backend RunManager to manage run lifecycle.
 *
 * @example
 * ```tsx
 * function AgentControls() {
 *   const { currentRunId, cancelRun, isRunning, canCancel } = useRunControl();
 *
 *   return (
 *     <button onClick={cancelRun} disabled={!canCancel}>
 *       {isRunning ? "Cancel" : "No active run"}
 *     </button>
 *   );
 * }
 * ```
 */

import { useState, useCallback, useMemo } from "react";
import { useCopilotChat } from "@copilotkit/react-core";
import type { RunState } from "@/types/ag-ui";

// Re-export types for backward compatibility
export type { RunStatus, RunState } from "@/types/ag-ui";

/**
 * Result of getRunState operation - distinguishes between "not found" and "error".
 */
export interface GetRunStateResult {
  /** The run state if found */
  run: RunState | null;
  /** Whether the run was not found (404) */
  notFound: boolean;
  /** Whether there was a network/server error */
  error: Error | null;
}

/**
 * Return type for useRunControl hook.
 */
export interface UseRunControlResult {
  /** Current run ID if any */
  currentRunId: string | null;
  /** Set the current run ID */
  setCurrentRunId: (runId: string | null) => void;
  /** Whether an agent is currently running */
  isRunning: boolean;
  /** Whether the current run can be cancelled */
  canCancel: boolean;
  /** Cancel the current run */
  cancelRun: () => Promise<boolean>;
  /** Resume a cancelled/paused run */
  resumeRun: (runId: string) => Promise<boolean>;
  /** Get state of a run with explicit not-found/error distinction */
  getRunState: (runId: string) => Promise<GetRunStateResult>;
  /** List active runs for the current tenant */
  listActiveRuns: () => Promise<RunState[]>;
  /** Error message if any operation failed */
  error: string | null;
  /** Clear the error */
  clearError: () => void;
}

/**
 * Configuration for useRunControl.
 */
export interface UseRunControlConfig {
  /** Base URL for the API (defaults to /api/v1) */
  baseUrl?: string;
  /** Tenant ID for multi-tenant operations */
  tenantId?: string;
  /** Callback when run is cancelled */
  onCancel?: (runId: string) => void;
  /** Callback when run is resumed */
  onResume?: (runId: string) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/**
 * Hook to control agent run lifecycle (cancel/resume).
 */
export function useRunControl(config: UseRunControlConfig = {}): UseRunControlResult {
  const {
    baseUrl = "/api/v1",
    tenantId,
    onCancel,
    onResume,
    onError,
  } = config;

  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { isLoading } = useCopilotChat();

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Cancel the current run
  const cancelRun = useCallback(async (): Promise<boolean> => {
    if (!currentRunId) {
      setError("No active run to cancel");
      return false;
    }

    try {
      const response = await fetch(`${baseUrl}/copilot/cancel/${currentRunId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(tenantId ? { "X-Tenant-ID": tenantId } : {}),
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to cancel run: ${response.status}`);
      }

      const result = await response.json();

      if (result.cancelled) {
        onCancel?.(currentRunId);
        setCurrentRunId(null);
        return true;
      }

      setError(result.message || "Failed to cancel run");
      return false;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      onError?.(error);
      return false;
    }
  }, [currentRunId, baseUrl, tenantId, onCancel, onError]);

  // Resume a cancelled/paused run
  const resumeRun = useCallback(async (runId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${baseUrl}/copilot/resume/${runId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(tenantId ? { "X-Tenant-ID": tenantId } : {}),
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to resume run: ${response.status}`);
      }

      setCurrentRunId(runId);
      onResume?.(runId);
      return true;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      onError?.(error);
      return false;
    }
  }, [baseUrl, tenantId, onResume, onError]);

  // Get state of a run with explicit not-found/error distinction
  const getRunState = useCallback(async (runId: string): Promise<GetRunStateResult> => {
    try {
      const response = await fetch(`${baseUrl}/copilot/run/${runId}`, {
        headers: {
          ...(tenantId ? { "X-Tenant-ID": tenantId } : {}),
        },
      });

      if (!response.ok) {
        if (response.status === 404) {
          // Not found is not an error - it's a valid state
          return { run: null, notFound: true, error: null };
        }
        const errorData = await response.json().catch(() => ({}));
        const error = new Error(errorData.detail || `Failed to get run state: ${response.status}`);
        setError(error.message);
        onError?.(error);
        return { run: null, notFound: false, error };
      }

      const run = await response.json();
      return { run, notFound: false, error: null };
    } catch (err) {
      // Network or parsing error
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      onError?.(error);
      return { run: null, notFound: false, error };
    }
  }, [baseUrl, tenantId, onError]);

  // List active runs
  const listActiveRuns = useCallback(async (): Promise<RunState[]> => {
    try {
      const url = new URL(`${baseUrl}/copilot/runs`, window.location.origin);
      if (tenantId) {
        url.searchParams.set("tenant_id", tenantId);
      }

      const response = await fetch(url.toString(), {
        headers: {
          ...(tenantId ? { "X-Tenant-ID": tenantId } : {}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to list runs: ${response.status}`);
      }

      const result = await response.json();
      return result.runs || [];
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      onError?.(error);
      return [];
    }
  }, [baseUrl, tenantId, onError]);

  // Derived state
  const isRunning = isLoading || currentRunId !== null;
  const canCancel = isRunning && currentRunId !== null;

  return useMemo(
    () => ({
      currentRunId,
      setCurrentRunId,
      isRunning,
      canCancel,
      cancelRun,
      resumeRun,
      getRunState,
      listActiveRuns,
      error,
      clearError,
    }),
    [
      currentRunId,
      isRunning,
      canCancel,
      cancelRun,
      resumeRun,
      getRunState,
      listActiveRuns,
      error,
      clearError,
    ]
  );
}

export default useRunControl;
