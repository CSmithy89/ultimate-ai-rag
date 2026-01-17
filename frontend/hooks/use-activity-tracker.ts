"use client";

/**
 * useActivityTracker - AG-UI Enhancement (Phase 5)
 *
 * Tracks ACTIVITY events from the AG-UI protocol and provides
 * real-time activity state for rendering progress UI.
 *
 * This hook integrates with CopilotKit's event stream to capture
 * ACTIVITY_SNAPSHOT and ACTIVITY_DELTA events, applying RFC 6902
 * JSON Patch operations to maintain current activity state.
 *
 * @example
 * ```tsx
 * function ProgressIndicator() {
 *   const { activity, isActive } = useActivityTracker();
 *
 *   if (!isActive || !activity) return null;
 *
 *   return <ActivityTrackerWidget activity={activity} />;
 * }
 * ```
 */

import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  ActivityState,
  JSONPatchOperation,
  ACTIVITY_STATE_KEYS,
  EMPTY_ACTIVITY,
  ACTIVITY_RESET_DELAY_MS,
} from "@/types/ag-ui";

// Re-export types for convenience
export type { ActivityState, JSONPatchOperation };

/**
 * Apply a single JSON Patch operation to an activity state.
 * Only applies operations to valid ActivityState keys for type safety.
 */
function applyPatchOperation(
  activity: ActivityState,
  op: JSONPatchOperation
): ActivityState {
  const path = op.path.replace(/^\//, ""); // Remove leading slash
  const pathParts = path.split("/");

  if (pathParts.length === 1) {
    // Top-level property - validate key
    const key = pathParts[0];

    // Type-safe validation: only allow known ActivityState keys
    if (!ACTIVITY_STATE_KEYS.has(key as keyof ActivityState)) {
      // Ignore unknown keys for security (prevents prototype pollution)
      return activity;
    }

    const validKey = key as keyof ActivityState;

    switch (op.op) {
      case "replace":
      case "add":
        // Validate value types for each key
        if (validKey === "progress" && typeof op.value !== "number") {
          return activity;
        }
        if (
          (validKey === "totalSteps" || validKey === "currentStep") &&
          typeof op.value !== "number"
        ) {
          return activity;
        }
        if (
          (validKey === "id" ||
            validKey === "type" ||
            validKey === "message") &&
          typeof op.value !== "string"
        ) {
          return activity;
        }
        return { ...activity, [validKey]: op.value };
      case "remove": {
        const copy = { ...activity };
        // Reset to default value instead of deleting
        if (validKey === "progress" || validKey === "totalSteps" || validKey === "currentStep") {
          (copy as Record<string, unknown>)[validKey] = 0;
        } else if (validKey === "metadata") {
          copy.metadata = {};
        } else {
          (copy as Record<string, unknown>)[validKey] = "";
        }
        return copy;
      }
      default:
        return activity;
    }
  } else if (pathParts[0] === "metadata" && pathParts.length === 2) {
    // Metadata property
    const metadataKey = pathParts[1];
    const newMetadata = { ...(activity.metadata || {}) };
    switch (op.op) {
      case "replace":
      case "add":
        newMetadata[metadataKey] = op.value;
        break;
      case "remove":
        delete newMetadata[metadataKey];
        break;
    }
    return { ...activity, metadata: newMetadata };
  }

  return activity;
}

/**
 * Apply multiple JSON Patch operations to activity state.
 */
function applyPatches(
  activity: ActivityState,
  patches: JSONPatchOperation[]
): ActivityState {
  return patches.reduce((acc, patch) => applyPatchOperation(acc, patch), activity);
}

/**
 * Return type for useActivityTracker hook.
 */
export interface UseActivityTrackerResult {
  /** Current activity state */
  activity: ActivityState | null;
  /** Whether an activity is currently in progress */
  isActive: boolean;
  /** Whether the current activity is complete */
  isComplete: boolean;
  /** Progress percentage (0-100) */
  progressPercent: number;
  /** Reset the activity tracker */
  reset: () => void;
  /** Manually set activity state (for testing/debugging) */
  setActivity: (activity: ActivityState | null) => void;
  /** Process an ACTIVITY_SNAPSHOT event */
  processSnapshot: (snapshot: Record<string, unknown>) => void;
  /** Process an ACTIVITY_DELTA event */
  processDelta: (delta: JSONPatchOperation[]) => void;
}

/**
 * Hook to track activity progress from AG-UI ACTIVITY events.
 */
export function useActivityTracker(): UseActivityTrackerResult {
  const [activity, setActivity] = useState<ActivityState | null>(null);
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track the activity ID that triggered the reset timer
  const resetActivityIdRef = useRef<string | null>(null);

  // Reset the activity tracker
  const reset = useCallback(() => {
    setActivity(null);
    resetActivityIdRef.current = null;
  }, []);

  // Process an ACTIVITY_SNAPSHOT event
  const processSnapshot = useCallback((snapshot: Record<string, unknown>) => {
    // Clear any pending reset timer since we have a new activity
    if (resetTimerRef.current) {
      clearTimeout(resetTimerRef.current);
      resetTimerRef.current = null;
    }

    const newActivity: ActivityState = {
      id: String(snapshot.id || ""),
      type: String(snapshot.type || ""),
      message: String(snapshot.message || ""),
      progress: Number(snapshot.progress) || 0,
      totalSteps: Number(snapshot.totalSteps) || 0,
      currentStep: Number(snapshot.currentStep) || 0,
      metadata: (snapshot.metadata as Record<string, unknown>) || {},
    };
    setActivity(newActivity);
    resetActivityIdRef.current = newActivity.id;
  }, []);

  // Process an ACTIVITY_DELTA event
  const processDelta = useCallback((delta: JSONPatchOperation[]) => {
    setActivity((prev) => {
      if (!prev) {
        // If no previous activity, create a new one with the delta
        const newActivity = applyPatches({ ...EMPTY_ACTIVITY }, delta);
        resetActivityIdRef.current = newActivity.id;
        return newActivity;
      }
      return applyPatches(prev, delta);
    });
  }, []);

  // Derived state - memoized separately to avoid unnecessary object recreation
  const isActive = activity !== null && activity.id !== "";
  const isComplete = activity !== null && activity.progress >= 1.0;
  const progressPercent = activity ? Math.round(activity.progress * 100) : 0;

  // Auto-reset after completion (with delay for UI feedback)
  // Uses ref to prevent race condition with new activities
  useEffect(() => {
    if (isComplete && activity) {
      const completedActivityId = activity.id;

      // Clear any existing timer
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current);
      }

      resetTimerRef.current = setTimeout(() => {
        // Only reset if the activity ID hasn't changed
        // This prevents resetting a new activity that started during the delay
        if (resetActivityIdRef.current === completedActivityId) {
          reset();
        }
        resetTimerRef.current = null;
      }, ACTIVITY_RESET_DELAY_MS);

      return () => {
        if (resetTimerRef.current) {
          clearTimeout(resetTimerRef.current);
          resetTimerRef.current = null;
        }
      };
    }
  }, [isComplete, activity?.id, reset]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current);
      }
    };
  }, []);

  // Memoize callbacks object separately from data
  // This prevents re-renders when only activity data changes
  const callbacks = useMemo(
    () => ({
      reset,
      setActivity,
      processSnapshot,
      processDelta,
    }),
    [reset, processSnapshot, processDelta]
  );

  // Memoize the full result
  return useMemo(
    () => ({
      activity,
      isActive,
      isComplete,
      progressPercent,
      ...callbacks,
    }),
    [activity, isActive, isComplete, progressPercent, callbacks]
  );
}

export default useActivityTracker;
