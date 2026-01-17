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

import { useState, useCallback, useEffect, useMemo } from "react";
import type { ActivityState } from "@/components/widgets/ActivityTrackerWidget";

/**
 * RFC 6902 JSON Patch operation.
 */
interface JSONPatchOperation {
  op: "add" | "remove" | "replace" | "move" | "copy" | "test";
  path: string;
  value?: unknown;
  from?: string;
}

/**
 * Apply a single JSON Patch operation to an activity state.
 */
function applyPatchOperation(
  activity: ActivityState,
  op: JSONPatchOperation
): ActivityState {
  const path = op.path.replace(/^\//, ""); // Remove leading slash
  const pathParts = path.split("/");

  if (pathParts.length === 1) {
    // Top-level property
    const key = pathParts[0] as keyof ActivityState;
    switch (op.op) {
      case "replace":
      case "add":
        return { ...activity, [key]: op.value };
      case "remove": {
        const copy = { ...activity };
        delete (copy as Record<string, unknown>)[key];
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
 * Default empty activity state.
 */
const emptyActivity: ActivityState = {
  id: "",
  type: "",
  message: "",
  progress: 0,
  totalSteps: 0,
  currentStep: 0,
  metadata: {},
};

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

  // Reset the activity tracker
  const reset = useCallback(() => {
    setActivity(null);
  }, []);

  // Process an ACTIVITY_SNAPSHOT event
  const processSnapshot = useCallback((snapshot: Record<string, unknown>) => {
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
  }, []);

  // Process an ACTIVITY_DELTA event
  const processDelta = useCallback((delta: JSONPatchOperation[]) => {
    setActivity((prev) => {
      if (!prev) {
        // If no previous activity, create a new one with the delta
        return applyPatches(emptyActivity, delta);
      }
      return applyPatches(prev, delta);
    });
  }, []);

  // Derived state
  const isActive = activity !== null && activity.id !== "";
  const isComplete = activity !== null && activity.progress >= 1.0;
  const progressPercent = activity ? Math.round(activity.progress * 100) : 0;

  // Auto-reset after completion (with delay for UI feedback)
  useEffect(() => {
    if (isComplete) {
      const timer = setTimeout(() => {
        reset();
      }, 3000); // Reset after 3 seconds
      return () => clearTimeout(timer);
    }
  }, [isComplete, reset]);

  return useMemo(
    () => ({
      activity,
      isActive,
      isComplete,
      progressPercent,
      reset,
      setActivity,
      processSnapshot,
      processDelta,
    }),
    [activity, isActive, isComplete, progressPercent, reset, processSnapshot, processDelta]
  );
}

export default useActivityTracker;
