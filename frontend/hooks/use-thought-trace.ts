"use client";

import { useState, useCallback } from "react";
import type { ThoughtStep } from "@/types/copilot";

/**
 * Options for the useThoughtTrace hook.
 */
interface UseThoughtTraceOptions {
  /** Maximum number of visible steps. Default: 10 */
  maxVisibleSteps?: number;
}

/**
 * Return type for the useThoughtTrace hook.
 */
interface UseThoughtTraceReturn {
  /** Array of thought steps (limited by maxVisibleSteps) */
  steps: ThoughtStep[];
  /** The current step in progress, or null if none */
  currentStep: ThoughtStep | null;
  /** Whether any step is currently in progress */
  isProcessing: boolean;
  /** Function to add a new step */
  addStep: (step: Omit<ThoughtStep, "timestamp">) => void;
  /** Function to update a step's status by index */
  updateStepStatus: (index: number, status: ThoughtStep["status"]) => void;
  /** Function to clear all steps */
  clearSteps: () => void;
}

/**
 * Custom hook for managing thought trace state externally.
 *
 * This hook provides a way to access and manage thought trace
 * state outside of the ThoughtTraceStepper component. It can be
 * used when you need programmatic control over the thought steps.
 *
 * Note: ThoughtTraceStepper uses useCoAgentStateRender directly
 * for automatic state sync with the backend. Use this hook only
 * when you need manual control over thought trace state.
 *
 * Story 6-2: Chat Sidebar Interface
 *
 * @param options - Configuration options
 * @returns Thought trace state and control functions
 *
 * @example
 * ```tsx
 * const { steps, currentStep, isProcessing, addStep, clearSteps } = useThoughtTrace({
 *   maxVisibleSteps: 5,
 * });
 *
 * // Add a new step
 * addStep({ step: "Analyzing query", status: "in_progress" });
 * ```
 */
export function useThoughtTrace(
  options: UseThoughtTraceOptions = {}
): UseThoughtTraceReturn {
  const { maxVisibleSteps = 10 } = options;
  const [steps, setSteps] = useState<ThoughtStep[]>([]);

  // Add a new step with automatic timestamp
  const addStep = useCallback((step: Omit<ThoughtStep, "timestamp">) => {
    const newStep: ThoughtStep = {
      ...step,
      timestamp: new Date().toISOString(),
    };
    setSteps((prev) => [...prev, newStep]);
  }, []);

  // Update a step's status by index
  const updateStepStatus = useCallback(
    (index: number, status: ThoughtStep["status"]) => {
      setSteps((prev) =>
        prev.map((step, i) => (i === index ? { ...step, status } : step))
      );
    },
    []
  );

  // Clear all steps
  const clearSteps = useCallback(() => setSteps([]), []);

  // Determine if any step is currently in progress
  const isProcessing = steps.some((s) => s.status === "in_progress");

  // Find the current step (first in_progress step)
  const currentStep = steps.find((s) => s.status === "in_progress") || null;

  return {
    steps: steps.slice(-maxVisibleSteps),
    currentStep,
    isProcessing,
    addStep,
    updateStepStatus,
    clearSteps,
  };
}
