/**
 * Tests for useThoughtTrace hook.
 * Story 6-2: Chat Sidebar Interface
 */

import { renderHook, act } from "@testing-library/react";
import { useThoughtTrace } from "../../hooks/use-thought-trace";

describe("useThoughtTrace", () => {
  it("returns initial empty state", () => {
    const { result } = renderHook(() => useThoughtTrace());

    expect(result.current.steps).toEqual([]);
    expect(result.current.currentStep).toBeNull();
    expect(result.current.isProcessing).toBe(false);
  });

  it("returns correct isProcessing when no steps are in progress", () => {
    const { result } = renderHook(() => useThoughtTrace());

    expect(result.current.isProcessing).toBe(false);
  });

  it("clearSteps resets the steps array", () => {
    const { result } = renderHook(() => useThoughtTrace());

    // Add a step first
    act(() => {
      result.current.addStep({ step: "Test step", status: "completed" });
    });

    expect(result.current.steps.length).toBe(1);

    act(() => {
      result.current.clearSteps();
    });

    expect(result.current.steps).toEqual([]);
  });

  it("respects maxVisibleSteps option", () => {
    const { result } = renderHook(() =>
      useThoughtTrace({ maxVisibleSteps: 3 })
    );

    // Add more steps than maxVisibleSteps
    act(() => {
      result.current.addStep({ step: "Step 1", status: "completed" });
      result.current.addStep({ step: "Step 2", status: "completed" });
      result.current.addStep({ step: "Step 3", status: "completed" });
      result.current.addStep({ step: "Step 4", status: "completed" });
      result.current.addStep({ step: "Step 5", status: "in_progress" });
    });

    // Should only show last 3 steps
    expect(result.current.steps.length).toBe(3);
    expect(result.current.steps[0].step).toBe("Step 3");
    expect(result.current.steps[2].step).toBe("Step 5");
  });

  it("returns null for currentStep when no steps are in progress", () => {
    const { result } = renderHook(() => useThoughtTrace());

    act(() => {
      result.current.addStep({ step: "Completed step", status: "completed" });
    });

    expect(result.current.currentStep).toBeNull();
  });

  it("provides a clearSteps function", () => {
    const { result } = renderHook(() => useThoughtTrace());

    expect(typeof result.current.clearSteps).toBe("function");
  });

  it("handles default options correctly", () => {
    const { result } = renderHook(() => useThoughtTrace());

    // Default maxVisibleSteps should be 10
    expect(result.current.steps).toEqual([]);
    expect(result.current.isProcessing).toBe(false);
  });

  it("addStep adds a new step with timestamp", () => {
    const { result } = renderHook(() => useThoughtTrace());

    act(() => {
      result.current.addStep({ step: "Analyzing query", status: "in_progress" });
    });

    expect(result.current.steps.length).toBe(1);
    expect(result.current.steps[0].step).toBe("Analyzing query");
    expect(result.current.steps[0].status).toBe("in_progress");
    expect(result.current.steps[0].timestamp).toBeDefined();
  });

  it("addStep preserves details when provided", () => {
    const { result } = renderHook(() => useThoughtTrace());

    act(() => {
      result.current.addStep({
        step: "Processing",
        status: "in_progress",
        details: "Some detailed information",
      });
    });

    expect(result.current.steps[0].details).toBe("Some detailed information");
  });

  it("updateStepStatus changes the status of a step", () => {
    const { result } = renderHook(() => useThoughtTrace());

    act(() => {
      result.current.addStep({ step: "Step 1", status: "pending" });
      result.current.addStep({ step: "Step 2", status: "pending" });
    });

    act(() => {
      result.current.updateStepStatus(0, "completed");
    });

    expect(result.current.steps[0].status).toBe("completed");
    expect(result.current.steps[1].status).toBe("pending");
  });

  it("isProcessing returns true when a step is in progress", () => {
    const { result } = renderHook(() => useThoughtTrace());

    act(() => {
      result.current.addStep({ step: "Processing step", status: "in_progress" });
    });

    expect(result.current.isProcessing).toBe(true);
  });

  it("currentStep returns the first in_progress step", () => {
    const { result } = renderHook(() => useThoughtTrace());

    act(() => {
      result.current.addStep({ step: "Completed step", status: "completed" });
      result.current.addStep({ step: "In progress step", status: "in_progress" });
      result.current.addStep({ step: "Pending step", status: "pending" });
    });

    expect(result.current.currentStep?.step).toBe("In progress step");
  });
});
