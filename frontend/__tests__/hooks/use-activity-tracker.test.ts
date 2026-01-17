/**
 * Tests for useActivityTracker hook
 */

import { renderHook, act } from "@testing-library/react";
import { useActivityTracker } from "@/hooks/use-activity-tracker";
import type { JSONPatchOperation } from "@/types/ag-ui";

describe("useActivityTracker", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe("initial state", () => {
    it("should return null activity initially", () => {
      const { result } = renderHook(() => useActivityTracker());
      expect(result.current.activity).toBeNull();
      expect(result.current.isActive).toBe(false);
      expect(result.current.isComplete).toBe(false);
      expect(result.current.progressPercent).toBe(0);
    });
  });

  describe("processSnapshot", () => {
    it("should set activity from snapshot", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "query_processing",
          message: "Processing...",
          progress: 0.5,
          totalSteps: 4,
          currentStep: 2,
        });
      });

      expect(result.current.activity).toEqual({
        id: "activity-123",
        type: "query_processing",
        message: "Processing...",
        progress: 0.5,
        totalSteps: 4,
        currentStep: 2,
        metadata: {},
      });
      expect(result.current.isActive).toBe(true);
      expect(result.current.progressPercent).toBe(50);
    });

    it("should handle missing fields with defaults", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-456",
        });
      });

      expect(result.current.activity).toEqual({
        id: "activity-456",
        type: "",
        message: "",
        progress: 0,
        totalSteps: 0,
        currentStep: 0,
        metadata: {},
      });
    });
  });

  describe("processDelta", () => {
    it("should apply replace operation", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Starting...",
          progress: 0,
          totalSteps: 3,
          currentStep: 0,
        });
      });

      act(() => {
        result.current.processDelta([
          { op: "replace", path: "/progress", value: 0.33 },
          { op: "replace", path: "/message", value: "Step 1 complete" },
          { op: "replace", path: "/currentStep", value: 1 },
        ]);
      });

      expect(result.current.activity?.progress).toBe(0.33);
      expect(result.current.activity?.message).toBe("Step 1 complete");
      expect(result.current.activity?.currentStep).toBe(1);
    });

    it("should ignore unknown keys for security", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Test",
          progress: 0,
          totalSteps: 1,
          currentStep: 0,
        });
      });

      const originalActivity = { ...result.current.activity };

      act(() => {
        result.current.processDelta([
          { op: "replace", path: "/__proto__", value: "hacked" },
          { op: "replace", path: "/constructor", value: "malicious" },
          { op: "add", path: "/unknownField", value: "test" },
        ] as JSONPatchOperation[]);
      });

      // Activity should be unchanged - all patches were ignored
      expect(result.current.activity?.id).toBe(originalActivity?.id);
      expect(result.current.activity?.type).toBe(originalActivity?.type);
      expect(result.current.activity?.message).toBe(originalActivity?.message);
      // Verify no unknown fields were added (only known keys should exist)
      const activity = result.current.activity as Record<string, unknown>;
      expect(Object.prototype.hasOwnProperty.call(activity, "unknownField")).toBe(false);
      // Note: __proto__ is always accessible on objects but shouldn't be an own property
      expect(Object.prototype.hasOwnProperty.call(activity, "__proto__")).toBe(false);
    });

    it("should validate value types", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Test",
          progress: 0.5,
          totalSteps: 3,
          currentStep: 1,
        });
      });

      act(() => {
        result.current.processDelta([
          { op: "replace", path: "/progress", value: "not a number" as unknown as number },
          { op: "replace", path: "/message", value: 12345 as unknown as string },
        ]);
      });

      // Values should not have changed due to type validation
      expect(result.current.activity?.progress).toBe(0.5);
      expect(result.current.activity?.message).toBe("Test");
    });

    it("should handle metadata operations", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Test",
          progress: 0,
          totalSteps: 1,
          currentStep: 0,
          metadata: { key1: "value1" },
        });
      });

      act(() => {
        result.current.processDelta([
          { op: "add", path: "/metadata/key2", value: "value2" },
          { op: "replace", path: "/metadata/key1", value: "updated" },
        ]);
      });

      expect(result.current.activity?.metadata).toEqual({
        key1: "updated",
        key2: "value2",
      });
    });
  });

  describe("auto-reset on completion", () => {
    it("should auto-reset after completion delay", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Complete",
          progress: 1.0,
          totalSteps: 1,
          currentStep: 1,
        });
      });

      expect(result.current.isComplete).toBe(true);
      expect(result.current.activity).not.toBeNull();

      // Advance past the reset delay
      act(() => {
        jest.advanceTimersByTime(3500);
      });

      expect(result.current.activity).toBeNull();
      expect(result.current.isActive).toBe(false);
    });

    it("should not reset if new activity starts during delay", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Complete",
          progress: 1.0,
          totalSteps: 1,
          currentStep: 1,
        });
      });

      // Start a new activity before the reset
      act(() => {
        jest.advanceTimersByTime(1000);
        result.current.processSnapshot({
          id: "activity-456",
          type: "retrieval",
          message: "New activity",
          progress: 0,
          totalSteps: 2,
          currentStep: 0,
        });
      });

      // Advance past original reset time
      act(() => {
        jest.advanceTimersByTime(3000);
      });

      // New activity should still be present
      expect(result.current.activity?.id).toBe("activity-456");
      expect(result.current.isActive).toBe(true);
    });
  });

  describe("reset", () => {
    it("should clear activity state", () => {
      const { result } = renderHook(() => useActivityTracker());

      act(() => {
        result.current.processSnapshot({
          id: "activity-123",
          type: "indexing",
          message: "Test",
          progress: 0.5,
          totalSteps: 1,
          currentStep: 0,
        });
      });

      expect(result.current.activity).not.toBeNull();

      act(() => {
        result.current.reset();
      });

      expect(result.current.activity).toBeNull();
      expect(result.current.isActive).toBe(false);
    });
  });
});
