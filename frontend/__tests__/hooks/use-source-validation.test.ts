/**
 * Tests for use-source-validation hook
 * Story 6-4: Human-in-the-Loop Source Validation
 */

import { renderHook, act } from "@testing-library/react";
import { useCopilotAction } from "@copilotkit/react-core";
import { useSourceValidation } from "@/hooks/use-source-validation";
import type { Source } from "@/types/copilot";

// Mock CopilotKit
jest.mock("@copilotkit/react-core", () => ({
  useCopilotAction: jest.fn(),
}));

const mockUseCopilotAction = useCopilotAction as jest.MockedFunction<typeof useCopilotAction>;

// Mock sources
const mockSources: Source[] = [
  {
    id: "source-1",
    title: "Document One",
    preview: "Preview of document one.",
    similarity: 0.95,
  },
  {
    id: "source-2",
    title: "Document Two",
    preview: "Preview of document two.",
    similarity: 0.65,
  },
  {
    id: "source-3",
    title: "Document Three",
    preview: "Preview of document three.",
    similarity: 0.35,
  },
];

describe("useSourceValidation", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Initial State", () => {
    it("initializes with default state", () => {
      const { result } = renderHook(() => useSourceValidation());

      expect(result.current.state.isValidating).toBe(false);
      expect(result.current.state.pendingSources).toEqual([]);
      expect(result.current.state.decisions.size).toBe(0);
      expect(result.current.state.approvedIds).toEqual([]);
      expect(result.current.state.rejectedIds).toEqual([]);
      expect(result.current.state.isSubmitting).toBe(false);
      expect(result.current.state.error).toBeNull();
      expect(result.current.isDialogOpen).toBe(false);
    });
  });

  describe("startValidation", () => {
    it("starts validation with provided sources", () => {
      const { result } = renderHook(() => useSourceValidation());

      act(() => {
        result.current.startValidation(mockSources);
      });

      expect(result.current.state.isValidating).toBe(true);
      expect(result.current.state.pendingSources).toEqual(mockSources);
      expect(result.current.state.decisions.size).toBe(3);
      expect(result.current.isDialogOpen).toBe(true);
    });

    it("initializes all decisions as pending", () => {
      const { result } = renderHook(() => useSourceValidation());

      act(() => {
        result.current.startValidation(mockSources);
      });

      expect(result.current.state.decisions.get("source-1")).toBe("pending");
      expect(result.current.state.decisions.get("source-2")).toBe("pending");
      expect(result.current.state.decisions.get("source-3")).toBe("pending");
    });
  });

  describe("Auto-approve/reject thresholds", () => {
    it("auto-approves sources above threshold", () => {
      const { result } = renderHook(() =>
        useSourceValidation({
          autoApproveThreshold: 0.9,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      // Source 1 (0.95) should be auto-approved
      expect(result.current.state.decisions.get("source-1")).toBe("approved");
      // Others remain pending
      expect(result.current.state.decisions.get("source-2")).toBe("pending");
      expect(result.current.state.decisions.get("source-3")).toBe("pending");
    });

    it("auto-rejects sources below threshold", () => {
      const { result } = renderHook(() =>
        useSourceValidation({
          autoRejectThreshold: 0.5,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      // Source 3 (0.35) should be auto-rejected
      expect(result.current.state.decisions.get("source-3")).toBe("rejected");
      // Others remain pending
      expect(result.current.state.decisions.get("source-1")).toBe("pending");
      expect(result.current.state.decisions.get("source-2")).toBe("pending");
    });

    it("combines auto-approve and auto-reject thresholds", () => {
      const { result } = renderHook(() =>
        useSourceValidation({
          autoApproveThreshold: 0.9,
          autoRejectThreshold: 0.5,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      expect(result.current.state.decisions.get("source-1")).toBe("approved");
      expect(result.current.state.decisions.get("source-2")).toBe("pending");
      expect(result.current.state.decisions.get("source-3")).toBe("rejected");
    });
  });

  describe("submitValidation", () => {
    it("updates state with approved IDs", () => {
      const onComplete = jest.fn();
      const { result } = renderHook(() =>
        useSourceValidation({
          onValidationComplete: onComplete,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      act(() => {
        result.current.submitValidation(["source-1", "source-2"]);
      });

      expect(result.current.state.isValidating).toBe(false);
      expect(result.current.state.approvedIds).toEqual(["source-1", "source-2"]);
      expect(onComplete).toHaveBeenCalledWith(["source-1", "source-2"]);
    });

    it("closes dialog after submission", () => {
      const { result } = renderHook(() => useSourceValidation());

      act(() => {
        result.current.startValidation(mockSources);
      });

      expect(result.current.isDialogOpen).toBe(true);

      act(() => {
        result.current.submitValidation(["source-1"]);
      });

      expect(result.current.isDialogOpen).toBe(false);
    });
  });

  describe("cancelValidation", () => {
    it("resets state and closes dialog", () => {
      const onCancelled = jest.fn();
      const { result } = renderHook(() =>
        useSourceValidation({
          onValidationCancelled: onCancelled,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      expect(result.current.isDialogOpen).toBe(true);

      act(() => {
        result.current.cancelValidation();
      });

      expect(result.current.state.isValidating).toBe(false);
      expect(result.current.state.pendingSources).toEqual([]);
      expect(result.current.isDialogOpen).toBe(false);
      expect(onCancelled).toHaveBeenCalled();
    });
  });

  describe("resetValidation", () => {
    it("resets all state to initial values", () => {
      const { result } = renderHook(() => useSourceValidation());

      act(() => {
        result.current.startValidation(mockSources);
      });

      act(() => {
        result.current.submitValidation(["source-1"]);
      });

      act(() => {
        result.current.resetValidation();
      });

      expect(result.current.state.isValidating).toBe(false);
      expect(result.current.state.pendingSources).toEqual([]);
      expect(result.current.state.decisions.size).toBe(0);
      expect(result.current.state.approvedIds).toEqual([]);
      expect(result.current.state.rejectedIds).toEqual([]);
    });
  });

  describe("Callbacks", () => {
    it("calls onValidationComplete with approved IDs", () => {
      const onComplete = jest.fn();
      const { result } = renderHook(() =>
        useSourceValidation({
          onValidationComplete: onComplete,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      act(() => {
        result.current.submitValidation(["source-1"]);
      });

      expect(onComplete).toHaveBeenCalledWith(["source-1"]);
    });

    it("calls onValidationCancelled when cancelled", () => {
      const onCancelled = jest.fn();
      const { result } = renderHook(() =>
        useSourceValidation({
          onValidationCancelled: onCancelled,
        })
      );

      act(() => {
        result.current.startValidation(mockSources);
      });

      act(() => {
        result.current.cancelValidation();
      });

      expect(onCancelled).toHaveBeenCalled();
    });
  });

  // Issue 6 Fix: Integration tests for useCopilotAction
  describe("CopilotKit Integration", () => {
    it("registers useCopilotAction with correct parameters", () => {
      renderHook(() => useSourceValidation());

      expect(mockUseCopilotAction).toHaveBeenCalledTimes(1);
      expect(mockUseCopilotAction).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "validate_sources",
          description: expect.stringContaining("human approval"),
          parameters: expect.arrayContaining([
            expect.objectContaining({
              name: "sources",
              type: "object[]",
              required: true,
            }),
            expect.objectContaining({
              name: "query",
              type: "string",
              required: false,
            }),
          ]),
          render: expect.any(Function),
        })
      );
    });

    it("useCopilotAction render callback returns React fragment when idle", () => {
      renderHook(() => useSourceValidation());

      const actionConfig = mockUseCopilotAction.mock.calls[0][0];
      const renderResult = actionConfig.render({ status: "idle", args: {} });

      // Should return a React fragment (empty element)
      expect(renderResult).toBeDefined();
      expect(renderResult.type).toBe(Symbol.for("react.fragment"));
    });

    it("useCopilotAction render callback triggers validation when executing with sources", () => {
      jest.useFakeTimers();
      const { result } = renderHook(() => useSourceValidation());

      const actionConfig = mockUseCopilotAction.mock.calls[0][0];
      
      // Simulate the render callback being called with executing status and sources
      actionConfig.render({
        status: "executing",
        args: { sources: mockSources },
      });

      // The setTimeout should have been scheduled
      act(() => {
        jest.runAllTimers();
      });

      // After the timeout, validation should have started
      expect(result.current.state.isValidating).toBe(true);
      expect(result.current.state.pendingSources).toEqual(mockSources);

      jest.useRealTimers();
    });

    it("useCopilotAction render callback does not trigger validation twice", () => {
      jest.useFakeTimers();
      const { result } = renderHook(() => useSourceValidation());

      const actionConfig = mockUseCopilotAction.mock.calls[0][0];
      
      // Call render twice with same executing status
      actionConfig.render({
        status: "executing",
        args: { sources: mockSources },
      });
      
      actionConfig.render({
        status: "executing",
        args: { sources: mockSources },
      });

      act(() => {
        jest.runAllTimers();
      });

      // Validation should only be triggered once
      expect(result.current.state.isValidating).toBe(true);
      expect(result.current.state.pendingSources).toEqual(mockSources);

      jest.useRealTimers();
    });

    it("useCopilotAction render callback ignores empty sources array", () => {
      jest.useFakeTimers();
      const { result } = renderHook(() => useSourceValidation());

      const actionConfig = mockUseCopilotAction.mock.calls[0][0];
      
      // Call render with empty sources
      actionConfig.render({
        status: "executing",
        args: { sources: [] },
      });

      act(() => {
        jest.runAllTimers();
      });

      // Validation should not be triggered
      expect(result.current.state.isValidating).toBe(false);
      expect(result.current.state.pendingSources).toEqual([]);

      jest.useRealTimers();
    });
  });
});
