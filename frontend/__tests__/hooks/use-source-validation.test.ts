/**
 * Tests for use-source-validation hook
 * Story 6-4: Human-in-the-Loop Source Validation
 * Story 21-A2: Migrate to useHumanInTheLoop Pattern
 */

import React from "react";
import { renderHook, act } from "@testing-library/react";
import { useHumanInTheLoop } from "@copilotkit/react-core";
import { useSourceValidation } from "@/hooks/use-source-validation";
import type { Source } from "@/types/copilot";

// Mock CopilotKit - Story 21-A2: Updated to mock useHumanInTheLoop
jest.mock("@copilotkit/react-core", () => ({
  useHumanInTheLoop: jest.fn(),
}));

// Mock SourceValidationDialog - needed since it's rendered inside the hook
jest.mock("@/components/copilot/SourceValidationDialog", () => {
  const mock = jest.fn(() => null);
  Object.defineProperty(mock, "name", { value: "SourceValidationDialog" });
  return { SourceValidationDialog: mock };
});

const mockUseHumanInTheLoop = useHumanInTheLoop as jest.MockedFunction<
  typeof useHumanInTheLoop
>;

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
      // Story 21-A2: isDialogOpen is now always false (dialog rendered inside hook)
      expect(result.current.isDialogOpen).toBe(false);
    });
  });

  describe("startValidation (deprecated)", () => {
    it("starts validation with provided sources", () => {
      const { result } = renderHook(() => useSourceValidation());

      act(() => {
        result.current.startValidation(mockSources);
      });

      expect(result.current.state.isValidating).toBe(true);
      expect(result.current.state.pendingSources).toEqual(mockSources);
      expect(result.current.state.decisions.size).toBe(3);
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

  describe("submitValidation (deprecated)", () => {
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

      act(() => {
        result.current.submitValidation(["source-1"]);
      });

      // Story 21-A2: isDialogOpen is now always false
      expect(result.current.isDialogOpen).toBe(false);
    });
  });

  describe("cancelValidation (deprecated)", () => {
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

      act(() => {
        result.current.cancelValidation();
      });

      expect(result.current.state.isValidating).toBe(false);
      expect(result.current.state.pendingSources).toEqual([]);
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

  // Story 21-A2: Updated tests for useHumanInTheLoop integration
  describe("CopilotKit useHumanInTheLoop Integration", () => {
    it("registers useHumanInTheLoop with correct parameters", () => {
      renderHook(() => useSourceValidation());

      expect(mockUseHumanInTheLoop).toHaveBeenCalledTimes(1);
      expect(mockUseHumanInTheLoop).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "validate_sources",
          description: expect.stringContaining("human approval"),
          parameters: expect.arrayContaining([
            expect.objectContaining({
              name: "sources",
              type: "object",
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

    it("render callback returns empty fragment when status is not executing", () => {
      renderHook(() => useSourceValidation());

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];

      // Test idle status (CopilotKit 1.x doesn't have idle, but test defensive code)
      const inProgressResult = hookConfig.render({
        status: "inProgress",
        args: {},
        respond: undefined,
        result: undefined,
      });
      // Should return empty fragment
      expect(inProgressResult.type).toBe(React.Fragment);
    });

    it("render callback shows completion message when status is complete with result", () => {
      renderHook(() => useSourceValidation());

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];

      // Test complete status with result
      const completeResult = hookConfig.render({
        status: "complete",
        args: { sources: mockSources },
        respond: undefined,
        result: { approved: ["source-1", "source-2"] },
      });

      // Should render completion message (div element)
      expect(completeResult).not.toBeNull();
      expect(completeResult.type).toBe("div");
      expect(completeResult.props.children).toContain("2 source(s)");
    });

    it("render callback shows cancellation message when result has empty approved array", () => {
      renderHook(() => useSourceValidation());

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];

      const completeResult = hookConfig.render({
        status: "complete",
        args: { sources: mockSources },
        respond: undefined,
        result: { approved: [] },
      });

      expect(completeResult).not.toBeNull();
      expect(completeResult.type).toBe("div");
      expect(completeResult.props.children).toContain("cancelled");
    });

    it("render callback renders dialog when status is executing with respond", () => {
      renderHook(() => useSourceValidation());

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // Simulate executing status with sources
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: mockSources },
        respond: mockRespond,
        result: undefined,
      });

      // Should render the SourceValidationDialog
      expect(renderResult).not.toBeNull();
      expect(renderResult.type.name).toBe("SourceValidationDialog");
      expect(renderResult.props.open).toBe(true);
      expect(renderResult.props.sources).toEqual(mockSources);
    });

    it("onSubmit callback calls respond with approved IDs", () => {
      const onComplete = jest.fn();
      renderHook(() =>
        useSourceValidation({
          onValidationComplete: onComplete,
        })
      );

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // Get the rendered dialog
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: mockSources },
        respond: mockRespond,
        result: undefined,
      });

      // Call onSubmit with approved IDs
      act(() => {
        renderResult.props.onSubmit(["source-1", "source-2"]);
      });

      // Verify respond was called correctly
      expect(mockRespond).toHaveBeenCalledWith({
        approved: ["source-1", "source-2"],
      });
      expect(onComplete).toHaveBeenCalledWith(["source-1", "source-2"]);
    });

    it("onCancel callback calls respond with empty array", () => {
      const onCancelled = jest.fn();
      renderHook(() =>
        useSourceValidation({
          onValidationCancelled: onCancelled,
        })
      );

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // Get the rendered dialog
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: mockSources },
        respond: mockRespond,
        result: undefined,
      });

      // Call onCancel
      act(() => {
        renderResult.props.onCancel();
      });

      // Verify respond was called with empty array
      expect(mockRespond).toHaveBeenCalledWith({ approved: [] });
      expect(onCancelled).toHaveBeenCalled();
    });

    it("auto-responds when all sources are auto-approved", () => {
      const onComplete = jest.fn();
      renderHook(() =>
        useSourceValidation({
          onValidationComplete: onComplete,
          autoApproveThreshold: 0.3, // All sources above this
        })
      );

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // All sources should be auto-approved (all have similarity > 0.3)
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: mockSources },
        respond: mockRespond,
        result: undefined,
      });

      // Should auto-respond and return empty fragment
      expect(renderResult.type).toBe(React.Fragment);
      expect(mockRespond).toHaveBeenCalledWith({
        approved: ["source-1", "source-2", "source-3"],
      });
      expect(onComplete).toHaveBeenCalledWith([
        "source-1",
        "source-2",
        "source-3",
      ]);
    });

    it("auto-responds when all sources are auto-rejected", () => {
      const onComplete = jest.fn();
      renderHook(() =>
        useSourceValidation({
          onValidationComplete: onComplete,
          autoRejectThreshold: 1.0, // All sources below this
        })
      );

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // All sources should be auto-rejected (all have similarity < 1.0)
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: mockSources },
        respond: mockRespond,
        result: undefined,
      });

      // Should auto-respond and return empty fragment
      expect(renderResult.type).toBe(React.Fragment);
      expect(mockRespond).toHaveBeenCalledWith({ approved: [] });
      expect(onComplete).toHaveBeenCalledWith([]);
    });

    it("shows dialog when some sources require manual review", () => {
      renderHook(() =>
        useSourceValidation({
          autoApproveThreshold: 0.9, // Only source-1 auto-approved
          autoRejectThreshold: 0.4, // Only source-3 auto-rejected
        })
      );

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // source-2 (0.65) is pending - needs manual review
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: mockSources },
        respond: mockRespond,
        result: undefined,
      });

      // Should show dialog since source-2 needs manual review
      expect(renderResult.type.name).toBe("SourceValidationDialog");
      expect(mockRespond).not.toHaveBeenCalled();
    });

    it("handles empty sources array gracefully", () => {
      renderHook(() => useSourceValidation());

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // Empty sources array
      const renderResult = hookConfig.render({
        status: "executing",
        args: { sources: [] },
        respond: mockRespond,
        result: undefined,
      });

      // Should render dialog with empty sources (not auto-respond)
      expect(renderResult.type.name).toBe("SourceValidationDialog");
      expect(renderResult.props.sources).toEqual([]);
    });

    it("handles missing sources in args gracefully", () => {
      renderHook(() => useSourceValidation());

      const hookConfig = mockUseHumanInTheLoop.mock.calls[0][0];
      const mockRespond = jest.fn();

      // Missing sources in args
      const renderResult = hookConfig.render({
        status: "executing",
        args: {},
        respond: mockRespond,
        result: undefined,
      });

      // Should render dialog with empty sources
      expect(renderResult.type.name).toBe("SourceValidationDialog");
      expect(renderResult.props.sources).toEqual([]);
    });
  });
});
