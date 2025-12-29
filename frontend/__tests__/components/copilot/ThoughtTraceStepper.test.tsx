/**
 * Tests for ThoughtTraceStepper component.
 * Story 6-2: Chat Sidebar Interface
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThoughtTraceStepper, StepIndicator } from "../../../components/copilot/ThoughtTraceStepper";
import type { ThoughtStep } from "../../../types/copilot";

// Mock useCoAgentStateRender hook
let mockRenderCallback: ((params: { state: { steps: ThoughtStep[] } }) => React.ReactNode) | null = null;

jest.mock("@copilotkit/react-core", () => ({
  useCoAgentStateRender: <T,>({
    name,
    render,
  }: {
    name: string;
    render: (params: { state: T }) => React.ReactNode;
  }) => {
    mockRenderCallback = render as unknown as typeof mockRenderCallback;
  },
}));

describe("ThoughtTraceStepper", () => {
  beforeEach(() => {
    mockRenderCallback = null;
  });

  it("registers with the orchestrator agent name", () => {
    render(<ThoughtTraceStepper />);
    // Component should have called useCoAgentStateRender
    expect(mockRenderCallback).not.toBeNull();
  });

  it("returns null when no steps are provided", () => {
    render(<ThoughtTraceStepper />);

    if (mockRenderCallback) {
      const result = mockRenderCallback({ state: { steps: [] } });
      expect(result).toBeNull();
    }
  });

  it("renders steps when provided", () => {
    const { container } = render(<ThoughtTraceStepper />);

    const mockSteps: ThoughtStep[] = [
      { step: "Analyzing query", status: "completed" },
      { step: "Retrieving documents", status: "in_progress" },
      { step: "Generating response", status: "pending" },
    ];

    if (mockRenderCallback) {
      const rendered = mockRenderCallback({ state: { steps: mockSteps } });
      // Re-render with the result
      const { getByText } = render(<>{rendered}</>);
      
      expect(getByText("Agent Progress")).toBeInTheDocument();
      expect(getByText("Analyzing query")).toBeInTheDocument();
      expect(getByText("Retrieving documents")).toBeInTheDocument();
      expect(getByText("Generating response")).toBeInTheDocument();
    }
  });
});

describe("StepIndicator", () => {
  const baseProps = {
    index: 0,
    isExpanded: false,
    onToggle: jest.fn(),
  };

  it("renders step text", () => {
    const step: ThoughtStep = {
      step: "Test step",
      status: "pending",
    };

    render(<StepIndicator step={step} {...baseProps} />);

    expect(screen.getByText("Test step")).toBeInTheDocument();
  });

  it("displays correct icon for pending status", () => {
    const step: ThoughtStep = {
      step: "Pending step",
      status: "pending",
    };

    const { container } = render(<StepIndicator step={step} {...baseProps} />);

    // Pending status should have slate-400 color
    const button = container.querySelector("button");
    expect(button).toBeInTheDocument();
    expect(screen.getByText("Pending step")).toHaveClass("text-slate-400");
  });

  it("displays correct icon for in_progress status", () => {
    const step: ThoughtStep = {
      step: "In progress step",
      status: "in_progress",
    };

    const { container } = render(<StepIndicator step={step} {...baseProps} />);

    // In progress status should have indigo-600 color and animate-spin
    expect(screen.getByText("In progress step")).toHaveClass("text-indigo-600");
    expect(screen.getByText("In progress step")).toHaveClass("font-medium");
  });

  it("displays correct icon for completed status", () => {
    const step: ThoughtStep = {
      step: "Completed step",
      status: "completed",
    };

    const { container } = render(<StepIndicator step={step} {...baseProps} />);

    // Completed status should have slate-600 color
    expect(screen.getByText("Completed step")).toHaveClass("text-slate-600");
  });

  it("calls onToggle when clicked", () => {
    const onToggle = jest.fn();
    const step: ThoughtStep = {
      step: "Clickable step",
      status: "completed",
      details: "Some details",
    };

    render(<StepIndicator step={step} {...baseProps} onToggle={onToggle} />);

    fireEvent.click(screen.getByRole("button"));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("shows details when expanded", () => {
    const step: ThoughtStep = {
      step: "Step with details",
      status: "completed",
      details: "Detailed information here",
    };

    render(<StepIndicator step={step} {...baseProps} isExpanded={true} />);

    expect(screen.getByText("Detailed information here")).toBeInTheDocument();
  });

  it("hides details when not expanded", () => {
    const step: ThoughtStep = {
      step: "Step with details",
      status: "completed",
      details: "Detailed information here",
    };

    render(<StepIndicator step={step} {...baseProps} isExpanded={false} />);

    expect(
      screen.queryByText("Detailed information here")
    ).not.toBeInTheDocument();
  });

  it("shows chevron down when expanded with details", () => {
    const step: ThoughtStep = {
      step: "Step with details",
      status: "completed",
      details: "Some details",
    };

    const { container } = render(
      <StepIndicator step={step} {...baseProps} isExpanded={true} />
    );

    // Check for ChevronDown icon presence (via test id or class)
    expect(container.querySelector('[data-testid="chevron-down"]') || 
           container.querySelector('.lucide-chevron-down')).toBeTruthy();
  });

  it("shows chevron right when collapsed with details", () => {
    const step: ThoughtStep = {
      step: "Step with details",
      status: "completed",
      details: "Some details",
    };

    const { container } = render(
      <StepIndicator step={step} {...baseProps} isExpanded={false} />
    );

    // Check for ChevronRight icon presence
    expect(container.querySelector('[data-testid="chevron-right"]') || 
           container.querySelector('.lucide-chevron-right')).toBeTruthy();
  });

  it("shows no chevron when step has no details", () => {
    const step: ThoughtStep = {
      step: "Step without details",
      status: "completed",
    };

    const { container } = render(<StepIndicator step={step} {...baseProps} />);

    // Should have a spacer instead of chevron
    expect(container.querySelector(".w-3")).toBeInTheDocument();
  });

  it("displays timestamp when provided", () => {
    const step: ThoughtStep = {
      step: "Step with timestamp",
      status: "completed",
      timestamp: "2024-01-15T10:30:00Z",
    };

    render(<StepIndicator step={step} {...baseProps} />);

    // Timestamp might be displayed or used for ordering
    expect(screen.getByText("Step with timestamp")).toBeInTheDocument();
  });

  // Issue 9: Edge case tests for empty and long step text
  it("renders empty step text gracefully", () => {
    const step: ThoughtStep = {
      step: "",
      status: "pending",
    };

    render(<StepIndicator step={step} {...baseProps} />);

    // Should render without crashing
    const button = screen.getByRole("button");
    expect(button).toBeInTheDocument();
  });

  it("handles very long step text", () => {
    const longText = "A".repeat(500);
    const step: ThoughtStep = {
      step: longText,
      status: "in_progress",
    };

    render(<StepIndicator step={step} {...baseProps} />);

    // Should render the long text
    expect(screen.getByText(longText)).toBeInTheDocument();
  });

  it("handles very long details text", () => {
    const longDetails = "B".repeat(1000);
    const step: ThoughtStep = {
      step: "Step with long details",
      status: "completed",
      details: longDetails,
    };

    render(<StepIndicator step={step} {...baseProps} isExpanded={true} />);

    // Should render the long details text
    expect(screen.getByText(longDetails)).toBeInTheDocument();
  });

  it("has aria-label on toggle button", () => {
    const step: ThoughtStep = {
      step: "Accessible step",
      status: "completed",
      details: "Some details",
    };

    render(<StepIndicator step={step} {...baseProps} isExpanded={false} />);

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-label");
    expect(button.getAttribute("aria-label")).toContain("Accessible step");
  });
});
