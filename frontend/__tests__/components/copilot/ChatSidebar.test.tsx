/**
 * Tests for ChatSidebar component.
 * Story 6-2: Chat Sidebar Interface
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { ChatSidebar } from "../../../components/copilot/ChatSidebar";

// Mock CopilotKit components
jest.mock("@copilotkit/react-ui", () => ({
  CopilotSidebar: ({
    children,
    labels,
    className,
    defaultOpen,
  }: {
    children?: React.ReactNode;
    labels?: { title?: string; initial?: string };
    className?: string;
    defaultOpen?: boolean;
  }) => (
    <div
      data-testid="copilot-sidebar"
      data-default-open={defaultOpen}
      className={className}
    >
      {labels?.title && <h2 data-testid="sidebar-title">{labels.title}</h2>}
      {labels?.initial && (
        <p data-testid="sidebar-initial">{labels.initial}</p>
      )}
      {children}
    </div>
  ),
}));

// Mock ThoughtTraceStepper
jest.mock("../../../components/copilot/ThoughtTraceStepper", () => ({
  ThoughtTraceStepper: () => (
    <div data-testid="thought-trace-stepper">Thought Trace Stepper</div>
  ),
}));

describe("ChatSidebar", () => {
  it("renders the CopilotSidebar component", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("copilot-sidebar")).toBeInTheDocument();
  });

  it("has correct title label", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("sidebar-title")).toHaveTextContent("AI Copilot");
  });

  it("has correct initial message label", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("sidebar-initial")).toHaveTextContent(
      "How can I help you today?"
    );
  });

  it("is open by default", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("copilot-sidebar")).toHaveAttribute(
      "data-default-open",
      "true"
    );
  });

  it("applies the copilot-sidebar class for styling", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("copilot-sidebar")).toHaveClass("copilot-sidebar");
  });

  it("includes ThoughtTraceStepper component", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("thought-trace-stepper")).toBeInTheDocument();
  });
});
