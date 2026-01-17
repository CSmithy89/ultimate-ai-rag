/**
 * Tests for ChatSidebar component.
 * Story 6-2: Chat Sidebar Interface
 * Story 6-3: Generative UI Components
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { ChatSidebar } from "../../../components/copilot/ChatSidebar";

// Mock next/navigation
jest.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
  }),
}));

// Mock useChatSuggestions hook
jest.mock("../../../hooks/use-chat-suggestions", () => ({
  useChatSuggestions: () => [],
}));

// Mock QuickActions component (has complex CopilotKit dependencies)
jest.mock("../../../components/copilot/QuickActions", () => ({
  QuickActions: ({ actions }: { actions: unknown[] }) => (
    <div data-testid="quick-actions">
      {actions?.length || 0} actions
    </div>
  ),
}));

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

// Mock GenerativeUIRenderer
jest.mock("../../../components/copilot/GenerativeUIRenderer", () => ({
  GenerativeUIRenderer: () => (
    <div data-testid="generative-ui-renderer">Generative UI Renderer</div>
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

  it("includes GenerativeUIRenderer component", () => {
    render(<ChatSidebar />);

    expect(screen.getByTestId("generative-ui-renderer")).toBeInTheDocument();
  });
});
