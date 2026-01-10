/**
 * EmbeddedChat component tests.
 *
 * Story 21-F2: Implement CopilotChat Embedded Component
 *
 * Tests cover:
 * - Component rendering
 * - Welcome message configuration
 * - Responsive height/width
 * - Container layout integration
 */

import { render, screen } from "@testing-library/react";
import { EmbeddedChat } from "@/components/copilot/EmbeddedChat";

// Mock CopilotKit components
jest.mock("@copilotkit/react-ui", () => ({
  CopilotChat: ({ children, labels, className }: {
    children: React.ReactNode;
    labels: { title: string; initial: string };
    className: string;
  }) => (
    <div
      data-testid="copilot-chat"
      data-title={labels.title}
      data-initial={labels.initial}
      className={className}
    >
      {children}
    </div>
  ),
}));

// Mock internal components
jest.mock("@/components/copilot/ThoughtTraceStepper", () => ({
  ThoughtTraceStepper: () => <div data-testid="thought-trace-stepper" />,
}));

jest.mock("@/components/copilot/CopilotErrorBoundary", () => ({
  CopilotErrorBoundary: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="error-boundary">{children}</div>
  ),
}));

jest.mock("@/components/copilot/GenerativeUIRenderer", () => ({
  GenerativeUIRenderer: () => <div data-testid="generative-ui-renderer" />,
}));

describe("EmbeddedChat", () => {
  describe("Rendering", () => {
    it("renders CopilotChat with default props", () => {
      render(<EmbeddedChat />);

      const chat = screen.getByTestId("copilot-chat");
      expect(chat).toBeInTheDocument();
      expect(chat).toHaveAttribute("data-title", "AI Assistant");
      expect(chat).toHaveAttribute(
        "data-initial",
        "Welcome! Ask me anything about your documents."
      );
    });

    it("renders with custom welcome message", () => {
      render(<EmbeddedChat welcomeMessage="Custom welcome!" />);

      const chat = screen.getByTestId("copilot-chat");
      expect(chat).toHaveAttribute("data-initial", "Custom welcome!");
    });

    it("renders with custom title", () => {
      render(<EmbeddedChat title="Custom Title" />);

      const chat = screen.getByTestId("copilot-chat");
      expect(chat).toHaveAttribute("data-title", "Custom Title");
    });

    it("renders child components", () => {
      render(<EmbeddedChat />);

      expect(screen.getByTestId("thought-trace-stepper")).toBeInTheDocument();
      expect(screen.getByTestId("generative-ui-renderer")).toBeInTheDocument();
    });

    it("wraps in error boundary", () => {
      render(<EmbeddedChat />);

      expect(screen.getByTestId("error-boundary")).toBeInTheDocument();
    });
  });

  describe("Layout", () => {
    it("renders container with embedded-chat-container class", () => {
      render(<EmbeddedChat />);

      const container = screen.getByTestId("copilot-chat").parentElement;
      expect(container).toHaveClass("embedded-chat-container");
    });

    it("applies custom className to container", () => {
      render(<EmbeddedChat className="h-[500px] border rounded-lg" />);

      const container = screen.getByTestId("copilot-chat").parentElement;
      expect(container).toHaveClass("h-[500px]");
      expect(container).toHaveClass("border");
      expect(container).toHaveClass("rounded-lg");
    });

    it("chat has h-full class for responsive height", () => {
      render(<EmbeddedChat />);

      const chat = screen.getByTestId("copilot-chat");
      expect(chat).toHaveClass("h-full");
    });
  });

  describe("Container Layouts", () => {
    it("works with flex container layout", () => {
      const { container } = render(
        <div className="flex flex-col h-screen">
          <EmbeddedChat className="flex-1" />
        </div>
      );

      const chatContainer = container.querySelector(".embedded-chat-container");
      expect(chatContainer).toHaveClass("flex-1");
    });

    it("works with grid container layout", () => {
      const { container } = render(
        <div className="grid grid-cols-2">
          <EmbeddedChat className="col-span-1" />
          <div>Other content</div>
        </div>
      );

      const chatContainer = container.querySelector(".embedded-chat-container");
      expect(chatContainer).toHaveClass("col-span-1");
    });

    it("works with fixed height container", () => {
      render(<EmbeddedChat className="h-[calc(100vh-8rem)]" />);

      const container = screen.getByTestId("copilot-chat").parentElement;
      expect(container).toHaveClass("h-[calc(100vh-8rem)]");
    });
  });
});
