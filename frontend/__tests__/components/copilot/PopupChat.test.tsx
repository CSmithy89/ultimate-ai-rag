/**
 * PopupChat component tests.
 *
 * Story 21-F1: Implement CopilotPopup Component
 *
 * Tests cover:
 * - Component rendering
 * - Position configuration
 * - Click-outside-to-close behavior
 * - Styling and responsive design
 */

import { render, screen } from "@testing-library/react";
import { PopupChat } from "@/components/copilot/PopupChat";

// Mock CopilotKit components
jest.mock("@copilotkit/react-ui", () => ({
  CopilotPopup: ({ children, labels, className, defaultOpen, clickOutsideToClose }: {
    children: React.ReactNode;
    labels: { title: string; initial: string };
    className: string;
    defaultOpen: boolean;
    clickOutsideToClose: boolean;
  }) => (
    <div
      data-testid="copilot-popup"
      data-title={labels.title}
      data-initial={labels.initial}
      data-default-open={defaultOpen}
      data-click-outside={clickOutsideToClose}
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

describe("PopupChat", () => {
  describe("Rendering", () => {
    it("renders CopilotPopup with default props", () => {
      render(<PopupChat />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup).toBeInTheDocument();
      expect(popup).toHaveAttribute("data-title", "RAG Assistant");
      expect(popup).toHaveAttribute("data-initial", "How can I help you today?");
    });

    it("renders with custom title and initial message", () => {
      render(
        <PopupChat
          title="Custom Assistant"
          initialMessage="Custom greeting"
        />
      );

      const popup = screen.getByTestId("copilot-popup");
      expect(popup).toHaveAttribute("data-title", "Custom Assistant");
      expect(popup).toHaveAttribute("data-initial", "Custom greeting");
    });

    it("renders child components", () => {
      render(<PopupChat />);

      expect(screen.getByTestId("thought-trace-stepper")).toBeInTheDocument();
      expect(screen.getByTestId("generative-ui-renderer")).toBeInTheDocument();
    });

    it("wraps in error boundary", () => {
      render(<PopupChat />);

      expect(screen.getByTestId("error-boundary")).toBeInTheDocument();
    });
  });

  describe("Position Configuration", () => {
    it("applies bottom-right position class by default", () => {
      render(<PopupChat />);

      const popup = screen.getByTestId("copilot-popup");
      // Default position (bottom-right) has no extra positioning classes
      expect(popup.className).toContain("copilot-popup");
    });

    it("applies bottom-left position class", () => {
      render(<PopupChat position="bottom-left" />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup.className).toContain("!left-4");
      expect(popup.className).toContain("!right-auto");
    });

    it("applies top-right position class", () => {
      render(<PopupChat position="top-right" />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup.className).toContain("!bottom-auto");
      expect(popup.className).toContain("!top-4");
    });

    it("applies top-left position class", () => {
      render(<PopupChat position="top-left" />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup.className).toContain("!left-4");
      expect(popup.className).toContain("!right-auto");
      expect(popup.className).toContain("!bottom-auto");
      expect(popup.className).toContain("!top-4");
    });
  });

  describe("Click-Outside Behavior", () => {
    it("enables click-outside-to-close by default", () => {
      render(<PopupChat />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup).toHaveAttribute("data-click-outside", "true");
    });

    it("allows disabling click-outside-to-close", () => {
      render(<PopupChat clickOutsideToClose={false} />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup).toHaveAttribute("data-click-outside", "false");
    });
  });

  describe("Default Open State", () => {
    it("is closed by default", () => {
      render(<PopupChat />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup).toHaveAttribute("data-default-open", "false");
    });

    it("can be open by default", () => {
      render(<PopupChat defaultOpen />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup).toHaveAttribute("data-default-open", "true");
    });
  });

  describe("Styling", () => {
    it("applies custom className", () => {
      render(<PopupChat className="custom-class" />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup.className).toContain("custom-class");
    });

    it("includes copilot-popup base class", () => {
      render(<PopupChat />);

      const popup = screen.getByTestId("copilot-popup");
      expect(popup.className).toContain("copilot-popup");
    });
  });
});
