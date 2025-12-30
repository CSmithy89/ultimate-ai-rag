/**
 * Tests for ActionPanel component
 * Story 6-5: Frontend Actions
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ActionPanel } from "@/components/copilot/ActionPanel";

describe("ActionPanel", () => {
  const mockActions = [
    {
      id: "action-1",
      type: "save" as const,
      status: "success" as const,
      timestamp: new Date().toISOString(),
      title: "Saved response",
    },
    {
      id: "action-2",
      type: "share" as const,
      status: "success" as const,
      timestamp: new Date().toISOString(),
      title: "Shared link",
      data: { shareUrl: "https://share.test/abc" },
    },
    {
      id: "action-3",
      type: "export" as const,
      status: "error" as const,
      timestamp: new Date().toISOString(),
      title: "Export failed",
      error: "Network error",
    },
    {
      id: "action-4",
      type: "bookmark" as const,
      status: "pending" as const,
      timestamp: new Date().toISOString(),
      title: "Bookmarking...",
    },
  ];

  const defaultProps = {
    actions: mockActions,
    isOpen: true,
    onClose: jest.fn(),
    onClearHistory: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Rendering", () => {
    it("renders nothing when isOpen is false", () => {
      const { container } = render(
        <ActionPanel {...defaultProps} isOpen={false} />
      );

      expect(container.firstChild).toBeNull();
    });

    it("renders panel when isOpen is true", () => {
      render(<ActionPanel {...defaultProps} />);

      expect(screen.getByText("Action History")).toBeInTheDocument();
    });

    it("renders all action items", () => {
      render(<ActionPanel {...defaultProps} />);

      expect(screen.getByText("Saved response")).toBeInTheDocument();
      expect(screen.getByText("Shared link")).toBeInTheDocument();
      expect(screen.getByText("Export failed")).toBeInTheDocument();
      expect(screen.getByText("Bookmarking...")).toBeInTheDocument();
    });

    it("renders empty state when no actions", () => {
      render(<ActionPanel {...defaultProps} actions={[]} />);

      expect(screen.getByText("No actions yet")).toBeInTheDocument();
    });
  });

  describe("Status Icons", () => {
    it("shows success styling for successful actions", () => {
      render(<ActionPanel {...defaultProps} />);

      // Success actions should have green styling - look for the outer wrapper with border class
      const savedItem = screen.getByText("Saved response");
      // The border class is on the outer ActionItem div which wraps the content
      const actionItemDiv = savedItem.closest(".border-emerald-200");
      expect(actionItemDiv).toBeInTheDocument();
    });

    it("shows error styling for failed actions", () => {
      render(<ActionPanel {...defaultProps} />);

      // Error actions should have red styling
      const errorItem = screen.getByText("Export failed");
      const actionItemDiv = errorItem.closest(".border-red-200");
      expect(actionItemDiv).toBeInTheDocument();
    });

    it("shows pending styling for in-progress actions", () => {
      render(<ActionPanel {...defaultProps} />);

      // Pending actions should have blue styling
      const pendingItem = screen.getByText("Bookmarking...");
      const actionItemDiv = pendingItem.closest(".border-blue-200");
      expect(actionItemDiv).toBeInTheDocument();
    });
  });

  describe("Interactions", () => {
    it("calls onClose when close button is clicked", () => {
      render(<ActionPanel {...defaultProps} />);

      fireEvent.click(screen.getByLabelText("Close action history"));

      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it("calls onClearHistory when clear button is clicked", () => {
      render(<ActionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText("Clear"));

      expect(defaultProps.onClearHistory).toHaveBeenCalled();
    });
  });

  describe("Action Details", () => {
    it("shows share URL for share actions", () => {
      render(<ActionPanel {...defaultProps} />);

      expect(screen.getByText("https://share.test/abc")).toBeInTheDocument();
    });

    it("shows error message for failed actions", () => {
      render(<ActionPanel {...defaultProps} />);

      expect(screen.getByText("Network error")).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("has proper role for panel", () => {
      render(<ActionPanel {...defaultProps} />);

      expect(screen.getByRole("complementary")).toBeInTheDocument();
    });

    it("has aria-label for close button", () => {
      render(<ActionPanel {...defaultProps} />);

      expect(screen.getByLabelText("Close action history")).toBeInTheDocument();
    });
  });
});
