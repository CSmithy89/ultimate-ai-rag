/**
 * Tests for SourceValidationDialog component
 * Story 6-4: Human-in-the-Loop Source Validation
 */

import React from "react";
import { render, screen, fireEvent, within } from "@testing-library/react";
import "@testing-library/jest-dom";
import { SourceValidationDialog } from "@/components/copilot/SourceValidationDialog";
import type { Source } from "@/types/copilot";

// Mock sources
const mockSources: Source[] = [
  {
    id: "source-1",
    title: "Document One",
    preview: "Preview of document one content.",
    similarity: 0.95,
    metadata: { type: "document" },
  },
  {
    id: "source-2",
    title: "Document Two",
    preview: "Preview of document two content.",
    similarity: 0.75,
    metadata: { type: "web" },
  },
  {
    id: "source-3",
    title: "Document Three",
    preview: "Preview of document three content.",
    similarity: 0.55,
    metadata: { type: "database" },
  },
];

describe("SourceValidationDialog", () => {
  const defaultProps = {
    open: true,
    sources: mockSources,
    onSubmit: jest.fn(),
    onCancel: jest.fn(),
    isSubmitting: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Rendering", () => {
    it("renders dialog when open is true", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByText("Review Retrieved Sources")).toBeInTheDocument();
    });

    it("does not render dialog when open is false", () => {
      render(<SourceValidationDialog {...defaultProps} open={false} />);

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("renders all source cards", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      expect(screen.getByText("Document One")).toBeInTheDocument();
      expect(screen.getByText("Document Two")).toBeInTheDocument();
      expect(screen.getByText("Document Three")).toBeInTheDocument();
    });

    it("renders custom title and description", () => {
      render(
        <SourceValidationDialog
          {...defaultProps}
          title="Custom Title"
          description="Custom description text."
        />
      );

      expect(screen.getByText("Custom Title")).toBeInTheDocument();
      expect(screen.getByText("Custom description text.")).toBeInTheDocument();
    });
  });

  describe("Statistics", () => {
    it("shows initial pending count", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      expect(screen.getByText("3 pending")).toBeInTheDocument();
      expect(screen.getByText("0 approved")).toBeInTheDocument();
      expect(screen.getByText("0 rejected")).toBeInTheDocument();
    });

    it("updates statistics when sources are approved", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      // Click approve on first source
      const approveButtons = screen.getAllByRole("button", {
        name: /approve source/i,
      });
      fireEvent.click(approveButtons[0]);

      expect(screen.getByText("2 pending")).toBeInTheDocument();
      expect(screen.getByText("1 approved")).toBeInTheDocument();
    });

    it("updates statistics when sources are rejected", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      // Click reject on first source
      const rejectButtons = screen.getAllByRole("button", {
        name: /reject source/i,
      });
      fireEvent.click(rejectButtons[0]);

      expect(screen.getByText("2 pending")).toBeInTheDocument();
      expect(screen.getByText("1 rejected")).toBeInTheDocument();
    });
  });

  describe("Quick Actions", () => {
    it("approve all button approves all pending sources", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      fireEvent.click(screen.getByText("Approve All Pending"));

      expect(screen.getByText("0 pending")).toBeInTheDocument();
      expect(screen.getByText("3 approved")).toBeInTheDocument();
    });

    it("reject all button rejects all pending sources", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      fireEvent.click(screen.getByText("Reject All Pending"));

      expect(screen.getByText("0 pending")).toBeInTheDocument();
      expect(screen.getByText("3 rejected")).toBeInTheDocument();
    });

    it("reset button returns all to pending", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      // First approve all
      fireEvent.click(screen.getByText("Approve All Pending"));
      expect(screen.getByText("3 approved")).toBeInTheDocument();

      // Then reset
      fireEvent.click(screen.getByText("Reset"));
      expect(screen.getByText("3 pending")).toBeInTheDocument();
    });

    it("disables approve all when no pending sources", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      fireEvent.click(screen.getByText("Approve All Pending"));

      expect(screen.getByText("Approve All Pending")).toBeDisabled();
    });

    it("disables reject all when no pending sources", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      fireEvent.click(screen.getByText("Reject All Pending"));

      expect(screen.getByText("Reject All Pending")).toBeDisabled();
    });
  });

  describe("Submit Behavior", () => {
    it("submit button is disabled when no sources approved", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      const submitButton = screen.getByRole("button", {
        name: /continue with 0 sources/i,
      });
      expect(submitButton).toBeDisabled();
    });

    it("submit button is enabled when at least one source approved", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      const approveButtons = screen.getAllByRole("button", {
        name: /approve source/i,
      });
      fireEvent.click(approveButtons[0]);

      const submitButton = screen.getByRole("button", {
        name: /continue with 1 source$/i,
      });
      expect(submitButton).toBeEnabled();
    });

    it("calls onSubmit with approved source IDs", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      // Approve first two sources
      const approveButtons = screen.getAllByRole("button", {
        name: /approve source/i,
      });
      fireEvent.click(approveButtons[0]);
      fireEvent.click(approveButtons[1]);

      // Submit
      const submitButton = screen.getByRole("button", {
        name: /continue with 2 sources/i,
      });
      fireEvent.click(submitButton);

      expect(defaultProps.onSubmit).toHaveBeenCalledWith([
        "source-1",
        "source-2",
      ]);
    });

    it("shows loading state when submitting", () => {
      render(<SourceValidationDialog {...defaultProps} isSubmitting={true} />);

      expect(screen.getByText("Submitting...")).toBeInTheDocument();
    });
  });

  describe("Skip and Cancel", () => {
    it("skip button submits all source IDs", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      fireEvent.click(screen.getByText("Skip & Use All"));

      expect(defaultProps.onSubmit).toHaveBeenCalledWith([
        "source-1",
        "source-2",
        "source-3",
      ]);
    });

    it("cancel button calls onCancel", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      fireEvent.click(screen.getByText("Cancel"));

      expect(defaultProps.onCancel).toHaveBeenCalled();
    });
  });

  describe("Source Card Interactions", () => {
    it("clicking source card toggles validation status", () => {
      render(<SourceValidationDialog {...defaultProps} />);

      // Find the first source card and click it
      const sourceCards = screen.getAllByRole("button", {
        name: /source \d+:/i,
      });
      
      // Initial state is pending, click to approve
      fireEvent.click(sourceCards[0]);
      expect(screen.getByText("1 approved")).toBeInTheDocument();

      // Click again to reject
      fireEvent.click(sourceCards[0]);
      expect(screen.getByText("1 rejected")).toBeInTheDocument();

      // Click again to return to pending
      fireEvent.click(sourceCards[0]);
      expect(screen.getByText("3 pending")).toBeInTheDocument();
    });
  });
});
