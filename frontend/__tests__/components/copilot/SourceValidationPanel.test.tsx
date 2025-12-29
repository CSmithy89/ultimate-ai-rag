/**
 * Tests for SourceValidationPanel component
 * Story 6-4: Human-in-the-Loop Source Validation
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { SourceValidationPanel } from "@/components/copilot/SourceValidationPanel";
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
];

describe("SourceValidationPanel", () => {
  const defaultProps = {
    sources: mockSources,
    onSubmit: jest.fn(),
    onSkip: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Rendering", () => {
    it("renders panel with header", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      expect(screen.getByText(/review sources/i)).toBeInTheDocument();
      // Check that source count is displayed (text is split across elements)
      expect(screen.getByText("2", { exact: true })).toBeInTheDocument();
    });

    it("renders all source cards when expanded", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      expect(screen.getByText("Document One")).toBeInTheDocument();
      expect(screen.getByText("Document Two")).toBeInTheDocument();
    });

    it("applies amber border for HITL attention", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      // The outer container div has the amber border
      const container = screen.getByText(/review sources/i).closest("button")?.parentElement;
      expect(container).toHaveClass("border-amber-400");
    });
  });

  describe("Collapse/Expand", () => {
    it("starts expanded by default", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      // Source cards should be visible
      expect(screen.getByText("Document One")).toBeInTheDocument();
    });

    it("starts collapsed when defaultCollapsed is true", () => {
      render(<SourceValidationPanel {...defaultProps} defaultCollapsed={true} />);

      // Source cards should not be visible
      expect(screen.queryByText("Document One")).not.toBeInTheDocument();
    });

    it("collapses when header is clicked", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      // Click header to collapse
      fireEvent.click(screen.getByText(/review sources/i));

      // Source cards should no longer be visible
      expect(screen.queryByText("Document One")).not.toBeInTheDocument();
    });

    it("expands when header is clicked while collapsed", () => {
      render(<SourceValidationPanel {...defaultProps} defaultCollapsed={true} />);

      // Click header to expand
      fireEvent.click(screen.getByText(/review sources/i));

      // Source cards should be visible
      expect(screen.getByText("Document One")).toBeInTheDocument();
    });

    it("shows mini stats when collapsed", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      // Approve one source first
      const approveButtons = screen.getAllByRole("button", {
        name: /approve source/i,
      });
      fireEvent.click(approveButtons[0]);

      // Collapse
      fireEvent.click(screen.getByText(/review sources/i));

      // Should show mini stats
      expect(screen.getByText("1 pending")).toBeInTheDocument();
    });
  });

  describe("Statistics", () => {
    it("shows statistics when expanded", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      expect(screen.getByText("2 pending")).toBeInTheDocument();
      expect(screen.getByText("0 approved")).toBeInTheDocument();
      expect(screen.getByText("0 rejected")).toBeInTheDocument();
    });

    it("updates statistics when sources are approved", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      const approveButtons = screen.getAllByRole("button", {
        name: /approve source/i,
      });
      fireEvent.click(approveButtons[0]);

      expect(screen.getByText("1 pending")).toBeInTheDocument();
      expect(screen.getByText("1 approved")).toBeInTheDocument();
    });
  });

  describe("Quick Actions", () => {
    it("approve all button approves all pending sources", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      fireEvent.click(screen.getByText("Approve All"));

      expect(screen.getByText("0 pending")).toBeInTheDocument();
      expect(screen.getByText("2 approved")).toBeInTheDocument();
    });

    it("reject all button rejects all pending sources", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      fireEvent.click(screen.getByText("Reject All"));

      expect(screen.getByText("0 pending")).toBeInTheDocument();
      expect(screen.getByText("2 rejected")).toBeInTheDocument();
    });
  });

  describe("Submit Behavior", () => {
    it("submit button is disabled when no sources approved", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      const submitButton = screen.getByRole("button", {
        name: /continue with 0 sources/i,
      });
      expect(submitButton).toBeDisabled();
    });

    it("submit button is enabled when at least one source approved", () => {
      render(<SourceValidationPanel {...defaultProps} />);

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
      render(<SourceValidationPanel {...defaultProps} />);

      // Approve first source
      const approveButtons = screen.getAllByRole("button", {
        name: /approve source/i,
      });
      fireEvent.click(approveButtons[0]);

      // Submit
      const submitButton = screen.getByRole("button", {
        name: /continue with 1 source$/i,
      });
      fireEvent.click(submitButton);

      expect(defaultProps.onSubmit).toHaveBeenCalledWith(["source-1"]);
    });
  });

  describe("Skip", () => {
    it("skip button calls onSkip when provided", () => {
      render(<SourceValidationPanel {...defaultProps} />);

      fireEvent.click(screen.getByText("Skip & Use All"));

      expect(defaultProps.onSkip).toHaveBeenCalled();
    });

    it("skip button calls onSubmit with all IDs when onSkip not provided", () => {
      const propsWithoutSkip = {
        sources: mockSources,
        onSubmit: jest.fn(),
      };

      render(<SourceValidationPanel {...propsWithoutSkip} />);

      fireEvent.click(screen.getByText("Skip & Use All"));

      expect(propsWithoutSkip.onSubmit).toHaveBeenCalledWith([
        "source-1",
        "source-2",
      ]);
    });
  });
});
