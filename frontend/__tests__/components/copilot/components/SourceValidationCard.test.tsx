/**
 * Tests for SourceValidationCard component
 * Story 6-4: Human-in-the-Loop Source Validation
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { SourceValidationCard } from "@/components/copilot/components/SourceValidationCard";
import type { Source } from "@/types/copilot";

// Mock source data
const mockSource: Source = {
  id: "source-1",
  title: "Test Document",
  preview: "This is a preview of the test document content.",
  similarity: 0.85,
  metadata: {
    type: "document",
    url: "https://example.com/doc",
  },
};

describe("SourceValidationCard", () => {
  const defaultProps = {
    source: mockSource,
    index: 0,
    validationStatus: "pending" as const,
    onToggle: jest.fn(),
    onApprove: jest.fn(),
    onReject: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Rendering", () => {
    it("renders source title and preview", () => {
      render(<SourceValidationCard {...defaultProps} />);

      expect(screen.getByText("Test Document")).toBeInTheDocument();
      expect(
        screen.getByText("This is a preview of the test document content.")
      ).toBeInTheDocument();
    });

    it("renders the source index badge", () => {
      render(<SourceValidationCard {...defaultProps} index={2} />);

      expect(screen.getByText("3")).toBeInTheDocument();
    });

    it("renders confidence percentage", () => {
      render(<SourceValidationCard {...defaultProps} />);

      expect(screen.getByText("85% match")).toBeInTheDocument();
    });

    it("renders external link when URL is provided", () => {
      render(<SourceValidationCard {...defaultProps} />);

      const link = screen.getByRole("link", { name: /view source/i });
      expect(link).toHaveAttribute("href", "https://example.com/doc");
      expect(link).toHaveAttribute("target", "_blank");
    });

    it("does not render external link when URL is not provided", () => {
      const sourceWithoutUrl: Source = {
        ...mockSource,
        metadata: { type: "document" },
      };
      render(
        <SourceValidationCard {...defaultProps} source={sourceWithoutUrl} />
      );

      expect(
        screen.queryByRole("link", { name: /view source/i })
      ).not.toBeInTheDocument();
    });
  });

  describe("Validation Status Styling", () => {
    it("renders with amber styling for pending status", () => {
      render(
        <SourceValidationCard {...defaultProps} validationStatus="pending" />
      );

      // Get card by aria-label (main interactive container)
      const card = screen.getByLabelText(/source 1:.*status: pending/i);
      expect(card).toHaveClass("border-amber-400");
    });

    it("renders with emerald styling for approved status", () => {
      render(
        <SourceValidationCard {...defaultProps} validationStatus="approved" />
      );

      const card = screen.getByLabelText(/source 1:.*status: approved/i);
      expect(card).toHaveClass("border-emerald-400");
    });

    it("renders with red styling for rejected status", () => {
      render(
        <SourceValidationCard {...defaultProps} validationStatus="rejected" />
      );

      const card = screen.getByLabelText(/source 1:.*status: rejected/i);
      expect(card).toHaveClass("border-red-300");
    });

    it("applies strikethrough to title when rejected", () => {
      render(
        <SourceValidationCard {...defaultProps} validationStatus="rejected" />
      );

      const title = screen.getByText("Test Document");
      expect(title).toHaveClass("line-through");
    });

    it("reduces opacity when rejected", () => {
      render(
        <SourceValidationCard {...defaultProps} validationStatus="rejected" />
      );

      const card = screen.getByLabelText(/source 1:.*status: rejected/i);
      expect(card).toHaveClass("opacity-60");
    });
  });

  describe("Interactions", () => {
    it("calls onToggle when card is clicked", () => {
      render(<SourceValidationCard {...defaultProps} />);

      const card = screen.getByLabelText(/source 1:/i);
      fireEvent.click(card);
      expect(defaultProps.onToggle).toHaveBeenCalledTimes(1);
    });

    it("calls onToggle when Enter key is pressed", () => {
      render(<SourceValidationCard {...defaultProps} />);

      const card = screen.getByLabelText(/source 1:/i);
      fireEvent.keyDown(card, { key: "Enter" });
      expect(defaultProps.onToggle).toHaveBeenCalledTimes(1);
    });

    it("calls onToggle when Space key is pressed", () => {
      render(<SourceValidationCard {...defaultProps} />);

      const card = screen.getByLabelText(/source 1:/i);
      fireEvent.keyDown(card, { key: " " });
      expect(defaultProps.onToggle).toHaveBeenCalledTimes(1);
    });

    it("calls onApprove when approve button is clicked", () => {
      render(<SourceValidationCard {...defaultProps} />);

      fireEvent.click(screen.getByRole("button", { name: /approve source/i }));
      expect(defaultProps.onApprove).toHaveBeenCalledTimes(1);
      expect(defaultProps.onToggle).not.toHaveBeenCalled();
    });

    it("calls onReject when reject button is clicked", () => {
      render(<SourceValidationCard {...defaultProps} />);

      fireEvent.click(screen.getByRole("button", { name: /reject source/i }));
      expect(defaultProps.onReject).toHaveBeenCalledTimes(1);
      expect(defaultProps.onToggle).not.toHaveBeenCalled();
    });

    it("does not trigger onToggle when external link is clicked", () => {
      render(<SourceValidationCard {...defaultProps} />);

      fireEvent.click(screen.getByRole("link", { name: /view source/i }));
      expect(defaultProps.onToggle).not.toHaveBeenCalled();
    });
  });

  describe("Accessibility", () => {
    it("has correct aria-label", () => {
      render(<SourceValidationCard {...defaultProps} />);

      const card = screen.getByLabelText(
        "Source 1: Test Document. Status: pending. Click to change."
      );
      expect(card).toBeInTheDocument();
    });

    it("is focusable", () => {
      render(<SourceValidationCard {...defaultProps} />);

      const card = screen.getByLabelText(/source 1:/i);
      expect(card).toHaveAttribute("tabIndex", "0");
    });

    it("approve button has aria-label", () => {
      render(<SourceValidationCard {...defaultProps} />);

      expect(
        screen.getByRole("button", { name: /approve source/i })
      ).toBeInTheDocument();
    });

    it("reject button has aria-label", () => {
      render(<SourceValidationCard {...defaultProps} />);

      expect(
        screen.getByRole("button", { name: /reject source/i })
      ).toBeInTheDocument();
    });
  });

  describe("Confidence Colors", () => {
    it("shows emerald color for high confidence (>= 90%)", () => {
      const highConfidenceSource = { ...mockSource, similarity: 0.95 };
      render(
        <SourceValidationCard {...defaultProps} source={highConfidenceSource} />
      );

      const badge = screen.getByText("95% match");
      expect(badge).toHaveClass("bg-emerald-100");
    });

    it("shows indigo color for medium-high confidence (>= 70%)", () => {
      const medHighSource = { ...mockSource, similarity: 0.75 };
      render(<SourceValidationCard {...defaultProps} source={medHighSource} />);

      const badge = screen.getByText("75% match");
      expect(badge).toHaveClass("bg-indigo-100");
    });

    it("shows amber color for medium confidence (>= 50%)", () => {
      const medSource = { ...mockSource, similarity: 0.55 };
      render(<SourceValidationCard {...defaultProps} source={medSource} />);

      const badge = screen.getByText("55% match");
      expect(badge).toHaveClass("bg-amber-100");
    });

    it("shows slate color for low confidence (< 50%)", () => {
      const lowSource = { ...mockSource, similarity: 0.35 };
      render(<SourceValidationCard {...defaultProps} source={lowSource} />);

      const badge = screen.getByText("35% match");
      expect(badge).toHaveClass("bg-slate-100");
    });
  });
});
