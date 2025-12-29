/**
 * Tests for SourceCard component.
 * Story 6-3: Generative UI Components
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { SourceCard } from "../../../../components/copilot/components/SourceCard";
import type { Source } from "../../../../types/copilot";

// Mock lucide-react icons
jest.mock("lucide-react", () => ({
  FileText: () => <span data-testid="icon-file-text">FileText</span>,
  Globe: () => <span data-testid="icon-globe">Globe</span>,
  Database: () => <span data-testid="icon-database">Database</span>,
  Share2: () => <span data-testid="icon-share2">Share2</span>,
  ExternalLink: () => <span data-testid="icon-external-link">ExternalLink</span>,
  CheckCircle2: () => <span data-testid="icon-check-circle">CheckCircle2</span>,
}));

const mockSource: Source = {
  id: "source-1",
  title: "Test Source Title",
  preview: "This is a preview of the source content that should be displayed.",
  similarity: 0.85,
  metadata: { type: "document" },
};

describe("SourceCard", () => {
  it("renders source title", () => {
    render(<SourceCard source={mockSource} />);
    expect(screen.getByText("Test Source Title")).toBeInTheDocument();
  });

  it("renders source preview", () => {
    render(<SourceCard source={mockSource} />);
    expect(screen.getByText(/This is a preview of the source content/)).toBeInTheDocument();
  });

  it("renders confidence percentage badge", () => {
    render(<SourceCard source={mockSource} />);
    expect(screen.getByText("85%")).toBeInTheDocument();
  });

  describe("confidence color coding", () => {
    it("renders emerald color for >= 90% confidence", () => {
      const highConfidenceSource = { ...mockSource, similarity: 0.95 };
      render(<SourceCard source={highConfidenceSource} />);
      const badge = screen.getByText("95%");
      expect(badge).toHaveClass("bg-emerald-100");
      expect(badge).toHaveClass("text-emerald-800");
    });

    it("renders indigo color for >= 70% confidence", () => {
      const medHighSource = { ...mockSource, similarity: 0.75 };
      render(<SourceCard source={medHighSource} />);
      const badge = screen.getByText("75%");
      expect(badge).toHaveClass("bg-indigo-100");
      expect(badge).toHaveClass("text-indigo-800");
    });

    it("renders amber color for >= 50% confidence", () => {
      const medSource = { ...mockSource, similarity: 0.55 };
      render(<SourceCard source={medSource} />);
      const badge = screen.getByText("55%");
      expect(badge).toHaveClass("bg-amber-100");
      expect(badge).toHaveClass("text-amber-800");
    });

    it("renders slate color for < 50% confidence", () => {
      const lowSource = { ...mockSource, similarity: 0.3 };
      render(<SourceCard source={lowSource} />);
      const badge = screen.getByText("30%");
      expect(badge).toHaveClass("bg-slate-100");
      expect(badge).toHaveClass("text-slate-800");
    });
  });

  describe("source type icons", () => {
    it("renders document icon for document type", () => {
      render(<SourceCard source={mockSource} />);
      expect(screen.getByTestId("icon-file-text")).toBeInTheDocument();
    });

    it("renders globe icon for web type", () => {
      const webSource = { ...mockSource, metadata: { type: "web" } };
      render(<SourceCard source={webSource} />);
      expect(screen.getByTestId("icon-globe")).toBeInTheDocument();
    });

    it("renders database icon for database type", () => {
      const dbSource = { ...mockSource, metadata: { type: "database" } };
      render(<SourceCard source={dbSource} />);
      expect(screen.getByTestId("icon-database")).toBeInTheDocument();
    });

    it("renders share2 icon for knowledge_graph type", () => {
      const kgSource = { ...mockSource, metadata: { type: "knowledge_graph" } };
      render(<SourceCard source={kgSource} />);
      expect(screen.getByTestId("icon-share2")).toBeInTheDocument();
    });

    it("renders default file icon when no type specified", () => {
      const noTypeSource = { ...mockSource, metadata: undefined };
      render(<SourceCard source={noTypeSource} />);
      expect(screen.getByTestId("icon-file-text")).toBeInTheDocument();
    });
  });

  describe("click handling", () => {
    it("calls onClick when card is clicked", () => {
      const handleClick = jest.fn();
      render(<SourceCard source={mockSource} onClick={handleClick} />);
      const card = screen.getByRole("button");
      fireEvent.click(card);
      expect(handleClick).toHaveBeenCalledWith(mockSource);
    });

    it("calls onClick when Enter key is pressed", () => {
      const handleClick = jest.fn();
      render(<SourceCard source={mockSource} onClick={handleClick} />);
      const card = screen.getByRole("button");
      fireEvent.keyDown(card, { key: "Enter" });
      expect(handleClick).toHaveBeenCalledWith(mockSource);
    });

    it("calls onClick when Space key is pressed", () => {
      const handleClick = jest.fn();
      render(<SourceCard source={mockSource} onClick={handleClick} />);
      const card = screen.getByRole("button");
      fireEvent.keyDown(card, { key: " " });
      expect(handleClick).toHaveBeenCalledWith(mockSource);
    });
  });

  describe("index display", () => {
    it("renders index number when provided", () => {
      render(<SourceCard source={mockSource} index={0} />);
      expect(screen.getByText("1")).toBeInTheDocument();
    });

    it("does not render index when not provided", () => {
      render(<SourceCard source={mockSource} />);
      expect(screen.queryByText("1")).not.toBeInTheDocument();
    });
  });

  describe("approval status", () => {
    it("shows approval icon when source is approved and showApprovalStatus is true", () => {
      const approvedSource = { ...mockSource, isApproved: true };
      render(<SourceCard source={approvedSource} showApprovalStatus={true} />);
      expect(screen.getByTestId("icon-check-circle")).toBeInTheDocument();
    });

    it("does not show approval icon when showApprovalStatus is false", () => {
      const approvedSource = { ...mockSource, isApproved: true };
      render(<SourceCard source={approvedSource} showApprovalStatus={false} />);
      expect(screen.queryByTestId("icon-check-circle")).not.toBeInTheDocument();
    });
  });

  describe("external link", () => {
    it("renders external link when URL is in metadata", () => {
      const sourceWithUrl = {
        ...mockSource,
        metadata: { type: "web", url: "https://example.com" },
      };
      render(<SourceCard source={sourceWithUrl} />);
      const link = screen.getByRole("link", { name: /view source/i });
      expect(link).toHaveAttribute("href", "https://example.com");
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    });

    it("does not render external link when no URL in metadata", () => {
      render(<SourceCard source={mockSource} />);
      expect(screen.queryByRole("link")).not.toBeInTheDocument();
    });
  });

  describe("highlighting", () => {
    it("applies highlight styles when isHighlighted is true", () => {
      render(<SourceCard source={mockSource} isHighlighted={true} />);
      const card = screen.getByRole("button");
      expect(card).toHaveClass("ring-2");
      expect(card).toHaveClass("ring-indigo-500");
    });
  });

  describe("accessibility", () => {
    it("has proper aria-label", () => {
      render(<SourceCard source={mockSource} />);
      const card = screen.getByRole("button");
      expect(card).toHaveAttribute(
        "aria-label",
        "Source: Test Source Title, confidence 85%"
      );
    });

    it("is focusable via keyboard", () => {
      render(<SourceCard source={mockSource} />);
      const card = screen.getByRole("button");
      expect(card).toHaveAttribute("tabIndex", "0");
    });
  });
});
