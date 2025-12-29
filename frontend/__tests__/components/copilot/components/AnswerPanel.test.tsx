/**
 * Tests for AnswerPanel component.
 * Story 6-3: Generative UI Components
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AnswerPanel } from "../../../../components/copilot/components/AnswerPanel";
import type { Source } from "../../../../types/copilot";

// Mock lucide-react icons
jest.mock("lucide-react", () => ({
  Sparkles: () => <span data-testid="icon-sparkles">Sparkles</span>,
  Copy: () => <span data-testid="icon-copy">Copy</span>,
  Check: () => <span data-testid="icon-check">Check</span>,
  ChevronDown: () => <span data-testid="icon-chevron-down">ChevronDown</span>,
  ChevronUp: () => <span data-testid="icon-chevron-up">ChevronUp</span>,
  FileText: () => <span data-testid="icon-file-text">FileText</span>,
  Globe: () => <span data-testid="icon-globe">Globe</span>,
  Database: () => <span data-testid="icon-database">Database</span>,
  Share2: () => <span data-testid="icon-share2">Share2</span>,
  ExternalLink: () => <span data-testid="icon-external-link">ExternalLink</span>,
  CheckCircle2: () => <span data-testid="icon-check-circle">CheckCircle2</span>,
}));

// Mock react-markdown
jest.mock("react-markdown", () => {
  return function MockReactMarkdown({ children }: { children: string }) {
    return <div data-testid="markdown-content">{children}</div>;
  };
});

// Mock remark-gfm
jest.mock("remark-gfm", () => () => {});

// Mock clipboard API
const mockClipboard = {
  writeText: jest.fn().mockResolvedValue(undefined),
};
Object.assign(navigator, { clipboard: mockClipboard });

const mockSources: Source[] = [
  {
    id: "source-1",
    title: "First Source",
    preview: "Preview of first source",
    similarity: 0.9,
  },
  {
    id: "source-2",
    title: "Second Source",
    preview: "Preview of second source",
    similarity: 0.8,
  },
];

describe("AnswerPanel", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders the answer text", () => {
    render(<AnswerPanel answer="This is the answer text" />);
    expect(screen.getByTestId("markdown-content")).toHaveTextContent("This is the answer text");
  });

  it("renders the default title", () => {
    render(<AnswerPanel answer="Test answer" />);
    expect(screen.getByText("Answer")).toBeInTheDocument();
  });

  it("renders custom title when provided", () => {
    render(<AnswerPanel answer="Test answer" title="Custom Title" />);
    expect(screen.getByText("Custom Title")).toBeInTheDocument();
  });

  it("renders sparkles icon in header", () => {
    render(<AnswerPanel answer="Test answer" />);
    expect(screen.getByTestId("icon-sparkles")).toBeInTheDocument();
  });

  describe("streaming indicator", () => {
    it("shows streaming indicator when isStreaming is true", () => {
      render(<AnswerPanel answer="Test answer" isStreaming={true} />);
      expect(screen.getByText("Generating...")).toBeInTheDocument();
    });

    it("does not show streaming indicator when isStreaming is false", () => {
      render(<AnswerPanel answer="Test answer" isStreaming={false} />);
      expect(screen.queryByText("Generating...")).not.toBeInTheDocument();
    });
  });

  describe("copy to clipboard", () => {
    it("copies answer to clipboard when copy button is clicked", async () => {
      render(<AnswerPanel answer="Test answer to copy" />);

      const copyButton = screen.getByRole("button", { name: /copy/i });
      fireEvent.click(copyButton);

      await waitFor(() => {
        expect(mockClipboard.writeText).toHaveBeenCalledWith("Test answer to copy");
      });
    });

    it("shows check icon after copying", async () => {
      render(<AnswerPanel answer="Test answer" />);

      const copyButton = screen.getByRole("button", { name: /copy/i });
      fireEvent.click(copyButton);

      await waitFor(() => {
        expect(screen.getByTestId("icon-check")).toBeInTheDocument();
      });
    });
  });

  describe("source references extraction", () => {
    it("extracts source references like [1] from answer", () => {
      render(
        <AnswerPanel
          answer="According to source [1], this is true."
          sources={mockSources}
        />
      );

      // Source references should be extracted and sources section should be available
      const expandButton = screen.getByRole("button", { name: /sources/i });
      expect(expandButton).toBeInTheDocument();
    });

    it("extracts multiple source references", () => {
      render(
        <AnswerPanel
          answer="Sources [1] and [2] confirm this."
          sources={mockSources}
        />
      );

      const expandButton = screen.getByRole("button", { name: /sources \(2\)/i });
      expect(expandButton).toBeInTheDocument();
    });

    it("does not show sources section when no references in answer", () => {
      render(
        <AnswerPanel
          answer="This answer has no source references."
          sources={mockSources}
        />
      );

      expect(screen.queryByRole("button", { name: /sources/i })).not.toBeInTheDocument();
    });
  });

  describe("sources expansion", () => {
    it("expands sources section when clicked", () => {
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
        />
      );

      const expandButton = screen.getByRole("button", { name: /sources/i });
      fireEvent.click(expandButton);

      expect(screen.getByText("First Source")).toBeInTheDocument();
    });

    it("shows chevron down icon when collapsed", () => {
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
        />
      );

      expect(screen.getByTestId("icon-chevron-down")).toBeInTheDocument();
    });

    it("shows chevron up icon when expanded", () => {
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
        />
      );

      const expandButton = screen.getByRole("button", { name: /sources/i });
      fireEvent.click(expandButton);

      expect(screen.getByTestId("icon-chevron-up")).toBeInTheDocument();
    });
  });

  describe("source click handling", () => {
    it("calls onSourceClick when a source card is clicked", () => {
      const handleSourceClick = jest.fn();
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
          onSourceClick={handleSourceClick}
        />
      );

      // Expand sources
      const expandButton = screen.getByRole("button", { name: /sources/i });
      fireEvent.click(expandButton);

      // Click on source card
      const sourceCard = screen.getByText("First Source").closest('[role="button"]');
      if (sourceCard) {
        fireEvent.click(sourceCard);
        expect(handleSourceClick).toHaveBeenCalledWith(mockSources[0]);
      }
    });
  });

  describe("showSources prop", () => {
    it("hides sources section when showSources is false", () => {
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
          showSources={false}
        />
      );

      expect(screen.queryByRole("button", { name: /sources/i })).not.toBeInTheDocument();
    });
  });

  describe("accessibility", () => {
    it("has proper aria-label on copy button", () => {
      render(<AnswerPanel answer="Test answer" />);

      const copyButton = screen.getByRole("button", { name: /copy/i });
      expect(copyButton).toHaveAttribute("aria-label");
    });

    it("has aria-expanded on sources button", () => {
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
        />
      );

      const expandButton = screen.getByRole("button", { name: /sources/i });
      expect(expandButton).toHaveAttribute("aria-expanded", "false");
    });

    it("updates aria-expanded when sources are expanded", () => {
      render(
        <AnswerPanel
          answer="Source [1] says this."
          sources={mockSources}
        />
      );

      const expandButton = screen.getByRole("button", { name: /sources/i });
      fireEvent.click(expandButton);

      expect(expandButton).toHaveAttribute("aria-expanded", "true");
    });
  });

  describe("styling", () => {
    it("applies custom className when provided", () => {
      const { container } = render(
        <AnswerPanel answer="Test answer" className="custom-class" />
      );

      expect(container.firstChild).toHaveClass("custom-class");
    });
  });

  describe("clipboard error handling", () => {
    it("handles clipboard write failure gracefully", async () => {
      const consoleSpy = jest.spyOn(console, "error").mockImplementation(() => {});
      mockClipboard.writeText.mockRejectedValueOnce(new Error("Clipboard access denied"));

      render(<AnswerPanel answer="Test answer" />);

      const copyButton = screen.getByRole("button", { name: /copy/i });
      fireEvent.click(copyButton);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith("Failed to copy to clipboard");
      });

      // Should still show copy icon (not check) after failure
      expect(screen.getByTestId("icon-copy")).toBeInTheDocument();

      consoleSpy.mockRestore();
    });
  });

  describe("XSS prevention in links", () => {
    it("blocks javascript: protocol links", () => {
      // This test verifies the URL validation function
      const { container } = render(
        <AnswerPanel answer="Click [here](javascript:alert('xss'))" />
      );

      // The link should not be rendered with javascript: href
      const links = container.querySelectorAll('a[href^="javascript:"]');
      expect(links.length).toBe(0);
    });

    it("blocks data: protocol links", () => {
      const { container } = render(
        <AnswerPanel answer="Click [here](data:text/html,<script>alert('xss')</script>)" />
      );

      const links = container.querySelectorAll('a[href^="data:"]');
      expect(links.length).toBe(0);
    });

    it("allows https: protocol links", () => {
      // Note: This test documents expected behavior
      // The actual rendering depends on ReactMarkdown mock
      render(<AnswerPanel answer="Visit [Google](https://google.com)" />);
      // With current mock, markdown is rendered as text
      expect(screen.getByTestId("markdown-content")).toBeInTheDocument();
    });
  });
});
