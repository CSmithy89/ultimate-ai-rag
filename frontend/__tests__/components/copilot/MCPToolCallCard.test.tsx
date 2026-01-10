/**
 * Tests for MCPToolCallCard component.
 * Story 21-A3: Implement Tool Call Visualization (AC1, AC2, AC3, AC4, AC7, AC8)
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { MCPToolCallCard } from "../../../components/copilot/MCPToolCallCard";

// Mock lucide-react icons
jest.mock("lucide-react", () => ({
  ChevronDown: ({ className, "data-testid": testId }: { className?: string; "data-testid"?: string }) => (
    <span className={className} data-testid={testId || "icon-chevron"}>
      ChevronDown
    </span>
  ),
  Loader2: ({ className, "data-testid": testId }: { className?: string; "data-testid"?: string }) => (
    <span className={className} data-testid={testId || "icon-loader"}>
      Loader2
    </span>
  ),
  Play: ({ className, "data-testid": testId }: { className?: string; "data-testid"?: string }) => (
    <span className={className} data-testid={testId || "icon-play"}>
      Play
    </span>
  ),
  CheckCircle: ({ className, "data-testid": testId }: { className?: string; "data-testid"?: string }) => (
    <span className={className} data-testid={testId || "icon-check"}>
      CheckCircle
    </span>
  ),
}));

describe("MCPToolCallCard", () => {
  const defaultProps = {
    name: "vector_search",
    args: { query: "test query", limit: 10 },
    status: "executing" as const,
  };

  describe("rendering", () => {
    it("renders tool name in header", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      expect(screen.getByText("vector_search")).toBeInTheDocument();
    });

    it("renders StatusBadge with correct status", () => {
      render(<MCPToolCallCard {...defaultProps} status="executing" />);
      expect(screen.getByText("Running")).toBeInTheDocument();
    });

    it("renders card container", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      expect(screen.getByTestId("mcp-tool-call-card")).toBeInTheDocument();
    });
  });

  describe("collapse/expand behavior", () => {
    it("starts collapsed by default", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      expect(screen.queryByTestId("tool-args")).not.toBeInTheDocument();
    });

    it("expands when header is clicked", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      expect(screen.getByTestId("tool-args")).toBeInTheDocument();
    });

    it("collapses when header is clicked again", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      fireEvent.click(header); // expand
      expect(screen.getByTestId("tool-args")).toBeInTheDocument();
      fireEvent.click(header); // collapse
      expect(screen.queryByTestId("tool-args")).not.toBeInTheDocument();
    });

    it("shows arguments when expanded", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const argsContent = screen.getByTestId("tool-args");
      expect(argsContent).toHaveTextContent("test query");
      expect(argsContent).toHaveTextContent("10");
    });

    it("rotates chevron icon when expanded", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const chevron = screen.getByTestId("icon-chevron");
      expect(chevron).toHaveClass("rotate-180");
    });
  });

  describe("keyboard navigation", () => {
    it("expands when Enter key is pressed", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      fireEvent.keyDown(header, { key: "Enter" });
      expect(screen.getByTestId("tool-args")).toBeInTheDocument();
    });

    it("expands when Space key is pressed", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      fireEvent.keyDown(header, { key: " " });
      expect(screen.getByTestId("tool-args")).toBeInTheDocument();
    });
  });

  describe("result display", () => {
    it("shows result only when status is complete", () => {
      const props = {
        ...defaultProps,
        status: "complete" as const,
        result: { documents: [{ id: 1, title: "Doc 1" }] },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      expect(screen.getByTestId("tool-result")).toBeInTheDocument();
    });

    it("does not show result when status is executing", () => {
      const props = {
        ...defaultProps,
        status: "executing" as const,
        result: { documents: [] },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      expect(screen.queryByTestId("tool-result")).not.toBeInTheDocument();
    });

    it("does not show result when status is inProgress", () => {
      const props = {
        ...defaultProps,
        status: "inProgress" as const,
        result: { documents: [] },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      expect(screen.queryByTestId("tool-result")).not.toBeInTheDocument();
    });
  });

  describe("sensitive data redaction", () => {
    it("redacts password in args", () => {
      const props = {
        ...defaultProps,
        args: { query: "test", password: "secret123" },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const argsContent = screen.getByTestId("tool-args");
      expect(argsContent).toHaveTextContent("[REDACTED]");
      expect(argsContent).not.toHaveTextContent("secret123");
    });

    it("redacts api_key in args", () => {
      const props = {
        ...defaultProps,
        args: { query: "test", api_key: "sk-12345" },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const argsContent = screen.getByTestId("tool-args");
      expect(argsContent).toHaveTextContent("[REDACTED]");
      expect(argsContent).not.toHaveTextContent("sk-12345");
    });

    it("redacts token in result", () => {
      const props = {
        ...defaultProps,
        status: "complete" as const,
        result: { data: "value", token: "abc123" },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const resultContent = screen.getByTestId("tool-result");
      expect(resultContent).toHaveTextContent("[REDACTED]");
      expect(resultContent).not.toHaveTextContent("abc123");
    });

    it("preserves non-sensitive args", () => {
      const props = {
        ...defaultProps,
        args: { query: "test query", limit: 10, enabled: true },
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const argsContent = screen.getByTestId("tool-args");
      expect(argsContent).toHaveTextContent("test query");
      expect(argsContent).toHaveTextContent("10");
      expect(argsContent).toHaveTextContent("true");
    });
  });

  describe("result truncation", () => {
    it("truncates long results to 500 characters", () => {
      const longResult = { data: "x".repeat(600) };
      const props = {
        ...defaultProps,
        status: "complete" as const,
        result: longResult,
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const resultContent = screen.getByTestId("tool-result");
      // Should show truncation indicator
      expect(resultContent).toHaveTextContent("...");
    });

    it("does not truncate short results", () => {
      const shortResult = { data: "short" };
      const props = {
        ...defaultProps,
        status: "complete" as const,
        result: shortResult,
      };
      render(<MCPToolCallCard {...props} />);
      const header = screen.getByRole("button");
      fireEvent.click(header);
      const resultContent = screen.getByTestId("tool-result");
      // Should not show truncation indicator
      const childSpans = resultContent.querySelectorAll("span");
      const truncationSpan = Array.from(childSpans).find((span) =>
        span.textContent?.includes("...")
      );
      expect(truncationSpan).toBeUndefined();
    });
  });

  describe("accessibility", () => {
    it("has aria-expanded attribute", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      expect(header).toHaveAttribute("aria-expanded", "false");
      fireEvent.click(header);
      expect(header).toHaveAttribute("aria-expanded", "true");
    });

    it("has aria-controls attribute", () => {
      render(<MCPToolCallCard {...defaultProps} />);
      const header = screen.getByRole("button");
      expect(header).toHaveAttribute("aria-controls", "tool-content-vector_search");
    });
  });

  describe("className prop", () => {
    it("accepts and applies className prop", () => {
      render(<MCPToolCallCard {...defaultProps} className="custom-class" />);
      const card = screen.getByTestId("mcp-tool-call-card");
      expect(card).toHaveClass("custom-class");
    });
  });

  describe("status transitions", () => {
    it("handles transition from inProgress to executing", () => {
      const { rerender } = render(
        <MCPToolCallCard {...defaultProps} status="inProgress" />
      );
      expect(screen.getByText("Preparing")).toBeInTheDocument();

      rerender(<MCPToolCallCard {...defaultProps} status="executing" />);
      expect(screen.getByText("Running")).toBeInTheDocument();
    });

    it("handles transition from executing to complete", () => {
      const { rerender } = render(
        <MCPToolCallCard {...defaultProps} status="executing" />
      );
      expect(screen.getByText("Running")).toBeInTheDocument();

      rerender(
        <MCPToolCallCard
          {...defaultProps}
          status="complete"
          result={{ success: true }}
        />
      );
      expect(screen.getByText("Complete")).toBeInTheDocument();
    });

    it("handles PascalCase status variants", () => {
      const { rerender } = render(
        <MCPToolCallCard {...defaultProps} status="InProgress" />
      );
      expect(screen.getByText("Preparing")).toBeInTheDocument();

      rerender(<MCPToolCallCard {...defaultProps} status="Executing" />);
      expect(screen.getByText("Running")).toBeInTheDocument();

      rerender(
        <MCPToolCallCard
          {...defaultProps}
          status="Complete"
          result={{ success: true }}
        />
      );
      expect(screen.getByText("Complete")).toBeInTheDocument();
    });
  });
});
