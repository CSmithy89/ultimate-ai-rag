/**
 * Tests for StatusBadge component.
 * Story 21-A3: Implement Tool Call Visualization (AC2, AC3, AC4)
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import {
  StatusBadge,
  isInProgress,
  isExecuting,
  isComplete,
  type ToolStatus,
} from "../../../components/copilot/StatusBadge";

// Mock lucide-react icons
jest.mock("lucide-react", () => ({
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

describe("StatusBadge", () => {
  describe("inProgress status", () => {
    it("renders Loader2 icon for lowercase inProgress", () => {
      render(<StatusBadge status="inProgress" />);
      expect(screen.getByTestId("icon-loader")).toBeInTheDocument();
      expect(screen.getByText("Preparing")).toBeInTheDocument();
    });

    it("renders Loader2 icon for PascalCase InProgress", () => {
      render(<StatusBadge status="InProgress" />);
      expect(screen.getByTestId("icon-loader")).toBeInTheDocument();
      expect(screen.getByText("Preparing")).toBeInTheDocument();
    });

    it("applies blue colors for inProgress", () => {
      render(<StatusBadge status="inProgress" />);
      const badge = screen.getByTestId("status-badge-inprogress");
      expect(badge).toHaveClass("bg-blue-100");
      expect(badge).toHaveClass("text-blue-800");
      expect(badge).toHaveClass("border-blue-200");
    });

    it("applies animate-spin to Loader2 icon", () => {
      render(<StatusBadge status="inProgress" />);
      const icon = screen.getByTestId("icon-loader");
      expect(icon).toHaveClass("animate-spin");
    });
  });

  describe("executing status", () => {
    it("renders Play icon for lowercase executing", () => {
      render(<StatusBadge status="executing" />);
      expect(screen.getByTestId("icon-play")).toBeInTheDocument();
      expect(screen.getByText("Running")).toBeInTheDocument();
    });

    it("renders Play icon for PascalCase Executing", () => {
      render(<StatusBadge status="Executing" />);
      expect(screen.getByTestId("icon-play")).toBeInTheDocument();
      expect(screen.getByText("Running")).toBeInTheDocument();
    });

    it("applies yellow colors for executing", () => {
      render(<StatusBadge status="executing" />);
      const badge = screen.getByTestId("status-badge-executing");
      expect(badge).toHaveClass("bg-yellow-100");
      expect(badge).toHaveClass("text-yellow-800");
      expect(badge).toHaveClass("border-yellow-200");
    });
  });

  describe("complete status", () => {
    it("renders CheckCircle icon for lowercase complete", () => {
      render(<StatusBadge status="complete" />);
      expect(screen.getByTestId("icon-check")).toBeInTheDocument();
      expect(screen.getByText("Complete")).toBeInTheDocument();
    });

    it("renders CheckCircle icon for PascalCase Complete", () => {
      render(<StatusBadge status="Complete" />);
      expect(screen.getByTestId("icon-check")).toBeInTheDocument();
      expect(screen.getByText("Complete")).toBeInTheDocument();
    });

    it("applies emerald colors for complete", () => {
      render(<StatusBadge status="complete" />);
      const badge = screen.getByTestId("status-badge-complete");
      expect(badge).toHaveClass("bg-emerald-100");
      expect(badge).toHaveClass("text-emerald-800");
      expect(badge).toHaveClass("border-emerald-200");
    });
  });

  describe("unknown status", () => {
    it("renders fallback badge with status text", () => {
      render(<StatusBadge status={"unknown" as ToolStatus} />);
      const badge = screen.getByTestId("status-badge-unknown");
      expect(badge).toHaveTextContent("unknown");
    });

    it("applies slate colors for unknown status", () => {
      render(<StatusBadge status={"unknown" as ToolStatus} />);
      const badge = screen.getByTestId("status-badge-unknown");
      expect(badge).toHaveClass("bg-slate-100");
      expect(badge).toHaveClass("text-slate-600");
    });
  });

  describe("className prop", () => {
    it("accepts and applies className prop for inProgress", () => {
      render(<StatusBadge status="inProgress" className="custom-class" />);
      const badge = screen.getByTestId("status-badge-inprogress");
      expect(badge).toHaveClass("custom-class");
    });

    it("accepts and applies className prop for executing", () => {
      render(<StatusBadge status="executing" className="custom-class" />);
      const badge = screen.getByTestId("status-badge-executing");
      expect(badge).toHaveClass("custom-class");
    });

    it("accepts and applies className prop for complete", () => {
      render(<StatusBadge status="complete" className="custom-class" />);
      const badge = screen.getByTestId("status-badge-complete");
      expect(badge).toHaveClass("custom-class");
    });
  });
});

describe("status helper functions", () => {
  describe("isInProgress", () => {
    it("returns true for lowercase inProgress", () => {
      expect(isInProgress("inProgress")).toBe(true);
    });

    it("returns true for PascalCase InProgress", () => {
      expect(isInProgress("InProgress")).toBe(true);
    });

    it("returns false for other statuses", () => {
      expect(isInProgress("executing")).toBe(false);
      expect(isInProgress("complete")).toBe(false);
      expect(isInProgress("Executing")).toBe(false);
      expect(isInProgress("Complete")).toBe(false);
    });
  });

  describe("isExecuting", () => {
    it("returns true for lowercase executing", () => {
      expect(isExecuting("executing")).toBe(true);
    });

    it("returns true for PascalCase Executing", () => {
      expect(isExecuting("Executing")).toBe(true);
    });

    it("returns false for other statuses", () => {
      expect(isExecuting("inProgress")).toBe(false);
      expect(isExecuting("complete")).toBe(false);
      expect(isExecuting("InProgress")).toBe(false);
      expect(isExecuting("Complete")).toBe(false);
    });
  });

  describe("isComplete", () => {
    it("returns true for lowercase complete", () => {
      expect(isComplete("complete")).toBe(true);
    });

    it("returns true for PascalCase Complete", () => {
      expect(isComplete("Complete")).toBe(true);
    });

    it("returns false for other statuses", () => {
      expect(isComplete("inProgress")).toBe(false);
      expect(isComplete("executing")).toBe(false);
      expect(isComplete("InProgress")).toBe(false);
      expect(isComplete("Executing")).toBe(false);
    });
  });
});
