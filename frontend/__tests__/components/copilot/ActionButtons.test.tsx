/**
 * Tests for ActionButtons component
 * Story 6-5: Frontend Actions
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ActionButtons } from "@/components/copilot/ActionButtons";

describe("ActionButtons", () => {
  const defaultProps = {
    content: {
      id: "test-content-1",
      content: "Test AI response",
      title: "Test Title",
      query: "What is the answer?",
    },
    actionStates: {
      save: "idle" as const,
      export: "idle" as const,
      share: "idle" as const,
      bookmark: "idle" as const,
      followUp: "idle" as const,
    },
    onSave: jest.fn(),
    onExport: jest.fn(),
    onShare: jest.fn(),
    onBookmark: jest.fn(),
    onFollowUp: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Rendering", () => {
    it("renders all action buttons", () => {
      render(<ActionButtons {...defaultProps} />);

      expect(screen.getByLabelText("Save to workspace")).toBeInTheDocument();
      expect(screen.getByLabelText("Export")).toBeInTheDocument();
      expect(screen.getByLabelText("Share")).toBeInTheDocument();
      expect(screen.getByLabelText("Bookmark")).toBeInTheDocument();
      expect(screen.getByLabelText("Follow up")).toBeInTheDocument();
    });

    it("renders in compact mode with smaller buttons", () => {
      render(<ActionButtons {...defaultProps} compact />);

      const saveButton = screen.getByLabelText("Save to workspace");
      expect(saveButton).toHaveClass("p-1.5");
    });

    it("renders export dropdown when export button is clicked", () => {
      render(<ActionButtons {...defaultProps} />);

      const exportButton = screen.getByLabelText("Export");
      fireEvent.click(exportButton);

      expect(screen.getByText("Markdown")).toBeInTheDocument();
      expect(screen.getByText("JSON")).toBeInTheDocument();
      expect(screen.getByText("PDF")).toBeInTheDocument();
    });
  });

  describe("Click Handlers", () => {
    it("calls onSave when save button is clicked", () => {
      render(<ActionButtons {...defaultProps} />);

      fireEvent.click(screen.getByLabelText("Save to workspace"));

      expect(defaultProps.onSave).toHaveBeenCalledWith(defaultProps.content);
    });

    it("calls onExport with correct format when export option is clicked", () => {
      render(<ActionButtons {...defaultProps} />);

      const exportButton = screen.getByLabelText("Export");
      fireEvent.click(exportButton);

      fireEvent.click(screen.getByText("Markdown"));

      expect(defaultProps.onExport).toHaveBeenCalledWith(
        defaultProps.content,
        "markdown"
      );
    });

    it("calls onShare when share button is clicked", () => {
      render(<ActionButtons {...defaultProps} />);

      fireEvent.click(screen.getByLabelText("Share"));

      expect(defaultProps.onShare).toHaveBeenCalledWith(defaultProps.content);
    });

    it("calls onBookmark when bookmark button is clicked", () => {
      render(<ActionButtons {...defaultProps} />);

      fireEvent.click(screen.getByLabelText("Bookmark"));

      expect(defaultProps.onBookmark).toHaveBeenCalledWith(defaultProps.content);
    });

    it("calls onFollowUp when follow-up button is clicked", () => {
      render(<ActionButtons {...defaultProps} />);

      fireEvent.click(screen.getByLabelText("Follow up"));

      expect(defaultProps.onFollowUp).toHaveBeenCalledWith(defaultProps.content);
    });
  });

  describe("Loading States", () => {
    it("shows loading spinner when save is loading", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, save: "loading" }}
        />
      );

      const saveButton = screen.getByLabelText("Save to workspace");
      expect(saveButton.querySelector(".animate-spin")).toBeInTheDocument();
    });

    it("disables save button when save is loading", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, save: "loading" }}
        />
      );

      expect(screen.getByLabelText("Save to workspace")).toBeDisabled();
    });

    it("disables export button when export is loading", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, export: "loading" }}
        />
      );

      expect(screen.getByLabelText("Export")).toBeDisabled();
    });

    it("disables share button when share is loading", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, share: "loading" }}
        />
      );

      expect(screen.getByLabelText("Share")).toBeDisabled();
    });

    it("disables bookmark button when bookmark is loading", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, bookmark: "loading" }}
        />
      );

      expect(screen.getByLabelText("Bookmark")).toBeDisabled();
    });
  });

  describe("Success States", () => {
    it("shows success styling when save succeeds", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, save: "success" }}
        />
      );

      const saveButton = screen.getByLabelText("Save to workspace");
      expect(saveButton).toHaveClass("text-emerald-600");
    });

    it("shows success styling when bookmark succeeds", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, bookmark: "success" }}
        />
      );

      const bookmarkButton = screen.getByLabelText("Bookmark");
      expect(bookmarkButton).toHaveClass("text-emerald-600");
    });
  });

  describe("Error States", () => {
    it("shows error styling when save fails", () => {
      render(
        <ActionButtons
          {...defaultProps}
          actionStates={{ ...defaultProps.actionStates, save: "error" }}
        />
      );

      const saveButton = screen.getByLabelText("Save to workspace");
      expect(saveButton).toHaveClass("text-red-600");
    });
  });

  describe("Accessibility", () => {
    it("has aria-labels on all buttons", () => {
      render(<ActionButtons {...defaultProps} />);

      expect(screen.getByLabelText("Save to workspace")).toBeInTheDocument();
      expect(screen.getByLabelText("Export")).toBeInTheDocument();
      expect(screen.getByLabelText("Share")).toBeInTheDocument();
      expect(screen.getByLabelText("Bookmark")).toBeInTheDocument();
      expect(screen.getByLabelText("Follow up")).toBeInTheDocument();
    });

    it("export dropdown has proper aria-expanded attribute", () => {
      render(<ActionButtons {...defaultProps} />);

      const exportButton = screen.getByLabelText("Export");
      expect(exportButton).toHaveAttribute("aria-expanded", "false");

      fireEvent.click(exportButton);
      expect(exportButton).toHaveAttribute("aria-expanded", "true");
    });
  });
});
