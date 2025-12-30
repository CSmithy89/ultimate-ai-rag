/**
 * Tests for Toaster component
 * Story 6-5: Frontend Actions
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { Toaster } from "@/components/ui/Toaster";
import { useToast } from "@/hooks/use-toast";

// Mock the useToast hook
jest.mock("@/hooks/use-toast", () => ({
  useToast: jest.fn(),
}));

const mockUseToast = useToast as jest.MockedFunction<typeof useToast>;

describe("Toaster", () => {
  const mockDismiss = jest.fn();
  const mockDismissAll = jest.fn();
  const mockToast = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseToast.mockReturnValue({
      toasts: [],
      toast: mockToast,
      dismiss: mockDismiss,
      dismissAll: mockDismissAll,
    });
  });

  describe("Rendering", () => {
    it("renders nothing when there are no toasts", () => {
      const { container } = render(<Toaster />);
      expect(container.firstChild).toBeNull();
    });

    it("renders toasts when present", () => {
      mockUseToast.mockReturnValue({
        toasts: [
          { id: "1", variant: "default", title: "Test Toast", description: "Test description" },
        ],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      expect(screen.getByText("Test Toast")).toBeInTheDocument();
      expect(screen.getByText("Test description")).toBeInTheDocument();
    });

    it("renders multiple toasts", () => {
      mockUseToast.mockReturnValue({
        toasts: [
          { id: "1", variant: "default", title: "Toast 1" },
          { id: "2", variant: "destructive", title: "Toast 2" },
          { id: "3", variant: "success", title: "Toast 3" },
        ],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      expect(screen.getByText("Toast 1")).toBeInTheDocument();
      expect(screen.getByText("Toast 2")).toBeInTheDocument();
      expect(screen.getByText("Toast 3")).toBeInTheDocument();
    });
  });

  describe("Toast Variants", () => {
    it("renders default variant with success styling", () => {
      mockUseToast.mockReturnValue({
        toasts: [{ id: "1", variant: "default", title: "Success" }],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      const alert = screen.getByRole("alert");
      expect(alert).toHaveClass("bg-white");
    });

    it("renders destructive variant with error styling", () => {
      mockUseToast.mockReturnValue({
        toasts: [{ id: "1", variant: "destructive", title: "Error" }],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      const alert = screen.getByRole("alert");
      expect(alert).toHaveClass("bg-red-50");
    });
  });

  describe("Dismiss Functionality", () => {
    it("calls dismiss when close button is clicked", () => {
      mockUseToast.mockReturnValue({
        toasts: [{ id: "test-id", variant: "default", title: "Dismissable" }],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      const dismissButton = screen.getByLabelText("Dismiss");
      fireEvent.click(dismissButton);

      expect(mockDismiss).toHaveBeenCalledWith("test-id");
    });
  });

  describe("Accessibility", () => {
    it("has proper aria-live attribute", () => {
      mockUseToast.mockReturnValue({
        toasts: [{ id: "1", variant: "default", title: "Accessible" }],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      const alert = screen.getByRole("alert");
      expect(alert).toHaveAttribute("aria-live", "assertive");
    });

    it("has notifications label on container", () => {
      mockUseToast.mockReturnValue({
        toasts: [{ id: "1", variant: "default", title: "Accessible" }],
        toast: mockToast,
        dismiss: mockDismiss,
        dismissAll: mockDismissAll,
      });

      render(<Toaster />);

      expect(screen.getByLabelText("Notifications")).toBeInTheDocument();
    });
  });
});
