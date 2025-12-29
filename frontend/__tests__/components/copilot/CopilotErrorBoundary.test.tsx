/**
 * Tests for CopilotErrorBoundary component.
 * Story 6-2: Chat Sidebar Interface
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { CopilotErrorBoundary } from "../../../components/copilot/CopilotErrorBoundary";

// Component that throws an error for testing
const ThrowError = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error("Test error message");
  }
  return <div data-testid="child-component">Child content</div>;
};

// Suppress console.error during error boundary tests
const originalConsoleError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalConsoleError;
});

describe("CopilotErrorBoundary", () => {
  it("renders children when there is no error", () => {
    render(
      <CopilotErrorBoundary>
        <ThrowError shouldThrow={false} />
      </CopilotErrorBoundary>
    );

    expect(screen.getByTestId("child-component")).toBeInTheDocument();
    expect(screen.getByText("Child content")).toBeInTheDocument();
  });

  it("renders fallback UI when child throws an error", () => {
    render(
      <CopilotErrorBoundary>
        <ThrowError shouldThrow={true} />
      </CopilotErrorBoundary>
    );

    expect(screen.queryByTestId("child-component")).not.toBeInTheDocument();
    expect(screen.getByText("Chat Unavailable")).toBeInTheDocument();
    expect(
      screen.getByText("The AI chat assistant encountered an error. Please try again.")
    ).toBeInTheDocument();
  });

  it("displays the error message in fallback UI", () => {
    render(
      <CopilotErrorBoundary>
        <ThrowError shouldThrow={true} />
      </CopilotErrorBoundary>
    );

    expect(screen.getByText("Test error message")).toBeInTheDocument();
  });

  it("renders custom fallback when provided", () => {
    const customFallback = <div data-testid="custom-fallback">Custom error UI</div>;

    render(
      <CopilotErrorBoundary fallback={customFallback}>
        <ThrowError shouldThrow={true} />
      </CopilotErrorBoundary>
    );

    expect(screen.getByTestId("custom-fallback")).toBeInTheDocument();
    expect(screen.getByText("Custom error UI")).toBeInTheDocument();
    expect(screen.queryByText("Chat Unavailable")).not.toBeInTheDocument();
  });

  it("has a retry button that attempts to recover", () => {
    render(
      <CopilotErrorBoundary>
        <ThrowError shouldThrow={true} />
      </CopilotErrorBoundary>
    );

    expect(screen.getByText("Chat Unavailable")).toBeInTheDocument();

    // Click retry button
    const retryButton = screen.getByRole("button", { name: /try again/i });
    expect(retryButton).toBeInTheDocument();
    fireEvent.click(retryButton);

    // After retry, it will try to render children again
    // Since shouldThrow is still true, it will show error again
    // But the state should have been reset first
    expect(screen.getByText("Chat Unavailable")).toBeInTheDocument();
  });

  it("has proper accessibility attributes on fallback UI", () => {
    render(
      <CopilotErrorBoundary>
        <ThrowError shouldThrow={true} />
      </CopilotErrorBoundary>
    );

    const alertContainer = screen.getByRole("alert");
    expect(alertContainer).toBeInTheDocument();
    expect(alertContainer).toHaveAttribute("aria-live", "assertive");
  });

  it("logs error to console when error is caught", () => {
    render(
      <CopilotErrorBoundary>
        <ThrowError shouldThrow={true} />
      </CopilotErrorBoundary>
    );

    expect(console.error).toHaveBeenCalled();
  });
});
