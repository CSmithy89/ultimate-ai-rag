/**
 * Tests for WidgetErrorBoundary component
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { WidgetErrorBoundary, SafeWidget } from "@/components/widgets/WidgetErrorBoundary";

// Component that throws an error
function ThrowingComponent(): React.ReactElement {
  throw new Error("Test error");
}

// Component that works normally
function WorkingComponent(): React.ReactElement {
  return <div data-testid="working">Working content</div>;
}

// Suppress console.error for cleaner test output
const originalConsoleError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalConsoleError;
});

describe("WidgetErrorBoundary", () => {
  describe("normal rendering", () => {
    it("should render children when no error", () => {
      render(
        <WidgetErrorBoundary>
          <WorkingComponent />
        </WidgetErrorBoundary>
      );

      expect(screen.getByTestId("working")).toBeInTheDocument();
      expect(screen.getByText("Working content")).toBeInTheDocument();
    });
  });

  describe("error handling", () => {
    it("should render fallback when child throws", () => {
      render(
        <WidgetErrorBoundary widgetName="TestWidget">
          <ThrowingComponent />
        </WidgetErrorBoundary>
      );

      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText("TestWidget failed to load")).toBeInTheDocument();
      expect(screen.getByText("Test error")).toBeInTheDocument();
    });

    it("should render generic message without widget name", () => {
      render(
        <WidgetErrorBoundary>
          <ThrowingComponent />
        </WidgetErrorBoundary>
      );

      expect(screen.getByText("Widget failed to load")).toBeInTheDocument();
    });

    it("should call onError callback", () => {
      const onError = jest.fn();

      render(
        <WidgetErrorBoundary onError={onError}>
          <ThrowingComponent />
        </WidgetErrorBoundary>
      );

      expect(onError).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({
          componentStack: expect.any(String),
        })
      );
    });

    it("should show retry button in fallback", () => {
      render(
        <WidgetErrorBoundary widgetName="TestWidget">
          <ThrowingComponent />
        </WidgetErrorBoundary>
      );

      const retryButton = screen.getByRole("button", { name: /try again/i });
      expect(retryButton).toBeInTheDocument();
    });

    it("should reset error state on retry click", () => {
      let shouldThrow = true;

      function ConditionalThrower(): React.ReactElement {
        if (shouldThrow) {
          throw new Error("Test error");
        }
        return <div data-testid="recovered">Recovered!</div>;
      }

      render(
        <WidgetErrorBoundary widgetName="TestWidget">
          <ConditionalThrower />
        </WidgetErrorBoundary>
      );

      expect(screen.getByRole("alert")).toBeInTheDocument();

      // Fix the component before retrying
      shouldThrow = false;

      // Click retry
      fireEvent.click(screen.getByRole("button", { name: /try again/i }));

      // Should re-render successfully
      expect(screen.getByTestId("recovered")).toBeInTheDocument();
    });
  });

  describe("custom fallback", () => {
    it("should render custom fallback when provided", () => {
      render(
        <WidgetErrorBoundary
          fallback={<div data-testid="custom-fallback">Custom error UI</div>}
        >
          <ThrowingComponent />
        </WidgetErrorBoundary>
      );

      expect(screen.getByTestId("custom-fallback")).toBeInTheDocument();
      expect(screen.getByText("Custom error UI")).toBeInTheDocument();
    });
  });

  describe("className prop", () => {
    it("should apply className to wrapper", () => {
      const { container } = render(
        <WidgetErrorBoundary className="custom-class">
          <ThrowingComponent />
        </WidgetErrorBoundary>
      );

      expect(container.firstChild).toHaveClass("custom-class");
    });
  });
});

describe("SafeWidget", () => {
  it("should render children normally", () => {
    render(
      <SafeWidget name="TestWidget">
        <WorkingComponent />
      </SafeWidget>
    );

    expect(screen.getByTestId("working")).toBeInTheDocument();
  });

  it("should catch errors and show fallback", () => {
    render(
      <SafeWidget name="TestWidget">
        <ThrowingComponent />
      </SafeWidget>
    );

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("TestWidget failed to load")).toBeInTheDocument();
  });

  it("should call onError callback", () => {
    const onError = jest.fn();

    render(
      <SafeWidget name="TestWidget" onError={onError}>
        <ThrowingComponent />
      </SafeWidget>
    );

    expect(onError).toHaveBeenCalled();
  });

  it("should support custom fallback", () => {
    render(
      <SafeWidget
        name="TestWidget"
        fallback={<div data-testid="safe-fallback">Safe fallback</div>}
      >
        <ThrowingComponent />
      </SafeWidget>
    );

    expect(screen.getByTestId("safe-fallback")).toBeInTheDocument();
  });
});
