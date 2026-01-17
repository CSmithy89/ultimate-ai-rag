"use client";

/**
 * WidgetErrorBoundary - Error Boundary for AG-UI Widgets
 *
 * Catches rendering errors in widget components and displays a fallback UI.
 * Prevents entire app crashes when a single widget fails to render.
 *
 * @example
 * ```tsx
 * <WidgetErrorBoundary widgetName="ChartWidget">
 *   <ChartWidget data={data} />
 * </WidgetErrorBoundary>
 * ```
 */

import React, { Component, ErrorInfo, ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * Props for WidgetErrorBoundary.
 */
interface WidgetErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Widget name for error reporting */
  widgetName?: string;
  /** Custom fallback component */
  fallback?: ReactNode;
  /** Callback when error is caught */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * State for error boundary.
 */
interface WidgetErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Default fallback component shown when a widget crashes.
 */
function DefaultErrorFallback({
  widgetName,
  error,
  onReset,
}: {
  widgetName?: string;
  error: Error | null;
  onReset?: () => void;
}): React.ReactElement {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center p-4",
        "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800",
        "rounded-lg text-center"
      )}
      role="alert"
      aria-live="assertive"
    >
      <svg
        className="h-8 w-8 text-red-500 mb-2"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <p className="text-sm font-medium text-red-800 dark:text-red-200">
        {widgetName ? `${widgetName} failed to load` : "Widget failed to load"}
      </p>
      {error && (
        <p className="text-xs text-red-600 dark:text-red-400 mt-1 max-w-xs truncate">
          {error.message}
        </p>
      )}
      {onReset && (
        <button
          onClick={onReset}
          className={cn(
            "mt-3 px-3 py-1.5 text-xs font-medium",
            "bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300",
            "rounded hover:bg-red-200 dark:hover:bg-red-900/60",
            "transition-colors"
          )}
        >
          Try again
        </button>
      )}
    </div>
  );
}

/**
 * Error boundary component for AG-UI widgets.
 *
 * Catches JavaScript errors anywhere in their child component tree,
 * logs those errors, and displays a fallback UI instead of crashing.
 */
export class WidgetErrorBoundary extends Component<
  WidgetErrorBoundaryProps,
  WidgetErrorBoundaryState
> {
  constructor(props: WidgetErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): WidgetErrorBoundaryState {
    // Update state so the next render shows the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console in development
    console.error(
      `[WidgetErrorBoundary] Error in ${this.props.widgetName || "widget"}:`,
      error,
      errorInfo
    );

    // Call optional error callback
    this.props.onError?.(error, errorInfo);
  }

  /**
   * Reset the error boundary state.
   * Allows the user to retry rendering the widget.
   */
  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    const { hasError, error } = this.state;
    const { children, widgetName, fallback, className } = this.props;

    if (hasError) {
      // Render custom fallback or default error UI
      if (fallback) {
        return <div className={className}>{fallback}</div>;
      }

      return (
        <div className={className}>
          <DefaultErrorFallback
            widgetName={widgetName}
            error={error}
            onReset={this.handleReset}
          />
        </div>
      );
    }

    return children;
  }
}

/**
 * Hook-style wrapper for using error boundary with widgets.
 *
 * @example
 * ```tsx
 * <SafeWidget name="ChartWidget">
 *   <ChartWidget data={data} />
 * </SafeWidget>
 * ```
 */
export function SafeWidget({
  name,
  children,
  className,
  onError,
  fallback,
}: {
  name: string;
  children: ReactNode;
  className?: string;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  fallback?: ReactNode;
}): React.ReactElement {
  return (
    <WidgetErrorBoundary
      widgetName={name}
      className={className}
      onError={onError}
      fallback={fallback}
    >
      {children}
    </WidgetErrorBoundary>
  );
}

export default WidgetErrorBoundary;
