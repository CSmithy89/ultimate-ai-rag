"use client";

import { Component, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

/**
 * Props for the CopilotErrorBoundary component.
 */
interface CopilotErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Optional fallback component to display on error */
  fallback?: ReactNode;
}

/**
 * State for the CopilotErrorBoundary component.
 */
interface CopilotErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary specifically for CopilotKit components.
 * Catches errors in the CopilotSidebar and provides a graceful
 * fallback UI with retry functionality.
 *
 * Story 6-2: Chat Sidebar Interface
 *
 * @example
 * ```tsx
 * <CopilotErrorBoundary>
 *   <CopilotSidebar />
 * </CopilotErrorBoundary>
 * ```
 */
export class CopilotErrorBoundary extends Component<
  CopilotErrorBoundaryProps,
  CopilotErrorBoundaryState
> {
  constructor(props: CopilotErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): CopilotErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to console in development
    console.error("CopilotErrorBoundary caught an error:", error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div
          className="fixed right-4 top-4 w-80 bg-white border border-slate-200 rounded-lg shadow-lg p-6"
          role="alert"
          aria-live="assertive"
        >
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="h-6 w-6 text-amber-500" aria-hidden="true" />
            <h2 className="text-lg font-semibold text-slate-900">
              Chat Unavailable
            </h2>
          </div>
          <p className="text-sm text-slate-600 mb-4">
            The AI chat assistant encountered an error. Please try again.
          </p>
          {this.state.error && (
            <pre className="text-xs text-slate-500 bg-slate-50 rounded p-2 mb-4 overflow-x-auto">
              {this.state.error.message}
            </pre>
          )}
          <button
            onClick={this.handleRetry}
            className="flex items-center gap-2 w-full justify-center px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
            type="button"
          >
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
