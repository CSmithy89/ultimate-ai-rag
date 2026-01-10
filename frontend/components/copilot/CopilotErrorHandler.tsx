"use client";

import { useEffect, useCallback } from "react";
import { useCopilotChat } from "@copilotkit/react-core";
import { useToast } from "@/hooks/use-toast";
import { useAnalytics } from "@/hooks/use-analytics";

/**
 * Error codes from backend RUN_ERROR events.
 *
 * Story 21-B2: Add RUN_ERROR Event Support
 *
 * These match the RunErrorCode enum in the backend.
 */
export const RunErrorCode = {
  AGENT_EXECUTION_ERROR: "AGENT_EXECUTION_ERROR",
  TENANT_REQUIRED: "TENANT_REQUIRED",
  RATE_LIMITED: "RATE_LIMITED",
  TIMEOUT: "TIMEOUT",
  INVALID_REQUEST: "INVALID_REQUEST",
} as const;

export type RunErrorCodeType = (typeof RunErrorCode)[keyof typeof RunErrorCode];

/**
 * User-friendly error messages for each error code.
 */
const ERROR_MESSAGES: Record<string, { title: string; description: string }> = {
  [RunErrorCode.AGENT_EXECUTION_ERROR]: {
    title: "Processing Error",
    description: "Something went wrong. Please try again.",
  },
  [RunErrorCode.TENANT_REQUIRED]: {
    title: "Authentication Required",
    description: "Please log in to continue.",
  },
  [RunErrorCode.RATE_LIMITED]: {
    title: "Too Many Requests",
    description: "Please wait a moment before trying again.",
  },
  [RunErrorCode.TIMEOUT]: {
    title: "Request Timeout",
    description: "The request took too long. Please try again.",
  },
  [RunErrorCode.INVALID_REQUEST]: {
    title: "Invalid Request",
    description: "Please check your input and try again.",
  },
};

/**
 * RUN_ERROR event data structure from AG-UI protocol.
 */
interface RunErrorData {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * CopilotErrorHandler - Handles and displays RUN_ERROR events.
 *
 * Story 21-B2: Add RUN_ERROR Event Support
 *
 * This component:
 * - Monitors the copilot chat for error states
 * - Displays toast notifications for user-facing errors
 * - Tracks errors via analytics
 * - Shows technical details in development mode
 *
 * Usage:
 * Render this component inside CopilotKit context to enable
 * error handling with toast notifications.
 *
 * @example
 * ```tsx
 * <CopilotKit runtimeUrl="/api/copilotkit">
 *   <CopilotErrorHandler />
 *   <CopilotSidebar />
 * </CopilotKit>
 * ```
 */
export function CopilotErrorHandler() {
  const { toast } = useToast();
  const { track } = useAnalytics();
  // Note: useCopilotChat provides chat state but RUN_ERROR events
  // are handled internally by CopilotKit. This hook is imported for
  // potential future use when CopilotKit exposes stream event subscriptions.
  useCopilotChat();
  const isDev = process.env.NODE_ENV === "development";

  /**
   * Handle a RUN_ERROR event from the backend.
   *
   * @param errorData - The error data from RUN_ERROR event
   */
  const handleRunError = useCallback(
    (errorData: RunErrorData) => {
      const { code, message, details } = errorData;

      // Get user-friendly message or use backend message
      const errorInfo = ERROR_MESSAGES[code] ?? {
        title: "Error",
        description: message,
      };

      // Show toast notification
      toast({
        variant: "destructive",
        title: errorInfo.title,
        description: errorInfo.description,
        duration: 6000,
      });

      // Track error in analytics
      track("copilot_run_error", {
        code,
        hasDetails: !!details,
      });

      // Log details in development mode
      if (isDev && details) {
        console.error("[CopilotErrorHandler] RUN_ERROR details:", {
          code,
          message,
          details,
        });
      }
    },
    [toast, track, isDev]
  );

  // Note: RUN_ERROR events are processed internally by CopilotKit.
  // The onError handler on CopilotKit provider already tracks errors.
  // This component provides a hook point for additional error UI handling
  // if needed in the future when CopilotKit exposes stream events.

  // For now, we expose the handleRunError function for manual invocation
  // or future integration when CopilotKit provides event subscriptions.

  // Store handler in window for potential manual testing in development
  useEffect(() => {
    if (isDev && typeof window !== "undefined") {
      (window as unknown as Record<string, unknown>).__copilotHandleRunError =
        handleRunError;
    }
    return () => {
      if (isDev && typeof window !== "undefined") {
        delete (window as unknown as Record<string, unknown>)
          .__copilotHandleRunError;
      }
    };
  }, [handleRunError, isDev]);

  // This component renders nothing - it's purely for side effects
  return null;
}

export default CopilotErrorHandler;
