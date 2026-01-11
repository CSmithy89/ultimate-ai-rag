"use client";

/**
 * AG-UI Error Handler for CopilotKit Integration
 *
 * This module provides error handling utilities for AG-UI RUN_ERROR events.
 * It includes a hook for handling errors and displaying appropriate toast notifications.
 *
 * @module ErrorHandler
 * @see Story 22-B2 - Implement Extended AG-UI Error Events
 * @see Epic 22 - Advanced Protocol Integration
 */

import { useCallback } from "react";
import { useToast } from "@/hooks/use-toast";
import {
  type AGUIErrorData,
  AGUIErrorCode,
  getErrorDisplayMessage,
  formatRetryMessage,
  getToastVariant,
  isRetryableError,
  requiresUserAction,
} from "@/lib/ag-ui-error-codes";

/**
 * Options for the AG-UI error handler hook.
 */
export interface UseAGUIErrorHandlerOptions {
  /**
   * Called when a retryable error is received.
   * Allows custom retry logic.
   */
  onRetryableError?: (error: AGUIErrorData) => void;

  /**
   * Called when a user action is required.
   * Allows custom handling (e.g., redirect to login).
   */
  onUserActionRequired?: (error: AGUIErrorData) => void;

  /**
   * Called for all errors.
   * Allows logging or analytics.
   */
  onError?: (error: AGUIErrorData) => void;

  /**
   * Whether to show toast notifications.
   * @default true
   */
  showToast?: boolean;
}

/**
 * Return type for the useAGUIErrorHandler hook.
 */
export interface UseAGUIErrorHandlerReturn {
  /**
   * Handle an AG-UI error event.
   * Displays appropriate toast notification and triggers callbacks.
   */
  handleError: (error: AGUIErrorData) => void;
}

/**
 * Hook for handling AG-UI error events.
 *
 * Provides centralized error handling with:
 * - Toast notifications with appropriate severity
 * - Retry guidance for rate-limited requests
 * - Callbacks for custom error handling
 *
 * @example
 * ```tsx
 * const { handleError } = useAGUIErrorHandler({
 *   onRetryableError: (error) => {
 *     // Custom retry logic
 *     setTimeout(() => retryRequest(), (error.retry_after ?? 60) * 1000);
 *   },
 *   onUserActionRequired: (error) => {
 *     if (error.code === AGUIErrorCode.TENANT_REQUIRED) {
 *       router.push('/login');
 *     }
 *   },
 * });
 *
 * // In your event handler
 * if (event.type === 'RUN_ERROR') {
 *   handleError(event.data);
 * }
 * ```
 */
export function useAGUIErrorHandler(
  options: UseAGUIErrorHandlerOptions = {}
): UseAGUIErrorHandlerReturn {
  const {
    onRetryableError,
    onUserActionRequired,
    onError,
    showToast = true,
  } = options;

  const { toast } = useToast();

  const handleError = useCallback(
    (error: AGUIErrorData) => {
      // Always call the general error callback
      onError?.(error);

      // Check for specific error types
      if (isRetryableError(error.code)) {
        onRetryableError?.(error);
      }

      if (requiresUserAction(error.code)) {
        onUserActionRequired?.(error);
      }

      // Show toast notification if enabled
      if (showToast) {
        const title = getErrorDisplayMessage(error);
        let description = error.message;

        // Add retry guidance for rate-limited errors
        if (error.retry_after !== undefined && error.retry_after > 0) {
          description = `${error.message} ${formatRetryMessage(error.retry_after)}`;
        }

        toast({
          title,
          description,
          variant: getToastVariant(error),
        });
      }
    },
    [onError, onRetryableError, onUserActionRequired, showToast, toast]
  );

  return { handleError };
}

/**
 * Parse an AG-UI event and extract error data if it's a RUN_ERROR event.
 *
 * @param event - The AG-UI event object
 * @returns Error data if the event is a RUN_ERROR, undefined otherwise
 */
export function parseAGUIError(
  event: { type?: string; event?: string; data?: unknown }
): AGUIErrorData | undefined {
  // Check for RUN_ERROR event type
  const eventType = event.event || event.type;
  if (eventType !== "RUN_ERROR") {
    return undefined;
  }

  // Validate error data structure
  const data = event.data as Record<string, unknown> | undefined;
  if (!data || typeof data.code !== "string" || typeof data.message !== "string") {
    return undefined;
  }

  return {
    code: data.code,
    message: data.message,
    http_status: typeof data.http_status === "number" ? data.http_status : 500,
    details: typeof data.details === "object" ? (data.details as Record<string, unknown>) : undefined,
    retry_after: typeof data.retry_after === "number" ? data.retry_after : undefined,
  };
}

/**
 * Check if an error is a specific error code.
 *
 * @param error - The error data
 * @param code - The error code to check
 * @returns True if the error matches the code
 */
export function isErrorCode(
  error: AGUIErrorData,
  code: AGUIErrorCode | string
): boolean {
  return error.code === code;
}

// Re-export types and utilities for convenience
export {
  AGUIErrorCode,
  type AGUIErrorData,
  getErrorDisplayMessage,
  formatRetryMessage,
  getToastVariant,
  isRetryableError,
  requiresUserAction,
} from "@/lib/ag-ui-error-codes";
