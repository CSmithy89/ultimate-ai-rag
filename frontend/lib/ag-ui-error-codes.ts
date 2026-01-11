/**
 * AG-UI Error Codes for CopilotKit Integration
 *
 * This module provides TypeScript types and utilities for handling AG-UI error events
 * from the backend. These error codes align with RFC 7807 Problem Details format.
 *
 * @module ag-ui-error-codes
 * @see Story 22-B2 - Implement Extended AG-UI Error Events
 * @see Epic 22 - Advanced Protocol Integration
 */

/**
 * Standardized AG-UI error codes.
 *
 * Each code maps to a specific HTTP status and error type:
 * - AGENT_EXECUTION_ERROR (500): Unhandled agent exception
 * - TENANT_REQUIRED (401): Missing tenant_id header
 * - TENANT_UNAUTHORIZED (403): Invalid or unauthorized tenant_id
 * - SESSION_NOT_FOUND (404): Invalid session reference
 * - RATE_LIMITED (429): Request/session/message limit exceeded
 * - TIMEOUT (504): Request processing timeout
 * - INVALID_REQUEST (400): Malformed or invalid request
 * - CAPABILITY_NOT_FOUND (404): Requested capability unavailable
 * - UPSTREAM_ERROR (502): External service failure
 * - SERVICE_UNAVAILABLE (503): System overloaded/unavailable
 */
export enum AGUIErrorCode {
  AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR",
  TENANT_REQUIRED = "TENANT_REQUIRED",
  TENANT_UNAUTHORIZED = "TENANT_UNAUTHORIZED",
  SESSION_NOT_FOUND = "SESSION_NOT_FOUND",
  RATE_LIMITED = "RATE_LIMITED",
  TIMEOUT = "TIMEOUT",
  INVALID_REQUEST = "INVALID_REQUEST",
  CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND",
  UPSTREAM_ERROR = "UPSTREAM_ERROR",
  SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE",
}

/**
 * HTTP status code mapping for each error code.
 */
export const ERROR_CODE_HTTP_STATUS: Record<AGUIErrorCode, number> = {
  [AGUIErrorCode.AGENT_EXECUTION_ERROR]: 500,
  [AGUIErrorCode.TENANT_REQUIRED]: 401,
  [AGUIErrorCode.TENANT_UNAUTHORIZED]: 403,
  [AGUIErrorCode.SESSION_NOT_FOUND]: 404,
  [AGUIErrorCode.RATE_LIMITED]: 429,
  [AGUIErrorCode.TIMEOUT]: 504,
  [AGUIErrorCode.INVALID_REQUEST]: 400,
  [AGUIErrorCode.CAPABILITY_NOT_FOUND]: 404,
  [AGUIErrorCode.UPSTREAM_ERROR]: 502,
  [AGUIErrorCode.SERVICE_UNAVAILABLE]: 503,
};

/**
 * User-friendly error messages for each error code.
 */
export const ERROR_MESSAGES: Record<AGUIErrorCode, string> = {
  [AGUIErrorCode.AGENT_EXECUTION_ERROR]: "Something went wrong. Please try again.",
  [AGUIErrorCode.TENANT_REQUIRED]: "Authentication required.",
  [AGUIErrorCode.TENANT_UNAUTHORIZED]: "Access denied.",
  [AGUIErrorCode.SESSION_NOT_FOUND]: "Session expired. Please refresh.",
  [AGUIErrorCode.RATE_LIMITED]: "Too many requests. Please wait.",
  [AGUIErrorCode.TIMEOUT]: "Request timed out. Please try again.",
  [AGUIErrorCode.INVALID_REQUEST]: "Invalid request. Please check your input.",
  [AGUIErrorCode.CAPABILITY_NOT_FOUND]: "Feature not available.",
  [AGUIErrorCode.UPSTREAM_ERROR]: "External service unavailable.",
  [AGUIErrorCode.SERVICE_UNAVAILABLE]: "Service temporarily unavailable.",
};

/**
 * Data structure for AG-UI error events.
 *
 * This interface represents the data payload of a RUN_ERROR event
 * from the AG-UI protocol.
 */
export interface AGUIErrorData {
  /** Standardized error code from AGUIErrorCode enum */
  code: string;
  /** Human-readable error message */
  message: string;
  /** HTTP status code for client-side handling */
  http_status: number;
  /** Optional additional details (only in debug mode) */
  details?: Record<string, unknown>;
  /** Optional retry hint in seconds (for rate limiting) */
  retry_after?: number;
}

/**
 * Check if an error code represents a retryable error.
 *
 * Retryable errors are transient failures that may succeed on retry:
 * - RATE_LIMITED: Wait for retry_after period
 * - TIMEOUT: Request may succeed with another attempt
 * - SERVICE_UNAVAILABLE: Service may recover
 *
 * @param code - The error code to check
 * @returns True if the error is retryable
 */
export function isRetryableError(code: AGUIErrorCode | string): boolean {
  const retryableCodes: string[] = [
    AGUIErrorCode.RATE_LIMITED,
    AGUIErrorCode.TIMEOUT,
    AGUIErrorCode.SERVICE_UNAVAILABLE,
  ];
  return retryableCodes.includes(code);
}

/**
 * Check if an error requires user action (not retryable).
 *
 * These errors require user intervention to resolve:
 * - TENANT_REQUIRED: User must authenticate
 * - TENANT_UNAUTHORIZED: User lacks permission
 * - INVALID_REQUEST: User must fix input
 * - SESSION_NOT_FOUND: User must start new session
 *
 * @param code - The error code to check
 * @returns True if user action is required
 */
export function requiresUserAction(code: AGUIErrorCode | string): boolean {
  const actionRequiredCodes: string[] = [
    AGUIErrorCode.TENANT_REQUIRED,
    AGUIErrorCode.TENANT_UNAUTHORIZED,
    AGUIErrorCode.INVALID_REQUEST,
    AGUIErrorCode.SESSION_NOT_FOUND,
  ];
  return actionRequiredCodes.includes(code);
}

/**
 * Get the display message for an error.
 *
 * Returns the user-friendly message for the error code, falling back
 * to the provided message if the code is unknown.
 *
 * @param error - The error data from the RUN_ERROR event
 * @returns User-friendly error message
 */
export function getErrorDisplayMessage(error: AGUIErrorData): string {
  const code = error.code as AGUIErrorCode;
  return ERROR_MESSAGES[code] || error.message;
}

/**
 * Format retry message for rate-limited errors.
 *
 * @param retryAfter - Seconds until retry is allowed
 * @returns Formatted retry message
 */
export function formatRetryMessage(retryAfter: number): string {
  if (retryAfter < 60) {
    return `Please retry in ${retryAfter} seconds.`;
  }
  const minutes = Math.ceil(retryAfter / 60);
  return `Please retry in ${minutes} minute${minutes > 1 ? "s" : ""}.`;
}

/**
 * Determine the toast variant based on error severity.
 *
 * @param error - The error data from the RUN_ERROR event
 * @returns Toast variant ("destructive" for server errors, "default" otherwise)
 */
export function getToastVariant(error: AGUIErrorData): "destructive" | "default" {
  return error.http_status >= 500 ? "destructive" : "default";
}
