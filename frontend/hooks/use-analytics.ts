"use client";

import { useCallback } from "react";
import { redactSensitiveKeys } from "@/lib/utils/redact";

/**
 * Properties for analytics events.
 * Keys can contain any JSON-serializable value.
 */
export interface AnalyticsProperties {
  [key: string]: unknown;
}

/**
 * Telemetry event payload sent to the backend.
 */
export interface TelemetryEventPayload {
  event: string;
  properties: AnalyticsProperties;
  timestamp: string;
}

/**
 * Return type for the useAnalytics hook.
 */
export interface UseAnalyticsReturn {
  /**
   * Track an analytics event.
   *
   * @param event - Event name (e.g., "copilot_message_sent")
   * @param properties - Optional event properties (PII is redacted)
   */
  track: (event: string, properties?: AnalyticsProperties) => void;
}

/**
 * useAnalytics provides a track() function for sending telemetry events.
 *
 * Story 21-B1: Configure Observability Hooks and Dev Console
 *
 * Features:
 * - Non-blocking fetch to /api/telemetry endpoint
 * - Console logging in development mode
 * - Graceful error handling (logs but doesn't block UI)
 * - PII redaction via redactSensitiveKeys utility
 *
 * Security:
 * - Sensitive keys (password, token, secret, etc.) are redacted before sending
 * - Message content should be sent as length only, not raw content
 * - Events include only tenant_id, never user_id (multi-tenancy)
 *
 * @example
 * ```tsx
 * const { track } = useAnalytics();
 *
 * // Track a simple event
 * track("copilot_chat_expanded");
 *
 * // Track event with properties
 * track("copilot_message_sent", { messageLength: 150 });
 *
 * // Track event with feedback
 * track("copilot_feedback", { messageId: "123", type: "positive" });
 * ```
 */
export function useAnalytics(): UseAnalyticsReturn {
  const track = useCallback(
    (event: string, properties?: AnalyticsProperties) => {
      // Redact sensitive keys from properties before sending
      const safeProperties = properties
        ? redactSensitiveKeys(properties as Record<string, unknown>)
        : {};

      const payload: TelemetryEventPayload = {
        event,
        properties: safeProperties,
        timestamp: new Date().toISOString(),
      };

      // Development logging for debugging
      if (process.env.NODE_ENV === "development") {
        console.log("[Analytics]", event, safeProperties);
      }

      // Non-blocking fetch to telemetry endpoint
      // We intentionally don't await this - telemetry should never block UI
      fetch("/api/telemetry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).catch((error) => {
        // Log telemetry failures but don't block the application
        console.error("[Analytics] Failed to send event:", error);
      });
    },
    []
  );

  return { track };
}

export default useAnalytics;
