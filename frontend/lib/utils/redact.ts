/**
 * Sensitive key redaction utility for tool arguments and results.
 *
 * Story 21-A3: Implement Tool Call Visualization
 *
 * Recursively redacts values of keys matching sensitive patterns to prevent
 * accidental exposure of credentials, tokens, and other sensitive data in
 * tool call visualizations.
 */

/**
 * Pattern matching sensitive keys that should be redacted in UI.
 * Matches: password, secret, token, key, auth, credential, api_key, api-key,
 * private_key, private-key, access_token, access-token
 */
export const SENSITIVE_PATTERNS =
  /password|secret|token|key|auth|credential|api[-_]?key|private[-_]?key|access[-_]?token/i;

/**
 * Recursively redact sensitive keys from an object.
 * Values of keys matching SENSITIVE_PATTERNS are replaced with "[REDACTED]".
 *
 * @param obj - Object to redact
 * @returns New object with sensitive values redacted
 *
 * @example
 * ```ts
 * const input = { password: "secret123", username: "admin" };
 * const output = redactSensitiveKeys(input);
 * // output: { password: "[REDACTED]", username: "admin" }
 * ```
 */
export function redactSensitiveKeys(
  obj: Record<string, unknown>
): Record<string, unknown> {
  // Handle null/undefined
  if (obj === null || obj === undefined) {
    return obj;
  }

  // Handle non-objects (primitive values)
  if (typeof obj !== "object") {
    return obj;
  }

  // Handle arrays
  if (Array.isArray(obj)) {
    return obj.map((item) =>
      typeof item === "object" && item !== null
        ? redactSensitiveKeys(item as Record<string, unknown>)
        : item
    ) as unknown as Record<string, unknown>;
  }

  // Handle objects
  return Object.fromEntries(
    Object.entries(obj).map(([key, value]) => [
      key,
      SENSITIVE_PATTERNS.test(key)
        ? "[REDACTED]"
        : typeof value === "object" && value !== null
          ? redactSensitiveKeys(value as Record<string, unknown>)
          : value,
    ])
  );
}
