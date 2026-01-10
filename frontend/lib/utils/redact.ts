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
 * Pattern matching sensitive key names that should be redacted in UI.
 *
 * Uses word boundaries (\b) to prevent false positives like:
 * - "monkey" matching "key"
 * - "author" matching "auth"
 * - "turkey" matching "key"
 *
 * Matches (case-insensitive):
 * - password, secret, credential(s)
 * - token, bearer, jwt, oauth
 * - api_key, api-key, private_key, private-key
 * - access_token, access-token, refresh_token, refresh-token
 * - auth_token, auth-token, client_secret, client-secret
 * - session_id, session-id, cookie
 * - signature, ssn
 *
 * @example
 * ```ts
 * SENSITIVE_KEY_PATTERNS.test("api_key");   // true
 * SENSITIVE_KEY_PATTERNS.test("password");  // true
 * SENSITIVE_KEY_PATTERNS.test("monkey");    // false (no word boundary match)
 * SENSITIVE_KEY_PATTERNS.test("author");    // false (no word boundary match)
 * ```
 */
export const SENSITIVE_KEY_PATTERNS =
  /\b(password|secret|credentials?|token|bearer|jwt|oauth|api[-_]?key|private[-_]?key|access[-_]?token|refresh[-_]?token|auth[-_]?token|client[-_]?secret|session[-_]?id|cookie|signature|ssn)\b/i;

/**
 * Pattern matching sensitive data embedded in string values.
 * Used to detect credentials in connection strings, config values, etc.
 *
 * @example
 * ```ts
 * // These values would be redacted:
 * "password=secret123"
 * "bearer eyJhbGciOiJIUzI1NiIs..."
 * "postgres://user:pass@host/db"
 * ```
 */
export const SENSITIVE_VALUE_PATTERNS =
  /(?:password|secret|token|bearer|jwt|auth|credential)\s*[=:]\s*\S+/gi;

/**
 * Pattern for connection strings with embedded credentials.
 * Matches common database and service URL patterns.
 */
const CONNECTION_STRING_PATTERN =
  /:\/\/[^:]+:[^@]+@/g;

/**
 * Safely stringify an object, handling circular references.
 * Used internally to prevent JSON.stringify crashes.
 *
 * @param obj - Object to stringify
 * @param space - Indentation spaces (optional)
 * @returns JSON string with circular refs replaced by "[Circular]"
 */
export function safeStringify(obj: unknown, space?: number): string {
  const seen = new WeakSet();
  try {
    return JSON.stringify(
      obj,
      (_key, value) => {
        if (typeof value === "object" && value !== null) {
          if (seen.has(value)) {
            return "[Circular]";
          }
          seen.add(value);
        }
        return value;
      },
      space
    );
  } catch {
    return "[Unstringifiable]";
  }
}

/**
 * Redact sensitive data from a string value.
 * Handles embedded credentials in connection strings and config values.
 *
 * @param value - String value to scan and redact
 * @returns String with sensitive data replaced by [REDACTED]
 */
function redactStringValue(value: string): string {
  let redacted = value;

  // Redact connection strings (user:pass@host)
  redacted = redacted.replace(CONNECTION_STRING_PATTERN, "://[REDACTED]@");

  // Redact key=value patterns with sensitive keys
  redacted = redacted.replace(SENSITIVE_VALUE_PATTERNS, "[REDACTED]");

  return redacted;
}

/**
 * Recursively redact sensitive keys and values from an object.
 *
 * Performs two layers of protection:
 * 1. Redacts values of keys matching SENSITIVE_KEY_PATTERNS
 * 2. Scans string values for embedded credentials (connection strings, etc.)
 *
 * Uses generic typing to preserve input type information.
 *
 * @param obj - Object to redact (preserves type)
 * @returns New object with sensitive values redacted
 *
 * @example
 * ```ts
 * // Key-based redaction
 * const input1 = { password: "secret123", username: "admin" };
 * const output1 = redactSensitiveKeys(input1);
 * // output1: { password: "[REDACTED]", username: "admin" }
 *
 * // Value-based redaction (connection strings)
 * const input2 = { config: "postgres://user:pass@host/db" };
 * const output2 = redactSensitiveKeys(input2);
 * // output2: { config: "postgres://[REDACTED]@host/db" }
 *
 * // Combined
 * const input3 = { api_key: "sk-123", data: "password=secret" };
 * const output3 = redactSensitiveKeys(input3);
 * // output3: { api_key: "[REDACTED]", data: "[REDACTED]" }
 * ```
 */
export function redactSensitiveKeys<T>(obj: T): T {
  // Handle null/undefined
  if (obj === null || obj === undefined) {
    return obj;
  }

  // Handle primitives
  if (typeof obj !== "object") {
    // Scan string values for embedded credentials
    if (typeof obj === "string") {
      return redactStringValue(obj) as T;
    }
    return obj;
  }

  // Handle arrays
  if (Array.isArray(obj)) {
    return obj.map((item) => redactSensitiveKeys(item)) as T;
  }

  // Handle objects
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
    // Check if key matches sensitive patterns
    if (SENSITIVE_KEY_PATTERNS.test(key)) {
      result[key] = "[REDACTED]";
    } else if (typeof value === "string") {
      // Scan string values for embedded credentials
      result[key] = redactStringValue(value);
    } else if (typeof value === "object" && value !== null) {
      // Recurse into nested objects
      result[key] = redactSensitiveKeys(value);
    } else {
      result[key] = value;
    }
  }

  return result as T;
}

// Re-export the old pattern name for backwards compatibility
// TODO: Remove in next major version
export const SENSITIVE_PATTERNS = SENSITIVE_KEY_PATTERNS;
