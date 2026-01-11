/**
 * Open-JSON-UI Sanitization Utilities
 *
 * DOMPurify-based sanitization for preventing XSS attacks in rendered content.
 * All text content from Open-JSON-UI payloads must be sanitized before rendering.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

import DOMPurify from "dompurify";

/**
 * Allowed HTML tags for inline formatting.
 * These tags provide basic text formatting without security risks.
 */
const ALLOWED_TAGS = ["b", "i", "em", "strong", "code", "br", "span"];

/**
 * Allowed HTML attributes.
 * Empty by default to prevent attribute-based attacks.
 */
const ALLOWED_ATTR: string[] = [];

/**
 * Sanitizes text content using DOMPurify with strict configuration.
 * Removes all potentially dangerous HTML while preserving safe inline formatting.
 *
 * @param content - Raw content that may contain HTML
 * @returns Sanitized content safe for rendering
 *
 * @example
 * ```ts
 * // XSS attempt is blocked
 * sanitizeContent('<script>alert("xss")</script>Hello')
 * // Returns: "Hello"
 *
 * // Safe formatting is preserved
 * sanitizeContent('<strong>Bold</strong> text')
 * // Returns: "<strong>Bold</strong> text"
 * ```
 */
export function sanitizeContent(content: string): string {
  if (typeof window === "undefined") {
    // Server-side: strip all HTML tags for safety
    return content.replace(/<[^>]*>/g, "");
  }

  return DOMPurify.sanitize(content, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
  });
}

/**
 * Sanitizes content for use as plain text (no HTML allowed).
 * Useful for contexts where HTML should never be rendered.
 *
 * @param content - Raw content
 * @returns Plain text with all HTML removed
 */
export function sanitizeToPlainText(content: string): string {
  if (typeof window === "undefined") {
    return content.replace(/<[^>]*>/g, "");
  }

  return DOMPurify.sanitize(content, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: [],
  });
}

/**
 * Validates a URL for safe usage in images and links.
 * Blocks javascript: and data: URLs to prevent XSS.
 *
 * @param url - URL to validate
 * @returns true if URL is safe, false otherwise
 *
 * @example
 * ```ts
 * isValidUrl("https://example.com/image.png") // true
 * isValidUrl("javascript:alert(1)") // false
 * isValidUrl("data:text/html,...") // false
 * ```
 */
export function isValidUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const safeProtocols = ["http:", "https:"];
    return safeProtocols.includes(parsed.protocol);
  } catch {
    return false;
  }
}

/**
 * Sanitizes a URL, returning empty string if invalid.
 *
 * @param url - URL to sanitize
 * @returns Sanitized URL or empty string
 */
export function sanitizeUrl(url: string): string {
  return isValidUrl(url) ? url : "";
}
