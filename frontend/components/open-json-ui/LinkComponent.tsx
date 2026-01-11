/**
 * Open-JSON-UI Link Component
 *
 * Renders hyperlinks with URL validation and target control.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { isValidUrl, sanitizeContent } from "@/lib/open-json-ui/sanitize";
import { ExternalLink, AlertCircle } from "lucide-react";
import type { LinkComponent as LinkComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for LinkComponent.
 */
export interface LinkComponentProps {
  /** Link component data */
  component: LinkComponentType;
}

/**
 * LinkComponent renders hyperlinks with security validation.
 *
 * Security: Only allows http/https URLs to prevent javascript: attacks.
 * External links (_blank) include proper security attributes.
 *
 * @param props - Component props
 * @returns Rendered anchor element or error state
 *
 * @example
 * ```tsx
 * <LinkComponent
 *   component={{
 *     type: "link",
 *     text: "Visit example",
 *     href: "https://example.com",
 *     target: "_blank"
 *   }}
 * />
 * ```
 */
export const LinkComponent = memo(function LinkComponent({
  component,
}: LinkComponentProps) {
  const sanitizedText = sanitizeContent(component.text);
  const isExternal = component.target === "_blank";

  // Validate URL for security
  if (!isValidUrl(component.href)) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1",
          "text-red-600 text-sm"
        )}
        data-testid="open-json-ui-link-invalid"
        role="alert"
      >
        <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
        <span>Invalid link URL</span>
      </span>
    );
  }

  return (
    <a
      href={component.href}
      target={component.target ?? "_self"}
      rel={isExternal ? "noopener noreferrer" : undefined}
      className={cn(
        "inline-flex items-center gap-1",
        "text-blue-600 hover:text-blue-800 underline underline-offset-2",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 rounded"
      )}
      data-testid="open-json-ui-link"
    >
      <span dangerouslySetInnerHTML={{ __html: sanitizedText }} />
      {isExternal && (
        <ExternalLink
          className="h-3.5 w-3.5 flex-shrink-0"
          aria-label="(opens in new tab)"
        />
      )}
    </a>
  );
});

export default LinkComponent;
