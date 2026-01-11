/**
 * Open-JSON-UI Text Component
 *
 * Renders plain text with optional styling variants.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { sanitizeContent } from "@/lib/open-json-ui/sanitize";
import type { TextComponent as TextComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for TextComponent.
 */
export interface TextComponentProps {
  /** Text component data */
  component: TextComponentType;
}

/**
 * Style variant mappings for text components.
 */
const styleVariants: Record<string, string> = {
  normal: "text-slate-900",
  muted: "text-slate-500",
  error: "text-red-600",
  success: "text-emerald-600",
};

/**
 * TextComponent renders text content with optional styling.
 *
 * @param props - Component props
 * @returns Rendered text paragraph
 *
 * @example
 * ```tsx
 * <TextComponent
 *   component={{ type: "text", content: "Hello world", style: "muted" }}
 * />
 * ```
 */
export const TextComponent = memo(function TextComponent({
  component,
}: TextComponentProps) {
  const sanitized = sanitizeContent(component.content);
  const styleClass = styleVariants[component.style ?? "normal"] ?? styleVariants.normal;

  return (
    <p
      className={cn("text-sm leading-relaxed", styleClass)}
      dangerouslySetInnerHTML={{ __html: sanitized }}
      data-testid="open-json-ui-text"
    />
  );
});

export default TextComponent;
