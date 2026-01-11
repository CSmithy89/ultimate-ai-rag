/**
 * Open-JSON-UI Heading Component
 *
 * Renders heading elements (h1-h6) with appropriate styling.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { sanitizeToPlainText } from "@/lib/open-json-ui/sanitize";
import type { HeadingComponent as HeadingComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for HeadingComponent.
 */
export interface HeadingComponentProps {
  /** Heading component data */
  component: HeadingComponentType;
}

/**
 * Style mappings for heading levels.
 */
const headingStyles: Record<number, string> = {
  1: "text-3xl font-bold tracking-tight",
  2: "text-2xl font-semibold tracking-tight",
  3: "text-xl font-semibold",
  4: "text-lg font-semibold",
  5: "text-base font-medium",
  6: "text-sm font-medium",
};

/**
 * HeadingComponent renders semantic heading elements.
 *
 * @param props - Component props
 * @returns Rendered heading element (h1-h6)
 *
 * @example
 * ```tsx
 * <HeadingComponent
 *   component={{ type: "heading", level: 2, content: "Section Title" }}
 * />
 * ```
 */
export const HeadingComponent = memo(function HeadingComponent({
  component,
}: HeadingComponentProps) {
  const sanitized = sanitizeToPlainText(component.content);
  const level = Math.min(Math.max(component.level, 1), 6);
  const Tag = `h${level}` as keyof JSX.IntrinsicElements;
  const styleClass = headingStyles[level] ?? headingStyles[6];

  return (
    <Tag
      className={cn("text-slate-900", styleClass)}
      data-testid="open-json-ui-heading"
    >
      {sanitized}
    </Tag>
  );
});

export default HeadingComponent;
