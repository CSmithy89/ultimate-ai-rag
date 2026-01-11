/**
 * Open-JSON-UI Code Component
 *
 * Renders code blocks with language-specific styling.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import type { CodeComponent as CodeComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for CodeComponent.
 */
export interface CodeComponentProps {
  /** Code component data */
  component: CodeComponentType;
}

/**
 * CodeComponent renders syntax-highlighted code blocks.
 *
 * Note: The language prop is added as a class for potential syntax highlighting
 * integration. Currently displays code with basic styling.
 *
 * @param props - Component props
 * @returns Rendered code block
 *
 * @example
 * ```tsx
 * <CodeComponent
 *   component={{
 *     type: "code",
 *     content: "const x = 1;",
 *     language: "typescript"
 *   }}
 * />
 * ```
 */
export const CodeComponent = memo(function CodeComponent({
  component,
}: CodeComponentProps) {
  const languageClass = component.language
    ? `language-${component.language}`
    : "";

  return (
    <pre
      className={cn(
        "bg-slate-900 text-slate-100 rounded-lg p-4 overflow-x-auto",
        "text-sm font-mono leading-relaxed"
      )}
      data-testid="open-json-ui-code"
    >
      <code className={languageClass}>{component.content}</code>
    </pre>
  );
});

export default CodeComponent;
