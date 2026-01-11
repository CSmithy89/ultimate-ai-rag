/**
 * Open-JSON-UI List Component
 *
 * Renders ordered or unordered lists with sanitized content.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { sanitizeContent } from "@/lib/open-json-ui/sanitize";
import type { ListComponent as ListComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for ListComponent.
 */
export interface ListComponentProps {
  /** List component data */
  component: ListComponentType;
}

/**
 * ListComponent renders ordered or unordered lists.
 *
 * @param props - Component props
 * @returns Rendered list element (ol or ul)
 *
 * @example
 * ```tsx
 * <ListComponent
 *   component={{
 *     type: "list",
 *     items: ["First item", "Second item"],
 *     ordered: true
 *   }}
 * />
 * ```
 */
export const ListComponent = memo(function ListComponent({
  component,
}: ListComponentProps) {
  const Tag = component.ordered ? "ol" : "ul";
  const listStyle = component.ordered
    ? "list-decimal list-inside"
    : "list-disc list-inside";

  return (
    <Tag
      className={cn("text-sm text-slate-700 space-y-1", listStyle)}
      data-testid="open-json-ui-list"
      role="list"
    >
      {component.items.map((item, index) => {
        const sanitized = sanitizeContent(item);
        return (
          <li
            key={index}
            className="leading-relaxed"
            dangerouslySetInnerHTML={{ __html: sanitized }}
          />
        );
      })}
    </Tag>
  );
});

export default ListComponent;
