/**
 * Open-JSON-UI Table Component
 *
 * Renders data tables with headers, rows, and optional caption.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { sanitizeContent } from "@/lib/open-json-ui/sanitize";
import type { TableComponent as TableComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for TableComponent.
 */
export interface TableComponentProps {
  /** Table component data */
  component: TableComponentType;
}

/**
 * TableComponent renders tabular data with proper accessibility.
 *
 * Implements a styled table similar to shadcn/ui Table component.
 *
 * @param props - Component props
 * @returns Rendered table element
 *
 * @example
 * ```tsx
 * <TableComponent
 *   component={{
 *     type: "table",
 *     headers: ["Name", "Value"],
 *     rows: [["foo", "1"], ["bar", "2"]],
 *     caption: "Data summary"
 *   }}
 * />
 * ```
 */
export const TableComponent = memo(function TableComponent({
  component,
}: TableComponentProps) {
  return (
    <div
      className="relative w-full overflow-auto"
      data-testid="open-json-ui-table"
    >
      <table className="w-full caption-bottom text-sm">
        {component.caption && (
          <caption className="mt-4 text-sm text-slate-500">
            {sanitizeContent(component.caption)}
          </caption>
        )}
        <thead className="[&_tr]:border-b">
          <tr className="border-b transition-colors hover:bg-slate-50/50">
            {component.headers.map((header, index) => (
              <th
                key={index}
                className={cn(
                  "h-10 px-4 text-left align-middle font-medium",
                  "text-slate-500 [&:has([role=checkbox])]:pr-0"
                )}
                scope="col"
                dangerouslySetInnerHTML={{ __html: sanitizeContent(header) }}
              />
            ))}
          </tr>
        </thead>
        <tbody className="[&_tr:last-child]:border-0">
          {component.rows.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              className="border-b transition-colors hover:bg-slate-50/50"
            >
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className="p-4 align-middle [&:has([role=checkbox])]:pr-0"
                  dangerouslySetInnerHTML={{ __html: sanitizeContent(cell) }}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
});

export default TableComponent;
