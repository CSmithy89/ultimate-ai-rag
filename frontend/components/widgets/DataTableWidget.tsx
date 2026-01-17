"use client";

/**
 * DataTableWidget - AG-UI Enhancement
 *
 * Displays tabular data with optional sorting, row selection, and actions.
 * Can be rendered from backend CUSTOM events or used directly.
 *
 * @example
 * ```tsx
 * <DataTableWidget
 *   columns={[
 *     { key: "name", header: "Name" },
 *     { key: "score", header: "Score", align: "right" },
 *   ]}
 *   rows={[
 *     { name: "Document A", score: 0.95 },
 *     { name: "Document B", score: 0.87 },
 *   ]}
 *   title="Retrieved Sources"
 *   showRowNumbers
 * />
 * ```
 */

import { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import type { DataTableWidgetProps } from "@/lib/widget-registry";

/**
 * Format a cell value for display.
 */
function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "number") {
    // Format numbers with reasonable precision
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

/**
 * DataTableWidget displays tabular data with headers, rows, and optional features.
 */
export function DataTableWidget({
  columns,
  rows,
  title,
  showRowNumbers = false,
  maxHeight,
  onRowClick,
}: DataTableWidgetProps) {
  // Track selected row for highlighting
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // Handle row click
  const handleRowClick = useCallback(
    (row: Record<string, unknown>, index: number) => {
      setSelectedIndex(index);
      onRowClick?.(row, index);
    },
    [onRowClick]
  );

  // Calculate column widths
  const columnWidths = useMemo(() => {
    return columns.map((col) => {
      if (col.width) {
        return typeof col.width === "number" ? `${col.width}px` : col.width;
      }
      return "auto";
    });
  }, [columns]);

  if (rows.length === 0) {
    return (
      <div className="w-full rounded-lg border bg-white p-4 shadow-sm">
        {title && (
          <h3 className="mb-3 text-sm font-medium text-gray-700">{title}</h3>
        )}
        <p className="text-center text-sm text-gray-500 py-4">No data available</p>
      </div>
    );
  }

  return (
    <div className="w-full rounded-lg border bg-white shadow-sm overflow-hidden">
      {/* Header with title */}
      {title && (
        <div className="border-b px-4 py-3">
          <h3 className="text-sm font-medium text-gray-700">{title}</h3>
        </div>
      )}

      {/* Table container with optional max height */}
      <div
        className="overflow-auto"
        style={{ maxHeight: maxHeight ?? "none" }}
      >
        <table className="w-full border-collapse">
          {/* Table header */}
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              {showRowNumbers && (
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-12">
                  #
                </th>
              )}
              {columns.map((col, colIndex) => (
                <th
                  key={col.key}
                  className={cn(
                    "px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wider",
                    col.align === "center" && "text-center",
                    col.align === "right" && "text-right",
                    !col.align && "text-left"
                  )}
                  style={{ width: columnWidths[colIndex] }}
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>

          {/* Table body */}
          <tbody className="bg-white divide-y divide-gray-100">
            {rows.map((row, rowIndex) => {
              const isSelected = selectedIndex === rowIndex;
              const isClickable = !!onRowClick;

              return (
                <tr
                  key={rowIndex}
                  onClick={isClickable ? () => handleRowClick(row, rowIndex) : undefined}
                  className={cn(
                    "transition-colors",
                    isClickable && "cursor-pointer hover:bg-gray-50",
                    isSelected && "bg-blue-50"
                  )}
                >
                  {showRowNumbers && (
                    <td className="px-3 py-2 text-xs text-gray-400 whitespace-nowrap">
                      {rowIndex + 1}
                    </td>
                  )}
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={cn(
                        "px-3 py-2 text-sm text-gray-900 whitespace-nowrap",
                        col.align === "center" && "text-center",
                        col.align === "right" && "text-right"
                      )}
                    >
                      {formatCellValue(row[col.key])}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer with row count */}
      <div className="border-t px-4 py-2 bg-gray-50">
        <p className="text-xs text-gray-500">
          {rows.length} {rows.length === 1 ? "row" : "rows"}
        </p>
      </div>
    </div>
  );
}

export default DataTableWidget;
