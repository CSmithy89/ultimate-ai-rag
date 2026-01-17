"use client";

/**
 * ApprovalDialogWidget - AG-UI Enhancement
 *
 * Displays an approval dialog for HITL workflows.
 * Can be rendered from backend CUSTOM events or used directly.
 *
 * @example
 * ```tsx
 * <ApprovalDialogWidget
 *   title="Approve Sources"
 *   description="Select sources to include in the response"
 *   items={[
 *     { id: "1", label: "Source A", description: "Preview..." },
 *     { id: "2", label: "Source B", description: "Preview..." },
 *   ]}
 *   actions={["approve", "reject", "skip"]}
 *   onRespond={(action, ids) => console.log(action, ids)}
 *   multiSelect
 * />
 * ```
 */

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import type { ApprovalDialogWidgetProps } from "@/lib/widget-registry";

/**
 * ApprovalDialogWidget displays a list of items for user approval.
 */
export function ApprovalDialogWidget({
  title,
  description,
  items,
  actions,
  onRespond,
  multiSelect = true,
}: ApprovalDialogWidgetProps) {
  // Track selected items
  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    () => new Set(items.map((item) => item.id))
  );

  // Toggle item selection
  const toggleItem = useCallback((itemId: string) => {
    if (!multiSelect) {
      // Single select mode - select only this item
      setSelectedIds(new Set([itemId]));
      return;
    }

    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(itemId)) {
        next.delete(itemId);
      } else {
        next.add(itemId);
      }
      return next;
    });
  }, [multiSelect]);

  // Select/deselect all
  const selectAll = useCallback(() => {
    setSelectedIds(new Set(items.map((item) => item.id)));
  }, [items]);

  const selectNone = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  // Handle action buttons
  const handleAction = useCallback(
    (action: "approve" | "reject" | "skip") => {
      const ids = action === "reject" ? [] : Array.from(selectedIds);
      if (action === "skip") {
        // Skip means approve all
        onRespond?.(action, items.map((item) => item.id));
      } else {
        onRespond?.(action, ids);
      }
    },
    [selectedIds, items, onRespond]
  );

  const hasApprove = actions.includes("approve");
  const hasReject = actions.includes("reject");
  const hasSkip = actions.includes("skip");

  return (
    <div className="w-full max-w-lg rounded-lg border bg-white shadow-lg">
      {/* Header */}
      <div className="border-b p-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        {description && (
          <p className="mt-1 text-sm text-gray-500">{description}</p>
        )}
      </div>

      {/* Selection controls */}
      {multiSelect && items.length > 1 && (
        <div className="flex items-center justify-between border-b px-4 py-2 bg-gray-50">
          <span className="text-sm text-gray-600">
            {selectedIds.size} of {items.length} selected
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={selectAll}
              className="text-sm text-blue-600 hover:underline"
            >
              Select All
            </button>
            <button
              type="button"
              onClick={selectNone}
              className="text-sm text-blue-600 hover:underline"
            >
              Select None
            </button>
          </div>
        </div>
      )}

      {/* Items list */}
      <div className="max-h-64 overflow-y-auto p-2">
        <div className="space-y-2">
          {items.map((item) => {
            const isSelected = selectedIds.has(item.id);

            return (
              <div
                key={item.id}
                onClick={() => toggleItem(item.id)}
                className={cn(
                  "flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                  isSelected
                    ? "bg-blue-50 border-blue-200"
                    : "bg-gray-50 border-transparent hover:border-gray-200"
                )}
              >
                {/* Checkbox/Radio */}
                <input
                  type={multiSelect ? "checkbox" : "radio"}
                  name="approval-items"
                  checked={isSelected}
                  onChange={() => toggleItem(item.id)}
                  className="mt-1 h-4 w-4 rounded border-gray-300"
                />

                {/* Item content */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">{item.label}</p>
                  {item.description && (
                    <p className="mt-0.5 text-xs text-gray-500 line-clamp-2">
                      {item.description}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex items-center justify-between gap-2 border-t p-4">
        {/* Reject button */}
        {hasReject && (
          <button
            type="button"
            onClick={() => handleAction("reject")}
            className="px-4 py-2 text-sm font-medium text-red-600 border border-red-200 rounded-md hover:bg-red-50 transition-colors"
          >
            Reject All
          </button>
        )}

        {/* Right side buttons */}
        <div className="flex gap-2 ml-auto">
          {hasSkip && (
            <button
              type="button"
              onClick={() => handleAction("skip")}
              className="px-4 py-2 text-sm font-medium text-gray-700 border border-gray-200 rounded-md hover:bg-gray-50 transition-colors"
            >
              Skip
            </button>
          )}
          {hasApprove && (
            <button
              type="button"
              onClick={() => handleAction("approve")}
              disabled={selectedIds.size === 0}
              className={cn(
                "px-4 py-2 text-sm font-medium text-white rounded-md transition-colors",
                selectedIds.size === 0
                  ? "bg-gray-300 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700"
              )}
            >
              Approve ({selectedIds.size})
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default ApprovalDialogWidget;
