/**
 * Open-JSON-UI Progress Component
 *
 * Renders progress bars with optional labels.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { sanitizeToPlainText } from "@/lib/open-json-ui/sanitize";
import type { ProgressComponent as ProgressComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for ProgressComponent.
 */
export interface ProgressComponentProps {
  /** Progress component data */
  component: ProgressComponentType;
}

/**
 * ProgressComponent renders a progress bar similar to shadcn/ui Progress.
 *
 * @param props - Component props
 * @returns Rendered progress indicator
 *
 * @example
 * ```tsx
 * <ProgressComponent
 *   component={{
 *     type: "progress",
 *     value: 65,
 *     label: "Upload progress"
 *   }}
 * />
 * ```
 */
export const ProgressComponent = memo(function ProgressComponent({
  component,
}: ProgressComponentProps) {
  // Clamp value between 0 and 100
  const value = Math.min(Math.max(component.value, 0), 100);
  const sanitizedLabel = component.label
    ? sanitizeToPlainText(component.label)
    : undefined;

  return (
    <div
      className="w-full space-y-1"
      data-testid="open-json-ui-progress"
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={sanitizedLabel ?? `Progress: ${value}%`}
    >
      {sanitizedLabel && (
        <div className="flex justify-between text-sm">
          <span className="text-slate-700">{sanitizedLabel}</span>
          <span className="text-slate-500">{value}%</span>
        </div>
      )}
      <div
        className={cn(
          "relative h-2 w-full overflow-hidden rounded-full",
          "bg-slate-100"
        )}
      >
        <div
          className={cn(
            "h-full flex-1 bg-slate-900 transition-all duration-300"
          )}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
});

export default ProgressComponent;
