"use client";

/**
 * StepProgressWidget - AG-UI Enhancement
 *
 * Displays agent processing steps with visual progress indication.
 * Can be rendered from backend CUSTOM events or used directly.
 *
 * @example
 * ```tsx
 * <StepProgressWidget
 *   steps={[
 *     { step: "Analyzing query", status: "completed" },
 *     { step: "Retrieving sources", status: "in_progress" },
 *     { step: "Generating answer", status: "pending" },
 *   ]}
 *   currentStep={1}
 *   title="Processing..."
 * />
 * ```
 */

import { useMemo } from "react";
import { cn } from "@/lib/utils";
import type { StepProgressWidgetProps } from "@/lib/widget-registry";

/**
 * Status icons for each step state.
 */
const StatusIcon = ({ status }: { status: "pending" | "in_progress" | "completed" }) => {
  switch (status) {
    case "completed":
      return (
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 text-green-600">
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      );
    case "in_progress":
      return (
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-600">
          <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        </div>
      );
    case "pending":
    default:
      return (
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-100 text-gray-400">
          <div className="h-2 w-2 rounded-full bg-current" />
        </div>
      );
  }
};

/**
 * StepProgressWidget displays a vertical list of processing steps
 * with visual indicators for each step's status.
 */
export function StepProgressWidget({
  steps,
  currentStep,
  title,
  showDetails = true,
}: StepProgressWidgetProps) {
  // Calculate progress percentage
  const progress = useMemo(() => {
    if (steps.length === 0) return 0;
    const completedCount = steps.filter((s) => s.status === "completed").length;
    return Math.round((completedCount / steps.length) * 100);
  }, [steps]);

  // Determine if all steps are complete
  const isComplete = useMemo(
    () => steps.every((s) => s.status === "completed"),
    [steps]
  );

  if (steps.length === 0) {
    return null;
  }

  return (
    <div className="w-full rounded-lg border bg-white p-4 shadow-sm">
      {/* Header with title and progress */}
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-700">
          {title || (isComplete ? "Complete" : "Processing...")}
        </h3>
        <span className="text-xs text-gray-500">{progress}%</span>
      </div>

      {/* Progress bar */}
      <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
        <div
          className={cn(
            "h-full transition-all duration-500",
            isComplete ? "bg-green-500" : "bg-blue-500"
          )}
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Steps list */}
      <div className="space-y-3">
        {steps.map((step, index) => (
          <div
            key={index}
            className={cn(
              "flex items-start gap-3 transition-opacity",
              step.status === "pending" && "opacity-50"
            )}
          >
            {/* Status icon */}
            <StatusIcon status={step.status} />

            {/* Step content */}
            <div className="flex-1 min-w-0">
              <p
                className={cn(
                  "text-sm",
                  step.status === "completed" && "text-gray-600",
                  step.status === "in_progress" && "text-blue-600 font-medium",
                  step.status === "pending" && "text-gray-400"
                )}
              >
                {step.step}
              </p>
              {showDetails && step.details && (
                <p className="mt-0.5 text-xs text-gray-400 line-clamp-2">
                  {step.details}
                </p>
              )}
            </div>

            {/* Step number indicator */}
            {currentStep !== undefined && index === currentStep && (
              <span className="flex-shrink-0 text-xs text-blue-500">
                Step {index + 1}/{steps.length}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default StepProgressWidget;
