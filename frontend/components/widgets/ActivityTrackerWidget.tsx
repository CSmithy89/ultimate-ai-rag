"use client";

/**
 * ActivityTrackerWidget - AG-UI Enhancement (Phase 5)
 *
 * Displays real-time activity progress from AG-UI ACTIVITY events.
 * Shows overall progress, current step, and metadata for long-running operations.
 *
 * @example
 * ```tsx
 * <ActivityTrackerWidget
 *   activity={{
 *     id: "activity-123",
 *     type: "query_processing",
 *     message: "Retrieving relevant information...",
 *     progress: 0.25,
 *     totalSteps: 4,
 *     currentStep: 1,
 *     metadata: { retrieval_strategy: "hybrid" },
 *   }}
 * />
 * ```
 */

import { useMemo } from "react";
import { cn } from "@/lib/utils";

/**
 * Activity state from AG-UI ACTIVITY events.
 */
export interface ActivityState {
  /** Unique activity identifier */
  id: string;
  /** Type of activity (e.g., "query_processing", "indexing") */
  type: string;
  /** Human-readable status message */
  message: string;
  /** Progress as a decimal (0.0 to 1.0) */
  progress: number;
  /** Total number of steps in the activity */
  totalSteps: number;
  /** Current step number (1-indexed) */
  currentStep: number;
  /** Additional metadata about the activity */
  metadata?: Record<string, unknown>;
}

/**
 * Props for ActivityTrackerWidget.
 */
export interface ActivityTrackerWidgetProps {
  /** Current activity state */
  activity: ActivityState;
  /** Whether to show metadata details */
  showMetadata?: boolean;
  /** Whether to show step indicators */
  showSteps?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Activity type icons mapping.
 */
const ActivityIcon = ({ type }: { type: string }) => {
  switch (type) {
    case "query_processing":
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
      );
    case "retrieval":
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
          />
        </svg>
      );
    case "hitl_validation":
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      );
    case "response_generation":
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
          />
        </svg>
      );
    case "indexing":
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"
          />
        </svg>
      );
    default:
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 10V3L4 14h7v7l9-11h-7z"
          />
        </svg>
      );
  }
};

/**
 * Format activity type to human-readable label.
 */
function formatActivityType(type: string): string {
  const typeMap: Record<string, string> = {
    query_processing: "Processing Query",
    retrieval: "Retrieving Data",
    hitl_validation: "Awaiting Validation",
    response_generation: "Generating Response",
    indexing: "Indexing Documents",
  };
  return typeMap[type] || type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Format metadata key for display.
 */
function formatMetadataKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * ActivityTrackerWidget displays real-time progress for long-running operations.
 */
export function ActivityTrackerWidget({
  activity,
  showMetadata = true,
  showSteps = true,
  className,
}: ActivityTrackerWidgetProps) {
  // Calculate progress percentage
  const progressPercent = useMemo(
    () => Math.round(activity.progress * 100),
    [activity.progress]
  );

  // Determine if activity is complete
  const isComplete = progressPercent >= 100;

  // Filter metadata for display (exclude internal fields)
  const displayMetadata = useMemo(() => {
    if (!activity.metadata) return [];
    return Object.entries(activity.metadata).filter(
      ([key]) => !key.startsWith("_") && key !== "query_preview"
    );
  }, [activity.metadata]);

  return (
    <div
      className={cn(
        "w-full rounded-lg border bg-white p-4 shadow-sm",
        isComplete && "border-green-200 bg-green-50/50",
        className
      )}
    >
      {/* Header with icon and type */}
      <div className="mb-3 flex items-center gap-2">
        <div
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-full",
            isComplete ? "bg-green-100 text-green-600" : "bg-blue-100 text-blue-600"
          )}
        >
          {isComplete ? (
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          ) : (
            <ActivityIcon type={activity.type} />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-gray-700 truncate">
            {formatActivityType(activity.type)}
          </h3>
          {showSteps && activity.totalSteps > 0 && (
            <p className="text-xs text-gray-500">
              Step {activity.currentStep} of {activity.totalSteps}
            </p>
          )}
        </div>
        <span
          className={cn(
            "text-sm font-medium",
            isComplete ? "text-green-600" : "text-blue-600"
          )}
        >
          {progressPercent}%
        </span>
      </div>

      {/* Progress bar */}
      <div className="mb-3 h-2 w-full overflow-hidden rounded-full bg-gray-100">
        <div
          className={cn(
            "h-full transition-all duration-300 ease-out",
            isComplete ? "bg-green-500" : "bg-blue-500"
          )}
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Status message */}
      <p
        className={cn(
          "text-sm",
          isComplete ? "text-green-700" : "text-gray-600"
        )}
      >
        {isComplete ? "Complete!" : activity.message}
      </p>

      {/* Metadata display */}
      {showMetadata && displayMetadata.length > 0 && (
        <div className="mt-3 border-t border-gray-100 pt-3">
          <div className="flex flex-wrap gap-2">
            {displayMetadata.map(([key, value]) => (
              <span
                key={key}
                className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
              >
                <span className="font-medium">{formatMetadataKey(key)}:</span>
                <span>{String(value)}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ActivityTrackerWidget;
