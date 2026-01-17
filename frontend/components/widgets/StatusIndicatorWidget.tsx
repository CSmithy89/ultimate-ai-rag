"use client";

/**
 * StatusIndicatorWidget - AG-UI Enhancement (Phase 5)
 *
 * Displays status indicators with icons, messages, and optional details.
 * Supports multiple status types: idle, loading, success, error, warning.
 *
 * @example
 * ```tsx
 * <StatusIndicatorWidget
 *   status="success"
 *   message="Operation completed successfully"
 *   details="Processed 150 documents in 2.3 seconds"
 * />
 * ```
 */

import { useMemo } from "react";
import { cn } from "@/lib/utils";

/**
 * Status types for the indicator.
 */
export type StatusType = "idle" | "loading" | "success" | "error" | "warning";

/**
 * Props for StatusIndicatorWidget.
 */
export interface StatusIndicatorWidgetProps {
  /** Current status */
  status: StatusType;
  /** Status message */
  message?: string;
  /** Optional details */
  details?: string;
  /** Whether to show animated indicator */
  animated?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Status configuration for styling and icons.
 */
interface StatusConfig {
  icon: React.ReactNode;
  bgColor: string;
  textColor: string;
  borderColor: string;
  iconColor: string;
  label: string;
}

/**
 * Get configuration for a status type.
 */
function getStatusConfig(status: StatusType, animated: boolean): StatusConfig {
  const configs: Record<StatusType, StatusConfig> = {
    idle: {
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <circle cx="12" cy="12" r="10" strokeWidth={2} />
        </svg>
      ),
      bgColor: "bg-slate-50",
      textColor: "text-slate-600",
      borderColor: "border-slate-200",
      iconColor: "text-slate-400",
      label: "Idle",
    },
    loading: {
      icon: (
        <svg
          className={cn("h-5 w-5", animated && "animate-spin")}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
      ),
      bgColor: "bg-blue-50",
      textColor: "text-blue-700",
      borderColor: "border-blue-200",
      iconColor: "text-blue-500",
      label: "Loading",
    },
    success: {
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      bgColor: "bg-green-50",
      textColor: "text-green-700",
      borderColor: "border-green-200",
      iconColor: "text-green-500",
      label: "Success",
    },
    error: {
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      bgColor: "bg-red-50",
      textColor: "text-red-700",
      borderColor: "border-red-200",
      iconColor: "text-red-500",
      label: "Error",
    },
    warning: {
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      ),
      bgColor: "bg-amber-50",
      textColor: "text-amber-700",
      borderColor: "border-amber-200",
      iconColor: "text-amber-500",
      label: "Warning",
    },
  };

  return configs[status];
}

/**
 * StatusIndicatorWidget displays status with icon, message, and details.
 */
export function StatusIndicatorWidget({
  status,
  message,
  details,
  animated = true,
  className,
}: StatusIndicatorWidgetProps) {
  const config = useMemo(() => getStatusConfig(status, animated), [status, animated]);

  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border p-4",
        config.bgColor,
        config.borderColor,
        className
      )}
      role="status"
      aria-live="polite"
    >
      <div className={cn("flex-shrink-0 mt-0.5", config.iconColor)}>
        {config.icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={cn("text-sm font-medium", config.textColor)}>
            {message || config.label}
          </span>
          {status === "loading" && animated && (
            <span className="flex gap-1">
              <span
                className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-bounce"
                style={{ animationDelay: "0ms" }}
              />
              <span
                className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
            </span>
          )}
        </div>
        {details && (
          <p className={cn("mt-1 text-sm opacity-75", config.textColor)}>
            {details}
          </p>
        )}
      </div>
    </div>
  );
}

export default StatusIndicatorWidget;
