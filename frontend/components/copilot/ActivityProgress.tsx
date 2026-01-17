"use client";

/**
 * ActivityProgress - AG-UI Enhancement (Phase 5)
 *
 * Displays real-time activity progress from AG-UI ACTIVITY events.
 * Uses a context provider to receive ACTIVITY_SNAPSHOT and ACTIVITY_DELTA
 * events and render progress UI.
 *
 * @example
 * ```tsx
 * // In ChatSidebar - automatic tracking via context
 * <ActivityProgressProvider>
 *   <ActivityProgress />
 * </ActivityProgressProvider>
 *
 * // Manual control
 * const { processSnapshot } = useActivityProgressControl();
 * processSnapshot({ id: "123", type: "query_processing", ... });
 * ```
 */

import React, { createContext, useContext, type ReactNode } from "react";
import { useCopilotChat } from "@copilotkit/react-core";
import { useActivityTracker, type UseActivityTrackerResult } from "@/hooks/use-activity-tracker";
import { ActivityTrackerWidget } from "@/components/widgets/ActivityTrackerWidget";
import { cn } from "@/lib/utils";

// ============================================
// CONTEXT
// ============================================

/**
 * Activity progress context type.
 */
type ActivityProgressContextType = UseActivityTrackerResult & {
  /** Whether the agent is currently loading */
  isAgentLoading: boolean;
};

const ActivityProgressContext = createContext<ActivityProgressContextType | null>(null);

/**
 * Hook to access activity progress context.
 */
export function useActivityProgressControl(): ActivityProgressContextType {
  const context = useContext(ActivityProgressContext);
  if (!context) {
    throw new Error("useActivityProgressControl must be used within ActivityProgressProvider");
  }
  return context;
}

// ============================================
// PROVIDER
// ============================================

/**
 * Props for ActivityProgressProvider.
 */
interface ActivityProgressProviderProps {
  children: ReactNode;
}

/**
 * Provider that manages activity progress state.
 * Wrap your CopilotKit components with this to enable activity tracking.
 */
export function ActivityProgressProvider({ children }: ActivityProgressProviderProps) {
  const { isLoading } = useCopilotChat();
  const activityTracker = useActivityTracker();

  const value: ActivityProgressContextType = {
    ...activityTracker,
    isAgentLoading: isLoading,
  };

  return (
    <ActivityProgressContext.Provider value={value}>
      {children}
    </ActivityProgressContext.Provider>
  );
}

// ============================================
// COMPONENT
// ============================================

/**
 * Props for ActivityProgress component.
 */
interface ActivityProgressProps {
  /** Custom class name for the wrapper */
  className?: string;
  /** Whether to show metadata in the activity widget */
  showMetadata?: boolean;
  /** Whether to show step indicators */
  showSteps?: boolean;
}

/**
 * ActivityProgress displays real-time progress for long-running agent operations.
 *
 * It renders the ActivityTrackerWidget when there's an active activity.
 * Activity state is managed via the ActivityProgressProvider context.
 *
 * Can be used in two ways:
 * 1. Inside ActivityProgressProvider for context-based state management
 * 2. Standalone - uses its own internal state
 */
export function ActivityProgress({
  className,
  showMetadata = true,
  showSteps = true,
}: ActivityProgressProps) {
  // Try to get context, fall back to internal hook if not in provider
  const contextValue = useContext(ActivityProgressContext);
  const internalTracker = useActivityTracker();
  const { isLoading } = useCopilotChat();

  // Use context if available, otherwise use internal state
  const { activity, isActive } = contextValue ?? internalTracker;
  const isAgentLoading = contextValue?.isAgentLoading ?? isLoading;

  // Show loading indicator when agent is working but no activity data yet
  if (isAgentLoading && !isActive) {
    return null; // ThoughtTraceStepper handles the general loading state
  }

  // Don't render if no active activity
  if (!isActive || !activity) {
    return null;
  }

  return (
    <div className={cn("px-4 py-2 border-t border-slate-200", className)}>
      <ActivityTrackerWidget
        activity={activity}
        showMetadata={showMetadata}
        showSteps={showSteps}
      />
    </div>
  );
}

// ============================================
// STANDALONE HOOK
// ============================================

/**
 * Standalone hook for activity tracking without the context provider.
 * Useful when you need to manually control activity state.
 */
export { useActivityTracker } from "@/hooks/use-activity-tracker";

export default ActivityProgress;
