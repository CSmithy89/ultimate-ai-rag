"use client";

import { useCoAgentStateRender } from "@copilotkit/react-core";
import type { ThoughtStep } from "@/types/copilot";
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Loader2,
  Circle,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

/**
 * Generate a stable key for a step.
 * Uses step id if available, otherwise falls back to step text + timestamp.
 */
function getStepKey(step: ThoughtStep, index: number): string {
  // Use id if available (for steps from backend with unique IDs)
  const stepWithId = step as ThoughtStep & { id?: string };
  if (stepWithId.id) {
    return stepWithId.id;
  }
  // Otherwise use step text + timestamp for stability
  return `${step.step}-${step.timestamp ?? index}`;
}

/**
 * Props for the StepIndicator component.
 */
interface StepIndicatorProps {
  step: ThoughtStep;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
}

/**
 * StepIndicator displays a single thought step with status icon
 * and optional expandable details.
 *
 * Status colors follow the design system:
 * - pending: Slate-400
 * - in_progress: Indigo-600 with animation
 * - completed: Emerald-500
 */
export function StepIndicator({
  step,
  index,
  isExpanded,
  onToggle,
}: StepIndicatorProps) {
  const StatusIcon = {
    pending: Circle,
    in_progress: Loader2,
    completed: CheckCircle2,
  }[step.status];

  const statusColors = {
    pending: "text-slate-400",
    in_progress: "text-indigo-600 animate-spin",
    completed: "text-emerald-500",
  };

  const statusLabel = step.status.replace("_", " ");
  const hasDetails = Boolean(step.details);
  const toggleLabel = hasDetails
    ? isExpanded
      ? `Collapse details for step: ${step.step}`
      : `Expand details for step: ${step.step}`
    : `Step: ${step.step}`;

  return (
    <div className="flex flex-col">
      <button
        onClick={onToggle}
        className="flex items-center gap-2 text-left hover:bg-slate-50 rounded p-1 -m-1"
        type="button"
        aria-expanded={hasDetails ? isExpanded : undefined}
        aria-label={toggleLabel}
      >
        {hasDetails ? (
          isExpanded ? (
            <ChevronDown
              className="h-3 w-3 text-slate-400"
              data-testid="chevron-down"
              aria-hidden="true"
            />
          ) : (
            <ChevronRight
              className="h-3 w-3 text-slate-400"
              data-testid="chevron-right"
              aria-hidden="true"
            />
          )
        ) : (
          <span className="w-3" aria-hidden="true" />
        )}
        <StatusIcon
          className={cn("h-4 w-4", statusColors[step.status])}
          aria-label={"Status: " + statusLabel}
        />
        <span
          className={cn(
            "text-sm",
            step.status === "in_progress" && "text-indigo-600 font-medium",
            step.status === "completed" && "text-slate-600",
            step.status === "pending" && "text-slate-400"
          )}
        >
          {step.step}
        </span>
      </button>
      {isExpanded && step.details && (
        // XSS Safety Note: React automatically escapes text content rendered
        // inside JSX elements including <pre> tags. The step.details value
        // is rendered as text, not HTML, so any HTML/script tags will be
        // displayed as literal text, preventing XSS attacks.
        <pre className="mt-1 ml-7 p-2 bg-slate-100 rounded text-xs text-slate-600 font-mono overflow-x-auto">
          {step.details}
        </pre>
      )}
    </div>
  );
}

/**
 * ThoughtTraceStepper displays the agent's thought process
 * as a vertical progress indicator with expandable steps.
 *
 * Uses CopilotKit's useCoAgentStateRender hook to receive
 * state updates from the backend orchestrator agent.
 *
 * Story 6-2: Chat Sidebar Interface
 */
export function ThoughtTraceStepper() {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  const toggleStep = (stepKey: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepKey)) {
        next.delete(stepKey);
      } else {
        next.add(stepKey);
      }
      return next;
    });
  };

  useCoAgentStateRender<{ steps: ThoughtStep[] }>({
    name: "orchestrator",
    render: ({ state }) => {
      if (!state?.steps?.length) return null;

      return (
        <div className="flex flex-col gap-2 p-4 font-mono text-sm border-t border-slate-200">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Agent Progress
          </h3>
          {state.steps.map((step, idx) => {
            const stepKey = getStepKey(step, idx);
            return (
              <StepIndicator
                key={stepKey}
                step={step}
                index={idx}
                isExpanded={expandedSteps.has(stepKey)}
                onToggle={() => toggleStep(stepKey)}
              />
            );
          })}
        </div>
      );
    },
  });

  // Render is handled by useCoAgentStateRender hook
  return null;
}
