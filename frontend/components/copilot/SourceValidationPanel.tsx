"use client";

import { memo, useState, useCallback, useMemo, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  CheckCheck,
  XSquare,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { SourceValidationCard } from "./components/SourceValidationCard";
import type { Source, ValidationDecision } from "@/types/copilot";

interface SourceValidationPanelProps {
  /** Sources requiring validation */
  sources: Source[];
  /** Callback when user submits validation decisions */
  onSubmit: (approvedSourceIds: string[]) => void;
  /** Callback to skip validation (approve all) */
  onSkip?: () => void;
  /** Whether the panel is collapsed */
  defaultCollapsed?: boolean;
  /** Optional class name */
  className?: string;
}

/**
 * SourceValidationPanel provides an inline interface for reviewing
 * and approving/rejecting retrieved sources within the chat flow.
 *
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * This is an alternative to SourceValidationDialog for non-modal HITL.
 * It renders inline within the chat message flow with collapse/expand.
 */
export const SourceValidationPanel = memo(function SourceValidationPanel({
  sources,
  onSubmit,
  onSkip,
  defaultCollapsed = false,
  className,
}: SourceValidationPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [decisions, setDecisions] = useState<Map<string, ValidationDecision>>(
    () => new Map(sources.map((s) => [s.id, "pending"]))
  );

  // Issue 1 Fix: Reset decisions when sources prop changes
  useEffect(() => {
    setDecisions(new Map(sources.map((s) => [s.id, "pending"])));
  }, [sources]);

  // Compute statistics
  const stats = useMemo(() => {
    const values = Array.from(decisions.values());
    return {
      approved: values.filter((v) => v === "approved").length,
      rejected: values.filter((v) => v === "rejected").length,
      pending: values.filter((v) => v === "pending").length,
      total: sources.length,
    };
  }, [decisions, sources.length]);

  // Toggle individual source decision
  const toggleDecision = useCallback((sourceId: string) => {
    setDecisions((prev) => {
      const newMap = new Map(prev);
      const current = newMap.get(sourceId) || "pending";
      const next: ValidationDecision =
        current === "pending"
          ? "approved"
          : current === "approved"
            ? "rejected"
            : "pending";
      newMap.set(sourceId, next);
      return newMap;
    });
  }, []);

  // Set specific decision
  const setDecision = useCallback(
    (sourceId: string, decision: ValidationDecision) => {
      setDecisions((prev) => {
        const newMap = new Map(prev);
        newMap.set(sourceId, decision);
        return newMap;
      });
    },
    []
  );

  // Approve all pending
  const approveAll = useCallback(() => {
    setDecisions((prev) => {
      const newMap = new Map(prev);
      for (const [id, decision] of newMap) {
        if (decision === "pending") {
          newMap.set(id, "approved");
        }
      }
      return newMap;
    });
  }, []);

  // Reject all pending
  const rejectAll = useCallback(() => {
    setDecisions((prev) => {
      const newMap = new Map(prev);
      for (const [id, decision] of newMap) {
        if (decision === "pending") {
          newMap.set(id, "rejected");
        }
      }
      return newMap;
    });
  }, []);

  // Handle submit
  const handleSubmit = useCallback(() => {
    const approvedIds = sources
      .filter((s) => decisions.get(s.id) === "approved")
      .map((s) => s.id);
    onSubmit(approvedIds);
  }, [sources, decisions, onSubmit]);

  // Handle skip
  const handleSkip = useCallback(() => {
    if (onSkip) {
      onSkip();
    } else {
      onSubmit(sources.map((s) => s.id));
    }
  }, [sources, onSubmit, onSkip]);

  return (
    <div
      className={cn(
        "border-2 border-amber-400 rounded-lg bg-amber-50/30 overflow-hidden",
        className
      )}
    >
      {/* Header - always visible */}
      <button
        type="button"
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-amber-50/50 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-amber-500" />
          <span className="font-medium text-slate-900">
            Review Sources ({sources.length})
          </span>
        </div>
        <div className="flex items-center gap-2">
          {/* Mini stats when collapsed */}
          {isCollapsed && (
            <div className="flex items-center gap-2 mr-2">
              {stats.approved > 0 && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                  {stats.approved} <CheckCircle2 className="h-3 w-3 ml-1" />
                </span>
              )}
              {stats.pending > 0 && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
                  {stats.pending} pending
                </span>
              )}
            </div>
          )}
          {isCollapsed ? (
            <ChevronDown className="h-5 w-5 text-slate-500" />
          ) : (
            <ChevronUp className="h-5 w-5 text-slate-500" />
          )}
        </div>
      </button>

      {/* Expandable content */}
      {!isCollapsed && (
        <div className="px-4 pb-4">
          {/* Statistics */}
          <div className="flex items-center gap-3 py-2 mb-3 border-b border-amber-200">
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
              <AlertTriangle className="h-3 w-3 mr-1" />
              {stats.pending} pending
            </span>
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              {stats.approved} approved
            </span>
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">
              <XCircle className="h-3 w-3 mr-1" />
              {stats.rejected} rejected
            </span>
          </div>

          {/* Quick actions */}
          <div className="flex items-center gap-2 mb-3">
            <button
              type="button"
              onClick={approveAll}
              disabled={stats.pending === 0}
              className={cn(
                "inline-flex items-center px-2 py-1 rounded-md text-xs font-medium transition-colors",
                "border border-emerald-300 text-emerald-700 hover:bg-emerald-50",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              <CheckCheck className="h-3 w-3 mr-1" />
              Approve All
            </button>
            <button
              type="button"
              onClick={rejectAll}
              disabled={stats.pending === 0}
              className={cn(
                "inline-flex items-center px-2 py-1 rounded-md text-xs font-medium transition-colors",
                "border border-red-300 text-red-700 hover:bg-red-50",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              <XSquare className="h-3 w-3 mr-1" />
              Reject All
            </button>
          </div>

          {/* Source cards */}
          <div className="max-h-[400px] overflow-y-auto">
            <div className="space-y-2">
              {sources.map((source, index) => (
                <SourceValidationCard
                  key={source.id}
                  source={source}
                  index={index}
                  validationStatus={decisions.get(source.id) || "pending"}
                  onToggle={() => toggleDecision(source.id)}
                  onApprove={() => setDecision(source.id, "approved")}
                  onReject={() => setDecision(source.id, "rejected")}
                />
              ))}
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center justify-between mt-4 pt-3 border-t border-amber-200">
            <button
              type="button"
              onClick={handleSkip}
              className={cn(
                "inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                "text-slate-600 hover:bg-slate-100"
              )}
            >
              Skip & Use All
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={stats.approved === 0}
              className={cn(
                "inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                "bg-indigo-600 text-white hover:bg-indigo-700",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              <CheckCircle2 className="h-4 w-4 mr-1" />
              Continue with {stats.approved} Source{stats.approved !== 1 ? "s" : ""}
            </button>
          </div>
        </div>
      )}
    </div>
  );
});

export default SourceValidationPanel;
