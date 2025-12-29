"use client";

import { memo, useState, useCallback, useMemo, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  CheckCheck,
  XSquare,
  Loader2,
  X,
} from "lucide-react";
import { SourceValidationCard } from "./components/SourceValidationCard";
import type { Source, ValidationDecision } from "@/types/copilot";

interface SourceValidationDialogProps {
  /** Whether the dialog is open */
  open: boolean;
  /** Sources requiring validation */
  sources: Source[];
  /** Callback when user submits validation decisions */
  onSubmit: (approvedSourceIds: string[]) => void;
  /** Callback to cancel validation (skip HITL) */
  onCancel?: () => void;
  /** Optional title override */
  title?: string;
  /** Optional description override */
  description?: string;
  /** Whether submission is in progress */
  isSubmitting?: boolean;
}

/**
 * SourceValidationDialog provides a modal interface for reviewing
 * and approving/rejecting retrieved sources before AI answer generation.
 *
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * Design System:
 * - Amber-400 (#FBBF24) for pending/attention items
 * - Emerald-500 (#10B981) for approved sources
 * - Red-500 for rejected sources
 * - Indigo-600 (#4F46E5) for primary actions
 */
export const SourceValidationDialog = memo(function SourceValidationDialog({
  open,
  sources,
  onSubmit,
  onCancel,
  title = "Review Retrieved Sources",
  description = "Review and approve the sources that should be used to generate your answer. Rejected sources will be excluded.",
  isSubmitting = false,
}: SourceValidationDialogProps) {
  // Track validation decisions for each source
  const [decisions, setDecisions] = useState<Map<string, ValidationDecision>>(
    () => new Map(sources.map((s) => [s.id, "pending"]))
  );

  // Issue 5 Fix: Focus trap refs
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousActiveElementRef = useRef<HTMLElement | null>(null);
  const firstFocusableRef = useRef<HTMLButtonElement>(null);
  const lastFocusableRef = useRef<HTMLButtonElement>(null);

  // Reset decisions when sources change
  useEffect(() => {
    setDecisions(new Map(sources.map((s) => [s.id, "pending"])));
  }, [sources]);

  // Issue 5 Fix: Focus trap and focus restoration
  useEffect(() => {
    if (open) {
      // Store the currently focused element to restore later
      previousActiveElementRef.current = document.activeElement as HTMLElement;
      
      // Focus the first focusable element when dialog opens
      // Use setTimeout to ensure the dialog is rendered
      const timer = setTimeout(() => {
        if (firstFocusableRef.current) {
          firstFocusableRef.current.focus();
        }
      }, 0);
      
      return () => clearTimeout(timer);
    } else {
      // Restore focus when dialog closes
      if (previousActiveElementRef.current) {
        previousActiveElementRef.current.focus();
        previousActiveElementRef.current = null;
      }
    }
  }, [open]);

  // Issue 5 Fix: Handle keyboard events for focus trap and Escape
  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onCancel?.();
      return;
    }

    if (event.key === "Tab") {
      // If shift+tab on first element, move to last
      if (event.shiftKey && document.activeElement === firstFocusableRef.current) {
        event.preventDefault();
        lastFocusableRef.current?.focus();
      }
      // If tab on last element, move to first
      else if (!event.shiftKey && document.activeElement === lastFocusableRef.current) {
        event.preventDefault();
        firstFocusableRef.current?.focus();
      }
    }
  }, [onCancel]);

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
      // Cycle: pending -> approved -> rejected -> pending
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

  // Set specific decision for a source
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

  // Approve all pending sources
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

  // Reject all pending sources
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

  // Reset all to pending
  const resetAll = useCallback(() => {
    setDecisions(new Map(sources.map((s) => [s.id, "pending"])));
  }, [sources]);

  // Handle submit
  const handleSubmit = useCallback(() => {
    // Issue 10 Fix: Double-submit guard
    if (isSubmitting) {
      return;
    }
    const approvedIds = sources
      .filter((s) => decisions.get(s.id) === "approved")
      .map((s) => s.id);
    onSubmit(approvedIds);
  }, [sources, decisions, onSubmit, isSubmitting]);

  // Handle skip (approve all automatically)
  const handleSkip = useCallback(() => {
    // Issue 10 Fix: Double-submit guard
    if (isSubmitting) {
      return;
    }
    const allIds = sources.map((s) => s.id);
    onSubmit(allIds);
  }, [sources, onSubmit, isSubmitting]);

  if (!open) {
    return null;
  }

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 z-50 flex items-center justify-center"
      role="dialog"
      aria-modal="true"
      aria-labelledby="validation-dialog-title"
      onKeyDown={handleKeyDown}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onCancel}
        aria-hidden="true"
      />

      {/* Dialog content */}
      <div className="relative bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[85vh] flex flex-col mx-4">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              <h2
                id="validation-dialog-title"
                className="text-lg font-semibold text-slate-900"
              >
                {title}
              </h2>
            </div>
            {onCancel && (
              <button
                ref={firstFocusableRef}
                type="button"
                onClick={onCancel}
                className="text-slate-400 hover:text-slate-600 transition-colors"
                aria-label="Close dialog"
              >
                <X className="h-5 w-5" />
              </button>
            )}
          </div>
          <p className="mt-1 text-sm text-slate-600">{description}</p>
        </div>

        {/* Issue 9 Fix: Statistics bar with ARIA live region */}
        <div
          className="flex items-center gap-4 px-6 py-3 border-b border-slate-100 bg-slate-50"
          aria-live="polite"
          aria-atomic="true"
        >
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
            <AlertTriangle className="h-3 w-3 mr-1" />
            {stats.pending} pending
          </span>
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            {stats.approved} approved
          </span>
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">
            <XCircle className="h-3 w-3 mr-1" />
            {stats.rejected} rejected
          </span>
        </div>

        {/* Quick actions */}
        <div className="flex items-center gap-2 px-6 py-3 border-b border-slate-100">
          <button
            type="button"
            onClick={approveAll}
            disabled={stats.pending === 0}
            className={cn(
              "inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
              "border border-emerald-300 text-emerald-700 hover:bg-emerald-50",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <CheckCheck className="h-4 w-4 mr-1" />
            Approve All Pending
          </button>
          <button
            type="button"
            onClick={rejectAll}
            disabled={stats.pending === 0}
            className={cn(
              "inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
              "border border-red-300 text-red-700 hover:bg-red-50",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <XSquare className="h-4 w-4 mr-1" />
            Reject All Pending
          </button>
          <button
            type="button"
            onClick={resetAll}
            disabled={stats.approved === 0 && stats.rejected === 0}
            className={cn(
              "inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
              "text-slate-600 hover:bg-slate-100",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            Reset
          </button>
        </div>

        {/* Source cards - scrollable */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="space-y-3">
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

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 bg-slate-50">
          <button
            type="button"
            onClick={handleSkip}
            disabled={isSubmitting}
            className={cn(
              "inline-flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors",
              "text-slate-600 hover:bg-slate-200",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            Skip & Use All
          </button>
          <div className="flex items-center gap-3">
            {onCancel && (
              <button
                type="button"
                onClick={onCancel}
                disabled={isSubmitting}
                className={cn(
                  "inline-flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors",
                  "border border-slate-300 text-slate-700 hover:bg-slate-100",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                Cancel
              </button>
            )}
            <button
              ref={lastFocusableRef}
              type="button"
              onClick={handleSubmit}
              disabled={isSubmitting || stats.approved === 0}
              className={cn(
                "inline-flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors",
                "bg-indigo-600 text-white hover:bg-indigo-700",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  Continue with {stats.approved} Source
                  {stats.approved !== 1 ? "s" : ""}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

export default SourceValidationDialog;
