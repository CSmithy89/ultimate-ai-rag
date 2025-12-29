# Story 6-4: Human-in-the-Loop Source Validation

Status: drafted
Epic: 6 - Interactive Copilot Experience
Priority: High
Depends on: Story 6-3 (Generative UI Components)

## User Story

As a **researcher**,
I want **to review and approve/reject retrieved sources before the AI generates an answer**,
So that **I can ensure the AI uses only trusted, relevant information and maintain control over the knowledge used in responses**.

## Acceptance Criteria

- Given the AI retrieves sources for a query
- When HITL validation is triggered before answer synthesis
- Then a modal/panel displays retrieved source cards awaiting validation
- And each card uses Amber-400 (#FBBF24) visual indicator for pending validation status
- And users can Approve or Reject each source individually
- And users can use "Approve All" / "Reject All" shortcuts
- And rejected sources are excluded from answer synthesis
- And the backend pauses generation until human approves/rejects sources
- And approval status is tracked in agent state
- And validation is non-blocking if user continues typing new queries
- And the UI follows the "Professional Forge" design direction

## Technical Approach

### 1. Create SourceValidationDialog Component

**File:** `frontend/components/copilot/SourceValidationDialog.tsx`

Create a modal dialog for reviewing and approving/rejecting sources before answer generation:

```typescript
"use client";

import { memo, useState, useCallback, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  CheckCheck,
  XSquare,
  Loader2,
} from "lucide-react";
import { SourceValidationCard } from "./components/SourceValidationCard";
import type { Source } from "@/types/copilot";

/**
 * Validation decision for a source.
 */
export type ValidationDecision = "approved" | "rejected" | "pending";

/**
 * Source with validation state.
 */
export interface ValidatableSource extends Source {
  validationStatus: ValidationDecision;
}

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

  // Reset decisions when sources change
  useMemo(() => {
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

  // Check if all sources have been reviewed
  const allReviewed = stats.pending === 0;

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
    const approvedIds = sources
      .filter((s) => decisions.get(s.id) === "approved")
      .map((s) => s.id);
    onSubmit(approvedIds);
  }, [sources, decisions, onSubmit]);

  // Handle skip (approve all automatically)
  const handleSkip = useCallback(() => {
    const allIds = sources.map((s) => s.id);
    onSubmit(allIds);
  }, [sources, onSubmit]);

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel?.()}>
      <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            {title}
          </DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        {/* Statistics bar */}
        <div className="flex items-center gap-4 py-2 px-1 border-b border-slate-100">
          <Badge
            variant="outline"
            className="bg-amber-50 text-amber-700 border-amber-200"
          >
            <AlertTriangle className="h-3 w-3 mr-1" />
            {stats.pending} pending
          </Badge>
          <Badge
            variant="outline"
            className="bg-emerald-50 text-emerald-700 border-emerald-200"
          >
            <CheckCircle2 className="h-3 w-3 mr-1" />
            {stats.approved} approved
          </Badge>
          <Badge
            variant="outline"
            className="bg-red-50 text-red-700 border-red-200"
          >
            <XCircle className="h-3 w-3 mr-1" />
            {stats.rejected} rejected
          </Badge>
        </div>

        {/* Quick actions */}
        <div className="flex items-center gap-2 py-2">
          <Button
            variant="outline"
            size="sm"
            onClick={approveAll}
            disabled={stats.pending === 0}
            className="text-emerald-700 border-emerald-300 hover:bg-emerald-50"
          >
            <CheckCheck className="h-4 w-4 mr-1" />
            Approve All Pending
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={rejectAll}
            disabled={stats.pending === 0}
            className="text-red-700 border-red-300 hover:bg-red-50"
          >
            <XSquare className="h-4 w-4 mr-1" />
            Reject All Pending
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetAll}
            disabled={stats.approved === 0 && stats.rejected === 0}
          >
            Reset
          </Button>
        </div>

        {/* Source cards */}
        <ScrollArea className="flex-1 -mx-6 px-6">
          <div className="space-y-3 py-2">
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
        </ScrollArea>

        <DialogFooter className="flex-row justify-between sm:justify-between border-t border-slate-100 pt-4">
          <Button variant="ghost" onClick={handleSkip} disabled={isSubmitting}>
            Skip & Use All
          </Button>
          <div className="flex gap-2">
            {onCancel && (
              <Button variant="outline" onClick={onCancel} disabled={isSubmitting}>
                Cancel
              </Button>
            )}
            <Button
              onClick={handleSubmit}
              disabled={isSubmitting || stats.approved === 0}
              className="bg-indigo-600 hover:bg-indigo-700"
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
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
});

export default SourceValidationDialog;
```

Key implementation details:
- Modal dialog using shadcn/ui Dialog component
- Three-state validation: pending (Amber), approved (Emerald), rejected (Red)
- Statistics bar showing counts of each state
- Quick actions: Approve All, Reject All, Reset
- Skip option to approve all without manual review
- Submit disabled until at least one source is approved
- Scrollable source list for many sources

### 2. Create SourceValidationCard Component

**File:** `frontend/components/copilot/components/SourceValidationCard.tsx`

Create an enhanced SourceCard with approve/reject buttons and Amber-400 pending indicator:

```typescript
"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  FileText,
  Globe,
  Database,
  Share2,
  ExternalLink,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Check,
  X,
} from "lucide-react";
import type { Source } from "@/types/copilot";
import type { ValidationDecision } from "../SourceValidationDialog";

/**
 * Source type to icon mapping.
 */
const sourceTypeIcons: Record<string, React.ElementType> = {
  document: FileText,
  web: Globe,
  database: Database,
  knowledge_graph: Share2,
  default: FileText,
};

/**
 * Get border and background colors based on validation status.
 * Design System:
 * - Pending: Amber-400 (#FBBF24) for HITL attention
 * - Approved: Emerald-500 (#10B981) for validated
 * - Rejected: Red-500 for excluded
 */
function getValidationStyles(status: ValidationDecision): string {
  switch (status) {
    case "approved":
      return "border-emerald-400 bg-emerald-50/50 ring-1 ring-emerald-200";
    case "rejected":
      return "border-red-300 bg-red-50/30 ring-1 ring-red-200 opacity-60";
    case "pending":
    default:
      return "border-amber-400 bg-amber-50/50 ring-1 ring-amber-200";
  }
}

/**
 * Get status icon based on validation status.
 */
function getStatusIcon(status: ValidationDecision): React.ReactNode {
  switch (status) {
    case "approved":
      return <CheckCircle2 className="h-5 w-5 text-emerald-500" />;
    case "rejected":
      return <XCircle className="h-5 w-5 text-red-500" />;
    case "pending":
    default:
      return <AlertCircle className="h-5 w-5 text-amber-500" />;
  }
}

/**
 * Get confidence level color based on similarity score.
 */
function getConfidenceColor(similarity: number): string {
  if (similarity >= 0.9) return "bg-emerald-100 text-emerald-800 border-emerald-200";
  if (similarity >= 0.7) return "bg-indigo-100 text-indigo-800 border-indigo-200";
  if (similarity >= 0.5) return "bg-amber-100 text-amber-800 border-amber-200";
  return "bg-slate-100 text-slate-800 border-slate-200";
}

interface SourceValidationCardProps {
  /** The source data to display */
  source: Source;
  /** 0-based index for numbering sources */
  index: number;
  /** Current validation status */
  validationStatus: ValidationDecision;
  /** Toggle between validation states */
  onToggle: () => void;
  /** Explicitly approve this source */
  onApprove: () => void;
  /** Explicitly reject this source */
  onReject: () => void;
}

/**
 * SourceValidationCard displays a source with approve/reject controls
 * for Human-in-the-Loop validation before answer generation.
 *
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * Features:
 * - Amber-400 visual indicator for pending validation
 * - Emerald-500 for approved sources
 * - Red-500 for rejected sources
 * - Approve/Reject buttons
 * - Click to toggle status
 * - Source preview with confidence indicator
 */
export const SourceValidationCard = memo(function SourceValidationCard({
  source,
  index,
  validationStatus,
  onToggle,
  onApprove,
  onReject,
}: SourceValidationCardProps) {
  const sourceType = (source.metadata?.type as string) || "default";
  const IconComponent = sourceTypeIcons[sourceType] || sourceTypeIcons.default;
  const confidencePercent = Math.round(source.similarity * 100);
  const sourceUrl = source.metadata?.url as string | undefined;

  return (
    <div
      className={cn(
        "rounded-lg border-2 p-4 transition-all duration-200",
        "cursor-pointer hover:shadow-md",
        getValidationStyles(validationStatus)
      )}
      onClick={onToggle}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onToggle();
        }
      }}
      aria-label={`Source ${index + 1}: ${source.title}. Status: ${validationStatus}. Click to change.`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        {/* Left side: status icon, index, source icon, title */}
        <div className="flex items-start gap-3 min-w-0 flex-1">
          {/* Status indicator */}
          <div className="flex-shrink-0 mt-0.5">
            {getStatusIcon(validationStatus)}
          </div>

          {/* Index badge */}
          <span
            className={cn(
              "flex-shrink-0 w-6 h-6 rounded-full text-white text-sm font-medium flex items-center justify-center",
              validationStatus === "rejected" ? "bg-slate-400" : "bg-indigo-600"
            )}
          >
            {index + 1}
          </span>

          {/* Source info */}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1">
              <IconComponent
                className="h-4 w-4 text-slate-500 flex-shrink-0"
                aria-hidden="true"
              />
              <h4
                className={cn(
                  "text-sm font-medium truncate",
                  validationStatus === "rejected"
                    ? "text-slate-500 line-through"
                    : "text-slate-900"
                )}
              >
                {source.title}
              </h4>
            </div>

            {/* Preview text */}
            <p
              className={cn(
                "text-sm line-clamp-2",
                validationStatus === "rejected"
                  ? "text-slate-400"
                  : "text-slate-600"
              )}
            >
              {source.preview}
            </p>

            {/* Metadata row */}
            <div className="flex items-center gap-3 mt-2">
              {/* Confidence badge */}
              <span
                className={cn(
                  "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono border",
                  getConfidenceColor(source.similarity)
                )}
              >
                {confidencePercent}% match
              </span>

              {/* External link */}
              {sourceUrl && (
                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800"
                  onClick={(e) => e.stopPropagation()}
                  aria-label={`View source: ${source.title}`}
                >
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                  View
                </a>
              )}
            </div>
          </div>
        </div>

        {/* Right side: action buttons */}
        <div
          className="flex flex-col gap-1 flex-shrink-0"
          onClick={(e) => e.stopPropagation()}
        >
          <Button
            variant={validationStatus === "approved" ? "default" : "outline"}
            size="sm"
            onClick={onApprove}
            className={cn(
              "h-8 w-8 p-0",
              validationStatus === "approved"
                ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                : "text-emerald-600 border-emerald-300 hover:bg-emerald-50"
            )}
            aria-label="Approve source"
          >
            <Check className="h-4 w-4" />
          </Button>
          <Button
            variant={validationStatus === "rejected" ? "default" : "outline"}
            size="sm"
            onClick={onReject}
            className={cn(
              "h-8 w-8 p-0",
              validationStatus === "rejected"
                ? "bg-red-500 hover:bg-red-600 text-white"
                : "text-red-600 border-red-300 hover:bg-red-50"
            )}
            aria-label="Reject source"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
});

export default SourceValidationCard;
```

Key implementation details:
- Amber-400 border and background for pending sources (per design system)
- Emerald-500 for approved, Red-500 for rejected
- Status icon prominently displayed on left
- Approve/Reject buttons on right side
- Click anywhere on card to toggle state
- Visual strikethrough on rejected source titles
- Reduced opacity for rejected sources
- Keyboard accessible (Enter/Space to toggle)

### 3. Create use-source-validation Hook

**File:** `frontend/hooks/use-source-validation.ts`

Create a custom hook for managing validation state and CopilotKit integration:

```typescript
"use client";

import { useState, useCallback, useRef } from "react";
import { useCopilotAction } from "@copilotkit/react-core";
import type { Source } from "@/types/copilot";
import type { ValidationDecision } from "@/components/copilot/SourceValidationDialog";

/**
 * State for source validation.
 */
export interface SourceValidationState {
  /** Whether validation is currently in progress */
  isValidating: boolean;
  /** Sources awaiting validation */
  pendingSources: Source[];
  /** Map of source ID to validation decision */
  decisions: Map<string, ValidationDecision>;
  /** IDs of approved sources */
  approvedIds: string[];
  /** IDs of rejected sources */
  rejectedIds: string[];
  /** Whether submission is in progress */
  isSubmitting: boolean;
  /** Error message if validation failed */
  error: string | null;
}

/**
 * Options for the useSourceValidation hook.
 */
export interface UseSourceValidationOptions {
  /** Callback when validation completes */
  onValidationComplete?: (approvedIds: string[]) => void;
  /** Callback when validation is cancelled */
  onValidationCancelled?: () => void;
  /** Whether to auto-approve sources below a confidence threshold */
  autoApproveThreshold?: number;
  /** Whether to auto-reject sources below a confidence threshold */
  autoRejectThreshold?: number;
}

/**
 * Return type for the useSourceValidation hook.
 */
export interface UseSourceValidationReturn {
  /** Current validation state */
  state: SourceValidationState;
  /** Open the validation dialog with sources */
  startValidation: (sources: Source[]) => void;
  /** Submit validation decisions */
  submitValidation: (approvedIds: string[]) => void;
  /** Cancel validation */
  cancelValidation: () => void;
  /** Reset validation state */
  resetValidation: () => void;
  /** Whether the validation dialog should be open */
  isDialogOpen: boolean;
}

/**
 * useSourceValidation hook manages Human-in-the-Loop source validation
 * state and integrates with CopilotKit's renderAndWait pattern.
 *
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * @example
 * ```tsx
 * function ChatWithHITL() {
 *   const {
 *     state,
 *     isDialogOpen,
 *     submitValidation,
 *     cancelValidation,
 *   } = useSourceValidation({
 *     onValidationComplete: (ids) => console.log("Approved:", ids),
 *   });
 *
 *   return (
 *     <>
 *       <ChatSidebar />
 *       <SourceValidationDialog
 *         open={isDialogOpen}
 *         sources={state.pendingSources}
 *         onSubmit={submitValidation}
 *         onCancel={cancelValidation}
 *       />
 *     </>
 *   );
 * }
 * ```
 */
export function useSourceValidation(
  options: UseSourceValidationOptions = {}
): UseSourceValidationReturn {
  const {
    onValidationComplete,
    onValidationCancelled,
    autoApproveThreshold,
    autoRejectThreshold,
  } = options;

  // Validation state
  const [state, setState] = useState<SourceValidationState>({
    isValidating: false,
    pendingSources: [],
    decisions: new Map(),
    approvedIds: [],
    rejectedIds: [],
    isSubmitting: false,
    error: null,
  });

  // Reference to the respond function from renderAndWait
  const respondRef = useRef<((response: { approved: string[] }) => void) | null>(
    null
  );

  // Start validation with a set of sources
  const startValidation = useCallback(
    (sources: Source[]) => {
      // Apply auto-approve/reject thresholds if configured
      const decisions = new Map<string, ValidationDecision>();
      let hasAutoDecisions = false;

      for (const source of sources) {
        if (autoApproveThreshold && source.similarity >= autoApproveThreshold) {
          decisions.set(source.id, "approved");
          hasAutoDecisions = true;
        } else if (
          autoRejectThreshold &&
          source.similarity < autoRejectThreshold
        ) {
          decisions.set(source.id, "rejected");
          hasAutoDecisions = true;
        } else {
          decisions.set(source.id, "pending");
        }
      }

      setState({
        isValidating: true,
        pendingSources: sources,
        decisions,
        approvedIds: [],
        rejectedIds: [],
        isSubmitting: false,
        error: null,
      });

      // Log auto-decisions if any
      if (hasAutoDecisions) {
        console.log("[HITL] Auto-applied validation decisions based on thresholds");
      }
    },
    [autoApproveThreshold, autoRejectThreshold]
  );

  // Submit validation decisions
  const submitValidation = useCallback(
    (approvedIds: string[]) => {
      setState((prev) => ({
        ...prev,
        isSubmitting: true,
      }));

      // Compute rejected IDs
      const rejectedIds = prev.pendingSources
        .map((s) => s.id)
        .filter((id) => !approvedIds.includes(id));

      // Call the respond function if available (renderAndWait pattern)
      if (respondRef.current) {
        respondRef.current({ approved: approvedIds });
        respondRef.current = null;
      }

      // Update state
      setState((prev) => ({
        ...prev,
        isValidating: false,
        isSubmitting: false,
        approvedIds,
        rejectedIds,
      }));

      // Call completion callback
      onValidationComplete?.(approvedIds);
    },
    [onValidationComplete]
  );

  // Cancel validation
  const cancelValidation = useCallback(() => {
    // If renderAndWait is active, respond with empty approved list
    if (respondRef.current) {
      respondRef.current({ approved: [] });
      respondRef.current = null;
    }

    setState({
      isValidating: false,
      pendingSources: [],
      decisions: new Map(),
      approvedIds: [],
      rejectedIds: [],
      isSubmitting: false,
      error: null,
    });

    onValidationCancelled?.();
  }, [onValidationCancelled]);

  // Reset validation state
  const resetValidation = useCallback(() => {
    setState({
      isValidating: false,
      pendingSources: [],
      decisions: new Map(),
      approvedIds: [],
      rejectedIds: [],
      isSubmitting: false,
      error: null,
    });
  }, []);

  // Register CopilotKit action with renderAndWait for HITL
  useCopilotAction({
    name: "validate_sources",
    description:
      "Request human approval for retrieved sources before answer generation",
    parameters: [
      {
        name: "sources",
        type: "object[]",
        description: "Array of sources requiring validation",
        required: true,
      },
      {
        name: "query",
        type: "string",
        description: "The original user query for context",
        required: false,
      },
    ],
    // Use renderAndWait to pause agent execution until user responds
    renderAndWait: ({ args, respond }) => {
      const sources = args.sources as Source[];

      // Store respond function for later use
      respondRef.current = respond;

      // Start validation with the sources
      startValidation(sources);

      // The actual UI is rendered by the parent component using the dialog
      // This returns null because we're managing the UI externally
      return null;
    },
  });

  return {
    state,
    startValidation,
    submitValidation,
    cancelValidation,
    resetValidation,
    isDialogOpen: state.isValidating,
  };
}

export default useSourceValidation;
```

Key implementation details:
- Uses `useCopilotAction` with `renderAndWait` pattern to pause agent execution
- Tracks validation state: pending sources, decisions, approved/rejected IDs
- Supports auto-approve/reject based on confidence thresholds
- Stores `respond` function reference to send decisions back to agent
- Provides callbacks for validation complete and cancelled events
- Returns dialog open state for conditional rendering

### 4. Create SourceValidationPanel Component

**File:** `frontend/components/copilot/SourceValidationPanel.tsx`

Create an inline panel alternative to the dialog for non-modal HITL:

```typescript
"use client";

import { memo, useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
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
import type { Source } from "@/types/copilot";
import type { ValidationDecision } from "./SourceValidationDialog";

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
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-amber-50/50 transition-colors"
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
                <Badge variant="outline" className="bg-emerald-50 text-emerald-700 text-xs">
                  {stats.approved} <CheckCircle2 className="h-3 w-3 ml-1" />
                </Badge>
              )}
              {stats.pending > 0 && (
                <Badge variant="outline" className="bg-amber-50 text-amber-700 text-xs">
                  {stats.pending} pending
                </Badge>
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
            <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
              <AlertTriangle className="h-3 w-3 mr-1" />
              {stats.pending} pending
            </Badge>
            <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              {stats.approved} approved
            </Badge>
            <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
              <XCircle className="h-3 w-3 mr-1" />
              {stats.rejected} rejected
            </Badge>
          </div>

          {/* Quick actions */}
          <div className="flex items-center gap-2 mb-3">
            <Button
              variant="outline"
              size="sm"
              onClick={approveAll}
              disabled={stats.pending === 0}
              className="text-emerald-700 border-emerald-300 hover:bg-emerald-50 text-xs"
            >
              <CheckCheck className="h-3 w-3 mr-1" />
              Approve All
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={rejectAll}
              disabled={stats.pending === 0}
              className="text-red-700 border-red-300 hover:bg-red-50 text-xs"
            >
              <XSquare className="h-3 w-3 mr-1" />
              Reject All
            </Button>
          </div>

          {/* Source cards */}
          <ScrollArea className="max-h-[400px]">
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
          </ScrollArea>

          {/* Action buttons */}
          <div className="flex items-center justify-between mt-4 pt-3 border-t border-amber-200">
            <Button variant="ghost" size="sm" onClick={handleSkip}>
              Skip & Use All
            </Button>
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={stats.approved === 0}
              className="bg-indigo-600 hover:bg-indigo-700"
            >
              <CheckCircle2 className="h-4 w-4 mr-1" />
              Continue with {stats.approved} Source{stats.approved !== 1 ? "s" : ""}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
});

export default SourceValidationPanel;
```

Key implementation details:
- Inline panel with Amber-400 border per design system
- Collapsible for non-blocking interaction
- Shows mini stats when collapsed
- Same approval workflow as dialog
- Renders within chat message flow

### 5. Update GenerativeUIRenderer with HITL Action

**File:** `frontend/components/copilot/GenerativeUIRenderer.tsx` (modify)

Add the HITL validation action using `renderAndWait`:

```typescript
"use client";

import { useGenerativeUI } from "@/hooks/use-generative-ui";
import { useSourceValidation } from "@/hooks/use-source-validation";
import { SourceValidationDialog } from "./SourceValidationDialog";
import { SourceValidationPanel } from "./SourceValidationPanel";
import type { Source } from "@/types/copilot";
import type { GraphPreviewNode } from "./components/GraphPreview";

interface GenerativeUIRendererProps {
  onSourceClick?: (source: Source) => void;
  onGraphNodeClick?: (node: GraphPreviewNode) => void;
  onGraphExpand?: () => void;
  /** Use modal dialog (true) or inline panel (false) for HITL */
  useModalForValidation?: boolean;
  /** Callback when HITL validation completes */
  onValidationComplete?: (approvedIds: string[]) => void;
}

/**
 * GenerativeUIRenderer initializes the generative UI action handlers
 * including Human-in-the-Loop source validation.
 *
 * Story 6-3: Generative UI Components
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * @example
 * ```tsx
 * <CopilotSidebar>
 *   <GenerativeUIRenderer
 *     useModalForValidation={true}
 *     onValidationComplete={(ids) => console.log("Approved:", ids)}
 *   />
 * </CopilotSidebar>
 * ```
 */
export function GenerativeUIRenderer({
  onSourceClick,
  onGraphNodeClick,
  onGraphExpand,
  useModalForValidation = true,
  onValidationComplete,
}: GenerativeUIRendererProps) {
  // Initialize generative UI hooks (Story 6-3)
  useGenerativeUI({
    onSourceClick,
    onGraphNodeClick,
    onGraphExpand,
  });

  // Initialize source validation hooks (Story 6-4)
  const {
    state: validationState,
    isDialogOpen,
    submitValidation,
    cancelValidation,
  } = useSourceValidation({
    onValidationComplete,
  });

  return (
    <>
      {/* Modal dialog for HITL validation */}
      {useModalForValidation && (
        <SourceValidationDialog
          open={isDialogOpen}
          sources={validationState.pendingSources}
          onSubmit={submitValidation}
          onCancel={cancelValidation}
          isSubmitting={validationState.isSubmitting}
        />
      )}

      {/* Inline panel for non-modal HITL (rendered in chat flow) */}
      {!useModalForValidation && isDialogOpen && (
        <SourceValidationPanel
          sources={validationState.pendingSources}
          onSubmit={submitValidation}
          onSkip={() => submitValidation(validationState.pendingSources.map((s) => s.id))}
        />
      )}
    </>
  );
}
```

### 6. Backend: Add HITL Checkpoint to Agent

**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` (modify)

Add support for HITL validation in the AG-UI bridge:

```python
from typing import List, Optional, Dict, Any, Callable, Awaitable
import asyncio
from dataclasses import dataclass, field
from enum import Enum

from agentic_rag_backend.models.copilot import (
    AGUIEvent,
    AGUIEventType,
    ToolCallEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ActionRequestEvent,
    StateSnapshotEvent,
)


class HITLStatus(str, Enum):
    """Status of Human-in-the-Loop validation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass
class HITLCheckpoint:
    """Represents a checkpoint waiting for human validation."""

    checkpoint_id: str
    sources: List[Dict[str, Any]]
    query: str
    status: HITLStatus = HITLStatus.PENDING
    approved_source_ids: List[str] = field(default_factory=list)
    rejected_source_ids: List[str] = field(default_factory=list)
    response_event: Optional[asyncio.Event] = field(default_factory=asyncio.Event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "sources": self.sources,
            "query": self.query,
            "status": self.status.value,
            "approved_source_ids": self.approved_source_ids,
            "rejected_source_ids": self.rejected_source_ids,
        }


class AGUIBridgeWithHITL:
    """
    AG-UI Protocol bridge with Human-in-the-Loop support.

    Story 6-4: Human-in-the-Loop Source Validation

    This extends the base AG-UI bridge to support:
    - Pausing generation for source validation
    - Receiving approval/rejection decisions from frontend
    - Resuming generation with approved sources only
    """

    def __init__(self):
        self._pending_checkpoints: Dict[str, HITLCheckpoint] = {}
        self._hitl_timeout: float = 300.0  # 5 minutes default timeout

    async def create_hitl_checkpoint(
        self,
        checkpoint_id: str,
        sources: List[Dict[str, Any]],
        query: str,
    ) -> List[AGUIEvent]:
        """
        Create a HITL checkpoint and return AG-UI events to trigger frontend validation.

        Args:
            checkpoint_id: Unique identifier for this checkpoint
            sources: List of source dicts to validate
            query: The original user query for context

        Returns:
            List of AG-UI events to send to frontend
        """
        checkpoint = HITLCheckpoint(
            checkpoint_id=checkpoint_id,
            sources=sources,
            query=query,
        )
        self._pending_checkpoints[checkpoint_id] = checkpoint

        # Create AG-UI events to trigger frontend renderAndWait
        events = [
            ToolCallEvent(
                tool_call_id=checkpoint_id,
                tool_name="validate_sources",
            ),
            ToolCallArgsEvent(
                tool_call_id=checkpoint_id,
                args={
                    "sources": sources,
                    "query": query,
                    "checkpoint_id": checkpoint_id,
                },
            ),
            # Note: We don't send ToolCallEndEvent until validation completes
        ]

        return events

    async def wait_for_validation(
        self,
        checkpoint_id: str,
        timeout: Optional[float] = None,
    ) -> HITLCheckpoint:
        """
        Wait for human validation decision on a checkpoint.

        Args:
            checkpoint_id: The checkpoint to wait for
            timeout: Optional timeout in seconds (default: 300s)

        Returns:
            The checkpoint with validation results

        Raises:
            asyncio.TimeoutError: If validation times out
            KeyError: If checkpoint not found
        """
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")

        timeout = timeout or self._hitl_timeout

        try:
            await asyncio.wait_for(
                checkpoint.response_event.wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # On timeout, treat as "skip" - approve all sources
            checkpoint.status = HITLStatus.SKIPPED
            checkpoint.approved_source_ids = [s["id"] for s in checkpoint.sources]

        return checkpoint

    def receive_validation_response(
        self,
        checkpoint_id: str,
        approved_source_ids: List[str],
    ) -> List[AGUIEvent]:
        """
        Receive validation response from frontend.

        Args:
            checkpoint_id: The checkpoint being responded to
            approved_source_ids: List of approved source IDs

        Returns:
            AG-UI events to signal completion
        """
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")

        # Update checkpoint with decisions
        all_source_ids = {s["id"] for s in checkpoint.sources}
        checkpoint.approved_source_ids = approved_source_ids
        checkpoint.rejected_source_ids = list(
            all_source_ids - set(approved_source_ids)
        )
        checkpoint.status = (
            HITLStatus.APPROVED if approved_source_ids else HITLStatus.REJECTED
        )

        # Signal waiting coroutine
        checkpoint.response_event.set()

        # Create completion events
        events = [
            ToolCallEndEvent(tool_call_id=checkpoint_id),
            StateSnapshotEvent(
                state={
                    "hitl_checkpoint": checkpoint.to_dict(),
                    "approved_sources": [
                        s for s in checkpoint.sources
                        if s["id"] in approved_source_ids
                    ],
                }
            ),
        ]

        return events

    def get_approved_sources(
        self,
        checkpoint_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get the approved sources from a completed checkpoint.

        Args:
            checkpoint_id: The completed checkpoint

        Returns:
            List of approved source dicts
        """
        checkpoint = self._pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            return []

        return [
            s for s in checkpoint.sources
            if s["id"] in checkpoint.approved_source_ids
        ]

    def cleanup_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a checkpoint from memory."""
        self._pending_checkpoints.pop(checkpoint_id, None)


# Helper function to create HITL events
def create_validate_sources_events(
    sources: List[Dict[str, Any]],
    query: str,
    checkpoint_id: Optional[str] = None,
) -> List[AGUIEvent]:
    """
    Create AG-UI events to trigger source validation on frontend.

    This is a convenience function for triggering HITL validation.
    """
    import uuid

    checkpoint_id = checkpoint_id or str(uuid.uuid4())

    return [
        ToolCallEvent(
            tool_call_id=checkpoint_id,
            tool_name="validate_sources",
        ),
        ToolCallArgsEvent(
            tool_call_id=checkpoint_id,
            args={
                "sources": sources,
                "query": query,
                "checkpoint_id": checkpoint_id,
            },
        ),
    ]
```

### 7. Backend: Add HITL Endpoint

**File:** `backend/src/agentic_rag_backend/api/routes/copilot.py` (modify)

Add endpoint for receiving validation decisions:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])


class ValidationResponseRequest(BaseModel):
    """Request body for HITL validation response."""

    checkpoint_id: str = Field(..., description="ID of the checkpoint being responded to")
    approved_source_ids: List[str] = Field(
        default_factory=list,
        description="List of approved source IDs"
    )


class ValidationResponseResult(BaseModel):
    """Response for HITL validation endpoint."""

    checkpoint_id: str
    status: str
    approved_count: int
    rejected_count: int


@router.post("/validation-response", response_model=ValidationResponseResult)
async def receive_validation_response(
    request: ValidationResponseRequest,
    # Inject AG-UI bridge instance
    # ag_ui_bridge: AGUIBridgeWithHITL = Depends(get_ag_ui_bridge),
) -> ValidationResponseResult:
    """
    Receive Human-in-the-Loop validation response from frontend.

    Story 6-4: Human-in-the-Loop Source Validation

    This endpoint receives the user's approval/rejection decisions
    and signals the waiting agent to continue with approved sources.
    """
    try:
        # TODO: Inject actual bridge from dependency
        # events = ag_ui_bridge.receive_validation_response(
        #     checkpoint_id=request.checkpoint_id,
        #     approved_source_ids=request.approved_source_ids,
        # )

        # For now, return mock response
        return ValidationResponseResult(
            checkpoint_id=request.checkpoint_id,
            status="approved" if request.approved_source_ids else "rejected",
            approved_count=len(request.approved_source_ids),
            rejected_count=0,  # Calculated from total - approved
        )

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {request.checkpoint_id} not found"
        )
```

### 8. Update Types

**File:** `frontend/types/copilot.ts` (modify)

Add types for HITL validation:

```typescript
// Add to existing types file

/**
 * Validation decision for a source in HITL.
 */
export type ValidationDecision = "approved" | "rejected" | "pending";

/**
 * Source with validation state for HITL.
 */
export interface ValidatableSource extends Source {
  validationStatus: ValidationDecision;
}

/**
 * State of a HITL validation checkpoint.
 */
export interface HITLCheckpoint {
  checkpointId: string;
  sources: Source[];
  query: string;
  status: "pending" | "approved" | "rejected" | "skipped";
  approvedSourceIds: string[];
  rejectedSourceIds: string[];
}

/**
 * Response format for validation submission.
 */
export interface ValidationResponse {
  checkpointId: string;
  status: string;
  approvedCount: number;
  rejectedCount: number;
}

// Zod schemas
export const ValidationDecisionSchema = z.enum(["approved", "rejected", "pending"]);

export const ValidatableSourceSchema = SourceSchema.extend({
  validationStatus: ValidationDecisionSchema,
});

export const HITLCheckpointSchema = z.object({
  checkpointId: z.string(),
  sources: z.array(SourceSchema),
  query: z.string(),
  status: z.enum(["pending", "approved", "rejected", "skipped"]),
  approvedSourceIds: z.array(z.string()),
  rejectedSourceIds: z.array(z.string()),
});

export const ValidationResponseSchema = z.object({
  checkpointId: z.string(),
  status: z.string(),
  approvedCount: z.number(),
  rejectedCount: z.number(),
});
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `frontend/components/copilot/SourceValidationDialog.tsx` | Modal dialog for HITL source validation |
| `frontend/components/copilot/SourceValidationPanel.tsx` | Inline panel alternative for non-modal HITL |
| `frontend/components/copilot/components/SourceValidationCard.tsx` | Enhanced SourceCard with approve/reject buttons and Amber-400 indicator |
| `frontend/hooks/use-source-validation.ts` | Custom hook for validation state and CopilotKit renderAndWait integration |

### Modified Files

| File | Change |
|------|--------|
| `frontend/types/copilot.ts` | Add ValidationDecision, ValidatableSource, HITLCheckpoint types |
| `frontend/components/copilot/GenerativeUIRenderer.tsx` | Add HITL validation dialog/panel rendering |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Add HITLCheckpoint class and validation methods |
| `backend/src/agentic_rag_backend/api/routes/copilot.py` | Add POST /validation-response endpoint |
| `backend/src/agentic_rag_backend/agents/orchestrator.py` | Add HITL checkpoint before answer synthesis |

## Dependencies

### Frontend Dependencies (npm)

Already installed from previous stories:
```json
{
  "@copilotkit/react-core": "^1.50.1",
  "@copilotkit/react-ui": "^1.50.1",
  "lucide-react": "^0.x.x"
}
```

shadcn/ui components required (already installed):
- Dialog
- Button
- Badge
- ScrollArea

### Backend Dependencies (pip)

No new dependencies required. Uses existing:
- Pydantic models
- FastAPI routing
- asyncio for HITL waiting

### Environment Variables

No new environment variables required.

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| SourceValidationCard renders pending state with Amber-400 | `frontend/__tests__/components/copilot/SourceValidationCard.test.tsx` |
| SourceValidationCard toggles between states | `frontend/__tests__/components/copilot/SourceValidationCard.test.tsx` |
| SourceValidationDialog approve/reject all works | `frontend/__tests__/components/copilot/SourceValidationDialog.test.tsx` |
| SourceValidationDialog statistics update correctly | `frontend/__tests__/components/copilot/SourceValidationDialog.test.tsx` |
| SourceValidationPanel collapse/expand works | `frontend/__tests__/components/copilot/SourceValidationPanel.test.tsx` |
| useSourceValidation starts validation correctly | `frontend/__tests__/hooks/use-source-validation.test.ts` |
| useSourceValidation submits approved IDs | `frontend/__tests__/hooks/use-source-validation.test.ts` |
| useSourceValidation auto-approve threshold works | `frontend/__tests__/hooks/use-source-validation.test.ts` |

### Integration Tests

| Test | Location |
|------|----------|
| HITL dialog opens when validate_sources action triggers | `frontend/__tests__/integration/hitl.test.tsx` |
| Approved sources are sent back to agent | `frontend/__tests__/integration/hitl.test.tsx` |
| Skip & Use All approves all sources | `frontend/__tests__/integration/hitl.test.tsx` |
| Cancel validation sends empty approved list | `frontend/__tests__/integration/hitl.test.tsx` |

### E2E Tests

| Test | Location |
|------|----------|
| Full HITL flow: query -> sources -> approval -> answer | `frontend/tests/e2e/hitl.spec.ts` |
| HITL blocks synthesis until decision | `frontend/tests/e2e/hitl.spec.ts` |
| Rejected sources excluded from answer | `frontend/tests/e2e/hitl.spec.ts` |
| Skip validation generates answer with all sources | `frontend/tests/e2e/hitl.spec.ts` |

### Manual Verification Steps

1. Start backend with `cd backend && uv run uvicorn agentic_rag_backend.main:app --reload`
2. Start frontend with `cd frontend && pnpm dev`
3. Open browser to `http://localhost:3000`
4. Submit a query that triggers source retrieval
5. Verify HITL validation dialog/panel appears:
   - Sources display with Amber-400 pending indicator
   - Statistics bar shows counts
   - Approve All / Reject All buttons work
   - Individual approve/reject buttons work
   - Click to toggle status works
6. Test approval flow:
   - Approve some sources, reject others
   - Click Continue - verify only approved sources used
   - Verify rejected sources not in answer
7. Test skip flow:
   - Click "Skip & Use All"
   - Verify all sources used in answer
8. Test cancel flow:
   - Open dialog, click Cancel
   - Verify no answer generated
9. Test keyboard accessibility:
   - Tab through all controls
   - Enter/Space to toggle source status
10. Verify design system compliance:
    - Amber-400 (#FBBF24) for pending items
    - Emerald-500 for approved
    - Red-500 for rejected
    - Indigo-600 for primary actions

## Definition of Done

- [ ] All acceptance criteria met
- [ ] SourceValidationDialog component created with modal UI
- [ ] SourceValidationPanel component created for inline HITL
- [ ] SourceValidationCard component created with Amber-400 pending indicator
- [ ] use-source-validation hook manages state and CopilotKit integration
- [ ] renderAndWait pattern pauses agent execution until validation
- [ ] Backend HITL checkpoint and validation response endpoint implemented
- [ ] Approval status tracked in agent state
- [ ] Skip option allows bypassing validation
- [ ] TypeScript types added for HITL data structures
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Manual verification completed
- [ ] No TypeScript errors
- [ ] Code follows project naming conventions
- [ ] Design system colors applied correctly (Amber-400 for HITL attention)
- [ ] Keyboard accessible

## Technical Notes

### CopilotKit renderAndWait Pattern

The `renderAndWait` option in `useCopilotAction` pauses agent execution until the frontend responds:

```typescript
useCopilotAction({
  name: "validate_sources",
  renderAndWait: ({ args, respond }) => {
    // respond() sends the decision back to the agent
    // Agent execution pauses until respond() is called
    return <ValidationUI onComplete={(ids) => respond({ approved: ids })} />;
  },
});
```

### Backend Async Waiting

The backend uses asyncio.Event to pause execution while waiting for frontend response:

```python
checkpoint = HITLCheckpoint(...)
await asyncio.wait_for(checkpoint.response_event.wait(), timeout=300)
# Execution resumes when response_event.set() is called
```

### Non-Blocking UX

The HITL validation is "non-blocking" in the sense that:
- Users can continue typing new messages while validation is open
- The collapsed panel doesn't prevent other interactions
- Skip option allows quick bypass

However, the agent's answer generation is blocked until validation completes.

### State Management

Validation state is managed locally in the hook, not in global state, because:
- Each validation session is independent
- State is transient (cleared after submission)
- No need for persistence across page refreshes

## Accessibility Considerations

- All interactive elements have proper ARIA labels
- Keyboard navigation: Tab through controls, Enter/Space to activate
- Color is not the only indicator (icons accompany status colors)
- Focus management: Focus returns to trigger after dialog closes
- Screen reader announcements for status changes

## Design System Colors

Per UX Design Specification:
- **Amber-400 (#FBBF24):** HITL attention/pending items - PRIMARY for this story
- **Emerald-500 (#10B981):** Approved/validated sources
- **Red-500 (#EF4444):** Rejected sources
- **Indigo-600 (#4F46E5):** Primary action buttons

Color usage in components:
- SourceValidationCard border: Amber-400 (pending), Emerald-500 (approved), Red-500 (rejected)
- Statistics badges: Matching colors for each status
- Action buttons: Emerald for approve, Red for reject, Indigo for submit

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| User abandons validation (closes tab) | Timeout auto-approves after 5 minutes |
| Large number of sources | ScrollArea with max-height, pagination for 20+ |
| Network error during submission | Retry logic with error message display |
| Concurrent validation requests | Unique checkpoint IDs prevent conflicts |
| SSR issues with Dialog | Ensure "use client" directive, Portal rendering |

## References

- [CopilotKit renderAndWait](https://docs.copilotkit.ai/reference/hooks/useCopilotAction#renderandwait)
- [shadcn/ui Dialog](https://ui.shadcn.com/docs/components/dialog)
- [Epic 6 Tech Spec](_bmad-output/implementation-artifacts/epic-6-tech-spec.md)
- [Story 6-3: Generative UI Components](_bmad-output/implementation-artifacts/stories/6-3-generative-ui-components.md)
- [UX Design Specification](_bmad-output/project-planning-artifacts/ux-design-specification.md)
