"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
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
import type { Source, ValidationDecision } from "@/types/copilot";

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

export interface SourceValidationCardProps {
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
 * - Keyboard accessible (Enter/Space to toggle)
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
          <button
            type="button"
            onClick={onApprove}
            className={cn(
              "h-8 w-8 rounded-md flex items-center justify-center transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-offset-1",
              validationStatus === "approved"
                ? "bg-emerald-500 hover:bg-emerald-600 text-white focus:ring-emerald-400"
                : "bg-white border border-emerald-300 text-emerald-600 hover:bg-emerald-50 focus:ring-emerald-400"
            )}
            aria-label="Approve source"
          >
            <Check className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onReject}
            className={cn(
              "h-8 w-8 rounded-md flex items-center justify-center transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-offset-1",
              validationStatus === "rejected"
                ? "bg-red-500 hover:bg-red-600 text-white focus:ring-red-400"
                : "bg-white border border-red-300 text-red-600 hover:bg-red-50 focus:ring-red-400"
            )}
            aria-label="Reject source"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
});

export default SourceValidationCard;
