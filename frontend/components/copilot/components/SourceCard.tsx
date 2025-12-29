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
} from "lucide-react";
import type { Source } from "@/types/copilot";

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
 * Get confidence level color based on similarity score.
 * Design System:
 * - >= 90%: Emerald (high confidence)
 * - >= 70%: Indigo (medium-high confidence)
 * - >= 50%: Amber (medium confidence)
 * - < 50%: Slate (low confidence)
 */
function getConfidenceColor(similarity: number): string {
  if (similarity >= 0.9) return "bg-emerald-100 text-emerald-800 border-emerald-200";
  if (similarity >= 0.7) return "bg-indigo-100 text-indigo-800 border-indigo-200";
  if (similarity >= 0.5) return "bg-amber-100 text-amber-800 border-amber-200";
  return "bg-slate-100 text-slate-800 border-slate-200";
}

interface SourceCardProps {
  /** The source data to display */
  source: Source;
  /** Optional 0-based index for numbering sources */
  index?: number;
  /** Callback when the card is clicked */
  onClick?: (source: Source) => void;
  /** Whether this card is currently highlighted/selected */
  isHighlighted?: boolean;
  /** Whether to show the approval status indicator */
  showApprovalStatus?: boolean;
}

/**
 * SourceCard displays citation information for a retrieved source.
 * Used in Generative UI to show sources referenced in AI responses.
 *
 * Story 6-3: Generative UI Components
 *
 * Features:
 * - Source type icon (document, web, database, knowledge_graph)
 * - Confidence badge with color-coded severity
 * - Truncated title and 2-line snippet preview
 * - Optional approval status indicator for HITL integration
 * - External link support for source URLs
 * - Keyboard accessible
 */
export const SourceCard = memo(function SourceCard({
  source,
  index,
  onClick,
  isHighlighted = false,
  showApprovalStatus = false,
}: SourceCardProps) {
  const sourceType = (source.metadata?.type as string) || "default";
  const IconComponent = sourceTypeIcons[sourceType] || sourceTypeIcons.default;
  const confidencePercent = Math.round(source.similarity * 100);
  const sourceUrl = source.metadata?.url as string | undefined;

  return (
    <div
      role="button"
      tabIndex={0}
      className={cn(
        "cursor-pointer transition-all duration-200 hover:shadow-md",
        "border border-slate-200 hover:border-indigo-300 rounded-lg",
        "bg-white p-3",
        isHighlighted && "ring-2 ring-indigo-500 ring-offset-2",
        onClick && "hover:bg-slate-50"
      )}
      onClick={() => onClick?.(source)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick?.(source);
        }
      }}
      aria-label={`Source: ${source.title}, confidence ${confidencePercent}%`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          {index !== undefined && (
            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-600 text-white text-xs font-medium flex items-center justify-center">
              {index + 1}
            </span>
          )}
          <IconComponent
            className="h-4 w-4 text-slate-500 flex-shrink-0"
            aria-hidden="true"
          />
          <h4 className="text-sm font-medium text-slate-900 truncate">
            {source.title}
          </h4>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {showApprovalStatus && source.isApproved && (
            <CheckCircle2
              className="h-4 w-4 text-emerald-500"
              aria-label="Approved"
            />
          )}
          <span
            className={cn(
              "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono border",
              getConfidenceColor(source.similarity)
            )}
          >
            {confidencePercent}%
          </span>
        </div>
      </div>

      {/* Preview text */}
      <p className="text-sm text-slate-600 line-clamp-2">{source.preview}</p>

      {/* External link */}
      {sourceUrl && (
        <a
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 mt-2 text-xs text-indigo-600 hover:text-indigo-800"
          onClick={(e) => e.stopPropagation()}
          aria-label={`View source: ${source.title}`}
        >
          <ExternalLink className="h-3 w-3" aria-hidden="true" />
          View source
        </a>
      )}
    </div>
  );
});

export default SourceCard;
