"use client";

import { memo, useState, useCallback } from "react";
import { ChevronDown, Search, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusBadge, type ToolStatus, isComplete } from "./StatusBadge";

/**
 * Search result item from vector_search tool
 */
interface SearchResultItem {
  /** Source title or document name */
  title?: string;
  /** Content preview or snippet */
  content?: string;
  /** Preview text (alternative to content) */
  preview?: string;
  /** Similarity score (0-1) */
  similarity?: number;
  /** Score (alternative to similarity) */
  score?: number;
  /** Source metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Vector search result structure
 */
interface VectorSearchResult {
  /** Array of search results */
  results?: SearchResultItem[];
  /** Total count of results */
  total?: number;
  /** Alternative: items array */
  items?: SearchResultItem[];
  /** Alternative: documents array */
  documents?: SearchResultItem[];
}

export interface VectorSearchCardProps {
  /** Search query string */
  query: string;
  /** Current execution status */
  status: ToolStatus;
  /** Search results (when complete) */
  results?: VectorSearchResult | SearchResultItem[] | unknown;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Extract results array from various result formats
 */
function extractResults(
  results: VectorSearchResult | SearchResultItem[] | unknown
): SearchResultItem[] {
  if (!results) return [];

  // Direct array of results
  if (Array.isArray(results)) {
    return results;
  }

  // Object with results/items/documents property
  if (typeof results === "object") {
    const r = results as VectorSearchResult;
    if (Array.isArray(r.results)) return r.results;
    if (Array.isArray(r.items)) return r.items;
    if (Array.isArray(r.documents)) return r.documents;
  }

  return [];
}

/**
 * Get display text for a result item
 */
function getResultPreview(item: SearchResultItem): string {
  const text = item.preview || item.content || item.title || "No preview";
  return text.length > 100 ? text.slice(0, 100) + "..." : text;
}

/**
 * Get similarity score as percentage
 */
function getScorePercent(item: SearchResultItem): number {
  const score = item.similarity ?? item.score ?? 0;
  return Math.round(score * 100);
}

/**
 * VectorSearchCard displays a specialized card for vector_search tool calls.
 *
 * Story 21-A3: Implement Tool Call Visualization
 *
 * Features:
 * - Prominent display of search query
 * - Result count when complete
 * - Abbreviated source list with scores
 * - Expand/collapse for detailed results
 * - RAG-friendly presentation
 *
 * This component provides a more user-friendly view compared to the generic
 * MCPToolCallCard, specifically tailored for vector search operations.
 */
export const VectorSearchCard = memo(function VectorSearchCard({
  query,
  status,
  results,
  className,
}: VectorSearchCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const searchResults = extractResults(results);
  const resultCount = searchResults.length;

  const handleToggle = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleToggle();
      }
    },
    [handleToggle]
  );

  return (
    <div
      className={cn(
        "my-2 border border-slate-200 rounded-lg bg-white overflow-hidden",
        className
      )}
      data-testid="vector-search-card"
    >
      {/* Header */}
      <button
        type="button"
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        className={cn(
          "w-full p-3 flex items-center justify-between",
          "hover:bg-slate-50 transition-colors cursor-pointer",
          "focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
        )}
        aria-expanded={isExpanded}
        aria-controls="vector-search-content"
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <StatusBadge status={status} />
          <Search
            className="h-4 w-4 text-indigo-600 flex-shrink-0"
            aria-hidden="true"
          />
          <span className="font-medium text-sm text-slate-700 truncate">
            vector_search
          </span>
          {isComplete(status) && (
            <span className="text-xs text-slate-500 flex-shrink-0">
              ({resultCount} result{resultCount !== 1 ? "s" : ""})
            </span>
          )}
        </div>
        <ChevronDown
          className={cn(
            "h-4 w-4 text-slate-500 transition-transform duration-200 flex-shrink-0",
            isExpanded && "rotate-180"
          )}
          aria-hidden="true"
          data-testid="icon-chevron"
        />
      </button>

      {/* Query display (always visible when expanded) */}
      {isExpanded && (
        <div
          id="vector-search-content"
          className="p-3 pt-0 space-y-3 border-t border-slate-100"
        >
          {/* Query */}
          <div>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
              Query
            </span>
            <div
              className="mt-1 p-2 bg-indigo-50 rounded border border-indigo-100 text-sm text-indigo-900"
              data-testid="search-query"
            >
              &ldquo;{query}&rdquo;
            </div>
          </div>

          {/* Results (when complete) */}
          {isComplete(status) && searchResults.length > 0 && (
            <div>
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Results
              </span>
              <div className="mt-1 space-y-2" data-testid="search-results">
                {searchResults.slice(0, 5).map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-2 p-2 bg-slate-50 rounded text-sm"
                  >
                    <FileText
                      className="h-4 w-4 text-slate-400 flex-shrink-0 mt-0.5"
                      aria-hidden="true"
                    />
                    <div className="flex-1 min-w-0">
                      {item.title && (
                        <div className="font-medium text-slate-700 truncate">
                          {item.title}
                        </div>
                      )}
                      <div className="text-slate-600 text-xs line-clamp-2">
                        {getResultPreview(item)}
                      </div>
                    </div>
                    <span className="flex-shrink-0 px-1.5 py-0.5 bg-slate-200 text-slate-700 text-xs rounded font-mono">
                      {getScorePercent(item)}%
                    </span>
                  </div>
                ))}
                {searchResults.length > 5 && (
                  <div className="text-xs text-slate-500 text-center py-1">
                    + {searchResults.length - 5} more result
                    {searchResults.length - 5 !== 1 ? "s" : ""}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* No results message */}
          {isComplete(status) && searchResults.length === 0 && (
            <div className="text-sm text-slate-500 italic">
              No results found for this query.
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default VectorSearchCard;
