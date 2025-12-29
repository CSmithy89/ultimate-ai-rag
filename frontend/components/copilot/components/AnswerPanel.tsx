"use client";

import { memo, useCallback, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { Sparkles, Copy, Check, ChevronDown, ChevronUp } from "lucide-react";
import { SourceCard } from "./SourceCard";
import type { Source } from "@/types/copilot";

interface AnswerPanelProps {
  /** The answer text with optional markdown formatting */
  answer: string;
  /** Optional sources referenced in the answer */
  sources?: Source[];
  /** Title for the answer panel */
  title?: string;
  /** Whether the answer is currently being streamed */
  isStreaming?: boolean;
  /** Whether to show the collapsible sources section */
  showSources?: boolean;
  /** Callback when a source card is clicked */
  onSourceClick?: (source: Source) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Validates a URL to ensure it uses a safe protocol.
 * Prevents XSS attacks via javascript:, data:, and other dangerous protocols.
 *
 * @param url - The URL to validate
 * @returns true if the URL is safe, false otherwise
 */
function isSafeUrl(url: string | undefined): boolean {
  if (!url) return false;

  // Allow relative URLs
  if (url.startsWith("/") && !url.startsWith("//")) {
    return true;
  }

  // Allow only http and https protocols
  try {
    const parsed = new URL(url, "https://example.com");
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    // If URL parsing fails, reject it
    return false;
  }
}

/**
 * AnswerPanel renders a formatted AI response with markdown support
 * and collapsible source references.
 *
 * Story 6-3: Generative UI Components
 *
 * Features:
 * - Markdown rendering with remark-gfm for GitHub-flavored markdown
 * - Custom styled code blocks
 * - Copy to clipboard functionality
 * - Streaming indicator for in-progress responses
 * - Collapsible source references section
 * - Automatic extraction of inline source citations like [1], [2]
 * - XSS protection for markdown links
 */
export const AnswerPanel = memo(function AnswerPanel({
  answer,
  sources = [],
  title = "Answer",
  isStreaming = false,
  showSources = true,
  onSourceClick,
  className,
}: AnswerPanelProps) {
  const [copied, setCopied] = useState(false);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  // Extract inline source references like [1], [2] from the answer
  const sourceReferences = useMemo(() => {
    const matches = answer.match(/\[(\d+)\]/g);
    if (!matches) return [];
    return [...new Set(matches.map((m) => parseInt(m.slice(1, -1), 10) - 1))];
  }, [answer]);

  const referencedSources = useMemo(() => {
    return sourceReferences
      .filter((idx) => idx >= 0 && idx < sources.length)
      .map((idx) => sources[idx]);
  }, [sourceReferences, sources]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API might fail in some contexts
      console.error("Failed to copy to clipboard");
    }
  }, [answer]);

  // Memoize ReactMarkdown components to prevent unnecessary re-renders
  const markdownComponents = useMemo(
    () => ({
      a: ({
        href,
        children,
        ...props
      }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => {
        // XSS Protection: Validate URL before rendering
        if (!isSafeUrl(href)) {
          // Render as plain text if URL is not safe
          return <span className="text-slate-600">{children}</span>;
        }

        return (
          <a
            href={href}
            className="text-indigo-600 hover:text-indigo-800 no-underline hover:underline"
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        );
      },
      code: ({
        className: codeClassName,
        children,
        ...props
      }: React.HTMLAttributes<HTMLElement>) => {
        const isInline = !codeClassName;
        return isInline ? (
          <code
            className="px-1 py-0.5 rounded bg-slate-100 text-slate-800 font-mono text-xs"
            {...props}
          >
            {children}
          </code>
        ) : (
          <code className={cn("font-mono text-sm", codeClassName)} {...props}>
            {children}
          </code>
        );
      },
      pre: ({
        children,
        ...props
      }: React.HTMLAttributes<HTMLPreElement>) => (
        <pre
          className="bg-slate-900 text-slate-100 rounded-lg p-4 overflow-x-auto"
          {...props}
        >
          {children}
        </pre>
      ),
    }),
    []
  );

  return (
    <div
      className={cn("border border-slate-200 rounded-lg bg-white", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-indigo-600" aria-hidden="true" />
          <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
          {isStreaming && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 animate-pulse">
              Generating...
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="p-1.5 rounded-md hover:bg-slate-100 transition-colors"
          aria-label={copied ? "Copied to clipboard" : "Copy answer to clipboard"}
        >
          {copied ? (
            <Check className="h-4 w-4 text-emerald-500" aria-hidden="true" />
          ) : (
            <Copy className="h-4 w-4 text-slate-500" aria-hidden="true" />
          )}
        </button>
      </div>

      {/* Markdown content */}
      <div className="px-4 pb-4">
        <div className="prose prose-sm prose-slate max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {answer}
          </ReactMarkdown>
        </div>

        {/* Source references section */}
        {showSources && referencedSources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-100">
            <button
              type="button"
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
              className="flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900"
              aria-expanded={sourcesExpanded}
              aria-controls="sources-list"
            >
              {sourcesExpanded ? (
                <ChevronUp className="h-4 w-4" aria-hidden="true" />
              ) : (
                <ChevronDown className="h-4 w-4" aria-hidden="true" />
              )}
              Sources ({referencedSources.length})
            </button>
            {sourcesExpanded && (
              <div id="sources-list" className="mt-3 space-y-2">
                {referencedSources.map((source, idx) => (
                  <SourceCard
                    key={source.id}
                    source={source}
                    index={sourceReferences[idx]}
                    onClick={onSourceClick}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

export default AnswerPanel;
