"use client";

/**
 * AITextarea component - AI-powered textarea with autocompletion.
 *
 * Story 21-F3: Implement CopilotTextarea Component
 *
 * Features:
 * - AI autocompletion in any textarea
 * - Inline suggestions while typing
 * - Tab to accept suggestions
 * - Ideal for document editing, notes, forms
 * - Configurable purpose/context
 */

import { CopilotTextarea } from "@copilotkit/react-textarea";
import "@copilotkit/react-textarea/styles.css";
import { cn } from "@/lib/utils";
import { forwardRef, memo } from "react";

export interface AutosuggestionsConfig {
  /** Purpose/context for AI suggestions */
  textareaPurpose: string;
  /** Additional chat API configurations */
  chatApiConfigs?: Record<string, unknown>;
}

export interface AITextareaProps {
  /** Controlled value */
  value: string;
  /** Change handler */
  onChange: (value: string) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Additional class names */
  className?: string;
  /** Whether the textarea is disabled */
  disabled?: boolean;
  /** Minimum rows */
  minRows?: number;
  /** Maximum rows (auto-grow limit) */
  maxRows?: number;
  /** Autosuggestions configuration */
  autosuggestionsConfig?: AutosuggestionsConfig;
  /** Aria label for accessibility */
  "aria-label"?: string;
  /** ID for the textarea */
  id?: string;
  /** Name for form submission */
  name?: string;
}

/**
 * AITextarea provides an AI-powered textarea with inline suggestions.
 *
 * Usage:
 * ```tsx
 * const [content, setContent] = useState("");
 *
 * <AITextarea
 *   value={content}
 *   onChange={setContent}
 *   placeholder="Start typing..."
 *   autosuggestionsConfig={{
 *     textareaPurpose: "Note-taking and documentation",
 *   }}
 * />
 * ```
 *
 * Tab to accept suggestions, Escape to dismiss.
 */
export const AITextarea = memo(
  forwardRef<HTMLTextAreaElement, AITextareaProps>(function AITextarea(
    {
      value,
      onChange,
      placeholder = "Start typing...",
      className,
      disabled = false,
      minRows = 3,
      maxRows = 10,
      autosuggestionsConfig,
      "aria-label": ariaLabel,
      id,
      name,
    },
    ref
  ) {
    const config = autosuggestionsConfig ?? {
      textareaPurpose: "General purpose text editor",
      chatApiConfigs: {},
    };

    return (
      <CopilotTextarea
        ref={ref}
        value={value}
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className={cn(
          "w-full rounded-md border border-slate-300",
          "px-3 py-2 text-sm",
          "focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          "resize-y",
          className
        )}
        style={{
          minHeight: `${minRows * 1.5}rem`,
          maxHeight: `${maxRows * 1.5}rem`,
        }}
        autosuggestionsConfig={{
          textareaPurpose: config.textareaPurpose,
          chatApiConfigs: config.chatApiConfigs ?? {},
        }}
        aria-label={ariaLabel}
        id={id}
        name={name}
      />
    );
  })
);

export default AITextarea;
