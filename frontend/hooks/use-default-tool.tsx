"use client";

import { useCallback, useRef } from "react";
import { useDefaultTool } from "@copilotkit/react-core";
import { useToast } from "@/hooks/use-toast";
import { redactSensitiveKeys } from "@/lib/utils/redact";
import type {
  DefaultToolRenderProps,
  DefaultToolStatus,
  DefaultToolHandlerUtilities,
} from "@/types/copilot";

/**
 * Check if the tool status indicates the tool is currently running.
 *
 * @param status - Tool execution status
 * @returns true if status is "inProgress" or "executing"
 */
export function isRunning(status: DefaultToolStatus): boolean {
  return status === "inProgress" || status === "executing";
}

/**
 * Check if the tool status indicates the tool has completed.
 *
 * @param status - Tool execution status
 * @returns true if status is "complete"
 */
export function isComplete(status: DefaultToolStatus): boolean {
  return status === "complete";
}

/**
 * Format a tool name for display by removing common prefixes.
 * MCP tools often have prefixes like "mcp_" or "server:".
 *
 * @param name - Raw tool name
 * @returns Formatted tool name for display
 *
 * @example
 * ```ts
 * formatToolName("mcp_vector_search") // "vector_search"
 * formatToolName("github:create_issue") // "create_issue"
 * formatToolName("search_docs") // "search_docs"
 * ```
 */
export function formatToolName(name: string): string {
  // Remove common MCP prefixes
  let formatted = name.replace(/^mcp_/i, "");
  // Remove server: prefix (e.g., "github:create_issue" -> "create_issue")
  if (formatted.includes(":")) {
    const parts = formatted.split(":");
    formatted = parts[parts.length - 1];
  }
  return formatted;
}

/**
 * Get default tool handler utilities without registering the hook.
 * Useful for testing and external usage.
 *
 * @returns Utility functions for default tool handling
 */
export function getDefaultToolUtilities(): DefaultToolHandlerUtilities {
  return {
    isRunning,
    isComplete,
    formatToolName,
  };
}

/**
 * useDefaultToolHandler provides a catch-all handler for unregistered backend tools.
 *
 * Story 21-A8: Implement useDefaultTool Catch-All
 *
 * This hook catches tool calls that don't have a specific handler registered via
 * `useFrontendTool`, `useHumanInTheLoop`, or `useRenderToolCall`. It provides:
 *
 * - Console logging for debugging (with sensitive data redaction)
 * - Toast notifications when tools complete
 * - Generic loading indicator during tool execution
 * - Graceful error handling that won't crash the UI
 *
 * Use Cases:
 * - New backend MCP tools work immediately without frontend deployment
 * - Third-party MCP tools (when MCP client is enabled) auto-support
 * - Debugging tool execution during development
 * - User feedback for background tool operations
 *
 * Relationship to Story 21-A3 (Tool Call Visualization):
 * - 21-A3's `useRenderToolCall` with "*" wildcard renders ALL tool calls visually
 * - 21-A8's `useDefaultTool` handles tools WITHOUT specific handlers
 * - Both can coexist - they serve complementary purposes
 *
 * @example
 * ```tsx
 * // In CopilotProvider or similar context wrapper
 * function CopilotContextProvider() {
 *   useDefaultToolHandler();
 *   return null;
 * }
 * ```
 *
 * @example
 * ```tsx
 * // Inside CopilotKit context
 * function MyApp() {
 *   useDefaultToolHandler();
 *   return <ChatSidebar />;
 * }
 * ```
 */
export function useDefaultToolHandler(): void {
  const { toast } = useToast();

  // Track which tools we've shown completion toast for to avoid duplicates
  // when render is called multiple times during status transitions
  const completedToolsRef = useRef<Set<string>>(new Set());

  // Stable callback for toast to avoid recreating render function
  const showCompletionToast = useCallback(
    (toolName: string, toolCallId: string) => {
      // Only show toast once per tool call
      if (completedToolsRef.current.has(toolCallId)) {
        return;
      }
      completedToolsRef.current.add(toolCallId);

      const displayName = formatToolName(toolName);
      toast({
        variant: "default",
        title: "Tool Executed",
        description: `${displayName} completed successfully`,
      });
    },
    [toast]
  );

  useDefaultTool({
    render: (props) => {
      const { name, args, status } = props as DefaultToolRenderProps;

      try {
        // Generate a unique ID for this tool call to track completion
        const toolCallId = `${name}-${JSON.stringify(args).slice(0, 50)}`;

        // Log for debugging (with sensitive data redaction)
        const redactedArgs = redactSensitiveKeys(args ?? {});
        console.log(`[DefaultTool] ${name}`, {
          status,
          args: redactedArgs,
        });

        // Show toast on completion
        if (isComplete(status)) {
          showCompletionToast(name, toolCallId);
          // Return empty fragment for completed tools - toast handles feedback
          return <></>;
        }

        // Show loading indicator during execution
        if (isRunning(status)) {
          const displayName = formatToolName(name);
          return (
            <div
              className="text-sm text-muted-foreground flex items-center gap-2 my-2 px-3 py-2 bg-slate-50 rounded-md border border-slate-200"
              data-testid="default-tool-loading"
              role="status"
              aria-live="polite"
            >
              <span
                className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse"
                aria-hidden="true"
              />
              <span>Running {displayName}...</span>
            </div>
          );
        }

        // Unknown status - return empty fragment
        return <></>;
      } catch (error) {
        // Log error but don't crash UI
        console.error("[DefaultTool] Render error:", error);
        return <></>;
      }
    },
  });
}

export default useDefaultToolHandler;
