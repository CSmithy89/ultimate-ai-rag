"use client";

import React from "react";
import { useRenderToolCall } from "@copilotkit/react-core";
import type {
  ActionRenderPropsNoArgs,
  CatchAllActionRenderProps,
} from "@copilotkit/react-core";
import { MCPToolCallCard } from "./MCPToolCallCard";
import { VectorSearchCard } from "./VectorSearchCard";
import { MCPUIRenderer, type MCPUIPayload } from "@/components/mcp-ui/MCPUIRenderer";
import { OpenJSONUIRenderer } from "@/components/open-json-ui/OpenJSONUIRenderer";
import { OpenJSONUIPayloadSchema } from "@/lib/open-json-ui/schema";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import type { ToolStatus } from "./StatusBadge";

/**
 * Valid status values from CopilotKit.
 * Includes both 1.x (lowercase) and 2.x (PascalCase) formats,
 * plus error states.
 * (Issue 3.3: normalizeStatus Blind-Casting)
 */
const VALID_STATUSES = new Set<string>([
  "inProgress", "executing", "complete",  // CopilotKit 1.x
  "InProgress", "Executing", "Complete",  // CopilotKit 2.x
  "error", "failed", "Error", "Failed",   // Error states
]);

/**
 * Normalize CopilotKit status to our ToolStatus type.
 * Validates input and provides fallback for unknown statuses.
 * (Issue 3.3: normalizeStatus Blind-Casting - Fixed)
 *
 * @param status - Status string from CopilotKit
 * @returns Valid ToolStatus, defaulting to "inProgress" for unknown values
 */
function normalizeStatus(status: string): ToolStatus {
  if (VALID_STATUSES.has(status)) {
    return status as ToolStatus;
  }
  // Log warning for unexpected status values
  if (process.env.NODE_ENV !== "production") {
    console.warn(`[tool-renderers] Unknown tool status: "${status}", defaulting to "inProgress"`);
  }
  return "inProgress";
}

/**
 * Safe render wrapper to catch errors in tool renderers.
 * (Issue 3.6: Missing Error Boundary for Tool Renderers)
 *
 * @param toolName - Name of the tool being rendered
 * @param renderFn - Function that renders the component
 * @returns Rendered component or error fallback
 */
function safeRender(
  toolName: string,
  renderFn: () => React.ReactElement
): React.ReactElement {
  try {
    return renderFn();
  } catch (error) {
    console.error(`[tool-renderers] Error rendering ${toolName}:`, error);
    // Return a minimal error card
    return (
      <div className="my-2 p-3 border border-red-200 rounded-lg bg-red-50 text-red-800 text-sm">
        <strong>Error rendering tool:</strong> {toolName}
      </div>
    );
  }
}

function isMCPUIResult(result: unknown): result is MCPUIPayload {
  if (!result || typeof result !== "object") {
    return false;
  }
  const payload = result as Record<string, unknown>;
  return (
    payload.type === "mcp_ui" &&
    typeof payload.ui_url === "string" &&
    typeof payload.tool_name === "string"
  );
}

function renderToolResult(
  toolName: string,
  args: Record<string, unknown> | undefined,
  status: ToolStatus,
  result: unknown
): React.ReactElement {
  const tenantId =
    typeof args?.tenant_id === "string" ? (args.tenant_id as string) : undefined;

  if (isMCPUIResult(result)) {
    return (
      <CopilotErrorBoundary
        fallback={
          <div className="my-2 p-3 border border-red-200 rounded-lg bg-red-50 text-red-800 text-sm">
            <strong>Tool UI error:</strong> {toolName}
          </div>
        }
      >
        <MCPUIRenderer payload={result} tenantId={tenantId} />
      </CopilotErrorBoundary>
    );
  }

  const openJsonParse = OpenJSONUIPayloadSchema.safeParse(result);
  if (openJsonParse.success) {
    return (
      <CopilotErrorBoundary
        fallback={
          <div className="my-2 p-3 border border-red-200 rounded-lg bg-red-50 text-red-800 text-sm">
            <strong>Tool UI error:</strong> {toolName}
          </div>
        }
      >
        <OpenJSONUIRenderer payload={openJsonParse.data} />
      </CopilotErrorBoundary>
    );
  }

  return (
    <MCPToolCallCard
      name={toolName}
      args={args || {}}
      status={status}
      result={result}
    />
  );
}

/**
 * useToolCallRenderers hook registers tool call renderers with CopilotKit.
 *
 * Story 21-A3: Implement Tool Call Visualization (AC5, AC6, AC10)
 *
 * This hook must be called inside a component that is within the CopilotKit context.
 * It registers custom renderers for MCP tool calls, providing visual feedback
 * when tools like vector_search, graph_search, ingest_url, etc. are executed.
 *
 * Registered renderers:
 * - vector_search: Specialized VectorSearchCard with query/results display
 * - graph_search: Generic MCPToolCallCard
 * - ingest_url: Generic MCPToolCallCard
 * - ingest_pdf: Generic MCPToolCallCard
 * - wildcard (*): Catches all unregistered tools with MCPToolCallCard
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   // Register tool renderers
 *   useToolCallRenderers();
 *
 *   return <div>My content</div>;
 * }
 * ```
 */
export function useToolCallRenderers(): void {
  // Vector search renderer - specialized card for RAG searches
  // Story 21-A3: AC5 - Specific renderer for vector_search
  useRenderToolCall({
    name: "vector_search",
    render: (props: ActionRenderPropsNoArgs) => {
      return safeRender("vector_search", () => {
        const { args, status, result } = props;
        const query = (args?.query as string) || (args?.text as string) || "";
        return (
          <VectorSearchCard
            query={query}
            status={normalizeStatus(status)}
            results={result}
          />
        );
      });
    },
  });

  // Graph search renderer - generic card
  useRenderToolCall({
    name: "graph_search",
    render: (props: ActionRenderPropsNoArgs) => {
      return safeRender("graph_search", () => {
        const { args, status, result } = props;
        return renderToolResult(
          "graph_search",
          (args as Record<string, unknown>) || {},
          normalizeStatus(status),
          result
        );
      });
    },
  });

  // Ingest URL renderer - generic card
  useRenderToolCall({
    name: "ingest_url",
    render: (props: ActionRenderPropsNoArgs) => {
      return safeRender("ingest_url", () => {
        const { args, status, result } = props;
        return renderToolResult(
          "ingest_url",
          (args as Record<string, unknown>) || {},
          normalizeStatus(status),
          result
        );
      });
    },
  });

  // Ingest PDF renderer - generic card
  useRenderToolCall({
    name: "ingest_pdf",
    render: (props: ActionRenderPropsNoArgs) => {
      return safeRender("ingest_pdf", () => {
        const { args, status, result } = props;
        return renderToolResult(
          "ingest_pdf",
          (args as Record<string, unknown>) || {},
          normalizeStatus(status),
          result
        );
      });
    },
  });

  // Wildcard renderer - catches all unregistered tools
  // Story 21-A3: AC6 - Wildcard catches any unmatched tools
  // Note: For wildcard ("*"), CopilotKit passes CatchAllActionRenderProps which includes `name`
  // but the TypeScript types don't reflect this. We cast to handle both cases safely.
  useRenderToolCall({
    name: "*",
    render: (props: ActionRenderPropsNoArgs) => {
      // For wildcard renderers, CopilotKit passes `name` property at runtime
      const catchAllProps = props as unknown as CatchAllActionRenderProps;
      const toolName = catchAllProps.name || "unknown_tool";
      return safeRender(toolName, () => {
        const { args, status, result } = props;
        return renderToolResult(
          toolName,
          (args as Record<string, unknown>) || {},
          normalizeStatus(status),
          result
        );
      });
    },
  });
}

/**
 * ToolCallRenderer component registers tool call renderers.
 *
 * Story 21-A3: Implement Tool Call Visualization
 *
 * This component should be included inside a CopilotKit context to enable
 * tool call visualization. It renders nothing but registers the renderers.
 *
 * @example
 * ```tsx
 * <CopilotKit>
 *   <ToolCallRenderer />
 *   {children}
 * </CopilotKit>
 * ```
 */
export function ToolCallRenderer(): null {
  useToolCallRenderers();
  return null;
}

export default ToolCallRenderer;
