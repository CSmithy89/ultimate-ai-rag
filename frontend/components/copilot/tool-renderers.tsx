"use client";

import React from "react";
import { useRenderToolCall } from "@copilotkit/react-core";
import type {
  ActionRenderPropsNoArgs,
  CatchAllActionRenderProps,
} from "@copilotkit/react-core";
import { MCPToolCallCard } from "./MCPToolCallCard";
import { VectorSearchCard } from "./VectorSearchCard";
import type { ToolStatus } from "./StatusBadge";

/**
 * Normalize CopilotKit status to our ToolStatus type.
 * CopilotKit 1.x uses: "inProgress" | "executing" | "complete"
 */
function normalizeStatus(status: string): ToolStatus {
  return status as ToolStatus;
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
      const { args, status, result } = props;
      const query = (args?.query as string) || (args?.text as string) || "";
      return (
        <VectorSearchCard
          query={query}
          status={normalizeStatus(status)}
          results={result}
        />
      );
    },
  });

  // Graph search renderer - generic card
  useRenderToolCall({
    name: "graph_search",
    render: (props: ActionRenderPropsNoArgs) => {
      const { args, status, result } = props;
      return (
        <MCPToolCallCard
          name="graph_search"
          args={(args as Record<string, unknown>) || {}}
          status={normalizeStatus(status)}
          result={result}
        />
      );
    },
  });

  // Ingest URL renderer - generic card
  useRenderToolCall({
    name: "ingest_url",
    render: (props: ActionRenderPropsNoArgs) => {
      const { args, status, result } = props;
      return (
        <MCPToolCallCard
          name="ingest_url"
          args={(args as Record<string, unknown>) || {}}
          status={normalizeStatus(status)}
          result={result}
        />
      );
    },
  });

  // Ingest PDF renderer - generic card
  useRenderToolCall({
    name: "ingest_pdf",
    render: (props: ActionRenderPropsNoArgs) => {
      const { args, status, result } = props;
      return (
        <MCPToolCallCard
          name="ingest_pdf"
          args={(args as Record<string, unknown>) || {}}
          status={normalizeStatus(status)}
          result={result}
        />
      );
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
      const { args, status, result } = props;
      return (
        <MCPToolCallCard
          name={toolName}
          args={(args as Record<string, unknown>) || {}}
          status={normalizeStatus(status)}
          result={result}
        />
      );
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
