"use client";

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import { ReactNode } from "react";

interface CopilotProviderProps {
  children: ReactNode;
}

/**
 * CopilotProvider wraps the application with CopilotKit context.
 *
 * Story 21-A3: Implement Tool Call Visualization (AC10)
 *
 * Features:
 * - Connects to the CopilotKit runtime at /api/copilotkit
 * - Tool call visualization is provided via useToolCallRenderers hook
 *   which should be called inside a component within CopilotKit context
 *
 * Tool renderers provide visual feedback when MCP tools are called:
 * - vector_search: Specialized card with query and results display
 * - graph_search, ingest_url, ingest_pdf: Generic MCP tool cards
 * - Wildcard ("*"): Catches all unregistered tools
 *
 * Note: To enable tool call visualization, include the ToolCallRenderer
 * component (or call useToolCallRenderers hook) inside your CopilotKit context.
 * This is typically done via GenerativeUIRenderer.
 */
export function CopilotProvider({ children }: CopilotProviderProps) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      {children}
    </CopilotKit>
  );
}
