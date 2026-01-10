"use client";

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import { ReactNode } from "react";
import { useDefaultToolHandler } from "@/hooks/use-default-tool";

interface CopilotProviderProps {
  children: ReactNode;
}

/**
 * CopilotContextProvider registers global CopilotKit hooks.
 *
 * This component must be rendered inside CopilotKit to register hooks
 * that need the CopilotKit context (like tool handlers).
 *
 * Story 21-A8: Registers useDefaultToolHandler for catch-all tool handling.
 */
function CopilotContextProvider() {
  // Register default tool handler for catch-all support
  // This catches tools without specific handlers and provides:
  // - Console logging for debugging (with redaction)
  // - Toast notifications on completion
  // - Generic loading indicator during execution
  useDefaultToolHandler();

  return null;
}

/**
 * CopilotProvider wraps the application with CopilotKit context.
 *
 * Story 21-A3: Implement Tool Call Visualization (AC10)
 * Story 21-A8: Implement useDefaultTool Catch-All
 *
 * Features:
 * - Connects to the CopilotKit runtime at /api/copilotkit
 * - Tool call visualization is provided via useToolCallRenderers hook
 *   which should be called inside a component within CopilotKit context
 * - Default tool handler catches unregistered backend tools (21-A8)
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
      <CopilotContextProvider />
      {children}
    </CopilotKit>
  );
}
