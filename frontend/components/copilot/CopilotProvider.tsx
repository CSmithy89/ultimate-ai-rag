"use client";

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import { ReactNode } from "react";
import { useDefaultToolHandler } from "@/hooks/use-default-tool";
import { useAnalytics } from "@/hooks/use-analytics";

interface CopilotProviderProps {
  children: ReactNode;
}

/**
 * Determine if dev console should be shown based on environment.
 *
 * Story 21-B1: Enable showDevConsole only when:
 * 1. NODE_ENV is development
 * 2. NEXT_PUBLIC_SHOW_DEV_CONSOLE is explicitly set to "true"
 */
const isDev = process.env.NODE_ENV === "development";
const showDevConsole =
  isDev && process.env.NEXT_PUBLIC_SHOW_DEV_CONSOLE === "true";

/**
 * CopilotKit Cloud API key or license key.
 *
 * Story 21-B1: Optional keys that enable observabilityHooks on
 * CopilotChat/CopilotSidebar components. Without these keys,
 * observability hooks are silently ignored but the chat still works.
 *
 * - publicApiKey: For CopilotKit Cloud users (ck_pub_xxx)
 * - publicLicenseKey: For self-hosted deployments
 */
const publicApiKey = process.env.NEXT_PUBLIC_COPILOTKIT_API_KEY || undefined;
const publicLicenseKey =
  process.env.NEXT_PUBLIC_COPILOTKIT_LICENSE_KEY || undefined;

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
 * CopilotErrorHandler - handles errors from CopilotKit.
 *
 * Story 21-B1: Emit structured error telemetry when CopilotKit errors occur.
 * Uses the onError callback which receives error events from CopilotKit.
 */
function useCopilotErrorHandler() {
  const { track } = useAnalytics();

  return (errorEvent: unknown) => {
    // CopilotKit error events may have different shapes
    const error = errorEvent as {
      type?: string;
      error?: Error | { message?: string };
      message?: string;
      context?: unknown;
      timestamp?: string;
    };

    // Extract error message safely
    const errorMessage =
      error.error instanceof Error
        ? error.error.message
        : error.error?.message ?? error.message ?? "Unknown error";

    // Track error in analytics
    track("copilot_error", {
      type: error.type ?? "unknown",
      error: errorMessage,
      context: error.context,
      timestamp: error.timestamp ?? new Date().toISOString(),
    });

    // Log to console in development
    if (isDev) {
      console.error("[CopilotKit Error]", errorEvent);
    }
  };
}

/**
 * CopilotProvider wraps the application with CopilotKit context.
 *
 * Story 21-A3: Implement Tool Call Visualization (AC10)
 * Story 21-A8: Implement useDefaultTool Catch-All
 * Story 21-B1: Configure Observability Hooks and Dev Console
 *
 * Features:
 * - Connects to the CopilotKit runtime at /api/copilotkit
 * - Tool call visualization is provided via useToolCallRenderers hook
 *   which should be called inside a component within CopilotKit context
 * - Default tool handler catches unregistered backend tools (21-A8)
 * - Dev console visible when NEXT_PUBLIC_SHOW_DEV_CONSOLE=true (21-B1)
 * - Error events tracked via analytics pipeline (21-B1)
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
  const handleError = useCopilotErrorHandler();

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      showDevConsole={showDevConsole}
      onError={handleError}
      // Optional: Enable observabilityHooks on CopilotChat/Sidebar
      // Without these, hooks are silently ignored but analytics still works
      // via our useAnalytics hook in the error handler
      {...(publicApiKey ? { publicApiKey } : {})}
      {...(publicLicenseKey ? { publicLicenseKey } : {})}
    >
      <CopilotContextProvider />
      {children}
    </CopilotKit>
  );
}
