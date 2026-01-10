/**
 * CopilotKit API Route
 *
 * Story 21-C3: Wire MCP Client to CopilotRuntime
 *
 * This route handles CopilotKit requests and optionally connects to
 * external MCP servers for tool access.
 */

import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";

const serviceAdapter = new ExperimentalEmptyAdapter();

/**
 * MCP server configuration from environment variables.
 * Each server needs a URL and optional API key.
 */
interface MCPServerConfig {
  name: string;
  url: string;
  apiKey?: string;
}

/**
 * Parse MCP server configurations from environment.
 * Format: MCP_SERVERS='[{"name":"github","url":"...","apiKey":"..."}]'
 */
function getMCPServers(): MCPServerConfig[] {
  const serversJson = process.env.MCP_SERVERS;
  if (!serversJson) {
    return [];
  }

  try {
    const servers = JSON.parse(serversJson);
    if (!Array.isArray(servers)) {
      console.warn("MCP_SERVERS must be a JSON array");
      return [];
    }
    return servers.filter(
      (s: unknown): s is MCPServerConfig =>
        typeof s === "object" &&
        s !== null &&
        typeof (s as MCPServerConfig).name === "string" &&
        typeof (s as MCPServerConfig).url === "string"
    );
  } catch (e) {
    console.warn("Failed to parse MCP_SERVERS:", e);
    return [];
  }
}

const mcpServers = getMCPServers();

const runtime = new CopilotRuntime({
  // Remote backend actions endpoint for tool execution
  // Story 21-C3: Backend handles MCP tool calls via /api/v1/copilot/tools/call
  remoteEndpoints: process.env.BACKEND_URL
    ? [
        {
          url: `${process.env.BACKEND_URL}/api/v1/copilot`,
        },
      ]
    : [],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

/**
 * GET endpoint to expose MCP configuration status for debugging.
 * Only enabled in development.
 */
export const GET = async () => {
  if (process.env.NODE_ENV !== "development") {
    return new Response("Not found", { status: 404 });
  }

  return Response.json({
    mcpEnabled: mcpServers.length > 0,
    serverCount: mcpServers.length,
    servers: mcpServers.map((s) => ({ name: s.name, url: s.url })),
    backendUrl: process.env.BACKEND_URL || null,
  });
};
