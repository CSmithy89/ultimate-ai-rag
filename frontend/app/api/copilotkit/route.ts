import { NextRequest, NextResponse } from "next/server";
import {
  CopilotRuntime,
  EmptyAdapter,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
// Import from the langgraph subpath to avoid deprecation error
import { HttpAgent } from "@ag-ui/client";
import OpenAI from "openai";

/**
 * CopilotKit API route using CopilotRuntime with LangGraphHttpAgent.
 *
 * Story 6-2: Implements the backend integration for CopilotKit.
 *
 * The backend at /api/v1/copilot implements the AG-UI protocol with SSE streaming.
 * CopilotRuntime handles protocol translation between CopilotKit and our backend.
 *
 * Environment variables:
 * - COPILOT_BACKEND_URL: Backend URL (default: http://localhost:8000)
 *   In Docker: http://backend:8000
 * - OPENAI_API_KEY: Required for OpenAIAdapter
 * - DEFAULT_TENANT_ID: Default tenant for multi-tenancy
 */

const BACKEND_URL = process.env.COPILOT_BACKEND_URL || "http://localhost:8000";
const DEFAULT_TENANT_ID =
  process.env.DEFAULT_TENANT_ID || "550e8400-e29b-41d4-a716-446655440000";

const SERVICE_ADAPTER = process.env.COPILOT_SERVICE_ADAPTER ?? "empty";

// Initialize OpenAI client (only used when explicitly configured).
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "dummy-key-for-passthrough",
});

const serviceAdapter =
  SERVICE_ADAPTER === "openai"
    ? new OpenAIAdapter({ openai } as any)
    : new EmptyAdapter();

// Configure the AG-UI HTTP agent to connect to our backend
const ragAgent = new HttpAgent({
  url: `${BACKEND_URL}/api/v1/copilot`,
  // Add custom headers for multi-tenancy
  headers: {
    "X-Tenant-ID": DEFAULT_TENANT_ID,
  },
});

// Create CopilotRuntime with our agent
const runtime = new CopilotRuntime({
  agents: {
    default: ragAgent,
  },
});

export const POST = async (req: NextRequest) => {
  try {
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime,
      serviceAdapter,
      endpoint: "/api/copilotkit",
    });

    return handleRequest(req);
  } catch (error) {
    console.error("CopilotKit runtime error:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";

    return NextResponse.json(
      {
        error: "CopilotKit runtime error",
        detail: errorMessage,
      },
      { status: 500 }
    );
  }
};

// Handle GET requests for health checks
export const GET = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    const data = await response.json();
    return NextResponse.json({
      status: "ok",
      backend: data,
      endpoint: `${BACKEND_URL}/api/v1/copilot`,
    });
  } catch {
    return NextResponse.json(
      {
        status: "error",
        error: "Backend not reachable",
        endpoint: `${BACKEND_URL}/api/v1/copilot`,
      },
      { status: 503 }
    );
  }
};
