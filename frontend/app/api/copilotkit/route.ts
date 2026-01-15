import { NextRequest, NextResponse } from "next/server";

/**
 * CopilotKit API route that proxies requests to the FastAPI backend.
 *
 * Story 6-2: Implements the backend proxy for CopilotKit integration.
 *
 * The backend at /api/v1/copilot implements the AG-UI protocol with SSE streaming,
 * handling LLM orchestration, tool execution, and multi-tenancy.
 *
 * Environment variables:
 * - COPILOT_BACKEND_URL: Backend URL (default: http://localhost:8000)
 *   In Docker: http://backend:8000
 */

const BACKEND_URL = process.env.COPILOT_BACKEND_URL || "http://localhost:8000";
const COPILOT_ENDPOINT = `${BACKEND_URL}/api/v1/copilot`;

export const POST = async (req: NextRequest) => {
  try {
    // Get the request body
    const body = await req.json();

    // Forward headers that might be needed (tenant ID, etc.)
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };

    // Forward tenant ID if present in the request config
    const tenantId = body?.config?.configurable?.tenant_id;
    if (tenantId) {
      headers["X-Tenant-ID"] = tenantId;
    }

    // Make request to backend
    const backendResponse = await fetch(COPILOT_ENDPOINT, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    // Check if the response is SSE (Server-Sent Events)
    const contentType = backendResponse.headers.get("content-type") || "";

    if (contentType.includes("text/event-stream")) {
      // Stream the SSE response
      const readable = backendResponse.body;

      if (!readable) {
        return NextResponse.json(
          { error: "No response body from backend" },
          { status: 502 }
        );
      }

      // Create a streaming response
      return new Response(readable, {
        status: backendResponse.status,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    // For non-SSE responses, forward as JSON
    const data = await backendResponse.json();
    return NextResponse.json(data, { status: backendResponse.status });
  } catch (error) {
    console.error("CopilotKit proxy error:", error);

    // Return a proper error response
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";

    return NextResponse.json(
      {
        error: "Failed to proxy request to backend",
        detail: errorMessage,
      },
      { status: 502 }
    );
  }
};

// Also handle GET requests for potential health checks
export const GET = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    const data = await response.json();
    return NextResponse.json({
      status: "ok",
      backend: data,
      endpoint: COPILOT_ENDPOINT,
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        error: "Backend not reachable",
        endpoint: COPILOT_ENDPOINT,
      },
      { status: 503 }
    );
  }
};
