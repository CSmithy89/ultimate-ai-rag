import { NextRequest, NextResponse } from "next/server";

/**
 * Telemetry API route - proxies telemetry events to the backend.
 *
 * Story 21-B1: Configure Observability Hooks and Dev Console
 *
 * This route receives telemetry events from the useAnalytics hook and
 * forwards them to the backend's /api/v1/telemetry endpoint.
 *
 * Security:
 * - Same-origin only (Next.js API route)
 * - PII already redacted by useAnalytics before reaching this point
 * - Backend performs additional sanitization
 *
 * Error handling:
 * - Gracefully handles backend unavailability
 * - Returns 202 Accepted immediately to avoid blocking UI
 */

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // Forward to backend telemetry endpoint
    // Fire-and-forget: don't wait for backend response
    // Use 2-second timeout to prevent hanging connections
    const tenantId = req.headers.get("x-tenant-id");
    fetch(`${BACKEND_URL}/api/v1/telemetry`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Forward origin for CORS validation
        Origin: req.headers.get("origin") ?? "",
        // Forward tenant ID for multi-tenancy (required per CLAUDE.md)
        ...(tenantId && { "X-Tenant-ID": tenantId }),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(2000), // 2 second timeout
    }).catch((error) => {
      // Log but don't fail the request (includes timeout errors)
      console.error("[Telemetry] Backend forward failed:", error);
    });

    // Return immediately with 202 Accepted
    // Telemetry should never block the UI
    return NextResponse.json(
      { status: "accepted" },
      { status: 202 }
    );
  } catch (error) {
    // Log error but return success to avoid blocking UI
    console.error("[Telemetry] Request processing error:", error);
    return NextResponse.json(
      { status: "accepted" },
      { status: 202 }
    );
  }
}
