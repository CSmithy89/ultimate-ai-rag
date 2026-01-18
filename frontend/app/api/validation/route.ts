import { NextRequest, NextResponse } from "next/server";

/**
 * API route to proxy HITL validation responses to the backend.
 *
 * This solves the URL mismatch issue where:
 * - Browser uses NEXT_PUBLIC_BACKEND_URL (external URL like localhost:8000)
 * - Server uses COPILOT_BACKEND_URL (internal URL like backend:8000 in Docker)
 *
 * By proxying through this route, validation responses always use the correct
 * internal backend URL.
 */

const BACKEND_URL = process.env.COPILOT_BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const tenantId = req.headers.get("X-Tenant-ID") ||
      process.env.NEXT_PUBLIC_TENANT_ID ||
      "550e8400-e29b-41d4-a716-446655440000";

    const response = await fetch(`${BACKEND_URL}/api/v1/copilot/validation-response`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.detail || "Validation failed" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("[Validation API] Error:", error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
