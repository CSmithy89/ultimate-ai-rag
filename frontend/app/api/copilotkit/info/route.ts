import { NextRequest, NextResponse } from "next/server";

/**
 * CopilotKit /info endpoint for runtime sync.
 *
 * This endpoint returns information about available agents and actions
 * that CopilotKit uses during runtime initialization.
 *
 * CopilotKit expects a "default" agent to be present. We also register
 * "orchestrator" for useCoAgentStateRender hooks.
 */
export const GET = async (req: NextRequest) => {
  console.log("[CopilotKit /info] GET request received", {
    url: req.url,
    headers: Object.fromEntries(req.headers.entries()),
  });
  return NextResponse.json({
    version: "1.0.0",
    agents: {
      default: {
        name: "default",
        description: "Default RAG assistant for document retrieval and Q&A",
      },
      orchestrator: {
        name: "orchestrator",
        description: "RAG orchestrator agent that handles document retrieval and Q&A",
      },
    },
    actions: [],
  });
};

// Also handle POST requests as some CopilotKit versions use POST
export const POST = async (req: NextRequest) => {
  const body = await req.text();
  console.log("[CopilotKit /info] POST request received", {
    url: req.url,
    headers: Object.fromEntries(req.headers.entries()),
    body: body.substring(0, 500),
  });
  return NextResponse.json({
    version: "1.0.0",
    agents: {
      default: {
        name: "default",
        description: "Default RAG assistant for document retrieval and Q&A",
      },
      orchestrator: {
        name: "orchestrator",
        description: "RAG orchestrator agent that handles document retrieval and Q&A",
      },
    },
    actions: [],
  });
};
