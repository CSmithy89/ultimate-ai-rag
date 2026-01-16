import { NextResponse } from "next/server";

/**
 * CopilotKit /info endpoint for runtime sync.
 *
 * This endpoint returns information about available agents and actions
 * that CopilotKit uses during runtime initialization.
 *
 * CopilotKit expects a "default" agent to be present. We also register
 * "orchestrator" for useCoAgentStateRender hooks.
 */
export const GET = async () => {
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
export const POST = async () => {
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
