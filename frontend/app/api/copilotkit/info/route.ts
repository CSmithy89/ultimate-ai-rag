import { NextResponse } from "next/server";

/**
 * CopilotKit /info endpoint for runtime sync.
 *
 * This endpoint returns information about available agents and actions
 * that CopilotKit uses during runtime initialization.
 *
 * The "orchestrator" agent is registered here to allow useCoAgentStateRender
 * hooks with name="orchestrator" to work properly.
 */
export const GET = async () => {
  return NextResponse.json({
    agents: {
      orchestrator: {
        name: "orchestrator",
        description: "RAG orchestrator agent that handles document retrieval and Q&A",
      },
    },
    actions: [],
  });
};
