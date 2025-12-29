"use client";

import { useCopilotAction, useCoAgentStateRender } from "@copilotkit/react-core";
import { useState, useCallback } from "react";
import { SourceCard } from "@/components/copilot/components/SourceCard";
import { AnswerPanel } from "@/components/copilot/components/AnswerPanel";
import {
  GraphPreview,
  type GraphPreviewNode,
  type GraphPreviewEdge,
} from "@/components/copilot/components/GraphPreview";
import type {
  Source,
  GenerativeUIState,
  GraphPreviewNode as TypedGraphPreviewNode,
  GraphPreviewEdge as TypedGraphPreviewEdge,
} from "@/types/copilot";
import {
  SourceSchema,
  GraphPreviewNodeSchema,
  GraphPreviewEdgeSchema,
} from "@/types/copilot";
import { z } from "zod";

// Zod schemas for action argument validation
const ShowSourcesArgsSchema = z.object({
  sources: z.array(SourceSchema).optional().default([]),
  title: z.string().optional(),
});

const ShowAnswerArgsSchema = z.object({
  answer: z.string().optional().default(""),
  sources: z.array(SourceSchema).optional(),
  title: z.string().optional(),
});

const ShowKnowledgeGraphArgsSchema = z.object({
  nodes: z.array(GraphPreviewNodeSchema).optional().default([]),
  edges: z.array(GraphPreviewEdgeSchema).optional().default([]),
  title: z.string().optional(),
});

/**
 * Error fallback component for invalid action arguments.
 */
function ValidationErrorFallback({ message }: { message: string }) {
  return (
    <div className="my-2 p-3 border border-red-200 rounded-lg bg-red-50 text-red-700 text-sm">
      <strong>Validation Error:</strong> {message}
    </div>
  );
}

/**
 * Options for the useGenerativeUI hook.
 */
interface UseGenerativeUIOptions {
  onSourceClick?: (source: Source) => void;
  onGraphNodeClick?: (node: GraphPreviewNode) => void;
  onGraphExpand?: () => void;
}

/**
 * useGenerativeUI hook registers CopilotKit actions for rendering
 * generative UI components within the chat flow.
 *
 * Story 6-3: Generative UI Components
 *
 * @example
 * ```tsx
 * function ChatWithGenerativeUI() {
 *   useGenerativeUI({
 *     onSourceClick: (source) => console.log("Clicked source:", source),
 *     onGraphExpand: () => openFullGraphModal(),
 *   });
 *
 *   return <ChatSidebar />;
 * }
 * ```
 */
export function useGenerativeUI(options: UseGenerativeUIOptions = {}) {
  const { onSourceClick, onGraphNodeClick, onGraphExpand } = options;
  const [state, setState] = useState<GenerativeUIState>({
    sources: [],
    answer: null,
    graphData: null,
  });

  // Callback to update state when actions complete
  const updateStateFromAction = useCallback(
    (updates: Partial<GenerativeUIState>) => {
      setState((prev) => ({ ...prev, ...updates }));
    },
    []
  );

  // Register action to show source citations
  useCopilotAction({
    name: "show_sources",
    description: "Display retrieved sources as citation cards",
    parameters: [
      {
        name: "sources",
        type: "object[]",
        description: "Array of source objects with id, title, preview, similarity",
        required: true,
      },
      {
        name: "title",
        type: "string",
        description: "Optional title for the sources section",
        required: false,
      },
    ],
    render: ({ status, args }) => {
      // Validate args with Zod
      const parseResult = ShowSourcesArgsSchema.safeParse(args);

      if (!parseResult.success) {
        return (
          <ValidationErrorFallback
            message={`Invalid sources data: ${parseResult.error.message}`}
          />
        );
      }

      const { sources, title } = parseResult.data;

      if (status === "executing" || status === "complete") {
        // Update state when action completes (Issue 10 fix)
        if (status === "complete") {
          updateStateFromAction({ sources });
        }

        return (
          <div className="space-y-2 my-2">
            {title && (
              <h4 className="text-sm font-medium text-slate-700">{title}</h4>
            )}
            {sources?.map((source: Source, idx: number) => (
              <SourceCard
                key={source.id}
                source={source}
                index={idx}
                onClick={onSourceClick}
              />
            ))}
          </div>
        );
      }

      // Return empty fragment instead of null for type compatibility
      return <></>;
    },
  });

  // Register action to show formatted answer
  useCopilotAction({
    name: "show_answer",
    description: "Display a formatted answer with markdown and source references",
    parameters: [
      {
        name: "answer",
        type: "string",
        description: "The answer text with optional markdown formatting",
        required: true,
      },
      {
        name: "sources",
        type: "object[]",
        description: "Optional sources referenced in the answer",
        required: false,
      },
      {
        name: "title",
        type: "string",
        description: "Optional title for the answer panel",
        required: false,
      },
    ],
    render: ({ status, args }) => {
      // Validate args with Zod
      const parseResult = ShowAnswerArgsSchema.safeParse(args);

      if (!parseResult.success) {
        return (
          <ValidationErrorFallback
            message={`Invalid answer data: ${parseResult.error.message}`}
          />
        );
      }

      const { answer, sources, title } = parseResult.data;

      // Update state when action completes (Issue 10 fix)
      if (status === "complete") {
        updateStateFromAction({ answer, sources: sources || [] });
      }

      return (
        <AnswerPanel
          answer={answer}
          sources={sources}
          title={title}
          isStreaming={status === "executing"}
          onSourceClick={onSourceClick}
          className="my-2"
        />
      );
    },
  });

  // Register action to show knowledge graph preview
  useCopilotAction({
    name: "show_knowledge_graph",
    description: "Display a mini knowledge graph visualization",
    parameters: [
      {
        name: "nodes",
        type: "object[]",
        description: "Graph nodes with id, label, and optional type",
        required: true,
      },
      {
        name: "edges",
        type: "object[]",
        description: "Graph edges with id, source, target, and optional label",
        required: true,
      },
      {
        name: "title",
        type: "string",
        description: "Optional title for the graph",
        required: false,
      },
    ],
    render: ({ status, args }) => {
      // Validate args with Zod
      const parseResult = ShowKnowledgeGraphArgsSchema.safeParse(args);

      if (!parseResult.success) {
        return (
          <ValidationErrorFallback
            message={`Invalid graph data: ${parseResult.error.message}`}
          />
        );
      }

      const { nodes, edges, title } = parseResult.data;

      if (status === "executing" || status === "complete") {
        // Update state when action completes (Issue 10 fix)
        if (status === "complete") {
          updateStateFromAction({ graphData: { nodes, edges } });
        }

        return (
          <GraphPreview
            nodes={nodes}
            edges={edges}
            title={title}
            onNodeClick={onGraphNodeClick}
            onExpand={onGraphExpand}
            className="my-2"
          />
        );
      }

      // Return empty fragment instead of null for type compatibility
      return <></>;
    },
  });

  // Register state renderer for agent state updates
  useCoAgentStateRender<{ generativeUI?: GenerativeUIState }>({
    name: "orchestrator",
    render: ({ state: agentState }) => {
      if (agentState?.generativeUI) {
        // Update local state with agent state
        setState(agentState.generativeUI);
      }
      // Return empty fragment instead of null for type compatibility
      return <></>;
    },
  });

  return {
    state,
    setState,
  };
}

// Re-export types from @/types/copilot instead of duplicating (Issue 4 fix)
export type {
  GenerativeUIState,
  GraphPreviewNode,
  GraphPreviewEdge,
} from "@/types/copilot";

export type { UseGenerativeUIOptions };
