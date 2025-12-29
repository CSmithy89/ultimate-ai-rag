# Story 6-3: Generative UI Components

Status: drafted
Epic: 6 - Interactive Copilot Experience
Priority: High
Depends on: Story 6-2 (Chat Sidebar Interface)

## User Story

As an **end-user**,
I want **the AI to render specialized UI components within the chat**,
So that **I can see interactive visualizations like source citations, formatted answers, and knowledge graph previews**.

## Acceptance Criteria

- Given the agent determines a visualization would help
- When it sends a Generative UI payload via AG-UI protocol
- Then the frontend dynamically renders the appropriate component
- And SourceCard displays citation information with source type, title, snippet, and confidence
- And AnswerPanel renders formatted responses with markdown support and source references
- And GraphPreview shows entity relationships using React Flow in a compact visualization
- And components are interactive (clickable, hoverable, expandable)
- And the UX follows the "Professional Forge" design direction (Indigo-600, Slate, Emerald-500)
- And all components use shadcn/ui patterns and Tailwind styling

## Technical Approach

### 1. Create SourceCard Component

**File:** `frontend/components/copilot/components/SourceCard.tsx`

Create a citation display component that shows source information with visual indicators:

```typescript
"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  FileText,
  Globe,
  Database,
  BookOpen,
  ExternalLink,
  CheckCircle2,
} from "lucide-react";
import type { Source } from "@/types/copilot";

/**
 * Source type to icon mapping.
 */
const sourceTypeIcons: Record<string, React.ElementType> = {
  document: FileText,
  web: Globe,
  database: Database,
  knowledge_graph: BookOpen,
  default: FileText,
};

/**
 * Get confidence level color based on similarity score.
 */
function getConfidenceColor(similarity: number): string {
  if (similarity >= 0.9) return "bg-emerald-100 text-emerald-800 border-emerald-200";
  if (similarity >= 0.7) return "bg-indigo-100 text-indigo-800 border-indigo-200";
  if (similarity >= 0.5) return "bg-amber-100 text-amber-800 border-amber-200";
  return "bg-slate-100 text-slate-800 border-slate-200";
}

interface SourceCardProps {
  source: Source;
  index?: number;
  onClick?: (source: Source) => void;
  isHighlighted?: boolean;
  showApprovalStatus?: boolean;
}

/**
 * SourceCard displays citation information for a retrieved source.
 * Used in Generative UI to show sources referenced in AI responses.
 *
 * Story 6-3: Generative UI Components
 */
export const SourceCard = memo(function SourceCard({
  source,
  index,
  onClick,
  isHighlighted = false,
  showApprovalStatus = false,
}: SourceCardProps) {
  const sourceType = (source.metadata?.type as string) || "default";
  const IconComponent = sourceTypeIcons[sourceType] || sourceTypeIcons.default;
  const confidencePercent = Math.round(source.similarity * 100);

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all duration-200 hover:shadow-md",
        "border border-slate-200 hover:border-indigo-300",
        isHighlighted && "ring-2 ring-indigo-500 ring-offset-2",
        onClick && "hover:bg-slate-50"
      )}
      onClick={() => onClick?.(source)}
    >
      <CardHeader className="pb-2 pt-3 px-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            {index !== undefined && (
              <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-600 text-white text-xs font-medium flex items-center justify-center">
                {index + 1}
              </span>
            )}
            <IconComponent className="h-4 w-4 text-slate-500 flex-shrink-0" />
            <CardTitle className="text-sm font-medium text-slate-900 truncate">
              {source.title}
            </CardTitle>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {showApprovalStatus && source.isApproved && (
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            )}
            <Badge
              variant="outline"
              className={cn("text-xs font-mono", getConfidenceColor(source.similarity))}
            >
              {confidencePercent}%
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-3 px-4">
        <p className="text-sm text-slate-600 line-clamp-2">{source.preview}</p>
        {source.metadata?.url && (
          <a
            href={source.metadata.url as string}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 mt-2 text-xs text-indigo-600 hover:text-indigo-800"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="h-3 w-3" />
            View source
          </a>
        )}
      </CardContent>
    </Card>
  );
});
```

Key implementation details:
- Memoized component for performance in lists
- Source type icons for visual differentiation
- Confidence badge with color-coded severity (Emerald > Indigo > Amber > Slate)
- Truncated title and 2-line snippet preview
- Optional approval status indicator for HITL integration
- External link support for source URLs

### 2. Create AnswerPanel Component

**File:** `frontend/components/copilot/components/AnswerPanel.tsx`

Create a formatted response panel with markdown rendering and source references:

```typescript
"use client";

import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sparkles, Copy, Check, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { SourceCard } from "./SourceCard";
import type { Source } from "@/types/copilot";

interface AnswerPanelProps {
  answer: string;
  sources?: Source[];
  title?: string;
  isStreaming?: boolean;
  showSources?: boolean;
  onSourceClick?: (source: Source) => void;
  className?: string;
}

/**
 * AnswerPanel renders a formatted AI response with markdown support
 * and collapsible source references.
 *
 * Story 6-3: Generative UI Components
 */
export const AnswerPanel = memo(function AnswerPanel({
  answer,
  sources = [],
  title = "Answer",
  isStreaming = false,
  showSources = true,
  onSourceClick,
  className,
}: AnswerPanelProps) {
  const [copied, setCopied] = useState(false);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  // Extract inline source references like [1], [2] from the answer
  const sourceReferences = useMemo(() => {
    const matches = answer.match(/\[(\d+)\]/g);
    if (!matches) return [];
    return [...new Set(matches.map((m) => parseInt(m.slice(1, -1), 10) - 1))];
  }, [answer]);

  const referencedSources = useMemo(() => {
    return sourceReferences
      .filter((idx) => idx >= 0 && idx < sources.length)
      .map((idx) => sources[idx]);
  }, [sourceReferences, sources]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card className={cn("border-slate-200", className)}>
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-600" />
            <CardTitle className="text-sm font-semibold text-slate-900">
              {title}
            </CardTitle>
            {isStreaming && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 animate-pulse">
                Generating...
              </span>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-8 w-8 p-0"
          >
            {copied ? (
              <Check className="h-4 w-4 text-emerald-500" />
            ) : (
              <Copy className="h-4 w-4 text-slate-500" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {/* Markdown rendered answer */}
        <div className="prose prose-sm prose-slate max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Custom styling for inline source references
              a: ({ href, children, ...props }) => (
                <a
                  href={href}
                  className="text-indigo-600 hover:text-indigo-800 no-underline hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                  {...props}
                >
                  {children}
                </a>
              ),
              code: ({ className, children, ...props }) => {
                const isInline = !className;
                return isInline ? (
                  <code
                    className="px-1 py-0.5 rounded bg-slate-100 text-slate-800 font-mono text-xs"
                    {...props}
                  >
                    {children}
                  </code>
                ) : (
                  <code
                    className={cn("font-mono text-sm", className)}
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
              pre: ({ children, ...props }) => (
                <pre
                  className="bg-slate-900 text-slate-100 rounded-lg p-4 overflow-x-auto"
                  {...props}
                >
                  {children}
                </pre>
              ),
            }}
          >
            {answer}
          </ReactMarkdown>
        </div>

        {/* Source references section */}
        {showSources && referencedSources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-100">
            <button
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
              className="flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900"
            >
              {sourcesExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
              Sources ({referencedSources.length})
            </button>
            {sourcesExpanded && (
              <div className="mt-3 space-y-2">
                {referencedSources.map((source, idx) => (
                  <SourceCard
                    key={source.id}
                    source={source}
                    index={sourceReferences[idx]}
                    onClick={onSourceClick}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
});
```

Key implementation details:
- Markdown rendering with remark-gfm for GitHub-flavored markdown
- Custom styled code blocks with syntax highlighting support
- Copy to clipboard functionality
- Streaming indicator for in-progress responses
- Collapsible source references section
- Automatic extraction of inline source citations like `[1]`, `[2]`

### 3. Create GraphPreview Component

**File:** `frontend/components/copilot/components/GraphPreview.tsx`

Create a mini knowledge graph visualization using React Flow:

```typescript
"use client";

import { memo, useMemo, useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
  MarkerType,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";

/**
 * Graph node from AG-UI payload.
 */
export interface GraphPreviewNode {
  id: string;
  label: string;
  type?: string;
  properties?: Record<string, unknown>;
}

/**
 * Graph edge from AG-UI payload.
 */
export interface GraphPreviewEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  type?: string;
}

interface GraphPreviewProps {
  nodes: GraphPreviewNode[];
  edges: GraphPreviewEdge[];
  title?: string;
  onExpand?: () => void;
  onNodeClick?: (node: GraphPreviewNode) => void;
  className?: string;
  height?: number;
}

/**
 * Entity type to color mapping for nodes.
 */
const entityColors: Record<string, string> = {
  PERSON: "#4F46E5",     // Indigo-600
  ORGANIZATION: "#0891B2", // Cyan-600
  CONCEPT: "#7C3AED",    // Violet-600
  DOCUMENT: "#059669",   // Emerald-600
  TECHNOLOGY: "#DC2626", // Red-600
  EVENT: "#D97706",      // Amber-600
  LOCATION: "#2563EB",   // Blue-600
  default: "#6B7280",    // Slate-500
};

/**
 * Custom node component for the graph preview.
 */
function PreviewNode({ data }: { data: { label: string; entityType?: string } }) {
  const color = entityColors[data.entityType || "default"] || entityColors.default;

  return (
    <div
      className="px-3 py-2 rounded-lg shadow-sm border-2 text-white text-xs font-medium max-w-[120px] truncate"
      style={{ backgroundColor: color, borderColor: color }}
    >
      {data.label}
    </div>
  );
}

const nodeTypes: NodeTypes = {
  preview: PreviewNode,
};

/**
 * Transform API nodes to React Flow format with circular layout.
 */
function transformNodes(nodes: GraphPreviewNode[]): Node[] {
  const centerX = 150;
  const centerY = 100;
  const radius = 80;

  // If only one node, place it in center
  if (nodes.length === 1) {
    return [
      {
        id: nodes[0].id,
        type: "preview",
        position: { x: centerX, y: centerY },
        data: { label: nodes[0].label, entityType: nodes[0].type },
      },
    ];
  }

  // Place first node in center, others in a circle around it
  return nodes.map((node, index) => {
    if (index === 0) {
      return {
        id: node.id,
        type: "preview",
        position: { x: centerX, y: centerY },
        data: { label: node.label, entityType: node.type },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
    }

    const angle = ((index - 1) * 2 * Math.PI) / (nodes.length - 1);
    return {
      id: node.id,
      type: "preview",
      position: {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      },
      data: { label: node.label, entityType: node.type },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    };
  });
}

/**
 * Transform API edges to React Flow format.
 */
function transformEdges(edges: GraphPreviewEdge[]): Edge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label,
    type: "smoothstep",
    animated: true,
    style: { stroke: "#94A3B8" }, // Slate-400
    labelStyle: { fontSize: 10, fill: "#64748B" }, // Slate-500
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 15,
      height: 15,
      color: "#94A3B8",
    },
  }));
}

/**
 * GraphPreview displays a mini knowledge graph visualization
 * within the chat flow using React Flow.
 *
 * Story 6-3: Generative UI Components
 */
export const GraphPreview = memo(function GraphPreview({
  nodes: apiNodes,
  edges: apiEdges,
  title = "Knowledge Graph",
  onExpand,
  onNodeClick,
  className,
  height = 200,
}: GraphPreviewProps) {
  // Transform data to React Flow format
  const flowNodes = useMemo(() => transformNodes(apiNodes), [apiNodes]);
  const flowEdges = useMemo(() => transformEdges(apiEdges), [apiEdges]);

  const [nodes, , onNodesChange] = useNodesState(flowNodes);
  const [edges, , onEdgesChange] = useEdgesState(flowEdges);

  // Handle node click
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        const apiNode = apiNodes.find((n) => n.id === node.id);
        if (apiNode) {
          onNodeClick(apiNode);
        }
      }
    },
    [apiNodes, onNodeClick]
  );

  return (
    <Card className={cn("border-slate-200 overflow-hidden", className)}>
      <CardHeader className="pb-2 pt-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-slate-900 flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-indigo-600" />
            {title}
          </CardTitle>
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-500 mr-2">
              {apiNodes.length} nodes, {apiEdges.length} edges
            </span>
            {onExpand && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onExpand}
                className="h-7 w-7 p-0"
              >
                <Maximize2 className="h-4 w-4 text-slate-500" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div style={{ height }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={handleNodeClick}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3, maxZoom: 1.5 }}
            zoomOnScroll={false}
            panOnDrag={false}
            nodesDraggable={false}
            nodesConnectable={false}
            proOptions={{ hideAttribution: true }}
            className="bg-slate-50"
          >
            <Controls
              showInteractive={false}
              className="bg-white border border-slate-200 rounded shadow-sm"
            />
            <Background color="#E2E8F0" gap={12} size={1} />
          </ReactFlow>
        </div>
      </CardContent>
    </Card>
  );
});

export default GraphPreview;
```

Key implementation details:
- Compact React Flow visualization within a Card
- Circular layout with center node for focused entity
- Entity type color coding matching existing KnowledgeGraph component
- Animated edges for visual interest
- Minimal controls (no pan/drag in preview mode)
- Expand button for full-screen view
- Node/edge count indicator

### 4. Create useGenerativeUI Hook

**File:** `frontend/hooks/use-generative-ui.ts`

Create a custom hook for registering render handlers with CopilotKit:

```typescript
"use client";

import { useCopilotAction, useCoAgentStateRender } from "@copilotkit/react-core";
import { useCallback, useState } from "react";
import { SourceCard } from "@/components/copilot/components/SourceCard";
import { AnswerPanel } from "@/components/copilot/components/AnswerPanel";
import { GraphPreview, type GraphPreviewNode, type GraphPreviewEdge } from "@/components/copilot/components/GraphPreview";
import type { Source } from "@/types/copilot";

/**
 * Parameters for the show_sources action.
 */
interface ShowSourcesParams {
  sources: Source[];
  title?: string;
}

/**
 * Parameters for the show_answer action.
 */
interface ShowAnswerParams {
  answer: string;
  sources?: Source[];
  title?: string;
}

/**
 * Parameters for the show_knowledge_graph action.
 */
interface ShowKnowledgeGraphParams {
  nodes: GraphPreviewNode[];
  edges: GraphPreviewEdge[];
  title?: string;
}

/**
 * State for generative UI components.
 */
interface GenerativeUIState {
  sources: Source[];
  answer: string | null;
  graphData: { nodes: GraphPreviewNode[]; edges: GraphPreviewEdge[] } | null;
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
      const { sources, title } = args as ShowSourcesParams;

      if (status === "executing" || status === "complete") {
        return (
          <div className="space-y-2 my-2">
            {title && (
              <h4 className="text-sm font-medium text-slate-700">{title}</h4>
            )}
            {sources.map((source: Source, idx: number) => (
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

      return null;
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
      const { answer, sources, title } = args as ShowAnswerParams;

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
      const { nodes, edges, title } = args as ShowKnowledgeGraphParams;

      if (status === "executing" || status === "complete") {
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

      return null;
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
      return null;
    },
  });

  return {
    state,
    setState,
  };
}
```

Key implementation details:
- Registers three CopilotKit actions: `show_sources`, `show_answer`, `show_knowledge_graph`
- Each action has a render function that displays the appropriate component
- Uses `useCoAgentStateRender` to sync with backend agent state
- Provides callbacks for source clicks, graph node clicks, and expand actions
- Status-aware rendering (executing vs complete)

### 5. Create GenerativeUIRenderer Component

**File:** `frontend/components/copilot/GenerativeUIRenderer.tsx`

Create a wrapper component that initializes the generative UI system:

```typescript
"use client";

import { useGenerativeUI } from "@/hooks/use-generative-ui";
import type { Source } from "@/types/copilot";
import type { GraphPreviewNode } from "./components/GraphPreview";

interface GenerativeUIRendererProps {
  onSourceClick?: (source: Source) => void;
  onGraphNodeClick?: (node: GraphPreviewNode) => void;
  onGraphExpand?: () => void;
}

/**
 * GenerativeUIRenderer initializes the generative UI action handlers
 * for rendering dynamic components within the chat flow.
 *
 * Include this component within your CopilotKit context to enable
 * generative UI capabilities.
 *
 * Story 6-3: Generative UI Components
 *
 * @example
 * ```tsx
 * <CopilotSidebar>
 *   <GenerativeUIRenderer
 *     onSourceClick={(source) => openSourceModal(source)}
 *     onGraphExpand={() => openGraphModal()}
 *   />
 * </CopilotSidebar>
 * ```
 */
export function GenerativeUIRenderer({
  onSourceClick,
  onGraphNodeClick,
  onGraphExpand,
}: GenerativeUIRendererProps) {
  // Initialize generative UI hooks
  useGenerativeUI({
    onSourceClick,
    onGraphNodeClick,
    onGraphExpand,
  });

  // This component doesn't render anything itself;
  // it just registers the action handlers
  return null;
}
```

### 6. Update Types

**File:** `frontend/types/copilot.ts` (modify)

Add types for generative UI:

```typescript
// Add to existing types

/**
 * Graph node for generative UI graph preview.
 */
export interface GraphPreviewNode {
  id: string;
  label: string;
  type?: string;
  properties?: Record<string, unknown>;
}

/**
 * Graph edge for generative UI graph preview.
 */
export interface GraphPreviewEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  type?: string;
}

/**
 * Generative UI state managed by the agent.
 */
export interface GenerativeUIState {
  sources: Source[];
  answer: string | null;
  graphData: {
    nodes: GraphPreviewNode[];
    edges: GraphPreviewEdge[];
  } | null;
}

// Zod schemas for validation
export const GraphPreviewNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string().optional(),
  properties: z.record(z.string(), z.unknown()).optional(),
});

export const GraphPreviewEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  label: z.string().optional(),
  type: z.string().optional(),
});

export const GenerativeUIStateSchema = z.object({
  sources: z.array(SourceSchema),
  answer: z.string().nullable(),
  graphData: z
    .object({
      nodes: z.array(GraphPreviewNodeSchema),
      edges: z.array(GraphPreviewEdgeSchema),
    })
    .nullable(),
});
```

### 7. Update ChatSidebar to Include GenerativeUIRenderer

**File:** `frontend/components/copilot/ChatSidebar.tsx` (modify)

```typescript
"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import { GenerativeUIRenderer } from "./GenerativeUIRenderer";

/**
 * ChatSidebar component wrapping CopilotKit's CopilotSidebar
 * with custom styling following the project's design system.
 *
 * Story 6-2: Chat Sidebar Interface
 * Story 6-3: Generative UI Components
 *
 * Design System:
 * - Primary (Indigo-600): #4F46E5
 * - Secondary (Emerald-500): #10B981
 * - Neutral: Slate colors
 */
export function ChatSidebar() {
  return (
    <CopilotErrorBoundary>
      <CopilotSidebar
        defaultOpen={true}
        labels={{
          title: "AI Copilot",
          initial: "How can I help you today?",
        }}
        className="copilot-sidebar"
      >
        <ThoughtTraceStepper />
        <GenerativeUIRenderer />
      </CopilotSidebar>
    </CopilotErrorBoundary>
  );
}
```

### 8. Backend: Add Generative UI Event Types

**File:** `backend/src/agentic_rag_backend/models/copilot.py` (modify)

Add event types for triggering generative UI from the backend:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ToolCallEvent(AGUIEvent):
    """Event for triggering a tool/action call that may render UI."""

    event: Literal[AGUIEventType.TOOL_CALL_START] = AGUIEventType.TOOL_CALL_START
    tool_call_id: str = Field(..., description="Unique ID for this tool call")
    tool_name: str = Field(..., description="Name of the tool being called")


class ToolCallArgsEvent(AGUIEvent):
    """Event containing arguments for a tool call."""

    event: Literal[AGUIEventType.TOOL_CALL_ARGS] = AGUIEventType.TOOL_CALL_ARGS
    tool_call_id: str = Field(..., description="Tool call ID this relates to")
    args: Dict[str, Any] = Field(..., description="Arguments for the tool")


class ToolCallEndEvent(AGUIEvent):
    """Event indicating tool call completion."""

    event: Literal[AGUIEventType.TOOL_CALL_END] = AGUIEventType.TOOL_CALL_END
    tool_call_id: str = Field(..., description="Tool call ID that completed")


# Generative UI helper functions
def create_show_sources_event(
    sources: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> List[AGUIEvent]:
    """Create AG-UI events to trigger show_sources action."""
    import uuid

    tool_call_id = str(uuid.uuid4())
    return [
        ToolCallEvent(tool_call_id=tool_call_id, tool_name="show_sources"),
        ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            args={"sources": sources, "title": title},
        ),
        ToolCallEndEvent(tool_call_id=tool_call_id),
    ]


def create_show_answer_event(
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    title: Optional[str] = None,
) -> List[AGUIEvent]:
    """Create AG-UI events to trigger show_answer action."""
    import uuid

    tool_call_id = str(uuid.uuid4())
    return [
        ToolCallEvent(tool_call_id=tool_call_id, tool_name="show_answer"),
        ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            args={"answer": answer, "sources": sources, "title": title},
        ),
        ToolCallEndEvent(tool_call_id=tool_call_id),
    ]


def create_show_knowledge_graph_event(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> List[AGUIEvent]:
    """Create AG-UI events to trigger show_knowledge_graph action."""
    import uuid

    tool_call_id = str(uuid.uuid4())
    return [
        ToolCallEvent(tool_call_id=tool_call_id, tool_name="show_knowledge_graph"),
        ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            args={"nodes": nodes, "edges": edges, "title": title},
        ),
        ToolCallEndEvent(tool_call_id=tool_call_id),
    ]
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `frontend/components/copilot/components/SourceCard.tsx` | Citation display component with source type, title, snippet, confidence |
| `frontend/components/copilot/components/AnswerPanel.tsx` | Formatted response with markdown rendering and source references |
| `frontend/components/copilot/components/GraphPreview.tsx` | Mini knowledge graph visualization using React Flow |
| `frontend/components/copilot/GenerativeUIRenderer.tsx` | Wrapper component that initializes generative UI handlers |
| `frontend/hooks/use-generative-ui.ts` | Custom hook for registering CopilotKit render actions |

### Modified Files

| File | Change |
|------|--------|
| `frontend/types/copilot.ts` | Add GraphPreviewNode, GraphPreviewEdge, GenerativeUIState types |
| `frontend/components/copilot/ChatSidebar.tsx` | Include GenerativeUIRenderer component |
| `backend/src/agentic_rag_backend/models/copilot.py` | Add ToolCall events and generative UI helper functions |

## Dependencies

### Frontend Dependencies (npm)

New dependencies required:
```json
{
  "react-markdown": "^9.0.0",
  "remark-gfm": "^4.0.0"
}
```

Already installed from previous stories:
```json
{
  "@copilotkit/react-core": "^1.50.1",
  "@copilotkit/react-ui": "^1.50.1",
  "reactflow": "^11.x.x",
  "lucide-react": "^0.x.x"
}
```

### Backend Dependencies (pip)

No new dependencies required. Uses existing:
- Pydantic models from Story 6-1
- AG-UI bridge from Story 6-1

### Environment Variables

No new environment variables required.

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| SourceCard renders with correct confidence colors | `frontend/__tests__/components/copilot/SourceCard.test.tsx` |
| SourceCard handles click events | `frontend/__tests__/components/copilot/SourceCard.test.tsx` |
| AnswerPanel renders markdown correctly | `frontend/__tests__/components/copilot/AnswerPanel.test.tsx` |
| AnswerPanel extracts source references | `frontend/__tests__/components/copilot/AnswerPanel.test.tsx` |
| AnswerPanel copy button works | `frontend/__tests__/components/copilot/AnswerPanel.test.tsx` |
| GraphPreview transforms nodes/edges correctly | `frontend/__tests__/components/copilot/GraphPreview.test.tsx` |
| GraphPreview handles empty data gracefully | `frontend/__tests__/components/copilot/GraphPreview.test.tsx` |
| useGenerativeUI registers all actions | `frontend/__tests__/hooks/use-generative-ui.test.ts` |

### Integration Tests

| Test | Location |
|------|----------|
| GenerativeUIRenderer initializes correctly within CopilotSidebar | `frontend/__tests__/integration/generative-ui.test.tsx` |
| Actions render correct components | `frontend/__tests__/integration/generative-ui.test.tsx` |
| Backend tool call events trigger frontend rendering | `frontend/__tests__/integration/generative-ui.test.tsx` |

### E2E Tests

| Test | Location |
|------|----------|
| Source cards appear when agent retrieves sources | `frontend/tests/e2e/generative-ui.spec.ts` |
| Answer panel renders with formatted markdown | `frontend/tests/e2e/generative-ui.spec.ts` |
| Graph preview displays and is interactive | `frontend/tests/e2e/generative-ui.spec.ts` |

### Manual Verification Steps

1. Start backend with `cd backend && uv run uvicorn agentic_rag_backend.main:app --reload`
2. Start frontend with `cd frontend && pnpm dev`
3. Open browser to `http://localhost:3000`
4. Open the chat sidebar and submit a query
5. Verify SourceCard components appear:
   - Source type icon displays correctly
   - Title is truncated if too long
   - Confidence badge shows percentage with appropriate color
   - Click on card triggers callback
6. Verify AnswerPanel component:
   - Markdown renders correctly (headers, code, lists)
   - Source references like `[1]` are parsed
   - Copy button copies text to clipboard
   - Sources section expands/collapses
7. Verify GraphPreview component:
   - Nodes display in circular layout
   - Entity type colors match design system
   - Edges are animated
   - Expand button triggers callback
   - Node click triggers callback
8. Test error cases:
   - Empty sources array renders gracefully
   - Invalid markdown doesn't crash
   - Graph with single node renders correctly

## Definition of Done

- [ ] All acceptance criteria met
- [ ] SourceCard component created with confidence badges and source type icons
- [ ] AnswerPanel component created with markdown rendering and source references
- [ ] GraphPreview component created with React Flow mini visualization
- [ ] GenerativeUIRenderer component created and integrated into ChatSidebar
- [ ] useGenerativeUI hook registers all CopilotKit actions
- [ ] Backend helper functions created for generating tool call events
- [ ] TypeScript types added for generative UI data structures
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Manual verification completed
- [ ] No TypeScript errors
- [ ] Code follows project naming conventions
- [ ] Components use shadcn/ui patterns and Tailwind styling
- [ ] Design system colors applied correctly

## Technical Notes

### CopilotKit Action Registration

CopilotKit actions with the `render` option allow the agent to trigger UI rendering from the backend:

```typescript
useCopilotAction({
  name: "action_name",
  parameters: [...],
  render: ({ status, args }) => <Component {...args} />,
});
```

The `status` can be:
- `"inProgress"` - Action is being executed
- `"executing"` - Action is actively running
- `"complete"` - Action finished successfully

### React Flow in Compact Mode

For the GraphPreview, we disable interactive features to keep it compact:
- `zoomOnScroll={false}` - Prevents accidental zooming
- `panOnDrag={false}` - Prevents accidental panning
- `nodesDraggable={false}` - Nodes stay in place
- `nodesConnectable={false}` - No edge editing

### Markdown Security

The `react-markdown` library is safe by default - it doesn't render raw HTML. For additional security, we don't use the `rehype-raw` plugin.

### Performance Considerations

- All generative UI components are memoized with `React.memo`
- GraphPreview uses `useMemo` for node/edge transformations
- Large source lists should be paginated (future enhancement)

## Accessibility Considerations

- SourceCard uses semantic button elements for click handling
- AnswerPanel copy button has proper aria labels
- GraphPreview provides keyboard navigation for controls
- All interactive elements have visible focus states
- Color is not the only indicator (icons accompany confidence levels)

## Design System Colors

Per UX Design Specification:
- **Primary (Indigo-600):** #4F46E5 - Node colors, action buttons
- **Secondary (Emerald-500):** #10B981 - High confidence, approved sources
- **Neutral (Slate):** #0F172A to #F8FAFC - Text, backgrounds
- **Accent (Amber-400):** #FBBF24 - Medium confidence warnings

Entity type colors in GraphPreview:
- PERSON: Indigo-600
- ORGANIZATION: Cyan-600
- CONCEPT: Violet-600
- DOCUMENT: Emerald-600
- TECHNOLOGY: Red-600
- EVENT: Amber-600
- LOCATION: Blue-600

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| React Flow bundle size | Use dynamic imports for GraphPreview if needed |
| Markdown XSS | Use react-markdown without rehype-raw |
| Large graph rendering | Limit nodes to 20 in preview, full view for more |
| CopilotKit action conflicts | Use unique action names with namespace prefix |
| SSR issues with React Flow | Ensure "use client" directive, dynamic import if needed |

## References

- [CopilotKit useCopilotAction](https://docs.copilotkit.ai/reference/hooks/useCopilotAction)
- [CopilotKit useCoAgentStateRender](https://docs.copilotkit.ai/reference/hooks/useCoAgentStateRender)
- [React Flow Documentation](https://reactflow.dev/docs)
- [React Markdown](https://github.com/remarkjs/react-markdown)
- [Epic 6 Tech Spec](_bmad-output/implementation-artifacts/epic-6-tech-spec.md)
- [Story 6-2: Chat Sidebar Interface](_bmad-output/implementation-artifacts/stories/6-2-chat-sidebar-interface.md)
- [UX Design Specification](_bmad-output/project-planning-artifacts/ux-design-specification.md)
