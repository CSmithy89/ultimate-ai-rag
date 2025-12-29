"use client";

import { memo, useMemo, useCallback, useEffect } from "react";
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
import { Maximize2 } from "lucide-react";

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
  /** Nodes to display in the graph */
  nodes: GraphPreviewNode[];
  /** Edges connecting the nodes */
  edges: GraphPreviewEdge[];
  /** Title for the graph panel */
  title?: string;
  /** Callback when expand button is clicked */
  onExpand?: () => void;
  /** Callback when a node is clicked */
  onNodeClick?: (node: GraphPreviewNode) => void;
  /** Additional CSS classes */
  className?: string;
  /** Height of the graph container in pixels */
  height?: number;
}

/**
 * Entity type to color mapping for nodes.
 * Matches existing entityColors from types/graphs.ts
 */
const entityColors: Record<string, string> = {
  Person: "#3B82F6", // Blue-500
  Organization: "#10B981", // Emerald-500
  Technology: "#6366F1", // Indigo-500
  Concept: "#8B5CF6", // Violet-500
  Location: "#F59E0B", // Amber-500
  Document: "#059669", // Emerald-600
  Event: "#D97706", // Amber-600
  default: "#6B7280", // Slate-500
};

/**
 * Custom node component for the graph preview.
 * Includes keyboard accessibility support.
 */
function PreviewNode({
  data,
}: {
  data: { label: string; entityType?: string; onClick?: () => void };
}) {
  const color =
    entityColors[data.entityType || "default"] || entityColors.default;

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        data.onClick?.();
      }
    },
    [data]
  );

  const ariaLabel = data.entityType
    ? `Graph node: ${data.label}, type: ${data.entityType}`
    : `Graph node: ${data.label}`;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={ariaLabel}
      onKeyDown={handleKeyDown}
      className="px-3 py-2 rounded-lg shadow-sm border-2 text-white text-xs font-medium max-w-[120px] truncate cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      style={{ backgroundColor: color, borderColor: color }}
      title={data.label}
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
function transformNodes(
  nodes: GraphPreviewNode[],
  onNodeClick?: (node: GraphPreviewNode) => void
): Node[] {
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
        data: {
          label: nodes[0].label,
          entityType: nodes[0].type,
          onClick: onNodeClick ? () => onNodeClick(nodes[0]) : undefined,
        },
      },
    ];
  }

  // If no nodes, return empty array
  if (nodes.length === 0) {
    return [];
  }

  // Place first node in center, others in a circle around it
  return nodes.map((node, index) => {
    if (index === 0) {
      return {
        id: node.id,
        type: "preview",
        position: { x: centerX, y: centerY },
        data: {
          label: node.label,
          entityType: node.type,
          onClick: onNodeClick ? () => onNodeClick(node) : undefined,
        },
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
      data: {
        label: node.label,
        entityType: node.type,
        onClick: onNodeClick ? () => onNodeClick(node) : undefined,
      },
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
 *
 * Features:
 * - Compact React Flow visualization within a Card
 * - Circular layout with center node for focused entity
 * - Entity type color coding matching existing KnowledgeGraph component
 * - Animated edges for visual interest
 * - Minimal controls (no pan/drag in preview mode)
 * - Expand button for full-screen view
 * - Node/edge count indicator
 * - Keyboard accessible nodes
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
  const flowNodes = useMemo(
    () => transformNodes(apiNodes, onNodeClick),
    [apiNodes, onNodeClick]
  );
  const flowEdges = useMemo(() => transformEdges(apiEdges), [apiEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  // Sync state when props change (Issue 2 fix: React Flow State Sync Bug)
  useEffect(() => {
    setNodes(flowNodes);
  }, [flowNodes, setNodes]);

  useEffect(() => {
    setEdges(flowEdges);
  }, [flowEdges, setEdges]);

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
    <div
      className={cn(
        "border border-slate-200 rounded-lg bg-white overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-100">
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <span
            className="inline-block w-2 h-2 rounded-full bg-indigo-600"
            aria-hidden="true"
          />
          {title}
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">
            {apiNodes.length} nodes, {apiEdges.length} edges
          </span>
          {onExpand && (
            <button
              type="button"
              onClick={onExpand}
              className="p-1 rounded-md hover:bg-slate-100 transition-colors"
              aria-label="Expand graph to full view"
            >
              <Maximize2
                className="h-4 w-4 text-slate-500"
                aria-hidden="true"
              />
            </button>
          )}
        </div>
      </div>

      {/* Graph visualization */}
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
    </div>
  );
});

export default GraphPreview;
