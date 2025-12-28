/**
 * Main Knowledge Graph visualization component using React Flow.
 * Story 4.4: Knowledge Graph Visualization
 */

'use client';

import { useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
  type EdgeTypes,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { EntityNode } from './EntityNode';
import { RelationshipEdge } from './RelationshipEdge';
import type {
  GraphData,
  GraphNode,
  GraphEdge,
  EntityType,
  RelationshipType,
  GraphFilterState,
} from '../../types/graphs';
import { entityColors } from '../../types/graphs';

// Register custom node and edge types
const nodeTypes: NodeTypes = {
  entity: EntityNode,
};

const edgeTypes: EdgeTypes = {
  relationship: RelationshipEdge,
};

// Default edge options
const defaultEdgeOptions = {
  type: 'relationship',
  markerEnd: {
    type: MarkerType.ArrowClosed,
    width: 20,
    height: 20,
  },
};

interface KnowledgeGraphProps {
  data: GraphData;
  filters?: GraphFilterState;
  onNodeClick?: (node: GraphNode) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
}

/**
 * Transform API graph nodes to React Flow nodes.
 */
function transformNodes(nodes: GraphNode[]): Node[] {
  // Simple grid layout for initial positioning
  const cols = Math.ceil(Math.sqrt(nodes.length));
  const spacing = 250;

  return nodes.map((node, index) => ({
    id: node.id,
    type: 'entity',
    position: {
      x: (index % cols) * spacing + Math.random() * 50,
      y: Math.floor(index / cols) * spacing + Math.random() * 50,
    },
    data: {
      label: node.label,
      entityType: node.type as EntityType,
      isOrphan: node.is_orphan,
      properties: node.properties,
    },
  }));
}

/**
 * Transform API graph edges to React Flow edges.
 */
function transformEdges(edges: GraphEdge[]): Edge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: 'relationship',
    label: edge.label,
    data: {
      relationshipType: edge.type as RelationshipType,
      properties: edge.properties,
    },
  }));
}

/**
 * Apply filters to graph data.
 */
function applyFilters(
  data: GraphData,
  filters?: GraphFilterState
): GraphData {
  if (!filters) return data;

  let filteredNodes = [...data.nodes];
  let filteredEdges = [...data.edges];

  // Filter by entity type
  if (filters.entityTypes.length > 0) {
    filteredNodes = filteredNodes.filter((node) =>
      filters.entityTypes.includes(node.type as EntityType)
    );
  }

  // Filter by orphan status
  if (filters.showOrphansOnly) {
    filteredNodes = filteredNodes.filter((node) => node.is_orphan);
  }

  // Filter by search query
  if (filters.searchQuery) {
    const query = filters.searchQuery.toLowerCase();
    filteredNodes = filteredNodes.filter((node) =>
      node.label.toLowerCase().includes(query)
    );
  }

  // Keep only edges between visible nodes
  const visibleNodeIds = new Set(filteredNodes.map((n) => n.id));
  filteredEdges = filteredEdges.filter(
    (edge) =>
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
  );

  // Filter by relationship type
  if (filters.relationshipTypes.length > 0) {
    filteredEdges = filteredEdges.filter((edge) =>
      filters.relationshipTypes.includes(edge.type as RelationshipType)
    );
  }

  return { nodes: filteredNodes, edges: filteredEdges };
}

/**
 * Main Knowledge Graph component.
 * Displays entities as nodes and relationships as edges using React Flow.
 */
export function KnowledgeGraph({
  data,
  filters,
  onNodeClick,
  onEdgeClick,
}: KnowledgeGraphProps) {
  // Apply filters to data
  const filteredData = useMemo(
    () => applyFilters(data, filters),
    [data, filters]
  );

  // Transform data to React Flow format
  const initialNodes = useMemo(
    () => transformNodes(filteredData.nodes),
    [filteredData.nodes]
  );
  const initialEdges = useMemo(
    () => transformEdges(filteredData.edges),
    [filteredData.edges]
  );

  // React Flow state hooks
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes and edges when data changes
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  // Handle node click
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        const graphNode = filteredData.nodes.find((n) => n.id === node.id);
        if (graphNode) {
          onNodeClick(graphNode);
        }
      }
    },
    [filteredData.nodes, onNodeClick]
  );

  // Handle edge click
  const handleEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      if (onEdgeClick) {
        const graphEdge = filteredData.edges.find((e) => e.id === edge.id);
        if (graphEdge) {
          onEdgeClick(graphEdge);
        }
      }
    },
    [filteredData.edges, onEdgeClick]
  );

  // MiniMap node color function
  const minimapNodeColor = useCallback((node: Node) => {
    const entityType = node.data?.entityType as EntityType;
    return entityColors[entityType] || '#6B7280';
  }, []);

  return (
    <div className="w-full h-full" style={{ minHeight: 500 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        fitViewOptions={{
          padding: 0.2,
          maxZoom: 1.5,
        }}
        attributionPosition="bottom-left"
      >
        <Controls />
        <MiniMap
          nodeColor={minimapNodeColor}
          nodeStrokeWidth={3}
          zoomable
          pannable
        />
        <Background color="#E5E7EB" gap={16} />
        
        {/* Arrow marker definition */}
        <svg>
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path
                d="M 0 0 L 10 5 L 0 10 z"
                fill="#6B7280"
              />
            </marker>
          </defs>
        </svg>
      </ReactFlow>
    </div>
  );
}

export default KnowledgeGraph;
