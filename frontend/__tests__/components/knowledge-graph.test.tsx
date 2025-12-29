/**
 * Tests for Knowledge Graph visualization components.
 * Story 4.4: Knowledge Graph Visualization
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import { ReactFlowProvider } from 'reactflow';
import { KnowledgeGraph } from '../../components/graphs/KnowledgeGraph';
import { EntityNode } from '../../components/graphs/EntityNode';
import { RelationshipEdge } from '../../components/graphs/RelationshipEdge';
import type { GraphData, EntityType, RelationshipType } from '../../types/graphs';

// Mock ReactFlow internals
jest.mock('reactflow', () => {
  const originalModule = jest.requireActual('reactflow');
  return {
    __esModule: true,
    ...originalModule,
    default: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="react-flow">{children}</div>
    ),
    ReactFlow: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="react-flow">{children}</div>
    ),
    Background: () => <div data-testid="background" />,
    Controls: () => <div data-testid="controls" />,
    MiniMap: () => <div data-testid="minimap" />,
    useNodesState: (initialNodes: unknown[]) => [initialNodes, jest.fn(), jest.fn()],
    useEdgesState: (initialEdges: unknown[]) => [initialEdges, jest.fn(), jest.fn()],
    Handle: ({ type, position }: { type: string; position: string }) => (
      <div data-testid={`handle-${type}-${position}`} />
    ),
    Position: { Top: 'top', Bottom: 'bottom', Left: 'left', Right: 'right' },
    MarkerType: { ArrowClosed: 'arrowclosed' },
    BaseEdge: ({ id }: { id: string }) => (
      <svg data-testid={`edge-${id}`}>
        <path />
      </svg>
    ),
    EdgeLabelRenderer: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="edge-label-renderer">{children}</div>
    ),
    getBezierPath: () => ['M0,0', 100, 100],
  };
});

// Mock data for tests
const mockGraphData: GraphData = {
  nodes: [
    {
      id: 'node-1',
      label: 'Test Entity 1',
      type: 'Person',
      is_orphan: false,
      properties: { description: 'A test person' },
    },
    {
      id: 'node-2',
      label: 'Test Entity 2',
      type: 'Organization',
      is_orphan: true,
      properties: { description: 'A test org' },
    },
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      type: 'RELATED_TO',
      label: 'RELATED_TO',
      properties: { confidence: 0.9 },
    },
  ],
};

describe('KnowledgeGraph', () => {
  it('renders with mock data', () => {
    render(
      <ReactFlowProvider>
        <KnowledgeGraph data={mockGraphData} />
      </ReactFlowProvider>
    );

    expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    expect(screen.getByTestId('controls')).toBeInTheDocument();
    expect(screen.getByTestId('minimap')).toBeInTheDocument();
    expect(screen.getByTestId('background')).toBeInTheDocument();
  });

  it('renders with empty data', () => {
    const emptyData: GraphData = { nodes: [], edges: [] };

    render(
      <ReactFlowProvider>
        <KnowledgeGraph data={emptyData} />
      </ReactFlowProvider>
    );

    expect(screen.getByTestId('react-flow')).toBeInTheDocument();
  });

  it('applies filters correctly', () => {
    const filters = {
      entityTypes: ['Person' as EntityType],
      relationshipTypes: [],
      showOrphansOnly: false,
      searchQuery: '',
    };

    render(
      <ReactFlowProvider>
        <KnowledgeGraph data={mockGraphData} filters={filters} />
      </ReactFlowProvider>
    );

    expect(screen.getByTestId('react-flow')).toBeInTheDocument();
  });
});

describe('EntityNode', () => {
  const mockNodeProps = {
    id: 'node-1',
    type: 'entity',
    data: {
      label: 'Test Person',
      entityType: 'Person' as EntityType,
      isOrphan: false,
      properties: { description: 'A test person' },
    },
    selected: false,
    dragging: false,
    xPos: 0,
    yPos: 0,
    isConnectable: true,
    zIndex: 0,
  };

  it('displays label and type', () => {
    render(<EntityNode {...mockNodeProps} />);

    expect(screen.getByText('Test Person')).toBeInTheDocument();
    expect(screen.getByText('Person')).toBeInTheDocument();
  });

  it('shows orphan indicator when isOrphan is true', () => {
    const orphanProps = {
      ...mockNodeProps,
      data: {
        ...mockNodeProps.data,
        isOrphan: true,
      },
    };

    render(<EntityNode {...orphanProps} />);

    expect(screen.getByText('No relationships')).toBeInTheDocument();
  });

  it('renders handles for connections', () => {
    render(<EntityNode {...mockNodeProps} />);

    expect(screen.getByTestId('handle-target-top')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source-bottom')).toBeInTheDocument();
  });
});

describe('RelationshipEdge', () => {
  const mockEdgeProps = {
    id: 'edge-1',
    source: 'node-1',
    target: 'node-2',
    sourceX: 0,
    sourceY: 0,
    targetX: 100,
    targetY: 100,
    sourcePosition: 'bottom' as const,
    targetPosition: 'top' as const,
    label: 'RELATED_TO',
    data: {
      relationshipType: 'RELATED_TO' as RelationshipType,
      properties: { confidence: 0.9 },
    },
    selected: false,
    animated: false,
    interactionWidth: 20,
  };

  it('displays relationship label', () => {
    render(<RelationshipEdge {...mockEdgeProps} />);

    expect(screen.getByText('RELATED_TO')).toBeInTheDocument();
  });

  it('renders edge path', () => {
    render(<RelationshipEdge {...mockEdgeProps} />);

    expect(screen.getByTestId('edge-edge-1')).toBeInTheDocument();
  });

  it('renders edge label renderer', () => {
    render(<RelationshipEdge {...mockEdgeProps} />);

    expect(screen.getByTestId('edge-label-renderer')).toBeInTheDocument();
  });
});
