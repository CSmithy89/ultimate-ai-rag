/**
 * Custom edge component for relationship visualization.
 * Story 4.4: Knowledge Graph Visualization
 */

'use client';

import { memo } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from 'reactflow';
import type { RelationshipType } from '../../types/graphs';

interface RelationshipEdgeData {
  relationshipType: RelationshipType;
  properties?: Record<string, unknown>;
}

// Color scheme for relationship types
const relationshipColors: Record<RelationshipType, string> = {
  MENTIONS: '#6B7280',      // Gray
  AUTHORED_BY: '#3B82F6',   // Blue
  PART_OF: '#10B981',       // Emerald
  USES: '#6366F1',          // Indigo
  RELATED_TO: '#8B5CF6',    // Violet
};

/**
 * Custom edge component for displaying relationships in the knowledge graph.
 * Shows relationship type label and uses color coding.
 */
function RelationshipEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  data,
  selected,
}: EdgeProps<RelationshipEdgeData>) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const relationshipType = data?.relationshipType || 'RELATED_TO';
  const edgeColor = relationshipColors[relationshipType] || relationshipColors.RELATED_TO;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: edgeColor,
          strokeWidth: selected ? 3 : 2,
          opacity: selected ? 1 : 0.7,
        }}
        markerEnd="url(#arrow)"
      />
      
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan pointer-events-auto absolute transform -translate-x-1/2 -translate-y-1/2"
          style={{
            left: labelX,
            top: labelY,
          }}
        >
          <div
            className="rounded px-2 py-0.5 text-xs font-medium shadow-sm"
            style={{
              backgroundColor: edgeColor,
              color: 'white',
              opacity: selected ? 1 : 0.9,
            }}
          >
            {label || relationshipType}
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

// Memoize for performance with large graphs
export const RelationshipEdge = memo(RelationshipEdgeComponent);
export default RelationshipEdge;
