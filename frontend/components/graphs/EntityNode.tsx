/**
 * Custom node component for entity visualization.
 * Story 4.4: Knowledge Graph Visualization
 */

'use client';

import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { EntityType } from '../../types/graphs';
import { entityColors } from '../../types/graphs';

interface EntityNodeData {
  label: string;
  entityType: EntityType;
  isOrphan: boolean;
  properties?: Record<string, unknown>;
}

/**
 * Custom node component for displaying entities in the knowledge graph.
 * Uses entity type-based coloring and orphan highlighting.
 */
function EntityNodeComponent({ data, selected }: NodeProps<EntityNodeData>) {
  const { label, entityType, isOrphan, properties } = data;
  
  // Get color based on entity type
  const backgroundColor = entityColors[entityType] || entityColors.Technology;
  
  // Orphan nodes get orange border
  const borderColor = isOrphan ? entityColors.orphan : backgroundColor;
  const ringStyle = isOrphan ? '0 0 0 3px rgba(249, 115, 22, 0.5)' : 'none';
  
  return (
    <div
      className="relative"
      style={{
        minWidth: 120,
        maxWidth: 200,
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: '#6B7280',
          width: 8,
          height: 8,
        }}
      />
      
      {/* Node content */}
      <div
        className="rounded-lg px-3 py-2 shadow-md transition-all"
        style={{
          backgroundColor,
          borderWidth: 2,
          borderStyle: 'solid',
          borderColor,
          boxShadow: selected
            ? '0 0 0 2px #3B82F6, 0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            : ringStyle,
        }}
      >
        {/* Entity type badge */}
        <div
          className="mb-1 text-xs font-medium uppercase tracking-wide opacity-80"
          style={{ color: 'white' }}
        >
          {entityType}
        </div>
        
        {/* Entity label */}
        <div
          className="text-sm font-semibold truncate"
          style={{ color: 'white' }}
          title={label}
        >
          {label}
        </div>
        
        {/* Orphan indicator */}
        {isOrphan && (
          <div
            className="mt-1 text-xs opacity-90"
            style={{ color: 'white' }}
          >
            No relationships
          </div>
        )}
      </div>
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: '#6B7280',
          width: 8,
          height: 8,
        }}
      />
    </div>
  );
}

// Memoize for performance with large graphs
export const EntityNode = memo(EntityNodeComponent);
export default EntityNode;
