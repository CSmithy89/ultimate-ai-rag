/**
 * Base workflow node component for the visual workflow editor.
 * Story 20-H6: Implement Visual Workflow Editor
 */

'use client';

import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { WorkflowNodeData } from '../../types/workflow';
import { nodeTypeColors, statusColors } from '../../types/workflow';

/**
 * Base workflow node component.
 * Renders a pipeline stage with input/output handles and status indicator.
 */
function WorkflowNodeComponent({
  data,
  selected,
}: NodeProps<WorkflowNodeData>) {
  const { label, nodeType, status, description, error } = data;
  const nodeColor = nodeTypeColors[nodeType];
  const statusColor = statusColors[status];

  // Get icon for node type
  const getIcon = () => {
    switch (nodeType) {
      case 'ingest': return '📥';
      case 'chunk': return '✂️';
      case 'embed': return '🔢';
      case 'extract': return '🏷️';
      case 'index': return '💾';
      case 'retrieve': return '🔍';
      case 'rerank': return '📊';
      case 'respond': return '💬';
      default: return '⚙️';
    }
  };

  return (
    <div
      className={`
        relative px-4 py-3 rounded-lg shadow-md
        border-2 transition-all duration-200
        min-w-[160px] max-w-[200px]
        ${selected ? 'shadow-lg scale-105' : ''}
      `}
      style={{
        backgroundColor: 'white',
        borderColor: selected ? nodeColor : '#E5E7EB',
      }}
    >
      {/* Input handle (top) - not for ingest nodes */}
      {nodeType !== 'ingest' && (
        <Handle
          type="target"
          position={Position.Top}
          className="w-3 h-3 rounded-full border-2 border-white"
          style={{ backgroundColor: nodeColor }}
        />
      )}

      {/* Status indicator */}
      <div
        className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-white"
        style={{ backgroundColor: statusColor }}
        title={status}
      />

      {/* Node content */}
      <div className="flex items-center gap-2">
        <span className="text-xl" role="img" aria-label={nodeType}>
          {getIcon()}
        </span>
        <div className="flex-1 min-w-0">
          <div
            className="font-semibold text-sm truncate"
            style={{ color: nodeColor }}
          >
            {label}
          </div>
          {description && (
            <div className="text-xs text-gray-500 truncate">
              {description}
            </div>
          )}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="mt-2 text-xs text-red-500 truncate" title={error}>
          {error}
        </div>
      )}

      {/* Output handle (bottom) - not for respond nodes */}
      {nodeType !== 'respond' && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="w-3 h-3 rounded-full border-2 border-white"
          style={{ backgroundColor: nodeColor }}
        />
      )}
    </div>
  );
}

export const WorkflowNode = memo(WorkflowNodeComponent);
export default WorkflowNode;
