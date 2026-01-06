/**
 * Workflow sidebar with node palette and configuration panel.
 * Story 20-H6: Implement Visual Workflow Editor
 */

'use client';

import { memo } from 'react';
import { nodePaletteItems, nodeTypeColors } from '../../types/workflow';
import type { WorkflowNodeType } from '../../types/workflow';

interface WorkflowSidebarProps {
  onDragStart: (event: React.DragEvent, nodeType: WorkflowNodeType) => void;
}

/**
 * Sidebar component with draggable node palette.
 */
function WorkflowSidebarComponent({ onDragStart }: WorkflowSidebarProps) {
  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="font-semibold text-gray-800">Pipeline Nodes</h2>
        <p className="text-xs text-gray-500 mt-1">
          Drag nodes to the canvas to build your RAG pipeline
        </p>
      </div>

      {/* Node palette */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {nodePaletteItems.map((item) => (
          <div
            key={item.type}
            draggable
            onDragStart={(e) => onDragStart(e, item.type)}
            className="
              flex items-center gap-3 p-3 rounded-lg
              border border-gray-200 bg-white
              cursor-grab active:cursor-grabbing
              hover:border-gray-300 hover:shadow-sm
              transition-all duration-150
            "
            style={{
              borderLeftWidth: '4px',
              borderLeftColor: nodeTypeColors[item.type],
            }}
          >
            <span className="text-xl" role="img" aria-label={item.type}>
              {item.icon}
            </span>
            <div className="flex-1 min-w-0">
              <div className="font-medium text-sm text-gray-800">
                {item.label}
              </div>
              <div className="text-xs text-gray-500 truncate">
                {item.description}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Help section */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <h3 className="font-medium text-sm text-gray-700 mb-2">Quick Tips</h3>
        <ul className="text-xs text-gray-500 space-y-1">
          <li>• Drag nodes from the palette</li>
          <li>• Connect nodes by dragging handles</li>
          <li>• Click a node to configure it</li>
          <li>• Use the toolbar to save/run</li>
        </ul>
      </div>
    </div>
  );
}

export const WorkflowSidebar = memo(WorkflowSidebarComponent);
export default WorkflowSidebar;
