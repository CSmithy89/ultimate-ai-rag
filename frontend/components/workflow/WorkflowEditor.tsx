/**
 * Main visual workflow editor component.
 * Story 20-H6: Implement Visual Workflow Editor
 */

'use client';

import { useCallback, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
  type ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { WorkflowNode } from './WorkflowNode';
import { WorkflowSidebar } from './WorkflowSidebar';
import { WorkflowToolbar } from './WorkflowToolbar';
import { useWorkflowStore } from './hooks/use-workflow-store';
import type { WorkflowNodeType, WorkflowNodeData } from '../../types/workflow';
import { nodeTypeColors } from '../../types/workflow';

// Register custom node types
const nodeTypes: NodeTypes = {
  workflowNode: WorkflowNode,
};

interface WorkflowEditorProps {
  enabled?: boolean;
}

/**
 * Main workflow editor component with drag-and-drop canvas.
 */
export function WorkflowEditor({ enabled = true }: WorkflowEditorProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null);

  const {
    nodes,
    edges,
    workflowName,
    execution,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    setWorkflowName,
    clearWorkflow,
    saveWorkflow,
    loadWorkflow,
    getSavedWorkflows,
    executeWorkflow,
  } = useWorkflowStore();

  /**
   * Handle drag start from sidebar.
   */
  const onDragStart = useCallback(
    (event: React.DragEvent, nodeType: WorkflowNodeType) => {
      event.dataTransfer.setData('application/reactflow', nodeType);
      event.dataTransfer.effectAllowed = 'move';
    },
    []
  );

  /**
   * Handle drag over canvas.
   */
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  /**
   * Valid node types for runtime validation.
   */
  const validNodeTypes: WorkflowNodeType[] = [
    'ingest', 'chunk', 'embed', 'extract', 'index', 'retrieve', 'rerank', 'respond'
  ];

  /**
   * Handle drop on canvas.
   */
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const nodeTypeRaw = event.dataTransfer.getData('application/reactflow');
      // Runtime validation to ensure type safety
      if (!nodeTypeRaw || !validNodeTypes.includes(nodeTypeRaw as WorkflowNodeType)) {
        return;
      }
      const nodeType = nodeTypeRaw as WorkflowNodeType;

      if (!reactFlowWrapper.current || !reactFlowInstance.current) {
        return;
      }

      // Get drop position
      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      addNode(nodeType, position);
    },
    [addNode]
  );

  /**
   * Store React Flow instance on init.
   */
  const onInit = useCallback((instance: ReactFlowInstance) => {
    reactFlowInstance.current = instance;
  }, []);

  /**
   * MiniMap node color function.
   */
  const minimapNodeColor = useCallback((node: { data?: WorkflowNodeData }) => {
    const nodeType = node.data?.nodeType;
    return nodeType ? nodeTypeColors[nodeType] : '#6B7280';
  }, []);

  // Show disabled message if feature is off
  if (!enabled) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-center p-8">
          <div className="text-6xl mb-4">🔒</div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">
            Visual Workflow Editor Disabled
          </h2>
          <p className="text-gray-500 max-w-md">
            The visual workflow editor is currently disabled.
            Set <code className="bg-gray-200 px-1 rounded">NEXT_PUBLIC_VISUAL_WORKFLOW_ENABLED=true</code> in your environment to enable this feature.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Toolbar */}
      <WorkflowToolbar
        workflowName={workflowName}
        onNameChange={setWorkflowName}
        onSave={saveWorkflow}
        onLoad={loadWorkflow}
        onClear={clearWorkflow}
        onRun={executeWorkflow}
        getSavedWorkflows={getSavedWorkflows}
        isRunning={execution?.status === 'running'}
      />

      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <WorkflowSidebar onDragStart={onDragStart} />

        {/* Canvas */}
        <div ref={reactFlowWrapper} className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={onInit}
            onDragOver={onDragOver}
            onDrop={onDrop}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{
              padding: 0.2,
              maxZoom: 1.5,
            }}
            defaultEdgeOptions={{
              type: 'smoothstep',
              style: { strokeWidth: 2, stroke: '#9CA3AF' },
            }}
            connectionLineStyle={{ strokeWidth: 2, stroke: '#3B82F6' }}
            attributionPosition="bottom-right"
          >
            <Controls />
            <MiniMap
              nodeColor={minimapNodeColor}
              nodeStrokeWidth={3}
              zoomable
              pannable
            />
            <Background color="#E5E7EB" gap={20} />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}

export default WorkflowEditor;
