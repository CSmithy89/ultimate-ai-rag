/**
 * Workflow state management hook.
 * Story 20-H6: Implement Visual Workflow Editor
 */

import { useCallback, useState } from 'react';
import {
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
} from 'reactflow';
import type {
  WorkflowNode,
  WorkflowEdge,
  WorkflowNodeData,
  Workflow,
  WorkflowExecution,
  WorkflowNodeType,
  NodeStatus,
} from '../../../types/workflow';
import { defaultNodeConfigs } from '../../../types/workflow';

const STORAGE_KEY = 'rag-workflow-drafts';

/**
 * Generate a unique ID for nodes and edges.
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Create a new workflow node.
 */
export function createWorkflowNode(
  type: WorkflowNodeType,
  position: { x: number; y: number }
): WorkflowNode {
  const nodeLabels: Record<WorkflowNodeType, string> = {
    ingest: 'Ingest',
    chunk: 'Chunk',
    embed: 'Embed',
    extract: 'Extract Entities',
    index: 'Index',
    retrieve: 'Retrieve',
    rerank: 'Rerank',
    respond: 'Respond',
  };

  return {
    id: generateId(),
    type: 'workflowNode',
    position,
    data: {
      label: nodeLabels[type],
      nodeType: type,
      status: 'idle' as NodeStatus,
      config: { ...defaultNodeConfigs[type] },
    },
  };
}

/**
 * Hook for managing workflow state.
 */
export function useWorkflowStore() {
  const [nodes, setNodes, onNodesChange] = useNodesState<WorkflowNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [workflowId, setWorkflowId] = useState<string>(generateId());
  const [workflowName, setWorkflowName] = useState<string>('Untitled Workflow');
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);

  /**
   * Handle new edge connections.
   */
  const onConnect = useCallback(
    (connection: Connection) => {
      // Validate connection (basic validation)
      if (!connection.source || !connection.target) return;

      // Add the edge with animation
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            type: 'smoothstep',
            animated: false,
            style: { strokeWidth: 2 },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  /**
   * Add a new node to the workflow.
   */
  const addNode = useCallback(
    (type: WorkflowNodeType, position: { x: number; y: number }) => {
      const newNode = createWorkflowNode(type, position);
      setNodes((nds) => [...nds, newNode]);
      return newNode;
    },
    [setNodes]
  );

  /**
   * Update a node's data.
   */
  const updateNodeData = useCallback(
    (nodeId: string, data: Partial<WorkflowNodeData>) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, ...data } }
            : node
        )
      );
    },
    [setNodes]
  );

  /**
   * Update a node's status.
   */
  const updateNodeStatus = useCallback(
    (nodeId: string, status: NodeStatus, error?: string) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, status, error } }
            : node
        )
      );
    },
    [setNodes]
  );

  /**
   * Remove a node and its connected edges.
   */
  const removeNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((node) => node.id !== nodeId));
      setEdges((eds) =>
        eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId)
      );
    },
    [setNodes, setEdges]
  );

  /**
   * Clear the workflow.
   */
  const clearWorkflow = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setWorkflowId(generateId());
    setWorkflowName('Untitled Workflow');
    setExecution(null);
  }, [setNodes, setEdges]);

  /**
   * Save workflow to localStorage.
   */
  const saveWorkflow = useCallback(() => {
    const workflow: Workflow = {
      id: workflowId,
      name: workflowName,
      nodes: nodes as WorkflowNode[],
      edges: edges as WorkflowEdge[],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    try {
      const existing = localStorage.getItem(STORAGE_KEY);
      const workflows: Workflow[] = existing ? JSON.parse(existing) : [];

      // Update existing or add new
      const index = workflows.findIndex((w) => w.id === workflowId);
      if (index >= 0) {
        workflows[index] = workflow;
      } else {
        workflows.push(workflow);
      }

      localStorage.setItem(STORAGE_KEY, JSON.stringify(workflows));
      return true;
    } catch (error) {
      console.error('Failed to save workflow:', error);
      return false;
    }
  }, [workflowId, workflowName, nodes, edges]);

  /**
   * Load workflow from localStorage.
   */
  const loadWorkflow = useCallback(
    (id: string) => {
      try {
        const existing = localStorage.getItem(STORAGE_KEY);
        if (!existing) return false;

        const workflows: Workflow[] = JSON.parse(existing);
        const workflow = workflows.find((w) => w.id === id);

        if (workflow) {
          setWorkflowId(workflow.id);
          setWorkflowName(workflow.name);
          setNodes(workflow.nodes);
          setEdges(workflow.edges);
          return true;
        }
        return false;
      } catch (error) {
        console.error('Failed to load workflow:', error);
        return false;
      }
    },
    [setNodes, setEdges]
  );

  /**
   * Get list of saved workflows.
   */
  const getSavedWorkflows = useCallback((): Workflow[] => {
    try {
      const existing = localStorage.getItem(STORAGE_KEY);
      return existing ? JSON.parse(existing) : [];
    } catch {
      return [];
    }
  }, []);

  /**
   * Delete a saved workflow.
   */
  const deleteWorkflow = useCallback((id: string) => {
    try {
      const existing = localStorage.getItem(STORAGE_KEY);
      if (!existing) return false;

      const workflows: Workflow[] = JSON.parse(existing);
      const filtered = workflows.filter((w) => w.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
      return true;
    } catch {
      return false;
    }
  }, []);

  /**
   * Simulate workflow execution (placeholder for real implementation).
   */
  const executeWorkflow = useCallback(async () => {
    // Reset all nodes to idle
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: { ...node.data, status: 'idle' as NodeStatus, error: undefined },
      }))
    );

    setExecution({
      workflowId,
      status: 'running',
      startedAt: new Date().toISOString(),
      results: {},
      errors: {},
    });

    // Get nodes in topological order (simplified: just process in order)
    const nodeOrder = [...nodes];

    for (const node of nodeOrder) {
      updateNodeStatus(node.id, 'running');

      // Simulate processing delay
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Simulate success (in real impl, would call backend)
      updateNodeStatus(node.id, 'success');
    }

    setExecution((prev) =>
      prev
        ? {
            ...prev,
            status: 'completed',
            completedAt: new Date().toISOString(),
          }
        : null
    );
  }, [workflowId, nodes, setNodes, updateNodeStatus]);

  return {
    // State
    nodes,
    edges,
    workflowId,
    workflowName,
    execution,

    // Node/edge change handlers
    onNodesChange,
    onEdgesChange,
    onConnect,

    // Node operations
    addNode,
    updateNodeData,
    updateNodeStatus,
    removeNode,

    // Workflow operations
    setWorkflowName,
    clearWorkflow,
    saveWorkflow,
    loadWorkflow,
    getSavedWorkflows,
    deleteWorkflow,
    executeWorkflow,
  };
}

export default useWorkflowStore;
