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
  return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
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
   * Validate edge connection between nodes.
   * Returns true if connection is valid.
   */
  const isValidConnection = useCallback(
    (source: string, target: string): boolean => {
      const sourceNode = nodes.find((n) => n.id === source);
      const targetNode = nodes.find((n) => n.id === target);

      if (!sourceNode || !targetNode) return false;

      const sourceType = sourceNode.data?.nodeType;
      const targetType = targetNode.data?.nodeType;

      // Respond nodes cannot have outgoing connections
      if (sourceType === 'respond') return false;

      // Ingest nodes cannot have incoming connections
      if (targetType === 'ingest') return false;

      // Prevent self-connections
      if (source === target) return false;

      // Prevent duplicate edges
      const isDuplicate = edges.some(
        (e) => e.source === source && e.target === target
      );
      if (isDuplicate) return false;

      return true;
    },
    [nodes, edges]
  );

  /**
   * Handle new edge connections.
   */
  const onConnect = useCallback(
    (connection: Connection) => {
      // Validate connection
      if (!connection.source || !connection.target) return;
      if (!isValidConnection(connection.source, connection.target)) return;

      // Add the edge
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
    [setEdges, isValidConnection]
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
   * Returns false if workflow is empty or save fails.
   */
  const saveWorkflow = useCallback(() => {
    // Validate: workflow must have at least one node
    if (nodes.length === 0) {
      console.warn('Cannot save empty workflow');
      return false;
    }

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
   * Validate workflow data structure.
   * Returns true if workflow has valid shape, false otherwise.
   */
  const isValidWorkflow = useCallback((data: unknown): data is Workflow => {
    if (!data || typeof data !== 'object') return false;
    const w = data as Record<string, unknown>;

    // Required string fields
    if (typeof w.id !== 'string' || !w.id) return false;
    if (typeof w.name !== 'string') return false;

    // Nodes must be array
    if (!Array.isArray(w.nodes)) return false;
    for (const node of w.nodes) {
      if (!node || typeof node !== 'object') return false;
      if (typeof node.id !== 'string') return false;
      if (!node.data || typeof node.data !== 'object') return false;
      if (typeof node.position !== 'object') return false;
    }

    // Edges must be array
    if (!Array.isArray(w.edges)) return false;
    for (const edge of w.edges) {
      if (!edge || typeof edge !== 'object') return false;
      if (typeof edge.source !== 'string') return false;
      if (typeof edge.target !== 'string') return false;
    }

    return true;
  }, []);

  /**
   * Load workflow from localStorage.
   * Validates data structure before loading to prevent crashes.
   */
  const loadWorkflow = useCallback(
    (id: string) => {
      try {
        const existing = localStorage.getItem(STORAGE_KEY);
        if (!existing) return false;

        const parsed = JSON.parse(existing);
        if (!Array.isArray(parsed)) {
          console.error('Invalid workflow storage format');
          return false;
        }

        const workflow = parsed.find((w: unknown) => {
          return w && typeof w === 'object' && (w as Record<string, unknown>).id === id;
        });

        if (!workflow) return false;

        // Validate workflow structure before loading
        if (!isValidWorkflow(workflow)) {
          console.error('Workflow data is corrupted or incompatible:', id);
          return false;
        }

        setWorkflowId(workflow.id);
        setWorkflowName(workflow.name);
        setNodes(workflow.nodes);
        setEdges(workflow.edges);
        return true;
      } catch (error) {
        console.error('Failed to load workflow:', error);
        return false;
      }
    },
    [setNodes, setEdges, isValidWorkflow]
  );

  /**
   * Get list of saved workflows.
   * Filters out any corrupted/invalid workflows.
   */
  const getSavedWorkflows = useCallback((): Workflow[] => {
    try {
      const existing = localStorage.getItem(STORAGE_KEY);
      if (!existing) return [];

      const parsed = JSON.parse(existing);
      if (!Array.isArray(parsed)) return [];

      // Filter to only valid workflows
      return parsed.filter((w: unknown) => isValidWorkflow(w)) as Workflow[];
    } catch {
      return [];
    }
  }, [isValidWorkflow]);

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
   * Compute topological order of nodes using Kahn's algorithm.
   * Returns nodes sorted by dependency order, or null if cycle detected.
   */
  const getTopologicalOrder = useCallback((): Node<WorkflowNodeData>[] | null => {
    // Build adjacency list and in-degree count
    const inDegree = new Map<string, number>();
    const adjacency = new Map<string, string[]>();

    // Initialize
    nodes.forEach((node) => {
      inDegree.set(node.id, 0);
      adjacency.set(node.id, []);
    });

    // Build graph from edges
    edges.forEach((edge) => {
      const targets = adjacency.get(edge.source);
      if (targets) {
        targets.push(edge.target);
      }
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    });

    // Find nodes with no incoming edges (start nodes)
    const queue: string[] = [];
    inDegree.forEach((degree, nodeId) => {
      if (degree === 0) queue.push(nodeId);
    });

    // Process nodes in topological order
    const result: Node<WorkflowNodeData>[] = [];
    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      const node = nodes.find((n) => n.id === nodeId);
      if (node) result.push(node);

      // Reduce in-degree for neighbors
      adjacency.get(nodeId)?.forEach((neighbor) => {
        const newDegree = (inDegree.get(neighbor) || 1) - 1;
        inDegree.set(neighbor, newDegree);
        if (newDegree === 0) queue.push(neighbor);
      });
    }

    // If not all nodes processed, there's a cycle
    if (result.length !== nodes.length) {
      console.warn('[Workflow Debug] Cycle detected in workflow graph');
      return null;
    }

    return result;
  }, [nodes, edges]);

  /**
   * Execute workflow with debug information.
   * Processes nodes in topological order based on edge connections.
   * Simulates pipeline execution (placeholder for real backend integration).
   */
  const executeWorkflow = useCallback(async () => {
    // Validate: workflow must have at least one node
    if (nodes.length === 0) {
      console.warn('Cannot execute empty workflow');
      return;
    }

    console.log(`[Workflow Debug] Starting execution: ${workflowName} (${workflowId})`);
    console.log(`[Workflow Debug] Nodes: ${nodes.length}, Edges: ${edges.length}`);

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

    // Get nodes in topological order (respects edge dependencies)
    const nodeOrder = getTopologicalOrder();
    if (!nodeOrder) {
      console.error('[Workflow Debug] Cannot execute: workflow contains a cycle');
      setExecution((prev) =>
        prev ? { ...prev, status: 'failed', completedAt: new Date().toISOString() } : null
      );
      return;
    }

    console.log(`[Workflow Debug] Execution order: ${nodeOrder.map((n) => (n.data as WorkflowNodeData).label).join(' → ')}`);

    for (let i = 0; i < nodeOrder.length; i++) {
      const node = nodeOrder[i];
      const nodeData = node.data as WorkflowNodeData;

      console.log(`[Workflow Debug] Step ${i + 1}/${nodeOrder.length}: ${nodeData.label} (${nodeData.nodeType})`);
      console.log(`[Workflow Debug] Config:`, nodeData.config);

      updateNodeStatus(node.id, 'running');

      // Simulate processing delay
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Simulate success (in real impl, would call backend)
      console.log(`[Workflow Debug] Completed: ${nodeData.label}`);
      updateNodeStatus(node.id, 'success');
    }

    console.log(`[Workflow Debug] Workflow execution completed successfully`);

    setExecution((prev) =>
      prev
        ? {
            ...prev,
            status: 'completed',
            completedAt: new Date().toISOString(),
          }
        : null
    );
  }, [workflowId, workflowName, nodes, edges, setNodes, updateNodeStatus, getTopologicalOrder]);

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
