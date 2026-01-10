/**
 * TypeScript types for Visual Workflow Editor (Story 20-H6).
 */

import type { Node, Edge } from 'reactflow';

/**
 * Types of workflow nodes representing RAG pipeline stages.
 */
export type WorkflowNodeType =
  | 'ingest'
  | 'chunk'
  | 'embed'
  | 'extract'
  | 'index'
  | 'retrieve'
  | 'rerank'
  | 'respond';

/**
 * Status of a workflow node during execution.
 */
export type NodeStatus = 'idle' | 'running' | 'success' | 'error' | 'skipped';

/**
 * Configuration for the ingest node.
 */
export interface IngestConfig {
  source: 'file' | 'url' | 's3' | 'confluence' | 'notion';
  batchSize?: number;
  filters?: string[];
}

/**
 * Configuration for the chunk node.
 */
export interface ChunkConfig {
  strategy: 'fixed' | 'semantic' | 'hierarchical';
  chunkSize: number;
  overlap: number;
  levels?: number[];
}

/**
 * Configuration for the embed node.
 */
export interface EmbedConfig {
  model: string;
  batchSize: number;
  dimensions?: number;
}

/**
 * Configuration for the extract node (entity extraction).
 */
export interface ExtractConfig {
  entityTypes: string[];
  model: string;
  confidenceThreshold: number;
}

/**
 * Configuration for the index node.
 */
export interface IndexConfig {
  target: 'pgvector' | 'neo4j' | 'hybrid';
  indexName?: string;
}

/**
 * Configuration for the retrieve node.
 */
export interface RetrieveConfig {
  method: 'vector' | 'graph' | 'hybrid' | 'bm42';
  topK: number;
  scoreThreshold?: number;
}

/**
 * Configuration for the rerank node.
 */
export interface RerankConfig {
  provider: 'cohere' | 'flashrank' | 'colbert' | 'graph';
  model?: string;
  topK: number;
}

/**
 * Configuration for the respond node.
 */
export interface RespondConfig {
  model: string;
  temperature: number;
  maxTokens?: number;
  systemPrompt?: string;
}

/**
 * Union of all node configurations.
 */
export type WorkflowNodeConfig =
  | IngestConfig
  | ChunkConfig
  | EmbedConfig
  | ExtractConfig
  | IndexConfig
  | RetrieveConfig
  | RerankConfig
  | RespondConfig;

/**
 * Data stored in a workflow node.
 */
export interface WorkflowNodeData {
  label: string;
  nodeType: WorkflowNodeType;
  status: NodeStatus;
  config: WorkflowNodeConfig;
  description?: string;
  error?: string;
  output?: unknown;
}

/**
 * A workflow node extending React Flow's Node type.
 */
export type WorkflowNode = Node<WorkflowNodeData>;

/**
 * Data stored in a workflow edge.
 */
export interface WorkflowEdgeData {
  animated?: boolean;
  label?: string;
}

/**
 * A workflow edge extending React Flow's Edge type.
 */
export type WorkflowEdge = Edge<WorkflowEdgeData>;

/**
 * Complete workflow definition.
 */
export interface Workflow {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  createdAt: string;
  updatedAt: string;
}

/**
 * Workflow execution state.
 */
export interface WorkflowExecution {
  workflowId: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  currentNodeId?: string;
  startedAt?: string;
  completedAt?: string;
  results: Record<string, unknown>;
  errors: Record<string, string>;
}

/**
 * Node palette item for the sidebar.
 */
export interface NodePaletteItem {
  type: WorkflowNodeType;
  label: string;
  description: string;
  icon: string;
  defaultConfig: WorkflowNodeConfig;
}

/**
 * Default configurations for each node type.
 */
export const defaultNodeConfigs: Record<WorkflowNodeType, WorkflowNodeConfig> = {
  ingest: {
    source: 'file',
    batchSize: 10,
  } as IngestConfig,
  chunk: {
    strategy: 'fixed',
    chunkSize: 512,
    overlap: 64,
  } as ChunkConfig,
  embed: {
    model: 'text-embedding-ada-002',
    batchSize: 100,
  } as EmbedConfig,
  extract: {
    entityTypes: ['person', 'organization', 'concept'],
    model: 'gpt-4o',
    confidenceThreshold: 0.7,
  } as ExtractConfig,
  index: {
    target: 'hybrid',
  } as IndexConfig,
  retrieve: {
    method: 'hybrid',
    topK: 10,
    scoreThreshold: 0.5,
  } as RetrieveConfig,
  rerank: {
    provider: 'flashrank',
    topK: 5,
  } as RerankConfig,
  respond: {
    model: 'gpt-4o-mini',
    temperature: 0.7,
    maxTokens: 2000,
  } as RespondConfig,
};

/**
 * Node palette items for the sidebar.
 */
export const nodePaletteItems: NodePaletteItem[] = [
  {
    type: 'ingest',
    label: 'Ingest',
    description: 'Import documents from various sources',
    icon: '📥',
    defaultConfig: defaultNodeConfigs.ingest,
  },
  {
    type: 'chunk',
    label: 'Chunk',
    description: 'Split documents into chunks',
    icon: '✂️',
    defaultConfig: defaultNodeConfigs.chunk,
  },
  {
    type: 'embed',
    label: 'Embed',
    description: 'Generate embeddings for chunks',
    icon: '🔢',
    defaultConfig: defaultNodeConfigs.embed,
  },
  {
    type: 'extract',
    label: 'Extract',
    description: 'Extract entities from text',
    icon: '🏷️',
    defaultConfig: defaultNodeConfigs.extract,
  },
  {
    type: 'index',
    label: 'Index',
    description: 'Store in vector/graph database',
    icon: '💾',
    defaultConfig: defaultNodeConfigs.index,
  },
  {
    type: 'retrieve',
    label: 'Retrieve',
    description: 'Search for relevant content',
    icon: '🔍',
    defaultConfig: defaultNodeConfigs.retrieve,
  },
  {
    type: 'rerank',
    label: 'Rerank',
    description: 'Reorder results by relevance',
    icon: '📊',
    defaultConfig: defaultNodeConfigs.rerank,
  },
  {
    type: 'respond',
    label: 'Respond',
    description: 'Generate response using LLM',
    icon: '💬',
    defaultConfig: defaultNodeConfigs.respond,
  },
];

/**
 * Colors for each node type.
 */
export const nodeTypeColors: Record<WorkflowNodeType, string> = {
  ingest: '#10B981',   // emerald
  chunk: '#3B82F6',    // blue
  embed: '#8B5CF6',    // violet
  extract: '#F59E0B',  // amber
  index: '#EF4444',    // red
  retrieve: '#06B6D4', // cyan
  rerank: '#EC4899',   // pink
  respond: '#14B8A6',  // teal
};

/**
 * Status colors for workflow nodes.
 */
export const statusColors: Record<NodeStatus, string> = {
  idle: '#6B7280',     // gray
  running: '#3B82F6',  // blue
  success: '#10B981',  // emerald
  error: '#EF4444',    // red
  skipped: '#9CA3AF',  // light gray
};
