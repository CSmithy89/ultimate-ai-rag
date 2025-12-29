/**
 * TypeScript types for Knowledge Graph visualization.
 * Story 4.4: Knowledge Graph Visualization
 */

import { z } from 'zod';

// Entity types in the knowledge graph
export type EntityType = 'Person' | 'Organization' | 'Technology' | 'Concept' | 'Location';

// Relationship types between entities
export type RelationshipType = 'MENTIONS' | 'AUTHORED_BY' | 'PART_OF' | 'USES' | 'RELATED_TO';

// Color scheme for entity types
export const entityColors: Record<EntityType | 'orphan', string> = {
  Person: '#3B82F6',        // Blue-500
  Organization: '#10B981',   // Emerald-500
  Technology: '#6366F1',     // Indigo-500
  Concept: '#8B5CF6',        // Violet-500
  Location: '#F59E0B',       // Amber-500
  orphan: '#F97316',         // Orange-500 (warning)
};

// Tailwind classes for entity types
export const entityColorClasses: Record<EntityType | 'orphan', string> = {
  Person: 'bg-blue-500 border-blue-600',
  Organization: 'bg-emerald-500 border-emerald-600',
  Technology: 'bg-indigo-500 border-indigo-600',
  Concept: 'bg-violet-500 border-violet-600',
  Location: 'bg-amber-500 border-amber-600',
  orphan: 'border-orange-400 ring-2 ring-orange-400/50',
};

// Zod schemas for validation
export const GraphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string(),
  properties: z.record(z.string(), z.unknown()).nullable().optional(),
  is_orphan: z.boolean(),
});

export const GraphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  type: z.string(),
  label: z.string(),
  properties: z.record(z.string(), z.unknown()).nullable().optional(),
});

export const GraphDataSchema = z.object({
  nodes: z.array(GraphNodeSchema),
  edges: z.array(GraphEdgeSchema),
});

export const GraphStatsSchema = z.object({
  node_count: z.number(),
  edge_count: z.number(),
  orphan_count: z.number(),
  entity_type_counts: z.record(z.string(), z.number()),
  relationship_type_counts: z.record(z.string(), z.number()),
});

export const GraphQueryOptionsSchema = z.object({
  tenantId: z.string().uuid(),
  limit: z.number().min(1).max(1000).optional(),
  offset: z.number().min(0).optional(),
  entityType: z.string().optional(),
  relationshipType: z.string().optional(),
  dateFrom: z.string().datetime().optional(),
});

// TypeScript types derived from Zod schemas
export type GraphNode = z.infer<typeof GraphNodeSchema>;
export type GraphEdge = z.infer<typeof GraphEdgeSchema>;
export type GraphData = z.infer<typeof GraphDataSchema>;
export type GraphStats = z.infer<typeof GraphStatsSchema>;
export type GraphQueryOptions = z.infer<typeof GraphQueryOptionsSchema>;

// API response types
export interface ApiMeta {
  requestId: string;
  timestamp: string;
}

export interface ApiResponse<T> {
  data: T;
  meta: ApiMeta;
}

export interface OrphansResponse {
  orphans: GraphNode[];
  count: number;
}

// React Flow specific types
export interface ReactFlowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    entityType: EntityType;
    isOrphan: boolean;
    properties?: Record<string, unknown>;
  };
}

export interface ReactFlowEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  label: string;
  data?: {
    relationshipType: RelationshipType;
    properties?: Record<string, unknown>;
  };
}

// Filter state type
export interface GraphFilterState {
  entityTypes: EntityType[];
  relationshipTypes: RelationshipType[];
  showOrphansOnly: boolean;
  searchQuery: string;
}
