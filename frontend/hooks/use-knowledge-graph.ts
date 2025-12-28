/**
 * TanStack Query hooks for Knowledge Graph data fetching.
 * Story 4.4: Knowledge Graph Visualization
 */

import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';
import type { GraphQueryOptions } from '../types/graphs';

/**
 * Hook to fetch knowledge graph data.
 * 
 * @param options - Query options including tenantId, filters, pagination
 * @returns TanStack Query result with graph data
 */
export function useKnowledgeGraph(options: GraphQueryOptions) {
  return useQuery({
    queryKey: ['knowledge', 'graph', options],
    queryFn: () => api.knowledge.getGraph(options),
    staleTime: 30000, // 30 seconds
    enabled: Boolean(options.tenantId),
  });
}

/**
 * Hook to fetch knowledge graph statistics.
 * 
 * @param tenantId - Tenant identifier
 * @returns TanStack Query result with graph stats
 */
export function useKnowledgeStats(tenantId: string) {
  return useQuery({
    queryKey: ['knowledge', 'stats', tenantId],
    queryFn: () => api.knowledge.getStats(tenantId),
    staleTime: 60000, // 1 minute
    enabled: Boolean(tenantId),
  });
}

/**
 * Hook to fetch orphan nodes.
 * 
 * @param tenantId - Tenant identifier
 * @param limit - Maximum number of orphans to return
 * @returns TanStack Query result with orphan nodes
 */
export function useKnowledgeOrphans(tenantId: string, limit?: number) {
  return useQuery({
    queryKey: ['knowledge', 'orphans', tenantId, limit],
    queryFn: () => api.knowledge.getOrphans(tenantId, limit),
    staleTime: 30000, // 30 seconds
    enabled: Boolean(tenantId),
  });
}
