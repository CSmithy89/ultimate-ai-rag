/**
 * API client for backend communication.
 * Story 4.4: Knowledge Graph Visualization
 */

import type {
  ApiResponse,
  GraphData,
  GraphQueryOptions,
  GraphStats,
  OrphansResponse,
} from '../types/graphs';
import {
  GraphDataSchema,
  GraphStatsSchema,
} from '../types/graphs';
import {
  CostEventSchema,
  CostSummarySchema,
  AlertConfigSchema,
  TrajectorySummarySchema,
  TrajectoryDetailSchema,
} from '../types/ops';
import { z } from 'zod';

// Schema for orphans response validation
const GraphOrphansSchema = z.object({
  orphans: z.array(z.object({
    id: z.string(),
    label: z.string(),
    type: z.string(),
    properties: z.record(z.string(), z.unknown()).nullable().optional(),
    is_orphan: z.boolean(),
  })),
  count: z.number(),
});

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

/**
 * Generic fetch wrapper with error handling.
 */
async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  const url = API_BASE_URL + endpoint;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || errorData.title || 'API error: ' + response.status
    );
  }

  return response.json();
}

/**
 * Convert camelCase to snake_case.
 */
function toSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => '_' + letter.toLowerCase());
}

/**
 * Build query string from options object.
 */
function buildQueryString(params: Record<string, unknown>): string {
  const searchParams = new URLSearchParams();
  
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      const snakeKey = toSnakeCase(key);
      searchParams.append(snakeKey, String(value));
    }
  }
  
  return searchParams.toString();
}

/**
 * Knowledge Graph API methods.
 */
export const knowledge = {
  /**
   * Fetch graph data for visualization.
   */
  async getGraph(options: GraphQueryOptions): Promise<GraphData> {
    const queryString = buildQueryString(options);
    const response = await fetchApi<GraphData>('/knowledge/graph?' + queryString);
    return GraphDataSchema.parse(response.data);
  },

  /**
   * Fetch graph statistics.
   */
  async getStats(tenantId: string): Promise<GraphStats> {
    const queryString = buildQueryString({ tenantId });
    const response = await fetchApi<GraphStats>('/knowledge/stats?' + queryString);
    return GraphStatsSchema.parse(response.data);
  },

  /**
   * Fetch orphan nodes.
   */
  async getOrphans(tenantId: string, limit?: number): Promise<OrphansResponse> {
    const queryString = buildQueryString({ tenantId, limit });
    const response = await fetchApi<OrphansResponse>('/knowledge/orphans?' + queryString);
    return GraphOrphansSchema.parse(response.data);
  },
};

/**
 * Ops API methods.
 */
export const ops = {
  async getCostSummary(tenantId: string, window: string) {
    const queryString = buildQueryString({ tenantId, window });
    const response = await fetchApi('/ops/costs/summary?' + queryString);
    return CostSummarySchema.parse(response.data);
  },

  async getCostEvents(tenantId: string, limit = 50, offset = 0) {
    const queryString = buildQueryString({ tenantId, limit, offset });
    const response = await fetchApi<{ events: unknown[] }>('/ops/costs/events?' + queryString);
    const events = z.array(CostEventSchema).parse(response.data.events);
    return events;
  },

  async getCostAlerts(tenantId: string) {
    const queryString = buildQueryString({ tenantId });
    const response = await fetchApi<{ alerts: unknown }>('/ops/costs/alerts?' + queryString);
    return AlertConfigSchema.parse(response.data.alerts);
  },

  async updateCostAlerts(payload: {
    tenant_id: string;
    daily_threshold_usd?: number | null;
    monthly_threshold_usd?: number | null;
    enabled: boolean;
  }) {
    const response = await fetchApi<{ alerts: unknown }>('/ops/costs/alerts', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    return AlertConfigSchema.parse(response.data.alerts);
  },

  async getTrajectories(
    tenantId: string,
    options?: { status?: string; agentType?: string; limit?: number; offset?: number }
  ) {
    const queryString = buildQueryString({
      tenantId,
      status: options?.status,
      agentType: options?.agentType,
      limit: options?.limit,
      offset: options?.offset,
    });
    const response = await fetchApi<{ trajectories: unknown[] }>(
      '/ops/trajectories?' + queryString
    );
    return z.array(TrajectorySummarySchema).parse(response.data.trajectories);
  },

  async getTrajectoryDetail(tenantId: string, trajectoryId: string) {
    const queryString = buildQueryString({ tenantId });
    const response = await fetchApi('/ops/trajectories/' + trajectoryId + '?' + queryString);
    return TrajectoryDetailSchema.parse(response.data);
  },
};

/**
 * Main API client object.
 */
export const api = {
  knowledge,
  ops,
};

export default api;
