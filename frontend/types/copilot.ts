/**
 * TypeScript types for CopilotKit integration.
 * Story 6-1: CopilotKit React Integration
 */

import { z } from 'zod';

/**
 * Source retrieved during RAG retrieval.
 */
export interface Source {
  id: string;
  title: string;
  preview: string;
  similarity: number;
  metadata?: Record<string, unknown>;
  isApproved?: boolean;
}

/**
 * A step in the agent's thought process.
 */
export interface ThoughtStep {
  step: string;
  status: "pending" | "in_progress" | "completed";
  timestamp?: string;
  details?: string;
}

/**
 * Current state of the AI agent.
 */
export interface AgentState {
  currentStep: string;
  thoughts: ThoughtStep[];
  retrievedSources: Source[];
  validatedSources: Source[];
  answer: string | null;
  trajectoryId: string | null;
}

/**
 * Result from a frontend action execution.
 */
export interface FrontendActionResult {
  success: boolean;
  error?: string;
  data?: unknown;
}

/**
 * Message in the CopilotKit conversation.
 */
export interface CopilotMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt?: string;
}

/**
 * AG-UI event types.
 */
export type AGUIEventType =
  | "RUN_STARTED"
  | "RUN_FINISHED"
  | "TEXT_MESSAGE_START"
  | "TEXT_MESSAGE_CONTENT"
  | "TEXT_MESSAGE_END"
  | "TOOL_CALL_START"
  | "TOOL_CALL_ARGS"
  | "TOOL_CALL_END"
  | "TOOL_CALL_RESULT"
  | "STATE_SNAPSHOT"
  | "ACTION_REQUEST";

/**
 * AG-UI event payload.
 */
export interface AGUIEvent {
  event: AGUIEventType;
  data: Record<string, unknown>;
}

// Zod schemas for validation
export const SourceSchema = z.object({
  id: z.string(),
  title: z.string(),
  preview: z.string(),
  similarity: z.number().min(0).max(1),
  metadata: z.record(z.string(), z.unknown()).optional(),
  isApproved: z.boolean().optional(),
});

export const ThoughtStepSchema = z.object({
  step: z.string(),
  status: z.enum(["pending", "in_progress", "completed"]),
  timestamp: z.string().optional(),
  details: z.string().optional(),
});

export const AgentStateSchema = z.object({
  currentStep: z.string(),
  thoughts: z.array(ThoughtStepSchema),
  retrievedSources: z.array(SourceSchema),
  validatedSources: z.array(SourceSchema),
  answer: z.string().nullable(),
  trajectoryId: z.string().nullable(),
});

export const FrontendActionResultSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),
  data: z.unknown().optional(),
});

// ============================================
// GENERATIVE UI TYPES - Story 6-3
// ============================================

/**
 * Graph node for generative UI graph preview.
 */
export interface GraphPreviewNode {
  id: string;
  label: string;
  type?: string;
  properties?: Record<string, unknown>;
}

/**
 * Graph edge for generative UI graph preview.
 */
export interface GraphPreviewEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  type?: string;
}

/**
 * Generative UI state managed by the agent.
 */
export interface GenerativeUIState {
  sources: Source[];
  answer: string | null;
  graphData: {
    nodes: GraphPreviewNode[];
    edges: GraphPreviewEdge[];
  } | null;
}

// Zod schemas for generative UI validation
export const GraphPreviewNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string().optional(),
  properties: z.record(z.string(), z.unknown()).optional(),
});

export const GraphPreviewEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  label: z.string().optional(),
  type: z.string().optional(),
});

export const GenerativeUIStateSchema = z.object({
  sources: z.array(SourceSchema),
  answer: z.string().nullable(),
  graphData: z
    .object({
      nodes: z.array(GraphPreviewNodeSchema),
      edges: z.array(GraphPreviewEdgeSchema),
    })
    .nullable(),
});
