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

// ============================================
// HITL VALIDATION TYPES - Story 6-4
// ============================================

/**
 * Validation decision for a source in HITL.
 */
export type ValidationDecision = "approved" | "rejected" | "pending";

/**
 * Source with validation state for HITL.
 */
export interface ValidatableSource extends Source {
  validationStatus: ValidationDecision;
}

/**
 * State of a HITL validation checkpoint.
 */
export interface HITLCheckpoint {
  checkpointId: string;
  sources: Source[];
  query: string;
  status: "pending" | "approved" | "rejected" | "skipped";
  approvedSourceIds: string[];
  rejectedSourceIds: string[];
}

/**
 * Response format for validation submission.
 */
export interface ValidationResponse {
  checkpointId: string;
  status: string;
  approvedCount: number;
  rejectedCount: number;
}

/**
 * State for source validation hook.
 */
export interface SourceValidationState {
  /** Whether validation is currently in progress */
  isValidating: boolean;
  /** Sources awaiting validation */
  pendingSources: Source[];
  /** Map of source ID to validation decision */
  decisions: Map<string, ValidationDecision>;
  /** IDs of approved sources */
  approvedIds: string[];
  /** IDs of rejected sources */
  rejectedIds: string[];
  /** Whether submission is in progress */
  isSubmitting: boolean;
  /** Error message if validation failed */
  error: string | null;
}

// Zod schemas for HITL validation
export const ValidationDecisionSchema = z.enum(["approved", "rejected", "pending"]);

export const ValidatableSourceSchema = SourceSchema.extend({
  validationStatus: ValidationDecisionSchema,
});

export const HITLCheckpointSchema = z.object({
  checkpointId: z.string(),
  sources: z.array(SourceSchema),
  query: z.string(),
  status: z.enum(["pending", "approved", "rejected", "skipped"]),
  approvedSourceIds: z.array(z.string()),
  rejectedSourceIds: z.array(z.string()),
});

export const ValidationResponseSchema = z.object({
  checkpointId: z.string(),
  status: z.string(),
  approvedCount: z.number(),
  rejectedCount: z.number(),
});

// ============================================
// FRONTEND ACTIONS TYPES - Story 6-5
// ============================================

/**
 * Action types supported by the frontend actions system.
 */
export type ActionType = "save" | "export" | "share" | "bookmark" | "followUp";

/**
 * State of an action.
 */
export type ActionState = "idle" | "loading" | "success" | "error";

/**
 * Export format options.
 */
export type ExportFormat = "markdown" | "pdf" | "json";

/**
 * Content that can be actioned upon.
 */
export interface ActionableContent {
  /** Unique ID for this content */
  id: string;
  /** The response text/content */
  content: string;
  /** Optional title for saved content */
  title?: string;
  /** Original query that generated this response */
  query?: string;
  /** Sources used in generating the response */
  sources?: Array<{
    id: string;
    title: string;
    url?: string;
  }>;
  /** Timestamp of the response */
  timestamp?: string;
  /** Session/conversation ID */
  sessionId?: string;
  /** Trajectory ID for this response */
  trajectoryId?: string;
}

/**
 * Action history item for tracking completed actions.
 */
export interface ActionHistoryItem {
  /** Unique ID for this action */
  id: string;
  /** Type of action performed */
  type: ActionType;
  /** Current status of the action */
  status: "pending" | "success" | "error";
  /** When the action was initiated */
  timestamp: string;
  /** Human-readable title */
  title: string;
  /** Error message if action failed */
  error?: string;
  /** Additional data from the action */
  data?: {
    shareUrl?: string;
    filename?: string;
    [key: string]: unknown;
  };
}

// Zod schemas for frontend actions validation
export const ActionTypeSchema = z.enum(["save", "export", "share", "bookmark", "followUp"]);

export const ActionStateSchema = z.enum(["idle", "loading", "success", "error"]);

export const ExportFormatSchema = z.enum(["markdown", "pdf", "json"]);

export const ActionableContentSchema = z.object({
  id: z.string(),
  content: z.string(),
  title: z.string().optional(),
  query: z.string().optional(),
  sources: z
    .array(
      z.object({
        id: z.string(),
        title: z.string(),
        url: z.string().optional(),
      })
    )
    .optional(),
  timestamp: z.string().optional(),
  sessionId: z.string().optional(),
  trajectoryId: z.string().optional(),
});

export const ActionHistoryItemSchema = z.object({
  id: z.string(),
  type: ActionTypeSchema,
  status: z.enum(["pending", "success", "error"]),
  timestamp: z.string(),
  title: z.string(),
  error: z.string().optional(),
  data: z
    .object({
      shareUrl: z.string().optional(),
      filename: z.string().optional(),
    })
    .passthrough()
    .optional(),
});

// ============================================
// COPILOT CONTEXT TYPES - Story 21-A4
// ============================================

/**
 * Page context information exposed to the AI.
 * Story 21-A4: Implement useCopilotReadable for App Context
 */
export interface PageContext {
  /** Current route path (e.g., "/knowledge", "/ops") */
  route: string;
  /** Human-readable page name */
  pageName: string;
  /** Optional page-specific metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Session context information exposed to the AI.
 * Only non-sensitive session data is included.
 * Story 21-A4: Implement useCopilotReadable for App Context
 */
export interface SessionContext {
  /** Tenant ID for multi-tenant context */
  tenantId: string | null;
  /** ISO timestamp when session started */
  sessionStart: string;
  /** Whether user is authenticated (not credentials) */
  isAuthenticated: boolean;
}

/**
 * User preferences for AI response formatting.
 * Story 21-A4: Implement useCopilotReadable for App Context
 */
export interface UserPreferences {
  /** Preferred response length: "brief", "medium", "detailed" */
  responseLength: "brief" | "medium" | "detailed";
  /** Whether to include source citations in responses */
  includeCitations: boolean;
  /** User's preferred language code (e.g., "en", "es") */
  language: string;
  /** User's expertise level for response complexity */
  expertiseLevel: "beginner" | "intermediate" | "expert";
}

/**
 * Query history item for tracking recent queries.
 * Story 21-A4: Implement useCopilotReadable for App Context
 */
export interface QueryHistoryItem {
  /** The query text */
  query: string;
  /** ISO timestamp when query was made */
  timestamp: string;
}

/**
 * Combined application context exposed to AI.
 * Story 21-A4: Implement useCopilotReadable for App Context
 */
export interface AppContext {
  /** Current page context */
  page: PageContext;
  /** Session context (non-sensitive) */
  session: SessionContext;
  /** User preferences */
  preferences: UserPreferences;
  /** Recent query history */
  recentQueries: QueryHistoryItem[];
}

// Zod schemas for context validation
export const PageContextSchema = z.object({
  route: z.string(),
  pageName: z.string(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export const SessionContextSchema = z.object({
  tenantId: z.string().nullable(),
  sessionStart: z.string(),
  isAuthenticated: z.boolean(),
});

export const UserPreferencesSchema = z.object({
  responseLength: z.enum(["brief", "medium", "detailed"]),
  includeCitations: z.boolean(),
  language: z.string(),
  expertiseLevel: z.enum(["beginner", "intermediate", "expert"]),
});

export const QueryHistoryItemSchema = z.object({
  query: z.string(),
  timestamp: z.string(),
});

export const AppContextSchema = z.object({
  page: PageContextSchema,
  session: SessionContextSchema,
  preferences: UserPreferencesSchema,
  recentQueries: z.array(QueryHistoryItemSchema),
});

// ============================================
// PROGRAMMATIC CHAT TYPES - Story 21-A6
// ============================================

/**
 * A chat message in the programmatic chat interface.
 * Story 21-A6: Implement useCopilotChat for Headless Control
 */
export interface ChatMessage {
  /** Unique message ID */
  id: string;
  /** Role of the message sender */
  role: "user" | "assistant" | "system";
  /** Message content text */
  content: string;
}

/**
 * Configuration for a quick action button.
 * Story 21-A6: Implement useCopilotChat for Headless Control
 */
export interface QuickActionConfig {
  /** Display label for the button */
  label: string;
  /** Message to send when clicked */
  message: string;
  /** Optional icon name (for future use) */
  icon?: string;
  /** Optional description for tooltip */
  description?: string;
}

/**
 * Return type for the useProgrammaticChat hook.
 * Story 21-A6: Implement useCopilotChat for Headless Control
 */
export interface ProgrammaticChatReturn {
  /** Array of visible messages in the conversation */
  messages: ChatMessage[];
  /** Number of messages in the conversation */
  messageCount: number;
  /** Whether the chat is currently generating a response */
  isLoading: boolean;
  /** Send a user message programmatically */
  sendMessage: (content: string) => Promise<void>;
  /** Regenerate the last assistant response */
  regenerateLastResponse: () => Promise<void>;
  /** Stop the current generation */
  stopGeneration: () => void;
  /** Clear all messages and reset the chat */
  clearHistory: () => void;
}

// Zod schemas for programmatic chat validation
export const ChatMessageSchema = z.object({
  id: z.string(),
  role: z.enum(["user", "assistant", "system"]),
  content: z.string(),
});

export const QuickActionConfigSchema = z.object({
  label: z.string(),
  message: z.string(),
  icon: z.string().optional(),
  description: z.string().optional(),
});
