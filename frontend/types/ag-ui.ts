/**
 * AG-UI Protocol Types - Shared TypeScript Types
 *
 * This file contains shared types used across AG-UI hooks and components.
 * Centralizing these types prevents circular dependencies.
 */

// ============================================
// CONSTANTS
// ============================================

/** Auto-reset delay for completed activities (ms) */
export const ACTIVITY_RESET_DELAY_MS = 3000;

/** Maximum file size for multimodal content (10MB) */
export const MAX_FILE_SIZE = 10 * 1024 * 1024;

/** Maximum number of attachments */
export const MAX_ATTACHMENTS = 5;

// ============================================
// ACTIVITY TRACKING TYPES (Phase 5)
// ============================================

/**
 * Activity state from AG-UI ACTIVITY events.
 */
export interface ActivityState {
  /** Unique activity identifier */
  id: string;
  /** Type of activity (e.g., "query_processing", "indexing") */
  type: string;
  /** Human-readable status message */
  message: string;
  /** Progress as a decimal (0.0 to 1.0) */
  progress: number;
  /** Total number of steps in the activity */
  totalSteps: number;
  /** Current step number (1-indexed) */
  currentStep: number;
  /** Additional metadata about the activity */
  metadata?: Record<string, unknown>;
}

/**
 * Valid keys for ActivityState (for type-safe patching).
 */
export const ACTIVITY_STATE_KEYS = new Set<keyof ActivityState>([
  "id",
  "type",
  "message",
  "progress",
  "totalSteps",
  "currentStep",
  "metadata",
]);

/**
 * Default empty activity state.
 */
export const EMPTY_ACTIVITY: ActivityState = {
  id: "",
  type: "",
  message: "",
  progress: 0,
  totalSteps: 0,
  currentStep: 0,
  metadata: {},
};

// ============================================
// JSON PATCH TYPES (RFC 6902)
// ============================================

/**
 * RFC 6902 JSON Patch operation types.
 */
export type JSONPatchOperationType =
  | "add"
  | "remove"
  | "replace"
  | "move"
  | "copy"
  | "test";

/**
 * RFC 6902 JSON Patch operation.
 */
export interface JSONPatchOperation {
  op: JSONPatchOperationType;
  path: string;
  value?: JSONPatchValue;
  from?: string;
}

/**
 * Allowed value types for JSON Patch operations on ActivityState.
 */
export type JSONPatchValue =
  | string
  | number
  | boolean
  | null
  | Record<string, unknown>;

// ============================================
// RUN CONTROL TYPES (Phase 6)
// ============================================

/**
 * Run status from backend.
 */
export type RunStatus =
  | "running"
  | "cancelled"
  | "completed"
  | "error"
  | "paused";

/**
 * Run state from backend.
 */
export interface RunState {
  run_id: string;
  thread_id: string;
  status: RunStatus;
  query: string;
  tenant_id?: string;
  session_id?: string;
  created_at: string;
  current_step: number;
  total_steps: number;
  partial_result?: string;
  error_message?: string;
}

// ============================================
// STEERING TYPES (Phase 6.2)
// ============================================

/**
 * Steering instruction context.
 */
export interface SteeringContext {
  [key: string]: unknown;
}

/**
 * Result of a steering operation.
 */
export interface SteeringResult {
  runId: string;
  status: string;
  message: string;
}

// ============================================
// MULTIMODAL TYPES (Phase 7.2)
// ============================================

/**
 * Supported media types for binary content.
 */
export const SUPPORTED_MEDIA_TYPES = {
  // Images
  "image/png": { extension: ".png", category: "image" },
  "image/jpeg": { extension: ".jpg", category: "image" },
  "image/gif": { extension: ".gif", category: "image" },
  "image/webp": { extension: ".webp", category: "image" },
  // Audio
  "audio/wav": { extension: ".wav", category: "audio" },
  "audio/mp3": { extension: ".mp3", category: "audio" },
  "audio/mpeg": { extension: ".mp3", category: "audio" },
  "audio/ogg": { extension: ".ogg", category: "audio" },
  // Documents
  "application/pdf": { extension: ".pdf", category: "document" },
  "text/plain": { extension: ".txt", category: "document" },
} as const;

export type SupportedMediaType = keyof typeof SUPPORTED_MEDIA_TYPES;

/**
 * Text content part of a multimodal message.
 */
export interface TextInputContent {
  type: "text";
  content: string;
}

/**
 * Binary content part of a multimodal message.
 */
export interface BinaryInputContent {
  type: "binary";
  media_type: string;
  data: string; // base64 encoded
  filename?: string;
}

/**
 * A content part (text or binary) in a multimodal message.
 */
export type MultimodalContentPart = TextInputContent | BinaryInputContent;

/**
 * An attachment ready to be sent.
 */
export interface Attachment {
  id: string;
  type: "binary";
  media_type: string;
  data: string;
  filename: string;
  size: number;
  preview?: string; // For images, a data URL for preview
}

// ============================================
// ERROR TYPES
// ============================================

/**
 * Error with additional context for API operations.
 */
export interface APIError extends Error {
  status?: number;
  code?: string;
  isNetworkError?: boolean;
  isNotFound?: boolean;
}

/**
 * Create an API error with context.
 */
export function createAPIError(
  message: string,
  options?: {
    status?: number;
    code?: string;
    isNetworkError?: boolean;
    isNotFound?: boolean;
  }
): APIError {
  const error = new Error(message) as APIError;
  error.status = options?.status;
  error.code = options?.code;
  error.isNetworkError = options?.isNetworkError ?? false;
  error.isNotFound = options?.isNotFound ?? false;
  return error;
}
