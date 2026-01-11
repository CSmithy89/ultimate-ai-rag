/**
 * MCP-UI Security Utilities
 *
 * Provides origin validation and postMessage schema validation
 * for secure MCP-UI iframe communication.
 *
 * Story 22-C1: Implement MCP-UI Renderer
 */

import { z } from 'zod';

/**
 * Zod schema for validating MCP-UI postMessage payloads.
 * Uses discriminated union for type-safe message handling.
 */
export const MCPUIMessageSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('mcp_ui_resize'),
    width: z.number().min(100).max(4000),
    height: z.number().min(50).max(4000),
  }),
  z.object({
    type: z.literal('mcp_ui_result'),
    result: z.unknown(),
  }),
  z.object({
    type: z.literal('mcp_ui_error'),
    error: z.string(),
  }),
]);

export type MCPUIMessage = z.infer<typeof MCPUIMessageSchema>;

/**
 * MCP-UI configuration from backend.
 */
export interface MCPUIConfig {
  enabled: boolean;
  allowed_origins: string[];
}

// Tenant-keyed cache for allowed origins (supports multi-tenancy)
const _allowedOriginsCache = new Map<string, Set<string>>();

/**
 * Load allowed origins from backend API.
 * Results are cached per-tenant to avoid repeated API calls while
 * respecting multi-tenancy isolation.
 *
 * @param tenantId - Tenant identifier for API call
 * @returns Set of allowed origin strings
 */
export async function loadAllowedOrigins(tenantId: string): Promise<Set<string>> {
  // Check tenant-specific cache
  const cached = _allowedOriginsCache.get(tenantId);
  if (cached !== undefined) {
    return cached;
  }

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

  try {
    const response = await fetch(`${apiUrl}/mcp/ui/config`, {
      headers: {
        'Content-Type': 'application/json',
        'X-Tenant-ID': tenantId,
      },
    });

    if (!response.ok) {
      console.warn('MCP-UI: Failed to load config, using fallback');
      // Don't cache error states - allow retry on next call
      return new Set<string>();
    }

    const config: MCPUIConfig = await response.json();
    const origins = new Set(config.allowed_origins);
    // Only cache successful responses
    _allowedOriginsCache.set(tenantId, origins);
    return origins;
  } catch (error) {
    console.warn('MCP-UI: Error loading config', error);
    // Don't cache error states - allow retry on next call
    return new Set<string>();
  }
}

/**
 * Get allowed origins from environment variable.
 * Used as fallback when backend is unavailable.
 *
 * @returns Set of allowed origin strings from NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS
 */
export function getEnvAllowedOrigins(): Set<string> {
  const envOrigins = process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS || '';
  return new Set(
    envOrigins
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
  );
}

/**
 * Check if an origin is in the allowed list.
 *
 * @param origin - Origin string to validate
 * @param allowedOrigins - Set of allowed origins (defaults to env-based)
 * @returns True if origin is allowed
 */
export function isAllowedOrigin(
  origin: string,
  allowedOrigins?: Set<string>
): boolean {
  const origins = allowedOrigins ?? getEnvAllowedOrigins();
  return origins.has(origin);
}

/**
 * Extract origin from a URL string.
 *
 * @param url - URL string to parse
 * @returns Origin string or null if parsing fails
 */
export function extractOrigin(url: string): string | null {
  try {
    return new URL(url).origin;
  } catch {
    return null;
  }
}

/**
 * Validate and parse a postMessage event.
 * Returns null if validation fails.
 *
 * @param event - MessageEvent from postMessage
 * @param allowedOrigins - Set of allowed origins
 * @returns Parsed MCPUIMessage or null
 */
export function validateMCPUIMessage(
  event: MessageEvent,
  allowedOrigins: Set<string>
): MCPUIMessage | null {
  // Validate origin
  if (!isAllowedOrigin(event.origin, allowedOrigins)) {
    console.warn('MCP-UI: Blocked message from untrusted origin', event.origin);
    return null;
  }

  // Validate message shape
  const result = MCPUIMessageSchema.safeParse(event.data);
  if (!result.success) {
    console.warn('MCP-UI: Invalid message shape', result.error);
    return null;
  }

  return result.data;
}

/**
 * Clear the cached allowed origins for all tenants.
 * Useful for testing or when configuration changes.
 *
 * @param tenantId - Optional tenant ID to clear specific tenant's cache.
 *                   If omitted, clears entire cache.
 */
export function clearAllowedOriginsCache(tenantId?: string): void {
  if (tenantId) {
    _allowedOriginsCache.delete(tenantId);
  } else {
    _allowedOriginsCache.clear();
  }
}
