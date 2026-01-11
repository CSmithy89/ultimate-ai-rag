/**
 * MCP-UI Renderer Component
 *
 * Renders MCP tool UIs in sandboxed iframes with proper security controls.
 * Validates origins against allowlist and handles postMessage communication.
 *
 * Story 22-C1: Implement MCP-UI Renderer
 */

'use client';

import { useRef, useState, useEffect, useCallback, memo } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useMCPUIBridge } from './MCPUIBridge';
import {
  extractOrigin,
  isAllowedOrigin,
  getEnvAllowedOrigins,
  loadAllowedOrigins,
} from '@/lib/mcp-ui-security';

/**
 * MCP-UI payload structure from backend.
 */
export interface MCPUIPayload {
  type: 'mcp_ui';
  tool_name: string;
  ui_url: string;
  ui_type: 'iframe';
  sandbox: string[];
  size: { width: number; height: number };
  allow: string[];
  data: Record<string, unknown>;
}

/**
 * Props for MCPUIRenderer component.
 */
export interface MCPUIRendererProps {
  /** MCP-UI payload from tool response */
  payload: MCPUIPayload;
  /** Tenant ID for loading config */
  tenantId?: string;
  /** Callback when iframe returns a result */
  onResult?: (result: unknown) => void;
  /** Callback when iframe reports an error */
  onError?: (error: string) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * MCPUIRenderer - Secure iframe renderer for MCP tool UIs.
 *
 * Renders external tool UIs in sandboxed iframes with:
 * - Origin validation against allowlist
 * - Minimal sandbox permissions
 * - PostMessage bridge for communication
 * - Security warning for blocked origins
 *
 * Security features:
 * - sandbox="allow-scripts" by default (no allow-same-origin to prevent sandbox escape)
 * - Origins validated before rendering
 * - postMessage origin checked before processing
 * - No access to parent DOM
 */
export const MCPUIRenderer = memo(function MCPUIRenderer({
  payload,
  tenantId,
  onResult,
  onError,
  className,
}: MCPUIRendererProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [dimensions, setDimensions] = useState(payload.size);
  const [allowedOrigins, setAllowedOrigins] = useState<Set<string>>(
    getEnvAllowedOrigins()
  );
  const [isBlocked, setIsBlocked] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [iframeLoaded, setIframeLoaded] = useState(false);

  // Load allowed origins from backend
  useEffect(() => {
    if (tenantId) {
      loadAllowedOrigins(tenantId).then((origins) => {
        setAllowedOrigins(origins);
        setIsLoading(false);
      });
    } else {
      setIsLoading(false);
    }
  }, [tenantId]);

  // Check if URL origin is allowed
  useEffect(() => {
    if (isLoading) return;

    const origin = extractOrigin(payload.ui_url);
    if (!origin || !isAllowedOrigin(origin, allowedOrigins)) {
      setIsBlocked(true);
      console.warn('MCP-UI: Blocked iframe from untrusted origin', origin);
    } else {
      setIsBlocked(false);
    }
  }, [payload.ui_url, allowedOrigins, isLoading]);

  // Handle resize messages
  const handleResize = useCallback((width: number, height: number) => {
    setDimensions({ width, height });
  }, []);

  // Handle result messages
  const handleResult = useCallback(
    (result: unknown) => {
      onResult?.(result);
    },
    [onResult]
  );

  // Handle error messages
  const handleError = useCallback(
    (error: string) => {
      console.error('MCP-UI error:', error);
      onError?.(error);
    },
    [onError]
  );

  // Setup postMessage bridge
  useMCPUIBridge(iframeRef, allowedOrigins, {
    onResize: handleResize,
    onResult: handleResult,
    onError: handleError,
  });

  // Send init data to iframe when it loads
  const handleIframeLoad = useCallback(() => {
    setIframeLoaded(true);
    if (iframeRef.current?.contentWindow && payload.data) {
      const origin = extractOrigin(payload.ui_url);
      if (origin) {
        iframeRef.current.contentWindow.postMessage(
          { type: 'mcp_ui_init', data: payload.data },
          origin
        );
      }
    }
  }, [payload.ui_url, payload.data]);

  // Show loading state
  if (isLoading) {
    return (
      <div
        className={cn(
          'my-2 border border-slate-200 rounded-lg bg-white p-4',
          className
        )}
        data-testid="mcp-ui-loading"
      >
        <div className="flex items-center gap-2 text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          <span className="text-sm">Loading MCP-UI configuration...</span>
        </div>
      </div>
    );
  }

  // Show security warning for blocked origins
  if (isBlocked) {
    const origin = extractOrigin(payload.ui_url);
    return (
      <div
        className={cn(
          'my-2 border border-red-300 rounded-lg bg-red-50 overflow-hidden',
          className
        )}
        data-testid="mcp-ui-blocked"
      >
        <div className="p-3 flex items-start gap-2">
          <AlertCircle
            className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5"
            aria-hidden="true"
          />
          <div>
            <h4 className="text-sm font-medium text-red-800">
              Security Warning
            </h4>
            <p className="text-sm text-red-700 mt-1">
              MCP-UI blocked: Untrusted origin{' '}
              <code className="bg-red-100 px-1 rounded text-xs">{origin}</code>
            </p>
            <p className="text-xs text-red-600 mt-1">
              Tool: {payload.tool_name}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Render sandboxed iframe
  return (
    <div
      className={cn(
        'my-2 border border-slate-200 rounded-lg bg-white overflow-hidden',
        className
      )}
      data-testid="mcp-ui-renderer"
    >
      {/* Header */}
      <div className="p-3 border-b border-slate-100 flex items-center gap-2">
        <span className="font-mono text-sm text-slate-700">
          {payload.tool_name}
        </span>
        {!iframeLoaded && (
          <Loader2
            className="h-3 w-3 animate-spin text-slate-400"
            aria-hidden="true"
          />
        )}
      </div>

      {/* Iframe container */}
      <div className="relative" style={{ minHeight: '100px' }}>
        <iframe
          ref={iframeRef}
          src={payload.ui_url}
          sandbox={payload.sandbox.join(' ')}
          allow={payload.allow.join('; ')}
          onLoad={handleIframeLoad}
          style={{
            width: dimensions.width,
            height: dimensions.height,
            border: 'none',
            display: 'block',
            maxWidth: '100%',
          }}
          title={`MCP-UI: ${payload.tool_name}`}
          data-testid="mcp-ui-iframe"
        />
      </div>
    </div>
  );
});

export default MCPUIRenderer;
