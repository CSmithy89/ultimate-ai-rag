/**
 * MCP-UI Bridge Hook
 *
 * Provides postMessage communication between the parent application
 * and MCP-UI iframes with proper origin validation.
 *
 * Story 22-C1: Implement MCP-UI Renderer
 */

'use client';

import { useEffect, type RefObject } from 'react';
import { validateMCPUIMessage, type MCPUIMessage } from '@/lib/mcp-ui-security';

/**
 * Callbacks for MCP-UI bridge messages.
 */
export interface MCPUIBridgeCallbacks {
  /** Called when iframe requests resize */
  onResize?: (width: number, height: number) => void;
  /** Called when iframe returns a result */
  onResult?: (result: unknown) => void;
  /** Called when iframe reports an error */
  onError?: (error: string) => void;
}

/**
 * Hook for managing postMessage communication with MCP-UI iframes.
 *
 * Validates all incoming messages against the allowed origins list
 * and the Zod schema before invoking callbacks.
 *
 * @param iframeRef - Reference to the iframe element
 * @param allowedOrigins - Set of allowed origin strings
 * @param callbacks - Callback handlers for different message types
 *
 * @example
 * ```tsx
 * const iframeRef = useRef<HTMLIFrameElement>(null);
 *
 * useMCPUIBridge(iframeRef, allowedOrigins, {
 *   onResize: (w, h) => setDimensions({ width: w, height: h }),
 *   onResult: (result) => console.log('Result:', result),
 *   onError: (error) => console.error('Error:', error),
 * });
 * ```
 */
export function useMCPUIBridge(
  iframeRef: RefObject<HTMLIFrameElement | null>,
  allowedOrigins: Set<string>,
  callbacks: MCPUIBridgeCallbacks
): void {
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      // Only process messages from the iframe
      if (
        iframeRef.current?.contentWindow &&
        event.source !== iframeRef.current.contentWindow
      ) {
        return;
      }

      // Validate origin and message shape
      const message = validateMCPUIMessage(event, allowedOrigins);
      if (!message) {
        return;
      }

      // Handle message by type
      handleMCPUIMessage(message, callbacks);
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [iframeRef, allowedOrigins, callbacks]);
}

/**
 * Process a validated MCP-UI message.
 */
function handleMCPUIMessage(
  message: MCPUIMessage,
  callbacks: MCPUIBridgeCallbacks
): void {
  switch (message.type) {
    case 'mcp_ui_resize':
      callbacks.onResize?.(message.width, message.height);
      break;

    case 'mcp_ui_result':
      callbacks.onResult?.(message.result);
      break;

    case 'mcp_ui_error':
      callbacks.onError?.(message.error);
      break;
  }
}

export default useMCPUIBridge;
