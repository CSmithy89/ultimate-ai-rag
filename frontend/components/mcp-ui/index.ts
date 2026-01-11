/**
 * MCP-UI Components
 *
 * Secure iframe rendering for MCP tool UIs with postMessage bridge.
 *
 * Story 22-C1: Implement MCP-UI Renderer
 */

export { MCPUIRenderer, type MCPUIRendererProps, type MCPUIPayload } from './MCPUIRenderer';
export { useMCPUIBridge, type MCPUIBridgeCallbacks } from './MCPUIBridge';
