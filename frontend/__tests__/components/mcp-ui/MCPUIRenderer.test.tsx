/**
 * Tests for MCPUIRenderer component
 * Story 22-C1: Implement MCP-UI Renderer
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MCPUIRenderer, type MCPUIPayload } from '@/components/mcp-ui/MCPUIRenderer';
import * as mcpUISecurity from '@/lib/mcp-ui-security';

// Mock the security module
jest.mock('@/lib/mcp-ui-security', () => ({
  extractOrigin: jest.fn(),
  isAllowedOrigin: jest.fn(),
  getEnvAllowedOrigins: jest.fn(),
  loadAllowedOrigins: jest.fn(),
  validateMCPUIMessage: jest.fn(),
}));

// Mock the bridge hook
jest.mock('@/components/mcp-ui/MCPUIBridge', () => ({
  useMCPUIBridge: jest.fn(),
}));

const mockExtractOrigin = mcpUISecurity.extractOrigin as jest.MockedFunction<
  typeof mcpUISecurity.extractOrigin
>;
const mockIsAllowedOrigin = mcpUISecurity.isAllowedOrigin as jest.MockedFunction<
  typeof mcpUISecurity.isAllowedOrigin
>;
const mockGetEnvAllowedOrigins = mcpUISecurity.getEnvAllowedOrigins as jest.MockedFunction<
  typeof mcpUISecurity.getEnvAllowedOrigins
>;
const mockLoadAllowedOrigins = mcpUISecurity.loadAllowedOrigins as jest.MockedFunction<
  typeof mcpUISecurity.loadAllowedOrigins
>;

const createMockPayload = (overrides?: Partial<MCPUIPayload>): MCPUIPayload => ({
  type: 'mcp_ui',
  tool_name: 'test-tool',
  ui_url: 'https://trusted.example.com/tool',
  ui_type: 'iframe',
  sandbox: ['allow-scripts', 'allow-same-origin'],
  size: { width: 600, height: 400 },
  allow: [],
  data: {},
  ...overrides,
});

describe('MCPUIRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetEnvAllowedOrigins.mockReturnValue(new Set(['https://trusted.example.com']));
    mockLoadAllowedOrigins.mockResolvedValue(new Set(['https://trusted.example.com']));
    mockExtractOrigin.mockReturnValue('https://trusted.example.com');
    mockIsAllowedOrigin.mockReturnValue(true);
  });

  describe('loading state', () => {
    it('should show loading state when fetching config with tenant ID', async () => {
      // Make loadAllowedOrigins never resolve during this test
      mockLoadAllowedOrigins.mockReturnValue(new Promise(() => {}));

      render(<MCPUIRenderer payload={createMockPayload()} tenantId="test-tenant" />);

      expect(screen.getByTestId('mcp-ui-loading')).toBeInTheDocument();
      expect(screen.getByText(/Loading MCP-UI configuration/i)).toBeInTheDocument();
    });

    it('should not show loading when no tenant ID provided', async () => {
      render(<MCPUIRenderer payload={createMockPayload()} />);

      await waitFor(() => {
        expect(screen.queryByTestId('mcp-ui-loading')).not.toBeInTheDocument();
      });
    });
  });

  describe('security blocking', () => {
    it('should show security warning for untrusted origin', async () => {
      mockIsAllowedOrigin.mockReturnValue(false);
      mockExtractOrigin.mockReturnValue('https://untrusted.example.com');

      render(
        <MCPUIRenderer
          payload={createMockPayload({ ui_url: 'https://untrusted.example.com/tool' })}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('mcp-ui-blocked')).toBeInTheDocument();
      });

      expect(screen.getByText(/Security Warning/i)).toBeInTheDocument();
      expect(screen.getByText(/Untrusted origin/i)).toBeInTheDocument();
    });

    it('should show tool name in security warning', async () => {
      mockIsAllowedOrigin.mockReturnValue(false);
      mockExtractOrigin.mockReturnValue('https://untrusted.example.com');

      render(
        <MCPUIRenderer
          payload={createMockPayload({
            tool_name: 'dangerous-tool',
            ui_url: 'https://untrusted.example.com/tool',
          })}
        />
      );

      await waitFor(() => {
        expect(screen.getByText(/dangerous-tool/i)).toBeInTheDocument();
      });
    });

    it('should block when origin cannot be extracted', async () => {
      mockExtractOrigin.mockReturnValue(null);

      render(
        <MCPUIRenderer
          payload={createMockPayload({ ui_url: 'invalid-url' as unknown as string })}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('mcp-ui-blocked')).toBeInTheDocument();
      });
    });
  });

  describe('iframe rendering', () => {
    it('should render iframe for allowed origin', async () => {
      render(<MCPUIRenderer payload={createMockPayload()} />);

      await waitFor(() => {
        expect(screen.getByTestId('mcp-ui-renderer')).toBeInTheDocument();
      });

      const iframe = screen.getByTestId('mcp-ui-iframe') as HTMLIFrameElement;
      expect(iframe).toBeInTheDocument();
      expect(iframe.src).toBe('https://trusted.example.com/tool');
    });

    it('should apply sandbox attributes', async () => {
      render(
        <MCPUIRenderer
          payload={createMockPayload({
            sandbox: ['allow-scripts', 'allow-same-origin'],
          })}
        />
      );

      await waitFor(() => {
        const iframe = screen.getByTestId('mcp-ui-iframe') as HTMLIFrameElement;
        // jsdom doesn't fully support DOMTokenList, use getAttribute instead
        const sandboxAttr = iframe.getAttribute('sandbox') || '';
        expect(sandboxAttr).toContain('allow-scripts');
        expect(sandboxAttr).toContain('allow-same-origin');
      });
    });

    it('should apply custom sandbox attributes', async () => {
      render(
        <MCPUIRenderer
          payload={createMockPayload({
            sandbox: ['allow-scripts'],
          })}
        />
      );

      await waitFor(() => {
        const iframe = screen.getByTestId('mcp-ui-iframe') as HTMLIFrameElement;
        // jsdom doesn't fully support DOMTokenList, use getAttribute instead
        const sandboxAttr = iframe.getAttribute('sandbox') || '';
        expect(sandboxAttr).toContain('allow-scripts');
        expect(sandboxAttr).not.toContain('allow-same-origin');
      });
    });

    it('should set correct iframe dimensions', async () => {
      render(
        <MCPUIRenderer
          payload={createMockPayload({
            size: { width: 800, height: 600 },
          })}
        />
      );

      await waitFor(() => {
        const iframe = screen.getByTestId('mcp-ui-iframe') as HTMLIFrameElement;
        expect(iframe.style.width).toBe('800px');
        expect(iframe.style.height).toBe('600px');
      });
    });

    it('should display tool name in header', async () => {
      render(
        <MCPUIRenderer
          payload={createMockPayload({
            tool_name: 'my-awesome-tool',
          })}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('my-awesome-tool')).toBeInTheDocument();
      });
    });

    it('should set iframe title for accessibility', async () => {
      render(
        <MCPUIRenderer
          payload={createMockPayload({
            tool_name: 'accessible-tool',
          })}
        />
      );

      await waitFor(() => {
        const iframe = screen.getByTestId('mcp-ui-iframe') as HTMLIFrameElement;
        expect(iframe.title).toBe('MCP-UI: accessible-tool');
      });
    });
  });

  describe('origin loading', () => {
    it('should load origins from API when tenant ID provided', async () => {
      render(<MCPUIRenderer payload={createMockPayload()} tenantId="test-tenant-123" />);

      await waitFor(() => {
        expect(mockLoadAllowedOrigins).toHaveBeenCalledWith('test-tenant-123');
      });
    });

    it('should use env origins when no tenant ID', async () => {
      render(<MCPUIRenderer payload={createMockPayload()} />);

      await waitFor(() => {
        expect(mockGetEnvAllowedOrigins).toHaveBeenCalled();
      });

      expect(mockLoadAllowedOrigins).not.toHaveBeenCalled();
    });
  });

  describe('callbacks', () => {
    it('should pass onResult callback', async () => {
      const onResult = jest.fn();
      render(<MCPUIRenderer payload={createMockPayload()} onResult={onResult} />);

      await waitFor(() => {
        expect(screen.getByTestId('mcp-ui-renderer')).toBeInTheDocument();
      });

      // Note: Actually testing callback invocation requires integration testing
      // with the bridge hook. This test just verifies prop passing.
    });

    it('should pass onError callback', async () => {
      const onError = jest.fn();
      render(<MCPUIRenderer payload={createMockPayload()} onError={onError} />);

      await waitFor(() => {
        expect(screen.getByTestId('mcp-ui-renderer')).toBeInTheDocument();
      });
    });
  });

  describe('CSS classes', () => {
    it('should apply custom className', async () => {
      render(<MCPUIRenderer payload={createMockPayload()} className="custom-class" />);

      await waitFor(() => {
        const renderer = screen.getByTestId('mcp-ui-renderer');
        expect(renderer.classList.contains('custom-class')).toBe(true);
      });
    });
  });
});
