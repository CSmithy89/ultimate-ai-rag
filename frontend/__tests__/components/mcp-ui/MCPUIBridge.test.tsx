/**
 * Tests for MCPUIBridge hook
 * Story 22-C1: Implement MCP-UI Renderer
 */

import React, { useRef } from 'react';
import { renderHook, act } from '@testing-library/react';
import { useMCPUIBridge, type MCPUIBridgeCallbacks } from '@/components/mcp-ui/MCPUIBridge';
import * as mcpUISecurity from '@/lib/mcp-ui-security';

// Mock the security module
jest.mock('@/lib/mcp-ui-security', () => ({
  validateMCPUIMessage: jest.fn(),
}));

const mockValidateMCPUIMessage = mcpUISecurity.validateMCPUIMessage as jest.MockedFunction<
  typeof mcpUISecurity.validateMCPUIMessage
>;

describe('useMCPUIBridge', () => {
  const allowedOrigins = new Set(['https://trusted.example.com']);

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    // Clean up any event listeners
    jest.restoreAllMocks();
  });

  const createMockIframeRef = () => {
    const contentWindow = {} as Window;
    const iframe = { contentWindow } as HTMLIFrameElement;
    return { current: iframe };
  };

  const renderBridgeHook = (callbacks: MCPUIBridgeCallbacks) => {
    const iframeRef = createMockIframeRef();

    const wrapper = ({ children }: { children: React.ReactNode }) => <>{children}</>;

    // Use a component that wraps the hook for proper ref handling
    const TestComponent = () => {
      useMCPUIBridge(
        iframeRef as React.RefObject<HTMLIFrameElement | null>,
        allowedOrigins,
        callbacks
      );
      return null;
    };

    return { iframeRef, TestComponent };
  };

  describe('resize messages', () => {
    it('should call onResize when resize message received', () => {
      const onResize = jest.fn();
      const { iframeRef } = renderBridgeHook({ onResize });

      mockValidateMCPUIMessage.mockReturnValue({
        type: 'mcp_ui_resize',
        width: 800,
        height: 600,
      });

      // Render with the hook
      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onResize }
        )
      );

      // Simulate postMessage event
      act(() => {
        const event = new MessageEvent('message', {
          origin: 'https://trusted.example.com',
          data: { type: 'mcp_ui_resize', width: 800, height: 600 },
          source: iframeRef.current?.contentWindow,
        });
        window.dispatchEvent(event);
      });

      expect(onResize).toHaveBeenCalledWith(800, 600);

      unmount();
    });
  });

  describe('result messages', () => {
    it('should call onResult when result message received', () => {
      const onResult = jest.fn();
      const iframeRef = createMockIframeRef();

      mockValidateMCPUIMessage.mockReturnValue({
        type: 'mcp_ui_result',
        result: { value: 42 },
      });

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onResult }
        )
      );

      act(() => {
        const event = new MessageEvent('message', {
          origin: 'https://trusted.example.com',
          data: { type: 'mcp_ui_result', result: { value: 42 } },
          source: iframeRef.current?.contentWindow,
        });
        window.dispatchEvent(event);
      });

      expect(onResult).toHaveBeenCalledWith({ value: 42 });

      unmount();
    });

    it('should handle various result types', () => {
      const onResult = jest.fn();
      const iframeRef = createMockIframeRef();

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onResult }
        )
      );

      const testResults = [
        'string result',
        42,
        { complex: 'object' },
        [1, 2, 3],
        null,
      ];

      for (const result of testResults) {
        mockValidateMCPUIMessage.mockReturnValue({
          type: 'mcp_ui_result',
          result,
        });

        act(() => {
          const event = new MessageEvent('message', {
            origin: 'https://trusted.example.com',
            data: { type: 'mcp_ui_result', result },
            source: iframeRef.current?.contentWindow,
          });
          window.dispatchEvent(event);
        });

        expect(onResult).toHaveBeenLastCalledWith(result);
      }

      unmount();
    });
  });

  describe('error messages', () => {
    it('should call onError when error message received', () => {
      const onError = jest.fn();
      const iframeRef = createMockIframeRef();

      mockValidateMCPUIMessage.mockReturnValue({
        type: 'mcp_ui_error',
        error: 'Something went wrong',
      });

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onError }
        )
      );

      act(() => {
        const event = new MessageEvent('message', {
          origin: 'https://trusted.example.com',
          data: { type: 'mcp_ui_error', error: 'Something went wrong' },
          source: iframeRef.current?.contentWindow,
        });
        window.dispatchEvent(event);
      });

      expect(onError).toHaveBeenCalledWith('Something went wrong');

      unmount();
    });
  });

  describe('origin validation', () => {
    it('should ignore messages from non-iframe sources', () => {
      const onResult = jest.fn();
      const iframeRef = createMockIframeRef();

      mockValidateMCPUIMessage.mockReturnValue({
        type: 'mcp_ui_result',
        result: 'ignored',
      });

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onResult }
        )
      );

      // Send message from different source (not the iframe)
      act(() => {
        const event = new MessageEvent('message', {
          origin: 'https://trusted.example.com',
          data: { type: 'mcp_ui_result', result: 'ignored' },
          source: window, // Different source
        });
        window.dispatchEvent(event);
      });

      expect(onResult).not.toHaveBeenCalled();

      unmount();
    });

    it('should ignore invalid messages', () => {
      const onResult = jest.fn();
      const iframeRef = createMockIframeRef();

      // Return null to indicate validation failure
      mockValidateMCPUIMessage.mockReturnValue(null);

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onResult }
        )
      );

      act(() => {
        const event = new MessageEvent('message', {
          origin: 'https://untrusted.example.com',
          data: { type: 'mcp_ui_result', result: 'ignored' },
          source: iframeRef.current?.contentWindow,
        });
        window.dispatchEvent(event);
      });

      expect(onResult).not.toHaveBeenCalled();

      unmount();
    });
  });

  describe('cleanup', () => {
    it('should remove event listener on unmount', () => {
      const removeEventListenerSpy = jest.spyOn(window, 'removeEventListener');
      const onResult = jest.fn();
      const iframeRef = createMockIframeRef();

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          { onResult }
        )
      );

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        'message',
        expect.any(Function)
      );

      removeEventListenerSpy.mockRestore();
    });
  });

  describe('optional callbacks', () => {
    it('should not throw when callbacks are undefined', () => {
      const iframeRef = createMockIframeRef();

      mockValidateMCPUIMessage.mockReturnValue({
        type: 'mcp_ui_resize',
        width: 800,
        height: 600,
      });

      const { unmount } = renderHook(() =>
        useMCPUIBridge(
          iframeRef as React.RefObject<HTMLIFrameElement | null>,
          allowedOrigins,
          {} // No callbacks provided
        )
      );

      // This should not throw
      expect(() => {
        act(() => {
          const event = new MessageEvent('message', {
            origin: 'https://trusted.example.com',
            data: { type: 'mcp_ui_resize', width: 800, height: 600 },
            source: iframeRef.current?.contentWindow,
          });
          window.dispatchEvent(event);
        });
      }).not.toThrow();

      unmount();
    });
  });
});
