/**
 * Tests for MCP-UI security utilities
 * Story 22-C1: Implement MCP-UI Renderer
 */

import {
  MCPUIMessageSchema,
  isAllowedOrigin,
  extractOrigin,
  validateMCPUIMessage,
  clearAllowedOriginsCache,
  loadAllowedOrigins,
  getEnvAllowedOrigins,
} from '@/lib/mcp-ui-security';

// Mock fetch for API tests
global.fetch = jest.fn();

const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;

describe('mcp-ui-security', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    clearAllowedOriginsCache();
    // Reset environment variable
    delete process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
  });

  describe('MCPUIMessageSchema', () => {
    it('should validate mcp_ui_resize message', () => {
      const message = { type: 'mcp_ui_resize', width: 800, height: 600 };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('mcp_ui_resize');
      }
    });

    it('should validate mcp_ui_result message', () => {
      const message = { type: 'mcp_ui_result', result: { value: 42 } };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('mcp_ui_result');
      }
    });

    it('should validate mcp_ui_error message', () => {
      const message = { type: 'mcp_ui_error', error: 'Something went wrong' };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('mcp_ui_error');
      }
    });

    it('should reject invalid message type', () => {
      const message = { type: 'invalid_type', data: {} };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(false);
    });

    it('should reject resize with width below minimum', () => {
      const message = { type: 'mcp_ui_resize', width: 50, height: 600 };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(false);
    });

    it('should reject resize with height below minimum', () => {
      const message = { type: 'mcp_ui_resize', width: 800, height: 10 };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(false);
    });

    it('should reject resize with width above maximum', () => {
      const message = { type: 'mcp_ui_resize', width: 5000, height: 600 };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(false);
    });
  });

  describe('extractOrigin', () => {
    it('should extract origin from valid URL', () => {
      expect(extractOrigin('https://example.com/path')).toBe('https://example.com');
    });

    it('should extract origin with port', () => {
      expect(extractOrigin('http://localhost:3000/path')).toBe('http://localhost:3000');
    });

    it('should return null for invalid URL', () => {
      expect(extractOrigin('not-a-url')).toBeNull();
    });

    it('should return null for empty string', () => {
      expect(extractOrigin('')).toBeNull();
    });
  });

  describe('isAllowedOrigin', () => {
    it('should return true for allowed origin', () => {
      const allowedOrigins = new Set(['https://example.com', 'https://tools.example.com']);
      expect(isAllowedOrigin('https://example.com', allowedOrigins)).toBe(true);
    });

    it('should return false for disallowed origin', () => {
      const allowedOrigins = new Set(['https://example.com']);
      expect(isAllowedOrigin('https://malicious.com', allowedOrigins)).toBe(false);
    });

    it('should return false for empty set', () => {
      expect(isAllowedOrigin('https://example.com', new Set())).toBe(false);
    });
  });

  describe('getEnvAllowedOrigins', () => {
    it('should parse comma-separated origins from env', () => {
      process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS =
        'https://a.com, https://b.com, https://c.com';
      const origins = getEnvAllowedOrigins();
      expect(origins.size).toBe(3);
      expect(origins.has('https://a.com')).toBe(true);
      expect(origins.has('https://b.com')).toBe(true);
      expect(origins.has('https://c.com')).toBe(true);
    });

    it('should handle empty env variable', () => {
      process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = '';
      const origins = getEnvAllowedOrigins();
      expect(origins.size).toBe(0);
    });

    it('should handle undefined env variable', () => {
      delete process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      const origins = getEnvAllowedOrigins();
      expect(origins.size).toBe(0);
    });

    it('should trim whitespace from origins', () => {
      process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = '  https://a.com  ,  https://b.com  ';
      const origins = getEnvAllowedOrigins();
      expect(origins.has('https://a.com')).toBe(true);
      expect(origins.has('https://b.com')).toBe(true);
    });
  });

  describe('loadAllowedOrigins', () => {
    it('should fetch origins from backend API', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          enabled: true,
          allowed_origins: ['https://api-origin.com'],
        }),
      } as Response);

      const origins = await loadAllowedOrigins('test-tenant');
      expect(origins.has('https://api-origin.com')).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/mcp/ui/config'),
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-Tenant-ID': 'test-tenant',
          }),
        })
      );
    });

    it('should cache loaded origins', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          enabled: true,
          allowed_origins: ['https://cached.com'],
        }),
      } as Response);

      // First call loads from API
      await loadAllowedOrigins('test-tenant');
      // Second call should use cache
      await loadAllowedOrigins('test-tenant');

      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should isolate cache per tenant', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            enabled: true,
            allowed_origins: ['https://tenant-a.com'],
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            enabled: true,
            allowed_origins: ['https://tenant-b.com'],
          }),
        } as Response);

      // Load for tenant A
      const originsA = await loadAllowedOrigins('tenant-a');
      // Load for tenant B
      const originsB = await loadAllowedOrigins('tenant-b');

      // Both should have made separate API calls
      expect(mockFetch).toHaveBeenCalledTimes(2);
      // Each tenant has their own origins
      expect(originsA.has('https://tenant-a.com')).toBe(true);
      expect(originsA.has('https://tenant-b.com')).toBe(false);
      expect(originsB.has('https://tenant-b.com')).toBe(true);
      expect(originsB.has('https://tenant-a.com')).toBe(false);
    });

    it('should return empty set on API failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      } as Response);

      const origins = await loadAllowedOrigins('test-tenant');
      expect(origins.size).toBe(0);
    });

    it('should not cache error states and allow retry', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            enabled: true,
            allowed_origins: ['https://recovered.com'],
          }),
        } as Response);

      // First call fails
      const originsFirst = await loadAllowedOrigins('error-tenant');
      expect(originsFirst.size).toBe(0);

      // Second call should retry (not use cached error)
      const originsSecond = await loadAllowedOrigins('error-tenant');
      expect(originsSecond.has('https://recovered.com')).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('should return empty set on network error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const origins = await loadAllowedOrigins('test-tenant');
      expect(origins.size).toBe(0);
    });
  });

  describe('validateMCPUIMessage', () => {
    const allowedOrigins = new Set(['https://trusted.com']);

    it('should validate message from allowed origin', () => {
      const event = {
        origin: 'https://trusted.com',
        data: { type: 'mcp_ui_resize', width: 800, height: 600 },
      } as MessageEvent;

      const result = validateMCPUIMessage(event, allowedOrigins);
      expect(result).not.toBeNull();
      expect(result?.type).toBe('mcp_ui_resize');
    });

    it('should reject message from disallowed origin', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const event = {
        origin: 'https://untrusted.com',
        data: { type: 'mcp_ui_resize', width: 800, height: 600 },
      } as MessageEvent;

      const result = validateMCPUIMessage(event, allowedOrigins);
      expect(result).toBeNull();
      expect(consoleSpy).toHaveBeenCalledWith(
        'MCP-UI: Blocked message from untrusted origin',
        'https://untrusted.com'
      );

      consoleSpy.mockRestore();
    });

    it('should reject message with invalid shape', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const event = {
        origin: 'https://trusted.com',
        data: { type: 'invalid_type' },
      } as MessageEvent;

      const result = validateMCPUIMessage(event, allowedOrigins);
      expect(result).toBeNull();
      expect(consoleSpy).toHaveBeenCalledWith(
        'MCP-UI: Invalid message shape',
        expect.any(Object)
      );

      consoleSpy.mockRestore();
    });

    it('should handle result messages with various data types', () => {
      const testCases = [
        { type: 'mcp_ui_result', result: 'string' },
        { type: 'mcp_ui_result', result: 42 },
        { type: 'mcp_ui_result', result: { complex: 'object' } },
        { type: 'mcp_ui_result', result: null },
        { type: 'mcp_ui_result', result: [1, 2, 3] },
      ];

      for (const data of testCases) {
        const event = {
          origin: 'https://trusted.com',
          data,
        } as MessageEvent;

        const result = validateMCPUIMessage(event, allowedOrigins);
        expect(result).not.toBeNull();
        expect(result?.type).toBe('mcp_ui_result');
      }
    });
  });

  describe('clearAllowedOriginsCache', () => {
    it('should clear the cache', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          enabled: true,
          allowed_origins: ['https://cached.com'],
        }),
      } as Response);

      // Load origins to populate cache
      await loadAllowedOrigins('test-tenant');
      expect(mockFetch).toHaveBeenCalledTimes(1);

      // Clear cache
      clearAllowedOriginsCache();

      // Load again - should make new API call
      await loadAllowedOrigins('test-tenant');
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });
});
