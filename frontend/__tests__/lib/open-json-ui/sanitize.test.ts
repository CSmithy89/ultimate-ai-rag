/**
 * Tests for Open-JSON-UI Sanitization Utilities
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

import {
  sanitizeContent,
  sanitizeToPlainText,
  isValidUrl,
  sanitizeUrl,
} from '@/lib/open-json-ui/sanitize';

// Mock DOMPurify for controlled testing
jest.mock('dompurify', () => ({
  sanitize: jest.fn((content: string, options?: { ALLOWED_TAGS?: string[]; ALLOWED_ATTR?: string[] }) => {
    const allowedTags = options?.ALLOWED_TAGS ?? ['b', 'i', 'em', 'strong', 'code', 'br', 'span'];

    // Start with a result that will be modified
    let result = content;

    // Always remove script tags and their content first (for security)
    result = result.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

    // Always remove style tags and their content
    result = result.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');

    // Always remove event handlers
    result = result.replace(/\s*on\w+\s*=\s*(['"])[^'"]*\1/gi, '');

    // Simple mock implementation that strips disallowed tags
    if (allowedTags.length === 0) {
      // Strip all HTML tags for plain text mode
      return result.replace(/<[^>]*>/g, '');
    }

    // For allowed tags mode, strip dangerous tags but keep safe ones

    // Remove iframe tags
    result = result.replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '');
    result = result.replace(/<iframe[^>]*\/?>/gi, '');

    // Remove object and embed tags
    result = result.replace(/<(object|embed)\b[^<]*(?:(?!<\/\1>)<[^<]*)*<\/\1>/gi, '');
    result = result.replace(/<(object|embed)[^>]*\/?>/gi, '');

    // Remove img tags (not in allowed list)
    if (!allowedTags.includes('img')) {
      result = result.replace(/<img[^>]*\/?>/gi, '');
    }

    // Remove a tags (not in allowed list)
    if (!allowedTags.includes('a')) {
      result = result.replace(/<a\b[^>]*>(.*?)<\/a>/gi, '$1');
    }

    // Remove div tags (not in allowed list)
    if (!allowedTags.includes('div')) {
      result = result.replace(/<\/?div[^>]*>/gi, '');
    }

    // Remove p tags (not in allowed list)
    if (!allowedTags.includes('p')) {
      result = result.replace(/<\/?p[^>]*>/gi, '');
    }

    return result;
  }),
}));

describe('Open-JSON-UI Sanitization', () => {
  describe('sanitizeContent', () => {
    it('should remove script tags', () => {
      const input = '<script>alert("xss")</script>Hello';
      const result = sanitizeContent(input);
      expect(result).not.toContain('<script');
      expect(result).not.toContain('alert');
      expect(result).toContain('Hello');
    });

    it('should remove inline script handlers', () => {
      const input = '<div onclick="alert(1)">Click me</div>';
      const result = sanitizeContent(input);
      expect(result).not.toContain('onclick');
      expect(result).not.toContain('alert');
    });

    it('should preserve bold tag', () => {
      const input = '<b>Bold text</b>';
      const result = sanitizeContent(input);
      expect(result).toContain('<b>');
      expect(result).toContain('Bold text');
      expect(result).toContain('</b>');
    });

    it('should preserve italic tag', () => {
      const input = '<i>Italic text</i>';
      const result = sanitizeContent(input);
      expect(result).toContain('<i>');
      expect(result).toContain('Italic text');
    });

    it('should preserve em tag', () => {
      const input = '<em>Emphasized</em>';
      const result = sanitizeContent(input);
      expect(result).toContain('<em>');
      expect(result).toContain('Emphasized');
    });

    it('should preserve strong tag', () => {
      const input = '<strong>Strong text</strong>';
      const result = sanitizeContent(input);
      expect(result).toContain('<strong>');
      expect(result).toContain('Strong text');
    });

    it('should preserve code tag', () => {
      const input = '<code>const x = 1;</code>';
      const result = sanitizeContent(input);
      expect(result).toContain('<code>');
      expect(result).toContain('const x = 1;');
    });

    it('should preserve br tag', () => {
      const input = 'Line 1<br>Line 2';
      const result = sanitizeContent(input);
      expect(result).toContain('<br>');
    });

    it('should preserve span tag', () => {
      const input = '<span>Wrapped text</span>';
      const result = sanitizeContent(input);
      expect(result).toContain('<span>');
      expect(result).toContain('Wrapped text');
    });

    it('should remove style tags', () => {
      const input = '<style>body { color: red; }</style>Content';
      const result = sanitizeContent(input);
      expect(result).not.toContain('<style');
      expect(result).not.toContain('color');
      expect(result).toContain('Content');
    });

    it('should remove iframe tags', () => {
      const input = '<iframe src="https://evil.com"></iframe>Safe content';
      const result = sanitizeContent(input);
      expect(result).not.toContain('<iframe');
      expect(result).not.toContain('evil.com');
      expect(result).toContain('Safe content');
    });

    it('should remove object tags', () => {
      const input = '<object data="file.swf">Flash content</object>Safe';
      const result = sanitizeContent(input);
      expect(result).not.toContain('<object');
      expect(result).toContain('Safe');
    });

    it('should remove embed tags', () => {
      const input = '<embed src="file.swf">Safe content';
      const result = sanitizeContent(input);
      expect(result).not.toContain('<embed');
    });

    it('should handle mixed safe and unsafe content', () => {
      const input = '<b>Bold</b><script>evil()</script><em>Emphasis</em>';
      const result = sanitizeContent(input);
      expect(result).toContain('<b>Bold</b>');
      expect(result).toContain('<em>Emphasis</em>');
      expect(result).not.toContain('<script');
    });

    it('should handle plain text without modification', () => {
      const input = 'Just plain text with no HTML';
      const result = sanitizeContent(input);
      expect(result).toBe(input);
    });

    it('should handle empty string', () => {
      const result = sanitizeContent('');
      expect(result).toBe('');
    });

    it('should handle nested tags correctly', () => {
      const input = '<b><i>Bold and italic</i></b>';
      const result = sanitizeContent(input);
      expect(result).toContain('<b>');
      expect(result).toContain('<i>');
      expect(result).toContain('Bold and italic');
    });
  });

  describe('sanitizeToPlainText', () => {
    it('should strip all HTML tags', () => {
      const input = '<b>Bold</b> and <i>italic</i> text';
      const result = sanitizeToPlainText(input);
      expect(result).not.toContain('<b>');
      expect(result).not.toContain('<i>');
      expect(result).toContain('Bold');
      expect(result).toContain('and');
      expect(result).toContain('italic');
      expect(result).toContain('text');
    });

    it('should remove script tags completely', () => {
      const input = '<script>alert(1)</script>Safe text';
      const result = sanitizeToPlainText(input);
      expect(result).not.toContain('script');
      expect(result).not.toContain('alert');
      expect(result).toContain('Safe text');
    });

    it('should handle complex nested HTML', () => {
      const input = '<div><p><b>Nested <em>content</em></b></p></div>';
      const result = sanitizeToPlainText(input);
      expect(result).not.toContain('<');
      expect(result).not.toContain('>');
      expect(result).toContain('Nested');
      expect(result).toContain('content');
    });

    it('should handle HTML entities preserved', () => {
      // After stripping tags, entities remain
      const input = '&lt;not a tag&gt;';
      const result = sanitizeToPlainText(input);
      // HTML entities should remain as-is
      expect(result).toBe('&lt;not a tag&gt;');
    });

    it('should handle empty string', () => {
      const result = sanitizeToPlainText('');
      expect(result).toBe('');
    });

    it('should handle plain text without modification', () => {
      const input = 'No HTML here';
      const result = sanitizeToPlainText(input);
      expect(result).toBe(input);
    });
  });

  describe('isValidUrl', () => {
    it('should accept https URLs', () => {
      expect(isValidUrl('https://example.com')).toBe(true);
      expect(isValidUrl('https://example.com/path')).toBe(true);
      expect(isValidUrl('https://example.com/path?query=value')).toBe(true);
      expect(isValidUrl('https://sub.example.com:8080/path')).toBe(true);
    });

    it('should accept http URLs', () => {
      expect(isValidUrl('http://example.com')).toBe(true);
      expect(isValidUrl('http://localhost:3000')).toBe(true);
      expect(isValidUrl('http://127.0.0.1:8080')).toBe(true);
    });

    it('should reject javascript: URLs', () => {
      expect(isValidUrl('javascript:alert(1)')).toBe(false);
      expect(isValidUrl('javascript:void(0)')).toBe(false);
      expect(isValidUrl('JAVASCRIPT:alert(1)')).toBe(false);
    });

    it('should reject data: URLs', () => {
      expect(isValidUrl('data:text/html,<script>alert(1)</script>')).toBe(false);
      expect(isValidUrl('data:image/png;base64,abc123')).toBe(false);
      expect(isValidUrl('DATA:text/plain,hello')).toBe(false);
    });

    it('should reject vbscript: URLs', () => {
      expect(isValidUrl('vbscript:msgbox(1)')).toBe(false);
    });

    it('should reject file: URLs', () => {
      expect(isValidUrl('file:///etc/passwd')).toBe(false);
    });

    it('should reject ftp: URLs', () => {
      expect(isValidUrl('ftp://example.com')).toBe(false);
    });

    it('should reject relative URLs', () => {
      expect(isValidUrl('/path/to/resource')).toBe(false);
      expect(isValidUrl('./relative/path')).toBe(false);
      expect(isValidUrl('../parent/path')).toBe(false);
    });

    it('should reject malformed URLs', () => {
      expect(isValidUrl('not-a-url')).toBe(false);
      expect(isValidUrl('')).toBe(false);
      expect(isValidUrl('just text')).toBe(false);
    });

    it('should reject protocol-relative URLs', () => {
      // Protocol-relative URLs should be rejected as they are not absolute URLs
      expect(isValidUrl('//example.com/path')).toBe(false);
    });

    it('should handle URLs with fragments', () => {
      expect(isValidUrl('https://example.com/page#section')).toBe(true);
    });

    it('should handle URLs with special characters', () => {
      expect(isValidUrl('https://example.com/path?q=hello%20world')).toBe(true);
      expect(isValidUrl('https://example.com/search?q=a+b')).toBe(true);
    });

    it('should handle international domain names', () => {
      expect(isValidUrl('https://xn--e1afmkfd.xn--p1ai')).toBe(true); // punycode
    });

    it('should handle IPv4 addresses', () => {
      expect(isValidUrl('http://192.168.1.1:8080')).toBe(true);
      expect(isValidUrl('https://10.0.0.1/api')).toBe(true);
    });

    it('should handle IPv6 addresses', () => {
      expect(isValidUrl('http://[::1]:8080')).toBe(true);
      expect(isValidUrl('https://[2001:db8::1]/path')).toBe(true);
    });
  });

  describe('sanitizeUrl', () => {
    it('should return valid https URL unchanged', () => {
      const url = 'https://example.com/image.png';
      expect(sanitizeUrl(url)).toBe(url);
    });

    it('should return valid http URL unchanged', () => {
      const url = 'http://localhost:3000/api';
      expect(sanitizeUrl(url)).toBe(url);
    });

    it('should return empty string for javascript: URL', () => {
      expect(sanitizeUrl('javascript:alert(1)')).toBe('');
    });

    it('should return empty string for data: URL', () => {
      expect(sanitizeUrl('data:text/html,<script>evil</script>')).toBe('');
    });

    it('should return empty string for invalid URL', () => {
      expect(sanitizeUrl('not-a-valid-url')).toBe('');
    });

    it('should return empty string for empty string', () => {
      expect(sanitizeUrl('')).toBe('');
    });

    it('should return empty string for relative URL', () => {
      expect(sanitizeUrl('/path/to/resource')).toBe('');
    });

    it('should handle URLs with query parameters', () => {
      const url = 'https://example.com/search?q=test&page=1';
      expect(sanitizeUrl(url)).toBe(url);
    });

    it('should handle URLs with fragments', () => {
      const url = 'https://example.com/page#section-1';
      expect(sanitizeUrl(url)).toBe(url);
    });

    it('should return empty string for file: URL', () => {
      expect(sanitizeUrl('file:///etc/passwd')).toBe('');
    });

    it('should return empty string for ftp: URL', () => {
      expect(sanitizeUrl('ftp://ftp.example.com/file.txt')).toBe('');
    });
  });
});
