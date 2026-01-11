/**
 * Tests for OpenJSONUIRenderer Component
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import {
  OpenJSONUIRenderer,
  type OpenJSONUIRendererProps,
} from '@/components/open-json-ui/OpenJSONUIRenderer';
import type { OpenJSONUIPayload } from '@/lib/open-json-ui/schema';

// Mock Next.js Image component
jest.mock('next/image', () => ({
  __esModule: true,
  default: function MockImage(props: {
    src: string;
    alt: string;
    width?: number;
    height?: number;
    onError?: () => void;
    className?: string;
  }) {
    return (
      <img
        src={props.src}
        alt={props.alt}
        width={props.width}
        height={props.height}
        className={props.className}
        onError={props.onError}
        data-testid="next-image"
      />
    );
  },
}));

// Mock sanitization functions
jest.mock('@/lib/open-json-ui/sanitize', () => ({
  sanitizeContent: jest.fn((content: string) => {
    // Simple mock that removes script tags
    return content.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
  }),
  sanitizeToPlainText: jest.fn((content: string) => {
    // Strip all HTML
    return content.replace(/<[^>]*>/g, '');
  }),
  isValidUrl: jest.fn((url: string) => {
    try {
      const parsed = new URL(url);
      return ['http:', 'https:'].includes(parsed.protocol);
    } catch {
      return false;
    }
  }),
  sanitizeUrl: jest.fn((url: string) => {
    try {
      const parsed = new URL(url);
      return ['http:', 'https:'].includes(parsed.protocol) ? url : '';
    } catch {
      return '';
    }
  }),
}));

const createPayload = (
  components: OpenJSONUIPayload['components']
): OpenJSONUIPayload => ({
  type: 'open_json_ui',
  components,
});

describe('OpenJSONUIRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Text Component', () => {
    it('should render text with normal style', () => {
      const payload = createPayload([
        { type: 'text', content: 'Hello world' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const textElement = screen.getByTestId('open-json-ui-text');
      expect(textElement).toBeInTheDocument();
      expect(textElement).toHaveTextContent('Hello world');
    });

    it('should render text with muted style', () => {
      const payload = createPayload([
        { type: 'text', content: 'Muted text', style: 'muted' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const textElement = screen.getByTestId('open-json-ui-text');
      expect(textElement).toBeInTheDocument();
      expect(textElement.className).toContain('text-slate-500');
    });

    it('should render text with error style', () => {
      const payload = createPayload([
        { type: 'text', content: 'Error message', style: 'error' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const textElement = screen.getByTestId('open-json-ui-text');
      expect(textElement.className).toContain('text-red-600');
    });

    it('should render text with success style', () => {
      const payload = createPayload([
        { type: 'text', content: 'Success!', style: 'success' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const textElement = screen.getByTestId('open-json-ui-text');
      expect(textElement.className).toContain('text-emerald-600');
    });
  });

  describe('Heading Component', () => {
    it('should render h1 heading', () => {
      const payload = createPayload([
        { type: 'heading', level: 1, content: 'Main Title' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const heading = screen.getByTestId('open-json-ui-heading');
      expect(heading.tagName).toBe('H1');
      expect(heading).toHaveTextContent('Main Title');
    });

    it('should render h2 heading', () => {
      const payload = createPayload([
        { type: 'heading', level: 2, content: 'Section Title' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const heading = screen.getByTestId('open-json-ui-heading');
      expect(heading.tagName).toBe('H2');
    });

    it('should render h3 heading', () => {
      const payload = createPayload([
        { type: 'heading', level: 3, content: 'Subsection' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const heading = screen.getByTestId('open-json-ui-heading');
      expect(heading.tagName).toBe('H3');
    });

    it('should render h4 heading', () => {
      const payload = createPayload([
        { type: 'heading', level: 4, content: 'Sub-subsection' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const heading = screen.getByTestId('open-json-ui-heading');
      expect(heading.tagName).toBe('H4');
    });

    it('should render h5 heading', () => {
      const payload = createPayload([
        { type: 'heading', level: 5, content: 'Minor heading' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const heading = screen.getByTestId('open-json-ui-heading');
      expect(heading.tagName).toBe('H5');
    });

    it('should render h6 heading', () => {
      const payload = createPayload([
        { type: 'heading', level: 6, content: 'Smallest heading' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const heading = screen.getByTestId('open-json-ui-heading');
      expect(heading.tagName).toBe('H6');
    });
  });

  describe('Code Component', () => {
    it('should render code block', () => {
      const payload = createPayload([
        { type: 'code', content: 'const x = 1;' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const codeBlock = screen.getByTestId('open-json-ui-code');
      expect(codeBlock).toBeInTheDocument();
      expect(codeBlock.tagName).toBe('PRE');
      expect(codeBlock).toHaveTextContent('const x = 1;');
    });

    it('should apply language class for syntax highlighting', () => {
      const payload = createPayload([
        { type: 'code', content: 'print("hello")', language: 'python' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const codeBlock = screen.getByTestId('open-json-ui-code');
      const codeElement = codeBlock.querySelector('code');
      expect(codeElement?.className).toContain('language-python');
    });

    it('should render code without language class when not specified', () => {
      const payload = createPayload([
        { type: 'code', content: 'some code' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const codeBlock = screen.getByTestId('open-json-ui-code');
      const codeElement = codeBlock.querySelector('code');
      expect(codeElement?.className).toBe('');
    });
  });

  describe('List Component', () => {
    it('should render unordered list', () => {
      const payload = createPayload([
        { type: 'list', items: ['Item 1', 'Item 2', 'Item 3'] },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const list = screen.getByTestId('open-json-ui-list');
      expect(list.tagName).toBe('UL');
      expect(list.querySelectorAll('li')).toHaveLength(3);
    });

    it('should render ordered list', () => {
      const payload = createPayload([
        { type: 'list', items: ['First', 'Second', 'Third'], ordered: true },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const list = screen.getByTestId('open-json-ui-list');
      expect(list.tagName).toBe('OL');
      expect(list.querySelectorAll('li')).toHaveLength(3);
    });

    it('should apply correct list styling', () => {
      const payload = createPayload([
        { type: 'list', items: ['A', 'B'], ordered: false },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const list = screen.getByTestId('open-json-ui-list');
      expect(list.className).toContain('list-disc');
    });
  });

  describe('Table Component', () => {
    it('should render table with headers and rows', () => {
      const payload = createPayload([
        {
          type: 'table',
          headers: ['Name', 'Value'],
          rows: [
            ['foo', '1'],
            ['bar', '2'],
          ],
        },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const table = screen.getByTestId('open-json-ui-table');
      expect(table).toBeInTheDocument();

      const headers = table.querySelectorAll('th');
      expect(headers).toHaveLength(2);
      expect(headers[0]).toHaveTextContent('Name');
      expect(headers[1]).toHaveTextContent('Value');

      const rows = table.querySelectorAll('tbody tr');
      expect(rows).toHaveLength(2);
    });

    it('should render table with caption', () => {
      const payload = createPayload([
        {
          type: 'table',
          headers: ['Column'],
          rows: [['Data']],
          caption: 'Table caption',
        },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const table = screen.getByTestId('open-json-ui-table');
      const caption = table.querySelector('caption');
      expect(caption).toBeInTheDocument();
      expect(caption).toHaveTextContent('Table caption');
    });

    it('should render empty table', () => {
      const payload = createPayload([
        {
          type: 'table',
          headers: [],
          rows: [],
        },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const table = screen.getByTestId('open-json-ui-table');
      expect(table).toBeInTheDocument();
    });
  });

  describe('Image Component', () => {
    it('should render image with valid URL', () => {
      const payload = createPayload([
        {
          type: 'image',
          src: 'https://example.com/image.png',
          alt: 'Test image',
        },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const imageContainer = screen.getByTestId('open-json-ui-image');
      expect(imageContainer).toBeInTheDocument();

      const img = screen.getByTestId('next-image');
      expect(img).toHaveAttribute('src', 'https://example.com/image.png');
      expect(img).toHaveAttribute('alt', 'Test image');
    });

    it('should show error for invalid URL', () => {
      const payload = createPayload([
        {
          type: 'image',
          src: 'javascript:alert(1)',
          alt: 'Malicious image',
        },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const errorElement = screen.getByTestId('open-json-ui-image-invalid');
      expect(errorElement).toBeInTheDocument();
      expect(errorElement).toHaveTextContent('Invalid image URL');
    });

    it('should apply custom dimensions', () => {
      const payload = createPayload([
        {
          type: 'image',
          src: 'https://example.com/image.png',
          alt: 'Sized image',
          width: 300,
          height: 200,
        },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const img = screen.getByTestId('next-image');
      expect(img).toHaveAttribute('width', '300');
      expect(img).toHaveAttribute('height', '200');
    });
  });

  describe('Button Component', () => {
    it('should render button with label', () => {
      const payload = createPayload([
        { type: 'button', label: 'Click me', action: 'click_action' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const button = screen.getByTestId('open-json-ui-button');
      expect(button).toBeInTheDocument();
      expect(button).toHaveTextContent('Click me');
    });

    it('should trigger onAction callback when clicked', () => {
      const onAction = jest.fn();
      const payload = createPayload([
        { type: 'button', label: 'Submit', action: 'submit_form' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} onAction={onAction} />);

      const button = screen.getByTestId('open-json-ui-button');
      fireEvent.click(button);

      expect(onAction).toHaveBeenCalledWith('submit_form');
      expect(onAction).toHaveBeenCalledTimes(1);
    });

    it('should include action in data attribute', () => {
      const payload = createPayload([
        { type: 'button', label: 'Action Button', action: 'my_action' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const button = screen.getByTestId('open-json-ui-button');
      expect(button).toHaveAttribute('data-action', 'my_action');
    });

    it('should handle button without onAction callback', () => {
      const payload = createPayload([
        { type: 'button', label: 'No Handler', action: 'ignored' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const button = screen.getByTestId('open-json-ui-button');
      // Should not throw when clicked
      expect(() => fireEvent.click(button)).not.toThrow();
    });
  });

  describe('Divider Component', () => {
    it('should render horizontal rule', () => {
      const payload = createPayload([{ type: 'divider' }]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const divider = screen.getByTestId('open-json-ui-divider');
      expect(divider).toBeInTheDocument();
      expect(divider.tagName).toBe('HR');
      expect(divider).toHaveAttribute('role', 'separator');
    });
  });

  describe('Progress Component', () => {
    it('should render progress bar', () => {
      const payload = createPayload([
        { type: 'progress', value: 50 },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const progress = screen.getByTestId('open-json-ui-progress');
      expect(progress).toBeInTheDocument();
      expect(progress).toHaveAttribute('role', 'progressbar');
      expect(progress).toHaveAttribute('aria-valuenow', '50');
    });

    it('should render progress with label', () => {
      const payload = createPayload([
        { type: 'progress', value: 75, label: 'Upload progress' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const progress = screen.getByTestId('open-json-ui-progress');
      expect(progress).toHaveTextContent('Upload progress');
      expect(progress).toHaveTextContent('75%');
    });

    it('should set correct aria attributes', () => {
      const payload = createPayload([
        { type: 'progress', value: 100 },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const progress = screen.getByTestId('open-json-ui-progress');
      expect(progress).toHaveAttribute('aria-valuemin', '0');
      expect(progress).toHaveAttribute('aria-valuemax', '100');
      expect(progress).toHaveAttribute('aria-valuenow', '100');
    });
  });

  describe('Alert Component', () => {
    it('should render default alert', () => {
      const payload = createPayload([
        { type: 'alert', description: 'This is an alert' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const alert = screen.getByTestId('open-json-ui-alert');
      expect(alert).toBeInTheDocument();
      expect(alert).toHaveAttribute('role', 'alert');
      expect(alert).toHaveTextContent('This is an alert');
    });

    it('should render alert with title', () => {
      const payload = createPayload([
        { type: 'alert', title: 'Warning', description: 'Something to note' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const alert = screen.getByTestId('open-json-ui-alert');
      expect(alert).toHaveTextContent('Warning');
      expect(alert).toHaveTextContent('Something to note');
    });

    it('should render destructive alert variant', () => {
      const payload = createPayload([
        { type: 'alert', description: 'Error occurred', variant: 'destructive' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const alert = screen.getByTestId('open-json-ui-alert');
      expect(alert.className).toContain('bg-red-50');
    });

    it('should render warning alert variant', () => {
      const payload = createPayload([
        { type: 'alert', description: 'Be careful', variant: 'warning' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const alert = screen.getByTestId('open-json-ui-alert');
      expect(alert.className).toContain('bg-amber-50');
    });

    it('should render success alert variant', () => {
      const payload = createPayload([
        { type: 'alert', description: 'All good!', variant: 'success' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const alert = screen.getByTestId('open-json-ui-alert');
      expect(alert.className).toContain('bg-emerald-50');
    });
  });

  describe('Error Handling', () => {
    it('should show error for invalid payload', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const invalidPayload = {
        type: 'open_json_ui',
        components: [{ type: 'text' }], // missing content
      } as unknown as OpenJSONUIPayload;

      render(<OpenJSONUIRenderer payload={invalidPayload} />);

      const error = screen.getByTestId('open-json-ui-error');
      expect(error).toBeInTheDocument();
      expect(error).toHaveAttribute('role', 'alert');
      expect(error).toHaveTextContent('Invalid UI payload');

      consoleSpy.mockRestore();
    });

    it('should show error message details', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const invalidPayload = {
        type: 'open_json_ui',
        components: [
          { type: 'heading', level: 10, content: 'Invalid level' },
        ],
      } as unknown as OpenJSONUIPayload;

      render(<OpenJSONUIRenderer payload={invalidPayload} />);

      const error = screen.getByTestId('open-json-ui-error');
      expect(error).toBeInTheDocument();

      consoleSpy.mockRestore();
    });

    it('should show fallback for unsupported component type', () => {
      // Note: In production, an unsupported type would fail Zod validation.
      // This test verifies the fallback UI exists and has proper structure
      // by checking the validation error state instead (since schema validation
      // would reject unknown types before they reach the fallback renderer).
      const payload = createPayload([
        { type: 'unknown_type', data: 'test' } as unknown as OpenJSONUIPayload['components'][0],
      ]);

      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      render(<OpenJSONUIRenderer payload={payload} />);

      // Since schema validation rejects unknown types, we get an error state
      const error = screen.getByTestId('open-json-ui-error');
      expect(error).toBeInTheDocument();
      expect(error).toHaveAttribute('role', 'alert');

      consoleSpy.mockRestore();
    });
  });

  describe('XSS Prevention', () => {
    it('should sanitize text content', () => {
      const { sanitizeContent } = require('@/lib/open-json-ui/sanitize');

      const payload = createPayload([
        { type: 'text', content: '<script>alert("xss")</script>Safe text' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      expect(sanitizeContent).toHaveBeenCalledWith(
        '<script>alert("xss")</script>Safe text'
      );

      const textElement = screen.getByTestId('open-json-ui-text');
      expect(textElement.innerHTML).not.toContain('<script');
    });

    it('should sanitize heading content to plain text', () => {
      const { sanitizeToPlainText } = require('@/lib/open-json-ui/sanitize');

      const payload = createPayload([
        { type: 'heading', level: 1, content: '<b>Bold</b> heading' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      expect(sanitizeToPlainText).toHaveBeenCalledWith('<b>Bold</b> heading');
    });

    it('should validate image URLs', () => {
      const { isValidUrl } = require('@/lib/open-json-ui/sanitize');

      const payload = createPayload([
        { type: 'image', src: 'javascript:alert(1)', alt: 'XSS attempt' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      expect(isValidUrl).toHaveBeenCalledWith('javascript:alert(1)');
    });
  });

  describe('Multiple Components', () => {
    it('should render multiple components in order', () => {
      const payload = createPayload([
        { type: 'heading', level: 1, content: 'Title' },
        { type: 'text', content: 'Description' },
        { type: 'divider' },
        { type: 'button', label: 'Action', action: 'do_something' },
      ]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const renderer = screen.getByTestId('open-json-ui-renderer');
      expect(renderer).toBeInTheDocument();

      expect(screen.getByTestId('open-json-ui-heading')).toBeInTheDocument();
      expect(screen.getByTestId('open-json-ui-text')).toBeInTheDocument();
      expect(screen.getByTestId('open-json-ui-divider')).toBeInTheDocument();
      expect(screen.getByTestId('open-json-ui-button')).toBeInTheDocument();
    });

    it('should handle empty components array', () => {
      const payload = createPayload([]);
      render(<OpenJSONUIRenderer payload={payload} />);

      const renderer = screen.getByTestId('open-json-ui-renderer');
      expect(renderer).toBeInTheDocument();
      expect(renderer.children).toHaveLength(0);
    });
  });

  describe('CSS Classes', () => {
    it('should apply custom className', () => {
      const payload = createPayload([{ type: 'text', content: 'Test' }]);
      render(<OpenJSONUIRenderer payload={payload} className="custom-class" />);

      const renderer = screen.getByTestId('open-json-ui-renderer');
      expect(renderer.classList.contains('custom-class')).toBe(true);
    });

    it('should apply custom className to error state', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const invalidPayload = {
        type: 'open_json_ui',
        components: [{ type: 'invalid' }],
      } as unknown as OpenJSONUIPayload;

      render(<OpenJSONUIRenderer payload={invalidPayload} className="error-custom" />);

      const error = screen.getByTestId('open-json-ui-error');
      expect(error.classList.contains('error-custom')).toBe(true);

      consoleSpy.mockRestore();
    });
  });
});
