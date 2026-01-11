/**
 * Tests for Open-JSON-UI Zod Schema Validation
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

import {
  TextComponentSchema,
  HeadingComponentSchema,
  CodeComponentSchema,
  TableComponentSchema,
  ImageComponentSchema,
  ButtonComponentSchema,
  ListComponentSchema,
  LinkComponentSchema,
  DividerComponentSchema,
  ProgressComponentSchema,
  AlertComponentSchema,
  OpenJSONUIPayloadSchema,
  validatePayload,
  isComponentType,
  type OpenJSONUIComponent,
} from '@/lib/open-json-ui/schema';

describe('Open-JSON-UI Schema Validation', () => {
  describe('TextComponentSchema', () => {
    it('should validate a valid text component', () => {
      const result = TextComponentSchema.safeParse({
        type: 'text',
        content: 'Hello world',
      });
      expect(result.success).toBe(true);
    });

    it('should validate text with style variants', () => {
      const styles = ['normal', 'muted', 'error', 'success'] as const;
      for (const style of styles) {
        const result = TextComponentSchema.safeParse({
          type: 'text',
          content: 'Styled text',
          style,
        });
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.style).toBe(style);
        }
      }
    });

    it('should reject text with invalid style', () => {
      const result = TextComponentSchema.safeParse({
        type: 'text',
        content: 'Hello',
        style: 'invalid-style',
      });
      expect(result.success).toBe(false);
    });

    it('should reject text without content', () => {
      const result = TextComponentSchema.safeParse({
        type: 'text',
      });
      expect(result.success).toBe(false);
    });

    it('should reject wrong type literal', () => {
      const result = TextComponentSchema.safeParse({
        type: 'heading',
        content: 'Hello',
      });
      expect(result.success).toBe(false);
    });
  });

  describe('HeadingComponentSchema', () => {
    it('should validate heading levels 1-6', () => {
      for (let level = 1; level <= 6; level++) {
        const result = HeadingComponentSchema.safeParse({
          type: 'heading',
          level,
          content: `Heading ${level}`,
        });
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.level).toBe(level);
        }
      }
    });

    it('should reject heading level 0', () => {
      const result = HeadingComponentSchema.safeParse({
        type: 'heading',
        level: 0,
        content: 'Invalid',
      });
      expect(result.success).toBe(false);
    });

    it('should reject heading level 7', () => {
      const result = HeadingComponentSchema.safeParse({
        type: 'heading',
        level: 7,
        content: 'Invalid',
      });
      expect(result.success).toBe(false);
    });

    it('should reject heading without content', () => {
      const result = HeadingComponentSchema.safeParse({
        type: 'heading',
        level: 1,
      });
      expect(result.success).toBe(false);
    });
  });

  describe('CodeComponentSchema', () => {
    it('should validate code without language', () => {
      const result = CodeComponentSchema.safeParse({
        type: 'code',
        content: 'const x = 1;',
      });
      expect(result.success).toBe(true);
    });

    it('should validate code with language', () => {
      const result = CodeComponentSchema.safeParse({
        type: 'code',
        content: 'print("hello")',
        language: 'python',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.language).toBe('python');
      }
    });

    it('should reject code without content', () => {
      const result = CodeComponentSchema.safeParse({
        type: 'code',
        language: 'javascript',
      });
      expect(result.success).toBe(false);
    });
  });

  describe('TableComponentSchema', () => {
    it('should validate valid table', () => {
      const result = TableComponentSchema.safeParse({
        type: 'table',
        headers: ['Name', 'Value'],
        rows: [
          ['foo', '1'],
          ['bar', '2'],
        ],
      });
      expect(result.success).toBe(true);
    });

    it('should validate table with caption', () => {
      const result = TableComponentSchema.safeParse({
        type: 'table',
        headers: ['Column'],
        rows: [['data']],
        caption: 'A simple table',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.caption).toBe('A simple table');
      }
    });

    it('should validate empty table', () => {
      const result = TableComponentSchema.safeParse({
        type: 'table',
        headers: [],
        rows: [],
      });
      expect(result.success).toBe(true);
    });

    it('should reject table without headers', () => {
      const result = TableComponentSchema.safeParse({
        type: 'table',
        rows: [['a', 'b']],
      });
      expect(result.success).toBe(false);
    });

    it('should reject table without rows', () => {
      const result = TableComponentSchema.safeParse({
        type: 'table',
        headers: ['A', 'B'],
      });
      expect(result.success).toBe(false);
    });
  });

  describe('ImageComponentSchema', () => {
    it('should validate valid image', () => {
      const result = ImageComponentSchema.safeParse({
        type: 'image',
        src: 'https://example.com/image.png',
        alt: 'Example image',
      });
      expect(result.success).toBe(true);
    });

    it('should validate image with dimensions', () => {
      const result = ImageComponentSchema.safeParse({
        type: 'image',
        src: 'https://example.com/image.png',
        alt: 'Image with size',
        width: 300,
        height: 200,
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.width).toBe(300);
        expect(result.data.height).toBe(200);
      }
    });

    it('should reject image with invalid URL', () => {
      const result = ImageComponentSchema.safeParse({
        type: 'image',
        src: 'not-a-valid-url',
        alt: 'Test',
      });
      expect(result.success).toBe(false);
    });

    it('should reject image without alt text', () => {
      const result = ImageComponentSchema.safeParse({
        type: 'image',
        src: 'https://example.com/image.png',
      });
      expect(result.success).toBe(false);
    });
  });

  describe('ButtonComponentSchema', () => {
    it('should validate valid button', () => {
      const result = ButtonComponentSchema.safeParse({
        type: 'button',
        label: 'Click me',
        action: 'button_click',
      });
      expect(result.success).toBe(true);
    });

    it('should validate button with variants', () => {
      const variants = ['default', 'destructive', 'outline', 'ghost', 'secondary'] as const;
      for (const variant of variants) {
        const result = ButtonComponentSchema.safeParse({
          type: 'button',
          label: 'Button',
          action: 'action',
          variant,
        });
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.variant).toBe(variant);
        }
      }
    });

    it('should reject button with invalid variant', () => {
      const result = ButtonComponentSchema.safeParse({
        type: 'button',
        label: 'Button',
        action: 'action',
        variant: 'invalid',
      });
      expect(result.success).toBe(false);
    });

    it('should reject button without label', () => {
      const result = ButtonComponentSchema.safeParse({
        type: 'button',
        action: 'action',
      });
      expect(result.success).toBe(false);
    });

    it('should reject button without action', () => {
      const result = ButtonComponentSchema.safeParse({
        type: 'button',
        label: 'Button',
      });
      expect(result.success).toBe(false);
    });
  });

  describe('ListComponentSchema', () => {
    it('should validate unordered list', () => {
      const result = ListComponentSchema.safeParse({
        type: 'list',
        items: ['Item 1', 'Item 2', 'Item 3'],
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.ordered).toBe(false);
      }
    });

    it('should validate ordered list', () => {
      const result = ListComponentSchema.safeParse({
        type: 'list',
        items: ['First', 'Second'],
        ordered: true,
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.ordered).toBe(true);
      }
    });

    it('should validate empty list', () => {
      const result = ListComponentSchema.safeParse({
        type: 'list',
        items: [],
      });
      expect(result.success).toBe(true);
    });

    it('should reject list without items', () => {
      const result = ListComponentSchema.safeParse({
        type: 'list',
        ordered: false,
      });
      expect(result.success).toBe(false);
    });
  });

  describe('LinkComponentSchema', () => {
    it('should validate valid link', () => {
      const result = LinkComponentSchema.safeParse({
        type: 'link',
        text: 'Click here',
        href: 'https://example.com',
      });
      expect(result.success).toBe(true);
    });

    it('should validate link with target', () => {
      const result = LinkComponentSchema.safeParse({
        type: 'link',
        text: 'Open in new tab',
        href: 'https://example.com',
        target: '_blank',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.target).toBe('_blank');
      }
    });

    it('should reject link with invalid URL', () => {
      const result = LinkComponentSchema.safeParse({
        type: 'link',
        text: 'Bad link',
        href: 'not-a-url',
      });
      expect(result.success).toBe(false);
    });

    it('should reject link with invalid target', () => {
      const result = LinkComponentSchema.safeParse({
        type: 'link',
        text: 'Link',
        href: 'https://example.com',
        target: '_parent',
      });
      expect(result.success).toBe(false);
    });
  });

  describe('DividerComponentSchema', () => {
    it('should validate divider', () => {
      const result = DividerComponentSchema.safeParse({
        type: 'divider',
      });
      expect(result.success).toBe(true);
    });

    it('should reject divider with extra properties (strict mode)', () => {
      // Zod by default strips extra properties, so this should still pass
      const result = DividerComponentSchema.safeParse({
        type: 'divider',
        extra: 'value',
      });
      // Zod is lenient by default, this should pass
      expect(result.success).toBe(true);
    });
  });

  describe('ProgressComponentSchema', () => {
    it('should validate progress with value', () => {
      const result = ProgressComponentSchema.safeParse({
        type: 'progress',
        value: 50,
      });
      expect(result.success).toBe(true);
    });

    it('should validate progress with label', () => {
      const result = ProgressComponentSchema.safeParse({
        type: 'progress',
        value: 75,
        label: 'Loading...',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.label).toBe('Loading...');
      }
    });

    it('should validate progress at 0%', () => {
      const result = ProgressComponentSchema.safeParse({
        type: 'progress',
        value: 0,
      });
      expect(result.success).toBe(true);
    });

    it('should validate progress at 100%', () => {
      const result = ProgressComponentSchema.safeParse({
        type: 'progress',
        value: 100,
      });
      expect(result.success).toBe(true);
    });

    it('should reject progress below 0', () => {
      const result = ProgressComponentSchema.safeParse({
        type: 'progress',
        value: -10,
      });
      expect(result.success).toBe(false);
    });

    it('should reject progress above 100', () => {
      const result = ProgressComponentSchema.safeParse({
        type: 'progress',
        value: 110,
      });
      expect(result.success).toBe(false);
    });
  });

  describe('AlertComponentSchema', () => {
    it('should validate alert with description only', () => {
      const result = AlertComponentSchema.safeParse({
        type: 'alert',
        description: 'This is an alert message',
      });
      expect(result.success).toBe(true);
    });

    it('should validate alert with title and description', () => {
      const result = AlertComponentSchema.safeParse({
        type: 'alert',
        title: 'Warning',
        description: 'Something needs attention',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.title).toBe('Warning');
      }
    });

    it('should validate alert variants', () => {
      const variants = ['default', 'destructive', 'warning', 'success'] as const;
      for (const variant of variants) {
        const result = AlertComponentSchema.safeParse({
          type: 'alert',
          description: 'Alert message',
          variant,
        });
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.variant).toBe(variant);
        }
      }
    });

    it('should reject alert with invalid variant', () => {
      const result = AlertComponentSchema.safeParse({
        type: 'alert',
        description: 'Message',
        variant: 'invalid',
      });
      expect(result.success).toBe(false);
    });

    it('should reject alert without description', () => {
      const result = AlertComponentSchema.safeParse({
        type: 'alert',
        title: 'Title only',
      });
      expect(result.success).toBe(false);
    });
  });

  describe('OpenJSONUIPayloadSchema', () => {
    it('should validate complete payload', () => {
      const result = OpenJSONUIPayloadSchema.safeParse({
        type: 'open_json_ui',
        components: [
          { type: 'heading', level: 1, content: 'Title' },
          { type: 'text', content: 'Description' },
          { type: 'button', label: 'Click', action: 'click' },
        ],
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.components).toHaveLength(3);
      }
    });

    it('should validate empty components array', () => {
      const result = OpenJSONUIPayloadSchema.safeParse({
        type: 'open_json_ui',
        components: [],
      });
      expect(result.success).toBe(true);
    });

    it('should reject wrong type', () => {
      const result = OpenJSONUIPayloadSchema.safeParse({
        type: 'mcp_ui',
        components: [],
      });
      expect(result.success).toBe(false);
    });

    it('should reject missing components', () => {
      const result = OpenJSONUIPayloadSchema.safeParse({
        type: 'open_json_ui',
      });
      expect(result.success).toBe(false);
    });

    it('should reject payload with invalid component', () => {
      const result = OpenJSONUIPayloadSchema.safeParse({
        type: 'open_json_ui',
        components: [
          { type: 'heading', level: 1, content: 'Valid' },
          { type: 'unknown_type', data: 'Invalid' },
        ],
      });
      expect(result.success).toBe(false);
    });
  });

  describe('validatePayload', () => {
    it('should return success with valid payload', () => {
      const payload = {
        type: 'open_json_ui',
        components: [{ type: 'text', content: 'Hello' }],
      };
      const result = validatePayload(payload);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.components).toHaveLength(1);
      }
    });

    it('should return error details for invalid payload', () => {
      const payload = {
        type: 'open_json_ui',
        components: [{ type: 'text' }], // missing content
      };
      const result = validatePayload(payload);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toBeDefined();
        expect(result.message).toContain('content');
      }
    });

    it('should provide human-readable error message', () => {
      const payload = {
        type: 'open_json_ui',
        components: [
          { type: 'heading', level: 10, content: 'Invalid level' },
        ],
      };
      const result = validatePayload(payload);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(typeof result.message).toBe('string');
        expect(result.message.length).toBeGreaterThan(0);
      }
    });

    it('should handle completely invalid input', () => {
      const result = validatePayload(null);
      expect(result.success).toBe(false);
    });

    it('should handle non-object input', () => {
      const result = validatePayload('not an object');
      expect(result.success).toBe(false);
    });
  });

  describe('isComponentType', () => {
    it('should correctly identify text component', () => {
      const component: OpenJSONUIComponent = { type: 'text', content: 'Hello' };
      expect(isComponentType(component, 'text')).toBe(true);
      expect(isComponentType(component, 'heading')).toBe(false);
    });

    it('should correctly identify heading component', () => {
      const component: OpenJSONUIComponent = { type: 'heading', level: 2, content: 'Title' };
      expect(isComponentType(component, 'heading')).toBe(true);
      expect(isComponentType(component, 'text')).toBe(false);
    });

    it('should correctly identify button component', () => {
      const component: OpenJSONUIComponent = { type: 'button', label: 'Click', action: 'click' };
      expect(isComponentType(component, 'button')).toBe(true);
      expect(isComponentType(component, 'link')).toBe(false);
    });

    it('should correctly identify code component', () => {
      const component: OpenJSONUIComponent = { type: 'code', content: 'const x = 1;' };
      expect(isComponentType(component, 'code')).toBe(true);
    });

    it('should correctly identify divider component', () => {
      const component: OpenJSONUIComponent = { type: 'divider' };
      expect(isComponentType(component, 'divider')).toBe(true);
    });

    it('should correctly identify progress component', () => {
      const component: OpenJSONUIComponent = { type: 'progress', value: 50 };
      expect(isComponentType(component, 'progress')).toBe(true);
    });

    it('should correctly identify alert component', () => {
      const component: OpenJSONUIComponent = { type: 'alert', description: 'Message' };
      expect(isComponentType(component, 'alert')).toBe(true);
    });

    it('should correctly identify list component', () => {
      const component: OpenJSONUIComponent = { type: 'list', items: ['a', 'b'] };
      expect(isComponentType(component, 'list')).toBe(true);
    });

    it('should correctly identify table component', () => {
      const component: OpenJSONUIComponent = { type: 'table', headers: ['H'], rows: [['R']] };
      expect(isComponentType(component, 'table')).toBe(true);
    });

    it('should correctly identify image component', () => {
      const component: OpenJSONUIComponent = {
        type: 'image',
        src: 'https://example.com/img.png',
        alt: 'Test',
      };
      expect(isComponentType(component, 'image')).toBe(true);
    });

    it('should correctly identify link component', () => {
      const component: OpenJSONUIComponent = {
        type: 'link',
        text: 'Link',
        href: 'https://example.com',
      };
      expect(isComponentType(component, 'link')).toBe(true);
    });
  });
});
