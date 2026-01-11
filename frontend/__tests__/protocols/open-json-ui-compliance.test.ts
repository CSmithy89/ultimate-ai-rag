/**
 * Open-JSON-UI Protocol Compliance Tests
 *
 * Story 22-D2: Implement Protocol Compliance Tests
 *
 * Verifies Open-JSON-UI protocol implementation:
 * - Component schema validation
 * - Payload structure compliance
 * - Sanitization requirements
 */

import { describe, it, expect } from "@jest/globals";

import {
  OpenJSONUIPayloadSchema,
  OpenJSONUIComponentSchema,
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
  validatePayload,
  isComponentType,
} from "@/lib/open-json-ui/schema";

// =============================================================================
// Component Schema Compliance Tests
// =============================================================================

describe("Open-JSON-UI Component Schema Compliance", () => {
  describe("TextComponent", () => {
    it("accepts valid text component", () => {
      const component = { type: "text", content: "Hello" };
      const result = TextComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts text with style", () => {
      const styles = ["normal", "muted", "error", "success"];
      for (const style of styles) {
        const component = { type: "text", content: "Hello", style };
        const result = TextComponentSchema.safeParse(component);
        expect(result.success).toBe(true);
      }
    });

    it("rejects invalid style", () => {
      const component = { type: "text", content: "Hello", style: "bold" };
      const result = TextComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("requires content field", () => {
      const component = { type: "text" };
      const result = TextComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });
  });

  describe("HeadingComponent", () => {
    it("accepts valid heading levels 1-6", () => {
      for (let level = 1; level <= 6; level++) {
        const component = { type: "heading", level, content: "Title" };
        const result = HeadingComponentSchema.safeParse(component);
        expect(result.success).toBe(true);
      }
    });

    it("rejects heading level 0", () => {
      const component = { type: "heading", level: 0, content: "Title" };
      const result = HeadingComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("rejects heading level 7", () => {
      const component = { type: "heading", level: 7, content: "Title" };
      const result = HeadingComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("requires level and content", () => {
      expect(HeadingComponentSchema.safeParse({ type: "heading" }).success).toBe(false);
      expect(HeadingComponentSchema.safeParse({ type: "heading", level: 1 }).success).toBe(false);
      expect(HeadingComponentSchema.safeParse({ type: "heading", content: "Title" }).success).toBe(false);
    });
  });

  describe("CodeComponent", () => {
    it("accepts code without language", () => {
      const component = { type: "code", content: "const x = 1;" };
      const result = CodeComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts code with language", () => {
      const component = { type: "code", content: "const x = 1;", language: "typescript" };
      const result = CodeComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts empty code content", () => {
      const component = { type: "code", content: "" };
      const result = CodeComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });
  });

  describe("TableComponent", () => {
    it("accepts valid table", () => {
      const component = {
        type: "table",
        headers: ["Name", "Value"],
        rows: [["Item 1", "100"], ["Item 2", "200"]],
      };
      const result = TableComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts table with caption", () => {
      const component = {
        type: "table",
        headers: ["A", "B"],
        rows: [["1", "2"]],
        caption: "Test table",
      };
      const result = TableComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts empty table", () => {
      const component = { type: "table", headers: [], rows: [] };
      const result = TableComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("requires headers and rows", () => {
      expect(TableComponentSchema.safeParse({ type: "table" }).success).toBe(false);
      expect(TableComponentSchema.safeParse({ type: "table", headers: [] }).success).toBe(false);
    });
  });

  describe("ImageComponent", () => {
    it("accepts valid image", () => {
      const component = {
        type: "image",
        src: "https://example.com/image.png",
        alt: "Test image",
      };
      const result = ImageComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts image with dimensions", () => {
      const component = {
        type: "image",
        src: "https://example.com/image.png",
        alt: "Test",
        width: 400,
        height: 300,
      };
      const result = ImageComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("rejects invalid URL", () => {
      const component = { type: "image", src: "not-a-url", alt: "Test" };
      const result = ImageComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("requires src and alt", () => {
      expect(ImageComponentSchema.safeParse({ type: "image" }).success).toBe(false);
      expect(ImageComponentSchema.safeParse({ type: "image", src: "https://x.com/a.png" }).success).toBe(false);
    });
  });

  describe("ButtonComponent", () => {
    it("accepts valid button", () => {
      const component = { type: "button", label: "Click Me", action: "submit" };
      const result = ButtonComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts button with variant", () => {
      const variants = ["default", "destructive", "outline", "ghost", "secondary"];
      for (const variant of variants) {
        const component = { type: "button", label: "Click", action: "test", variant };
        const result = ButtonComponentSchema.safeParse(component);
        expect(result.success).toBe(true);
      }
    });

    it("rejects invalid variant", () => {
      const component = { type: "button", label: "Click", action: "test", variant: "primary" };
      const result = ButtonComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("requires label and action", () => {
      expect(ButtonComponentSchema.safeParse({ type: "button" }).success).toBe(false);
      expect(ButtonComponentSchema.safeParse({ type: "button", label: "Click" }).success).toBe(false);
    });
  });

  describe("ListComponent", () => {
    it("accepts ordered list", () => {
      const component = { type: "list", items: ["A", "B", "C"], ordered: true };
      const result = ListComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts unordered list", () => {
      const component = { type: "list", items: ["A", "B", "C"], ordered: false };
      const result = ListComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("defaults ordered to false", () => {
      const component = { type: "list", items: ["A", "B"] };
      const result = ListComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.ordered).toBe(false);
      }
    });

    it("accepts empty list", () => {
      const component = { type: "list", items: [] };
      const result = ListComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });
  });

  describe("LinkComponent", () => {
    it("accepts valid link", () => {
      const component = { type: "link", text: "Click here", href: "https://example.com" };
      const result = LinkComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts link with target", () => {
      const component = { type: "link", text: "Click", href: "https://example.com", target: "_blank" };
      const result = LinkComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("rejects invalid URL", () => {
      const component = { type: "link", text: "Click", href: "not-a-url" };
      const result = LinkComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("rejects invalid target", () => {
      const component = { type: "link", text: "Click", href: "https://x.com", target: "_parent" };
      const result = LinkComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });
  });

  describe("DividerComponent", () => {
    it("accepts divider", () => {
      const component = { type: "divider" };
      const result = DividerComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });
  });

  describe("ProgressComponent", () => {
    it("accepts valid progress", () => {
      const component = { type: "progress", value: 50 };
      const result = ProgressComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts progress with label", () => {
      const component = { type: "progress", value: 75, label: "Loading..." };
      const result = ProgressComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts boundary values 0 and 100", () => {
      expect(ProgressComponentSchema.safeParse({ type: "progress", value: 0 }).success).toBe(true);
      expect(ProgressComponentSchema.safeParse({ type: "progress", value: 100 }).success).toBe(true);
    });

    it("rejects values outside 0-100", () => {
      expect(ProgressComponentSchema.safeParse({ type: "progress", value: -1 }).success).toBe(false);
      expect(ProgressComponentSchema.safeParse({ type: "progress", value: 101 }).success).toBe(false);
    });
  });

  describe("AlertComponent", () => {
    it("accepts valid alert", () => {
      const component = { type: "alert", description: "This is an alert" };
      const result = AlertComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts alert with title", () => {
      const component = { type: "alert", title: "Warning", description: "Message" };
      const result = AlertComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    });

    it("accepts alert with variant", () => {
      const variants = ["default", "destructive", "warning", "success"];
      for (const variant of variants) {
        const component = { type: "alert", description: "Test", variant };
        const result = AlertComponentSchema.safeParse(component);
        expect(result.success).toBe(true);
      }
    });

    it("rejects invalid variant", () => {
      const component = { type: "alert", description: "Test", variant: "info" };
      const result = AlertComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });

    it("requires description", () => {
      const component = { type: "alert" };
      const result = AlertComponentSchema.safeParse(component);
      expect(result.success).toBe(false);
    });
  });
});

// =============================================================================
// Payload Structure Compliance Tests
// =============================================================================

describe("Open-JSON-UI Payload Structure Compliance", () => {
  it("accepts valid payload with components", () => {
    const payload = {
      type: "open_json_ui",
      components: [
        { type: "heading", level: 1, content: "Title" },
        { type: "text", content: "Hello" },
      ],
    };
    const result = OpenJSONUIPayloadSchema.safeParse(payload);
    expect(result.success).toBe(true);
  });

  it("accepts empty components array", () => {
    const payload = { type: "open_json_ui", components: [] };
    const result = OpenJSONUIPayloadSchema.safeParse(payload);
    expect(result.success).toBe(true);
  });

  it("requires type to be open_json_ui", () => {
    const payload = { type: "other_type", components: [] };
    const result = OpenJSONUIPayloadSchema.safeParse(payload);
    expect(result.success).toBe(false);
  });

  it("requires components array", () => {
    const payload = { type: "open_json_ui" };
    const result = OpenJSONUIPayloadSchema.safeParse(payload);
    expect(result.success).toBe(false);
  });

  it("rejects invalid component in array", () => {
    const payload = {
      type: "open_json_ui",
      components: [
        { type: "text", content: "Valid" },
        { type: "invalid", data: "test" },
      ],
    };
    const result = OpenJSONUIPayloadSchema.safeParse(payload);
    expect(result.success).toBe(false);
  });

  it("validates all components in array", () => {
    const payload = {
      type: "open_json_ui",
      components: [
        { type: "text" },  // Missing content
      ],
    };
    const result = OpenJSONUIPayloadSchema.safeParse(payload);
    expect(result.success).toBe(false);
  });
});

// =============================================================================
// Discriminated Union Compliance Tests
// =============================================================================

describe("Open-JSON-UI Discriminated Union Compliance", () => {
  it("distinguishes components by type field", () => {
    const textComponent = { type: "text", content: "Hello" };
    const buttonComponent = { type: "button", label: "Click", action: "test" };

    const textResult = OpenJSONUIComponentSchema.safeParse(textComponent);
    const buttonResult = OpenJSONUIComponentSchema.safeParse(buttonComponent);

    expect(textResult.success).toBe(true);
    expect(buttonResult.success).toBe(true);

    if (textResult.success && buttonResult.success) {
      expect(textResult.data.type).toBe("text");
      expect(buttonResult.data.type).toBe("button");
    }
  });

  it("supports all 11 component types", () => {
    const components = [
      { type: "text", content: "Hello" },
      { type: "heading", level: 1, content: "Title" },
      { type: "code", content: "const x = 1;" },
      { type: "table", headers: ["A"], rows: [["1"]] },
      { type: "image", src: "https://example.com/img.png", alt: "Image" },
      { type: "button", label: "Click", action: "test" },
      { type: "list", items: ["A", "B"] },
      { type: "link", text: "Link", href: "https://example.com" },
      { type: "divider" },
      { type: "progress", value: 50 },
      { type: "alert", description: "Alert" },
    ];

    for (const component of components) {
      const result = OpenJSONUIComponentSchema.safeParse(component);
      expect(result.success).toBe(true);
    }
  });
});

// =============================================================================
// Validation Helper Compliance Tests
// =============================================================================

describe("Open-JSON-UI Validation Helper Compliance", () => {
  it("returns success: true for valid payload", () => {
    const payload = {
      type: "open_json_ui",
      components: [{ type: "text", content: "Hello" }],
    };
    const result = validatePayload(payload);
    expect(result.success).toBe(true);
  });

  it("returns success: false with error message for invalid payload", () => {
    const payload = { type: "wrong" };
    const result = validatePayload(payload);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.message).toBeTruthy();
      expect(result.error).toBeDefined();
    }
  });

  it("provides descriptive error path", () => {
    const payload = {
      type: "open_json_ui",
      components: [{ type: "progress", value: 150 }],  // Invalid value
    };
    const result = validatePayload(payload);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.message).toContain("components");
    }
  });
});

// =============================================================================
// Type Guard Compliance Tests
// =============================================================================

describe("Open-JSON-UI Type Guard Compliance", () => {
  it("correctly identifies component types", () => {
    const textComponent = { type: "text" as const, content: "Hello" };
    const buttonComponent = { type: "button" as const, label: "Click", action: "test" };

    expect(isComponentType(textComponent, "text")).toBe(true);
    expect(isComponentType(textComponent, "button")).toBe(false);
    expect(isComponentType(buttonComponent, "button")).toBe(true);
    expect(isComponentType(buttonComponent, "text")).toBe(false);
  });
});
