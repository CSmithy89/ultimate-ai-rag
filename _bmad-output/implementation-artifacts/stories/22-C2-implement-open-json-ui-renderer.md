# Story 22-C2: Implement Open-JSON-UI Renderer

Status: pending

Epic: 22 - Advanced Protocol Integration
Priority: P2 - MEDIUM
Story Points: 5
Owner: Frontend

## Story

As a **developer building cross-platform AI experiences**,
I want **to render OpenAI-style declarative UI payloads using a secure, schema-validated renderer**,
So that **agent responses can include rich, interactive UI components that work consistently across different platforms while preventing arbitrary code execution**.

## Background

Epic 22 extends protocol integration with cross-platform UI capabilities. Open-JSON-UI provides an OpenAI-style declarative approach to UI rendering, complementing the MCP-UI iframe-based approach (22-C1). This enables agents to return structured UI payloads that render consistently without requiring external iframe hosting.

### Open-JSON-UI Protocol Context

Since OpenAI does not provide a formal UI specification, this project defines an internal spec (version 1.0.0-internal) based on OpenAI patterns and shadcn/ui components:

- **Declarative Components**: JSON-defined UI elements that map to React components
- **Schema Validation**: Zod on frontend, Pydantic on backend for type safety
- **Security**: DOMPurify sanitization prevents XSS; no arbitrary code execution
- **Interactivity**: Button clicks and other actions trigger callbacks to the agent

### Component Types

| Type | Description | Mapping |
|------|-------------|---------|
| `text` | Plain text block with optional styling | `<p>` with className |
| `heading` | Heading levels h1-h6 | `<h1>`-`<h6>` |
| `code` | Syntax-highlighted code block | `<pre><code>` |
| `list` | Ordered or unordered list | `<ol>`/`<ul>` |
| `table` | Data table with headers and rows | shadcn Table |
| `image` | Image with alt text and dimensions | Next.js Image |
| `button` | Interactive action button | shadcn Button |
| `link` | Hyperlink with target control | `<a>` |
| `divider` | Horizontal rule separator | `<hr>` |
| `progress` | Progress bar indicator | shadcn Progress |
| `alert` | Alert/notification box | shadcn Alert |

### Related Prior Work

| Epic/Story | Relationship |
|------------|-------------|
| Epic 21-D: A2UI Widget Rendering | UI rendering patterns (completed) |
| Story 22-C1: MCP-UI Renderer | Iframe-based alternative (sibling story) |

## Acceptance Criteria

1. **Given** an Open-JSON-UI payload with a `text` component, **when** rendered by OpenJSONUIRenderer, **then** the text is displayed with the specified style (normal, muted, error, success).

2. **Given** an Open-JSON-UI payload with a `heading` component (levels 1-6), **when** rendered, **then** the correct heading element (`<h1>`-`<h6>`) is output.

3. **Given** an Open-JSON-UI payload with a `code` component including a language, **when** rendered, **then** the code block displays with syntax-appropriate formatting and the language class applied.

4. **Given** an Open-JSON-UI payload with a `table` component, **when** rendered, **then** a shadcn Table component displays with proper headers, rows, and optional caption.

5. **Given** an Open-JSON-UI payload with an `image` component, **when** rendered, **then** Next.js Image component displays with proper alt text, dimensions, and optimization.

6. **Given** an Open-JSON-UI payload with a `button` component, **when** the user clicks it, **then** the `onAction` callback is invoked with the button's action string.

7. **Given** an Open-JSON-UI payload with a `list` component (ordered or unordered), **when** rendered, **then** the correct list type (`<ol>` or `<ul>`) displays with all items.

8. **Given** an Open-JSON-UI payload with a `divider` component, **when** rendered, **then** a horizontal rule separator displays.

9. **Given** an Open-JSON-UI payload with a `progress` component, **when** rendered, **then** a shadcn Progress bar displays with the specified value (0-100).

10. **Given** an Open-JSON-UI payload with an `alert` component, **when** rendered, **then** a shadcn Alert displays with the specified variant (default, destructive, warning, success).

11. **Given** an Open-JSON-UI payload, **when** validated against the Zod schema, **then** invalid component types or missing required fields are rejected with descriptive errors.

12. **Given** text content containing HTML or script tags, **when** rendered, **then** DOMPurify sanitizes the content to prevent XSS attacks.

13. **Given** an Open-JSON-UI payload with an unsupported component type, **when** rendered, **then** a fallback message displays indicating the unsupported type.

14. **Given** all components, **when** rendered, **then** they meet WCAG 2.1 AA accessibility requirements (proper ARIA labels, keyboard navigation, focus management).

15. **Given** all components, **when** tests are run, **then** unit tests pass with >85% coverage.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Not Applicable** - Pure frontend rendering, no tenant data
- [x] Input validation / schema enforcement: **Addressed** - Zod schema validates all payloads
- [x] Tests (unit/integration): **Addressed** - Unit tests for each component type, integration tests for composite payloads
- [x] Error handling + logging: **Addressed** - Fallback rendering for unsupported types, console warnings for validation failures
- [x] Documentation updates: **Addressed** - JSDoc on all public components and types
- [x] Accessibility: **Addressed** - WCAG 2.1 AA compliance for all rendered components

## Security Checklist

- [ ] **DOMPurify sanitization**: All text content sanitized before rendering
- [ ] **No dangerouslySetInnerHTML without sanitization**: Never render raw HTML
- [ ] **URL validation**: Image src and link href validated against URL schema
- [ ] **No arbitrary code execution**: Button actions are string identifiers, not executable code
- [ ] **Content Security Policy compliant**: No inline scripts or styles that violate CSP
- [ ] **Zod schema validation**: All payloads validated before processing

## Tasks / Subtasks

- [ ] **Task 1: Define Open-JSON-UI Zod Schemas** (AC: 11, 12)
  - [ ] Create `frontend/lib/open-json-ui/schema.ts`
  - [ ] Define discriminated union schema for all component types
  - [ ] Define OpenJSONUIPayload wrapper schema
  - [ ] Add validation helper function with error messages

- [ ] **Task 2: Create DOMPurify Sanitization Utility** (AC: 12)
  - [ ] Create `frontend/lib/open-json-ui/sanitize.ts`
  - [ ] Configure DOMPurify with safe allowed tags
  - [ ] Export sanitizeContent function for text/HTML content

- [ ] **Task 3: Implement Text, Heading, Divider Components** (AC: 1, 2, 8)
  - [ ] Create `frontend/components/open-json-ui/TextComponent.tsx`
  - [ ] Create `frontend/components/open-json-ui/HeadingComponent.tsx`
  - [ ] Create `frontend/components/open-json-ui/DividerComponent.tsx`

- [ ] **Task 4: Implement Code Component** (AC: 3)
  - [ ] Create `frontend/components/open-json-ui/CodeComponent.tsx`
  - [ ] Apply language-specific CSS classes
  - [ ] Ensure proper overflow handling for long lines

- [ ] **Task 5: Implement List Component** (AC: 7)
  - [ ] Create `frontend/components/open-json-ui/ListComponent.tsx`
  - [ ] Support ordered and unordered variants
  - [ ] Sanitize list item content

- [ ] **Task 6: Implement Table Component** (AC: 4)
  - [ ] Create `frontend/components/open-json-ui/TableComponent.tsx`
  - [ ] Use shadcn Table, TableHeader, TableBody, TableRow, TableCell
  - [ ] Support optional caption

- [ ] **Task 7: Implement Image Component** (AC: 5)
  - [ ] Create `frontend/components/open-json-ui/ImageComponent.tsx`
  - [ ] Use Next.js Image with optimization
  - [ ] Validate image URL before rendering
  - [ ] Provide fallback for failed loads

- [ ] **Task 8: Implement Button Component** (AC: 6)
  - [ ] Create `frontend/components/open-json-ui/ButtonComponent.tsx`
  - [ ] Use shadcn Button with variant support
  - [ ] Wire onClick to onAction callback with action string

- [ ] **Task 9: Implement Progress and Alert Components** (AC: 9, 10)
  - [ ] Create `frontend/components/open-json-ui/ProgressComponent.tsx`
  - [ ] Create `frontend/components/open-json-ui/AlertComponent.tsx`
  - [ ] Use shadcn Progress and Alert components

- [ ] **Task 10: Create Main OpenJSONUIRenderer Component** (AC: 13, 14)
  - [ ] Create `frontend/components/open-json-ui/OpenJSONUIRenderer.tsx`
  - [ ] Map component types to React components
  - [ ] Implement fallback for unsupported types
  - [ ] Create index.ts barrel export

- [ ] **Task 11: Add Backend Open-JSON-UI Models** (AC: 11)
  - [ ] Create `backend/src/agentic_rag_backend/protocols/open_json_ui.py`
  - [ ] Define Pydantic models for each component type
  - [ ] Create factory function for building payloads

- [ ] **Task 12: Add Frontend Unit Tests** (AC: 15)
  - [ ] Create `frontend/__tests__/components/open-json-ui/` directory
  - [ ] Test each component type renders correctly
  - [ ] Test sanitization removes malicious content
  - [ ] Test button actions trigger callbacks
  - [ ] Test validation rejects invalid schemas

- [ ] **Task 13: Add Accessibility Testing** (AC: 14)
  - [ ] Add jest-axe tests for all components
  - [ ] Verify proper ARIA attributes
  - [ ] Test keyboard navigation for interactive elements

- [ ] **Task 14: Add Backend Unit Tests** (AC: 15)
  - [ ] Create `backend/tests/unit/protocols/test_open_json_ui.py`
  - [ ] Test Pydantic model validation
  - [ ] Test payload factory function

## Technical Notes

### Frontend Schema Structure

```typescript
// frontend/lib/open-json-ui/schema.ts
import { z } from "zod";

const TextComponentSchema = z.object({
  type: z.literal("text"),
  content: z.string(),
  style: z.enum(["normal", "muted", "error", "success"]).optional(),
});

const HeadingComponentSchema = z.object({
  type: z.literal("heading"),
  level: z.number().min(1).max(6),
  content: z.string(),
});

const CodeComponentSchema = z.object({
  type: z.literal("code"),
  content: z.string(),
  language: z.string().optional(),
});

const TableComponentSchema = z.object({
  type: z.literal("table"),
  headers: z.array(z.string()),
  rows: z.array(z.array(z.string())),
  caption: z.string().optional(),
});

const ImageComponentSchema = z.object({
  type: z.literal("image"),
  src: z.string().url(),
  alt: z.string(),
  width: z.number().optional(),
  height: z.number().optional(),
});

const ButtonComponentSchema = z.object({
  type: z.literal("button"),
  label: z.string(),
  action: z.string(),
  variant: z.enum(["default", "destructive", "outline", "ghost", "secondary"]).optional(),
});

const ListComponentSchema = z.object({
  type: z.literal("list"),
  items: z.array(z.string()),
  ordered: z.boolean().default(false),
});

const DividerComponentSchema = z.object({
  type: z.literal("divider"),
});

const ProgressComponentSchema = z.object({
  type: z.literal("progress"),
  value: z.number().min(0).max(100),
  label: z.string().optional(),
});

const AlertComponentSchema = z.object({
  type: z.literal("alert"),
  title: z.string().optional(),
  description: z.string(),
  variant: z.enum(["default", "destructive", "warning", "success"]).optional(),
});

export const OpenJSONUIComponentSchema = z.discriminatedUnion("type", [
  TextComponentSchema,
  HeadingComponentSchema,
  CodeComponentSchema,
  TableComponentSchema,
  ImageComponentSchema,
  ButtonComponentSchema,
  ListComponentSchema,
  DividerComponentSchema,
  ProgressComponentSchema,
  AlertComponentSchema,
]);

export const OpenJSONUIPayloadSchema = z.object({
  type: z.literal("open_json_ui"),
  components: z.array(OpenJSONUIComponentSchema),
});

export type OpenJSONUIComponent = z.infer<typeof OpenJSONUIComponentSchema>;
export type OpenJSONUIPayload = z.infer<typeof OpenJSONUIPayloadSchema>;
```

### Sanitization Utility

```typescript
// frontend/lib/open-json-ui/sanitize.ts
import DOMPurify from "dompurify";

const ALLOWED_TAGS = ["b", "i", "em", "strong", "code", "br", "span"];
const ALLOWED_ATTR: string[] = [];

export function sanitizeContent(content: string): string {
  return DOMPurify.sanitize(content, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
  });
}
```

### OpenJSONUIRenderer Component

```tsx
// frontend/components/open-json-ui/OpenJSONUIRenderer.tsx
interface OpenJSONUIRendererProps {
  payload: OpenJSONUIPayload;
  onAction?: (action: string) => void;
}

export function OpenJSONUIRenderer({ payload, onAction }: OpenJSONUIRendererProps) {
  // Validate payload
  const parsed = OpenJSONUIPayloadSchema.safeParse(payload);
  if (!parsed.success) {
    console.warn("Open-JSON-UI: Invalid payload", parsed.error);
    return <div className="text-destructive">Invalid UI payload</div>;
  }

  return (
    <div className="space-y-2 my-2">
      {parsed.data.components.map((component, idx) => (
        <OpenJSONUIComponent
          key={idx}
          component={component}
          onAction={onAction}
        />
      ))}
    </div>
  );
}
```

### Backend Pydantic Models

```python
# backend/src/agentic_rag_backend/protocols/open_json_ui.py
from pydantic import BaseModel
from typing import Literal, Any

class OpenJSONUIText(BaseModel):
    type: Literal["text"] = "text"
    content: str
    style: str | None = None

class OpenJSONUIButton(BaseModel):
    type: Literal["button"] = "button"
    label: str
    action: str
    variant: str | None = None

class OpenJSONUIProgress(BaseModel):
    type: Literal["progress"] = "progress"
    value: int  # 0-100
    label: str | None = None

class OpenJSONUIAlert(BaseModel):
    type: Literal["alert"] = "alert"
    description: str
    title: str | None = None
    variant: str | None = None

class OpenJSONUIPayload(BaseModel):
    type: str = "open_json_ui"
    components: list[dict[str, Any]]

def create_open_json_ui(components: list[BaseModel]) -> OpenJSONUIPayload:
    """Create Open-JSON-UI payload from component models."""
    return OpenJSONUIPayload(
        components=[c.model_dump() for c in components]
    )
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `frontend/lib/open-json-ui/schema.ts` | Create | Zod schemas for all component types |
| `frontend/lib/open-json-ui/sanitize.ts` | Create | DOMPurify sanitization utility |
| `frontend/lib/open-json-ui/index.ts` | Create | Barrel export |
| `frontend/components/open-json-ui/TextComponent.tsx` | Create | Text renderer |
| `frontend/components/open-json-ui/HeadingComponent.tsx` | Create | Heading renderer |
| `frontend/components/open-json-ui/CodeComponent.tsx` | Create | Code block renderer |
| `frontend/components/open-json-ui/ListComponent.tsx` | Create | List renderer |
| `frontend/components/open-json-ui/TableComponent.tsx` | Create | Table renderer |
| `frontend/components/open-json-ui/ImageComponent.tsx` | Create | Image renderer |
| `frontend/components/open-json-ui/ButtonComponent.tsx` | Create | Button renderer |
| `frontend/components/open-json-ui/DividerComponent.tsx` | Create | Divider renderer |
| `frontend/components/open-json-ui/ProgressComponent.tsx` | Create | Progress bar renderer |
| `frontend/components/open-json-ui/AlertComponent.tsx` | Create | Alert renderer |
| `frontend/components/open-json-ui/OpenJSONUIRenderer.tsx` | Create | Main renderer component |
| `frontend/components/open-json-ui/index.ts` | Create | Barrel export |
| `backend/src/agentic_rag_backend/protocols/open_json_ui.py` | Create | Pydantic models |
| `frontend/__tests__/components/open-json-ui/*.test.tsx` | Create | Component tests |
| `frontend/__tests__/lib/open-json-ui/*.test.ts` | Create | Schema and sanitization tests |
| `backend/tests/unit/protocols/test_open_json_ui.py` | Create | Backend model tests |
| `.env.example` | Modify | Add OPEN_JSON_UI_ENABLED variable |

### Dependencies

- `dompurify` - HTML sanitization (add to frontend)
- `@types/dompurify` - TypeScript types (add to frontend devDependencies)
- `zod` - Already installed, schema validation
- `pydantic` - Already installed, backend models
- `jest-axe` - Accessibility testing (add to frontend devDependencies)

### Environment Variables

```bash
# .env
OPEN_JSON_UI_ENABLED=true|false  # Feature flag for Open-JSON-UI rendering
```

## Dependencies

- **Epic 21-D completed** - A2UI widget rendering patterns for reference
- **Story 22-C1** - Can be developed in parallel; shares similar rendering concepts

## Definition of Done

- [ ] Zod schema defined for all 10+ component types
- [ ] DOMPurify sanitization utility created and tested
- [ ] All component renderers implemented
- [ ] OpenJSONUIRenderer dispatches to correct component
- [ ] Button actions trigger onAction callback
- [ ] Fallback rendering for unsupported types
- [ ] Backend Pydantic models for payload generation
- [ ] Unit tests pass with >85% coverage
- [ ] Accessibility tests pass (jest-axe)
- [ ] DOMPurify dependency added to package.json
- [ ] .env.example updated with feature flag
- [ ] Code review approved
- [ ] Story file updated with Dev Notes

## Dev Notes

*(To be filled during implementation)*

## Dev Agent Record

### Agent Model Used

*(To be filled during implementation)*

### Completion Notes List

*(To be filled during implementation)*

### File List

*(To be filled during implementation)*

## Test Outcomes

*(To be filled during implementation)*
