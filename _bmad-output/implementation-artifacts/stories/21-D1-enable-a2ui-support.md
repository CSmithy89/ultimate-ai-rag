# Story 21-D1: Enable A2UI Support

## Status: Review

## Story

As a backend developer, I need to enable A2UI protocol support so that agents can emit rich UI widget payloads in their responses for enhanced frontend rendering.

## Implementation Summary

### New Files Created

1. **`backend/src/agentic_rag_backend/protocols/a2ui.py`** - A2UI protocol implementation
   - `A2UIWidget`: Base model for all widget types
   - `A2UIAction`: Action button model
   - `A2UIFormField`: Form field definition model
   - Factory functions for all widget types:
     - `create_a2ui_card()`: Create card widgets
     - `create_a2ui_table()`: Create table widgets
     - `create_a2ui_form()`: Create form widgets
     - `create_a2ui_chart()`: Create chart widgets
     - `create_a2ui_image()`: Create image widgets
     - `create_a2ui_list()`: Create list widgets
   - `widgets_to_state()`: Utility for STATE_SNAPSHOT events

2. **`backend/tests/protocols/test_a2ui.py`** - A2UI tests (23 tests)
   - Tests for all models and factory functions
   - Tests for serialization and state conversion

### Files Modified

1. **`backend/src/agentic_rag_backend/protocols/__init__.py`**
   - Added exports for all A2UI classes and functions

## Supported Widget Types

| Type | Purpose | Factory Function |
|------|---------|------------------|
| `card` | Display content card | `create_a2ui_card()` |
| `table` | Tabular data | `create_a2ui_table()` |
| `form` | Input collection | `create_a2ui_form()` |
| `chart` | Data visualization | `create_a2ui_chart()` |
| `image` | Image display | `create_a2ui_image()` |
| `list` | Item listing | `create_a2ui_list()` |

## Usage Example

```python
from agentic_rag_backend.protocols import (
    create_a2ui_card,
    create_a2ui_table,
    widgets_to_state,
)
from agentic_rag_backend.models.copilot import StateSnapshotEvent

# Create widgets
card = create_a2ui_card(
    title="Search Results",
    content=f"Found {len(sources)} relevant documents",
)
table = create_a2ui_table(
    headers=["Source", "Score"],
    rows=[[s.name, str(s.score)] for s in sources],
)

# Emit via STATE_SNAPSHOT
yield StateSnapshotEvent(state=widgets_to_state([card, table]))
```

## Acceptance Criteria

- [x] A2UI widget models defined with Pydantic validation
- [x] Helper functions for all 6 widget types
- [x] Widgets serializable for STATE_SNAPSHOT events
- [x] Schema validation for widget properties
- [x] Tests verify widget serialization (23 tests)

## Files Changed

- `backend/src/agentic_rag_backend/protocols/a2ui.py` (new)
- `backend/src/agentic_rag_backend/protocols/__init__.py` (updated)
- `backend/tests/protocols/test_a2ui.py` (new)
