# Story 20-D1: Enhance Table/Layout Extraction

Status: done

## Story

As a developer building AI-powered document applications,
I want enhanced table and layout extraction from documents,
so that complex document structures like tables, sections, and figures are properly preserved and searchable.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group D: Document Intelligence (RAGFlow Approach). It enhances the existing Docling integration to better handle complex document structures.

**Competitive Positioning**: This feature competes with RAGFlow's document intelligence capabilities, providing better extraction of structured content from PDFs and documents.

**Why This Matters**:
- **Table Preservation**: Tables often contain critical data that gets mangled by simple text extraction
- **Layout Understanding**: Document sections, headers, and hierarchy improve retrieval relevance
- **Structured Access**: Tables as structured data enable programmatic querying
- **Markdown Embedding**: Tables converted to markdown can be embedded for semantic search

**Dependencies**:
- Epic 4 (Knowledge Ingestion Pipeline) - Provides base Docling integration
- Epic 19 (Quality Foundation) - COMPLETED
- Existing parser infrastructure

**Enables**:
- Story 20-D2 (Multimodal Ingestion) - Uses enhanced parser as foundation
- Better table-aware retrieval

## Acceptance Criteria

1. Given a PDF with tables, when parsed, then tables are extracted with headers and rows.
2. Given an extracted table, when accessed, then markdown representation is available.
3. Given an extracted table, when accessed, then structured data (list of dicts) is available.
4. Given ENHANCED_DOCLING_ENABLED=true (default), when documents are parsed, then enhanced extraction runs.
5. Given DOCLING_TABLE_AS_MARKDOWN=true, when tables are found, then they are stored as markdown chunks.
6. Layout preservation maintains section hierarchy and page information.
7. All operations enforce tenant isolation via `tenant_id` filtering.
8. Enhanced parsing adds <200ms latency over standard Docling parsing.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/indexing/
+-- enhanced_docling.py    # NEW: Enhanced document parsing
+-- __init__.py            # Update exports
```

### Core Components

1. **ExtractedTable Dataclass** - Represents a parsed table with:
   - id, page_number, position
   - headers, rows
   - caption, markdown representation
   - structured_data (list of dicts)

2. **DocumentLayout Dataclass** - Full document structure:
   - sections, tables, figures, footnotes
   - headers, page_count

3. **EnhancedDoclingParser Class** - Enhanced parsing with:
   - `parse_document()` - Main entry point
   - `_extract_tables()` - Table extraction with structure
   - `_table_to_markdown()` - Convert tables to markdown
   - `table_to_chunks()` - Convert table to searchable chunks

### Configuration

```bash
ENHANCED_DOCLING_ENABLED=true|false     # Default: true
DOCLING_TABLE_EXTRACTION=true|false     # Default: true
DOCLING_PRESERVE_LAYOUT=true|false      # Default: true
DOCLING_TABLE_AS_MARKDOWN=true|false    # Default: true
```

### API Integration

Enhanced parsing results include table and layout info:

```json
{
  "data": {
    "document_id": "doc_123",
    "layout": {
      "page_count": 10,
      "sections": [...],
      "tables": [
        {
          "id": "table_0",
          "page_number": 3,
          "headers": ["Name", "Value", "Unit"],
          "rows": [["Revenue", "5.2B", "USD"]],
          "markdown": "| Name | Value | Unit |..."
        }
      ]
    }
  }
}
```

## Tasks / Subtasks

- [ ] Create `enhanced_docling.py` module
- [ ] Implement ExtractedTable dataclass
  - [ ] Fields: id, page_number, position, headers, rows, caption, markdown, structured_data
  - [ ] `to_dict()` and `from_dict()` methods
- [ ] Implement DocumentLayout dataclass
  - [ ] Fields: sections, tables, figures, footnotes, headers, page_count
- [ ] Implement EnhancedDoclingParser class
  - [ ] `__init__()` with configuration
  - [ ] `parse_document()` main method
  - [ ] `_extract_tables()` table extraction
  - [ ] `_table_to_markdown()` markdown conversion
  - [ ] `table_to_chunks()` create searchable chunks from tables
- [ ] Add configuration to settings
  - [ ] ENHANCED_DOCLING_ENABLED
  - [ ] DOCLING_TABLE_EXTRACTION
  - [ ] DOCLING_PRESERVE_LAYOUT
  - [ ] DOCLING_TABLE_AS_MARKDOWN
- [ ] Update `indexing/__init__.py` exports
- [ ] Write unit tests for ExtractedTable
- [ ] Write unit tests for DocumentLayout
- [ ] Write unit tests for EnhancedDoclingParser
- [ ] Write integration tests with sample documents

## Testing Requirements

### Unit Tests

- ExtractedTable serialization/deserialization
- DocumentLayout creation with various configurations
- Table to markdown conversion (various table sizes)
- Table to structured data conversion
- Empty table handling
- Unicode in table cells

### Integration Tests

- End-to-end parsing with real Docling (if available)
- Table extraction from sample PDF
- Layout preservation verification

### Performance Tests

- Enhanced parsing latency < 200ms over baseline

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] All tasks completed
- [ ] Unit test coverage >= 80%
- [ ] Integration tests pass (if Docling available)
- [ ] Performance target met
- [ ] Configuration documented
- [ ] Code review approved
- [ ] No regressions in existing parsing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-D1 section)
- Use existing Docling infrastructure from `backend/src/agentic_rag_backend/indexing/parser.py`
- Tables should be converted to markdown for embedding
- Structured data enables programmatic access to table content
- Consider caching parsed tables for large documents
- Layout sections should maintain parent-child hierarchy

### RAGFlow Inspiration

RAGFlow's document intelligence includes:
- Layout analysis for proper section extraction
- Table structure recognition with cell merging
- Figure/chart detection with captions
- Formula extraction and rendering
