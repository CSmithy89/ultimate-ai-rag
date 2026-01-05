# Story 20-D2: Implement Multimodal Ingestion

Status: done

## Story

As a developer building AI-powered document applications,
I want multimodal document ingestion capabilities,
so that I can process various document types including Word, Excel, PowerPoint, and images.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group D: Document Intelligence (RAGFlow Approach). It adds support for additional document types beyond PDF.

**Competitive Positioning**: This feature provides comprehensive document ingestion similar to enterprise RAG systems, supporting the full range of business documents.

**Why This Matters**:
- **Business Documents**: Most organizations use Office documents extensively
- **Spreadsheet Data**: Excel files contain structured data valuable for RAG
- **Presentations**: PowerPoint slides contain key insights and summaries
- **Document Type Detection**: Automatic detection reduces user friction

**Dependencies**:
- Story 20-D1 (Enhanced Table/Layout Extraction) - Provides base Docling parser enhancements
- python-docx, openpyxl, python-pptx libraries for Office parsing

**Enables**:
- Processing of complete document repositories
- Enterprise document ingestion workflows

## Acceptance Criteria

1. Given a Word document (.docx), when ingested, then text and tables are extracted.
2. Given an Excel file (.xlsx), when ingested, then all sheets are parsed with cell data.
3. Given a PowerPoint file (.pptx), when ingested, then slides, text, and notes are extracted.
4. Document type is auto-detected from file extension or MIME type.
5. Given MULTIMODAL_INGESTION_ENABLED=true (default: false), when documents are uploaded, then multimodal processing runs.
6. Given OFFICE_DOCS_ENABLED=true, when Office documents are uploaded, then they are processed.
7. All operations enforce tenant isolation via `tenant_id` filtering.
8. Multimodal ingestion adds <500ms latency for typical documents.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/indexing/
+-- multimodal.py           # NEW: Multimodal ingestion
+-- __init__.py             # Update exports
```

### Core Components

1. **DocumentType Enum** - Supported document types:
   - PDF, WORD (docx), EXCEL (xlsx), POWERPOINT (pptx)
   - IMAGE, MARKDOWN, TEXT

2. **OfficeParser Class** - Parse Office documents:
   - `parse_word()` - Extract text, tables from Word
   - `parse_excel()` - Extract sheets, cells from Excel
   - `parse_powerpoint()` - Extract slides, text, notes from PowerPoint

3. **MultimodalIngester Class** - Unified ingestion:
   - `ingest()` - Main entry point
   - `_detect_type()` - Auto-detect document type
   - Route to appropriate parser based on type

### Configuration

```bash
MULTIMODAL_INGESTION_ENABLED=true|false      # Default: false
OFFICE_DOCS_ENABLED=true|false               # Default: true (when multimodal enabled)
IMAGE_INGESTION_ENABLED=true|false           # Default: false (future)
FORMULA_EXTRACTION_ENABLED=true|false        # Default: false (future)
```

## Tasks / Subtasks

- [ ] Create `multimodal.py` module
- [ ] Implement DocumentType enum
- [ ] Implement OfficeParser class
  - [ ] `parse_word()` using python-docx
  - [ ] `parse_excel()` using openpyxl
  - [ ] `parse_powerpoint()` using python-pptx
- [ ] Implement MultimodalIngester class
  - [ ] `__init__()` with parser dependencies
  - [ ] `ingest()` main method
  - [ ] `_detect_type()` type detection
- [ ] Add configuration to settings
  - [ ] MULTIMODAL_INGESTION_ENABLED
  - [ ] OFFICE_DOCS_ENABLED
- [ ] Update `indexing/__init__.py` exports
- [ ] Write unit tests for DocumentType
- [ ] Write unit tests for OfficeParser
- [ ] Write unit tests for MultimodalIngester
- [ ] Test tenant isolation in all operations

## Testing Requirements

### Unit Tests

- DocumentType enum completeness
- Word document parsing (text, tables)
- Excel parsing (sheets, cells)
- PowerPoint parsing (slides, notes)
- Document type detection from extension
- Empty document handling
- Malformed document handling

### Integration Tests

- End-to-end ingestion with real Office files
- Multi-document batch ingestion

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] All tasks completed
- [ ] Unit test coverage >= 80%
- [ ] Performance target met (<500ms)
- [ ] Configuration documented
- [ ] Code review approved
- [ ] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-D2 section)
- Use python-docx for Word, openpyxl for Excel, python-pptx for PowerPoint
- Consider async processing for large documents
- Tables from Word should use similar structure to PDF tables
- Excel sheets should preserve cell formatting info
