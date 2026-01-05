# Story 19-J3: Implement dedicated PDF parsing tool

Status: done

## Story

As a developer using the MCP interface,
I want a dedicated PDF ingestion tool that uses Docling,
so that PDF structure (headings, tables, page numbers) is preserved.

## Acceptance Criteria

1. `rag.ingest_pdf` MCP tool uses Docling-based parsing.
2. Tables are extracted with structure preserved.
3. Headings create chunk boundaries (when chunking by page/section).
4. Page numbers are tracked in metadata.
5. Tool exposes PDF-specific configuration options.

## Tasks / Subtasks

- [ ] Add `rag.ingest_pdf` MCP tool.
- [ ] Support page-level chunking and table extraction.
- [ ] Track page numbers in metadata.

## Dev Notes

- Update `backend/src/agentic_rag_backend/mcp_server/tools/rag.py`.
- Reuse `parse_pdf` from `indexing/parser.py`.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group J: 19-J3)
- `backend/src/agentic_rag_backend/mcp_server/tools/rag.py`
- `backend/src/agentic_rag_backend/indexing/parser.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

 - Added rag.ingest_pdf MCP tool using Docling parsing and page chunking.
 - Tables and headings are preserved in page-level chunks with page metadata.

### File List

 - `backend/src/agentic_rag_backend/mcp_server/tools/rag.py`

## Senior Developer Review

Outcome: APPROVE

- MCP tool uses Docling parsing and supports page-level chunking.
- Metadata includes page numbers and PDF-specific options.
