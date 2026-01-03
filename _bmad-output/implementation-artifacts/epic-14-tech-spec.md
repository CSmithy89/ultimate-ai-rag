# Epic 14 Tech Spec: Connectivity (MCP Wrapper Architecture)

**Date:** 2025-12-31
**Updated:** 2026-01-03 (Party Mode Analysis)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 14 exposes the RAG engine to external tools via an MCP server and strengthens A2A agent connectivity for multi-agent orchestration. This positions the project as a portable engine that can be consumed by other clients.

### Key Decision (2026-01-03)

**WRAP Graphiti MCP, don't duplicate.**

Graphiti already has a tested, maintained MCP server. We wrap it and extend with RAG-specific tools.

**Full Architecture Guide:** `docs/guides/mcp-wrapper-architecture.md`

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED MCP SERVER                        │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐  ┌──────────────────────────────┐   │
│  │ GRAPHITI MCP       │  │ RAG EXTENSIONS MCP           │   │
│  │ (Wrapped)          │  │ (New)                        │   │
│  ├────────────────────┤  ├──────────────────────────────┤   │
│  │ • add_memory       │  │ • vector_search              │   │
│  │ • search_nodes     │  │ • hybrid_retrieve            │   │
│  │ • search_facts     │  │ • ingest_url                 │   │
│  │ • delete_episode   │  │ • ingest_pdf                 │   │
│  │ • clear_graph      │  │ • ingest_youtube             │   │
│  │                    │  │ • query_with_reranking       │   │
│  │                    │  │ • explain_answer             │   │
│  └────────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Goals

- Deliver an MCP server with stable tool endpoints.
- **Wrap Graphiti MCP to avoid duplication.**
- **Extend with RAG-specific tools (vector, ingestion, reranking).**
- Implement a robust A2A protocol for agent collaboration and session handling.

### Scope

**In scope**
- MCP wrapper around Graphiti MCP.
- RAG extension tools: `vector_search`, `hybrid_retrieve`, `ingest_url`, `ingest_pdf`, `ingest_youtube`, `query_with_reranking`, `explain_answer`.
- A2A protocol improvements: versioning, session state, and error handling.

**Out of scope**
- Codebase intelligence (Epic 15).
- Framework adapters (Epic 16).

---

## Stories

### Story 14-1: Expose RAG Engine via MCP Server

**Objective:** Provide external access to core retrieval and ingestion tools by wrapping Graphiti MCP and extending with RAG-specific tools.

**Implementation Approach:**
1. Wrap Graphiti MCP server (proxy all existing tools)
2. Add RAG extension tools alongside

**Tools to Expose:**

| Tool | Source | Description |
|------|--------|-------------|
| `add_memory` | Graphiti | Ingest knowledge into graph |
| `search_nodes` | Graphiti | Find entities |
| `search_facts` | Graphiti | Find relationships |
| `vector_search` | **NEW** | pgvector semantic search |
| `hybrid_retrieve` | **NEW** | Combined vector + graph |
| `ingest_url` | **NEW** | Crawl4AI/Apify ingestion |
| `ingest_pdf` | **NEW** | Docling PDF processing |
| `ingest_youtube` | **NEW** | YouTube transcript extraction |
| `query_with_reranking` | **NEW** | Cross-encoder reranked results |
| `explain_answer` | **NEW** | Trajectory/explainability |

**Acceptance Criteria**
- MCP server exposes tool definitions with JSON-RPC 2.0 compatibility.
- **Graphiti MCP tools are wrapped, not duplicated.**
- RAG extension tools are added alongside Graphiti tools.
- Authentication and rate limiting are configurable.
- MCP server usage is logged with request and response metadata.
- **All tools enforce tenant isolation.**

### Story 14-2: Implement Robust A2A Protocol

**Objective:** Strengthen agent-to-agent collaboration.

**Acceptance Criteria**
- A2A interactions include session identifiers and protocol versioning.
- Failures return RFC 7807 formatted errors.
- A2A workflows support retries and timeouts with clear logs.

---

## Technical Notes

- **Wrap, don't duplicate:** Use Graphiti MCP as base, extend with RAG tools.
- Define stable tool schemas for MCP server endpoints.
- A2A protocol should align with existing orchestration patterns.

## Dependencies

- Current MCP implementation (Epic 7).
- Graphiti MCP server (Epic 5).
- Orchestrator session handling (Epic 2).

## Risks

- Protocol drift if MCP or A2A specs change.
- Security concerns around exposing ingestion endpoints.
- **Mitigation:** Authentication, rate limiting, tenant isolation.

## Success Metrics

- External clients can call MCP tools without custom adapters.
- A2A workflows succeed reliably under simulated network failures.
- **Graphiti tools work through wrapper without modification.**

## References

- `docs/guides/mcp-wrapper-architecture.md` - **Full architecture guide**
- `docs/roadmap-decisions-2026-01-03.md` - Decision rationale
- [Graphiti MCP Documentation](https://deepwiki.com/getzep/graphiti#8)
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
