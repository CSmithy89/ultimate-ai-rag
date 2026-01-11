# Epic 11: Code Cleanup & Migration

**Status:** Ready for Development
**Priority:** High
**Estimated Effort:** 1-2 Sprints
**Created:** 2025-12-31
**Source:** Tech Debt Audit (Epics 1-8)

---

## Overview

This epic addresses code-level technical debt: deprecated APIs, legacy code still present, incomplete migrations, and missing features. The goal is to clean up the codebase and complete deferred work.

### Business Value

- Python 3.12+ compatibility (datetime deprecation)
- Reduced codebase size (~1,000 lines removed)
- Complete Graphiti migration (no parallel systems)
- Working HITL and workspace features

### Success Criteria

1. Zero deprecation warnings in Python 3.12+
2. Legacy indexing modules deleted (~1,000 lines)
3. Graphiti data migration 100% complete
4. HITL validation and workspace persistence functional

---

## Stories

### Story 11.1: Fix datetime.utcnow() Deprecation

**As a** Developer,
**I want** all datetime.utcnow() calls replaced with timezone-aware alternatives,
**So that** the codebase is compatible with Python 3.12+.

**Acceptance Criteria:**
1. Given deprecated datetime.utcnow() exists, when replaced, then datetime.now(timezone.utc) is used
2. Given all files are updated, when deprecation warnings are checked, then zero warnings appear
3. Given timezone-aware datetimes are used, when serialized, then ISO8601 format includes timezone

**Tasks:**
- [ ] Audit all uses of datetime.utcnow() in codebase
- [ ] Replace with datetime.now(timezone.utc) in:
  - `models/documents.py`
  - `indexing/crawler.py`
  - `agents/indexer.py`
  - `api/routes/ingest.py`
  - Any other files found
- [ ] Update tests to use timezone-aware datetimes
- [ ] Add linting rule to prevent future utcnow() usage

**Story Points:** 3

---

### Story 11.2: Delete Deprecated Legacy Modules

**As a** Developer,
**I want** deprecated legacy indexing modules removed,
**So that** the codebase is smaller and easier to maintain.

**Acceptance Criteria:**
1. Given legacy modules are deprecated, when deleted, then ~1,000 lines are removed
2. Given modules are deleted, when tests run, then all tests pass
3. Given imports existed, when cleaned up, then no dead imports remain

**Tasks:**
- [ ] Delete `backend/src/agentic_rag_backend/indexing/entity_extractor.py` (352 lines)
- [ ] Delete `backend/src/agentic_rag_backend/indexing/graph_builder.py` (295 lines)
- [ ] Delete `backend/src/agentic_rag_backend/indexing/embeddings.py` (228 lines)
- [ ] Clean up imports in `indexing/__init__.py`
- [ ] Remove any remaining references to deleted modules
- [ ] Update tests that imported deprecated modules
- [ ] Verify all tests pass after deletion

**Story Points:** 5

---

### Story 11.3: Complete Graphiti Data Migration

**As a** Developer,
**I want** all knowledge graph data migrated to Graphiti format,
**So that** we can disable the legacy pipeline entirely.

**Acceptance Criteria:**
1. Given legacy entities exist, when migration runs, then 100% are migrated to Graphiti
2. Given migration completes, when validation runs, then entity counts match
3. Given both systems ran in parallel, when cutover completes, then only Graphiti is active
4. Given migration is validated, when feature flags are removed, then legacy settings are deleted

**Tasks:**
- [ ] Review/create `backend/scripts/migrate_to_graphiti.py`
- [ ] Add entity type classification for legacy entities
- [ ] Run migration on development data
- [ ] Validate entity and relationship counts
- [ ] Archive legacy data (backup before deletion)
- [ ] Remove feature flags: INGESTION_BACKEND, RETRIEVAL_BACKEND
- [ ] Update configuration documentation

**Story Points:** 8

---

### Story 11.4: Wire HITL Validation Endpoint

**As a** Developer,
**I want** the HITL validation endpoint connected to real AG-UI bridge,
**So that** human-in-the-loop source validation works in production.

**Acceptance Criteria:**
1. Given HITL validation is triggered, when user reviews sources, then checkpoints are persisted
2. Given validation completes, when result is submitted, then AG-UI bridge receives the decision
3. Given timeout occurs, when fallback runs, then default behavior is applied
4. Given validation history exists, when queried, then checkpoints are retrievable

**Tasks:**
- [ ] Review current HITL validation stub implementation
- [ ] Wire to real AG-UI bridge event flow
- [ ] Add checkpoint persistence (PostgreSQL or Redis)
- [ ] Implement timeout handling and fallback
- [ ] Add validation result endpoints
- [ ] Write integration tests for HITL flow

**Story Points:** 5

---

### Story 11.5: Implement Workspace Persistence

**As a** User,
**I want** share and bookmark features to actually persist data,
**So that** I can save and share my workspaces.

**Acceptance Criteria:**
1. Given user bookmarks a workspace, when data is saved, then it persists across sessions
2. Given user shares a workspace, when link is generated, then it is accessible by others
3. Given workspace actions exist, when clicked, then they perform the expected operation
4. If feature is deferred, then UI actions are hidden until implemented

**Tasks:**
- [ ] Decide: implement persistence OR hide UI actions
- [ ] If implementing:
  - [ ] Create workspace persistence model
  - [ ] Add PostgreSQL table for bookmarks/shares
  - [ ] Implement save/load endpoints
  - [ ] Wire UI actions to real endpoints
- [ ] If deferring:
  - [ ] Hide share/bookmark buttons in UI
  - [ ] Document decision and future implementation plan

**Story Points:** 5 (implement) or 2 (defer)

---

### Story 11.6: Replace HTML-to-Markdown Regex

**As a** Developer,
**I want** HTML-to-Markdown conversion using a proper library,
**So that** HTML parsing is robust and maintainable.

**Acceptance Criteria:**
1. Given HTML content is received, when converted, then proper markdown is produced
2. Given complex HTML (tables, nested lists), when converted, then structure is preserved
3. Given the library is used, when tested, then edge cases pass

**Tasks:**
- [ ] Add markdownify or html2text to dependencies
- [ ] Replace regex-based conversion in `indexing/crawler.py`
- [ ] Test with various HTML structures
- [ ] Update unit tests

**Story Points:** 2

---

### Story 11.7: Configure Neo4j Connection Pooling

**As a** DevOps Engineer,
**I want** Neo4j connection pooling configured for production,
**So that** the system scales under load.

**Acceptance Criteria:**
1. Given connection pool is configured, when multiple requests arrive, then connections are reused
2. Given pool settings exist, when load increases, then pool scales appropriately
3. Given configuration is documented, when deploying, then pool settings are clear

**Tasks:**
- [ ] Review Neo4j driver pool configuration options
- [ ] Add pool settings to `config.py` (max_size, min_size, timeout)
- [ ] Configure in `db/neo4j.py` client initialization
- [ ] Add connection pool metrics logging
- [ ] Document production pool settings

**Story Points:** 3

---

### Story 11.8: Add LLM Token Usage Monitoring

**As a** Developer,
**I want** token usage tracked for all LLM calls,
**So that** costs are visible and budgets can be enforced.

**Acceptance Criteria:**
1. Given LLM call is made, when response is received, then token counts are logged
2. Given usage is tracked, when metrics are queried, then per-request and aggregate counts are available
3. Given budget threshold exists, when exceeded, then alert is triggered

**Tasks:**
- [ ] Add token counting to entity extraction calls (GPT-4o)
- [ ] Add token counting to Graphiti LLM calls
- [ ] Create usage logging format (structured JSON)
- [ ] Add usage metrics endpoint
- [ ] Implement budget alerting (optional)

**Story Points:** 5

---

### Story 11.9: Decide A2A Session Persistence Strategy

**As a** Architect,
**I want** a decision on A2A session persistence,
**So that** sessions are not lost on restart.

**Acceptance Criteria:**
1. Given decision is made, when documented, then strategy is clear (in-memory vs Redis)
2. Given strategy is implemented, when server restarts, then active sessions are handled appropriately
3. Given persistence exists, when session is recovered, then state is consistent

**Tasks:**
- [ ] Analyze session data requirements and lifetime
- [ ] Evaluate in-memory vs Redis persistence trade-offs
- [ ] Document decision with rationale
- [ ] If Redis: implement session serialization/deserialization
- [ ] If in-memory: document session loss on restart and recovery strategy

**Story Points:** 5

---

## Dependencies

- Epic 10 (testing infrastructure) recommended to validate changes
- Epic 9 (process) helps ensure quality

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Migration data loss | High | Full backup before migration, validation counts |
| Breaking changes from deletions | Medium | Comprehensive test coverage before deletion |
| HITL complexity | Medium | Incremental implementation with feature flags |

---

## Definition of Done

- [ ] All 9 stories completed and reviewed
- [ ] Zero deprecation warnings
- [ ] ~1,000 lines of legacy code removed
- [ ] Graphiti migration complete, feature flags removed
- [ ] HITL and workspace features functional or explicitly deferred
- [ ] All tests passing

---

## References

- Tech Debt Audit: `_bmad-output/implementation-artifacts/tech-debt-audit-2025-12-31.md`
- Code debt items: E4-02, E4-03, E4-04, E4-05, E4-06, E5-02, E5-04, E5-06, E6-01, E6-02, E7-01
