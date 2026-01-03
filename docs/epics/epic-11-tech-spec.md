# Epic 11 Tech Spec: Code Cleanup & Migration

**Version:** 1.0  
**Created:** 2025-12-31  
**Status:** Ready for Implementation

---

## Overview

Epic 11 focuses on debt retirement and migration completeness: removing deprecated code paths, finishing Graphiti migration, and stabilizing core infrastructure (HITL, workspace persistence, Neo4j pooling, token monitoring, and A2A session persistence).

### Business Value

- Reduce maintenance burden by removing legacy modules and deprecated APIs.
- Improve reliability and data consistency with completed Graphiti migration.
- Strengthen production readiness via persistence and pooling improvements.
- Minimize future regressions by eliminating known deprecations.

---

## Scope and Goals

- Fix Python datetime deprecations across the codebase.
- Remove legacy ingestion/retrieval modules that should no longer be used.
- Complete Graphiti migration and ensure all paths use Graphiti-first behaviors.
- Wire HITL validation to backend and persistence layers.
- Implement workspace persistence for save/share/export data.
- Replace brittle HTML-to-Markdown regex with a proper parser.
- Configure Neo4j connection pooling for stability under load.
- Add LLM token usage monitoring to improve cost tracking fidelity.
- Persist A2A sessions to avoid loss on restart.

---

## Architecture Decisions

### 1. Deprecation Removal and Replacement
**Decision:** Replace deprecated APIs and legacy modules rather than extending them.
**Rationale:** Avoids compounding debt and ensures migration is final.

### 2. Graphiti Migration Completion
**Decision:** Treat Graphiti as the default path for ingestion and retrieval, with legacy modules removed.
**Rationale:** Keeps the pipeline consistent and reduces duplication.

### 3. Persistence for User-Facing Flows
**Decision:** Persist HITL and workspace state so user actions survive restarts.
**Rationale:** Improves reliability and aligns with NFR8 (stateless recovery).

### 4. Neo4j Connection Pooling
**Decision:** Configure Neo4j pooling parameters to support concurrent workloads.
**Rationale:** Prevents driver exhaustion and improves throughput under load.

---

## Component Changes

| Area | Change |
| --- | --- |
| datetime usage | Replace deprecated `datetime.utcnow()` patterns |
| legacy modules | Remove deprecated ingestion/retrieval code |
| Graphiti | Finish migration and unify ingestion/retrieval |
| HITL | Wire validation endpoints and persistence |
| workspace | Persist save/share/export data |
| parser | Replace HTML regex with a proper parser |
| Neo4j | Configure connection pooling |
| LLM usage | Add token monitoring instrumentation |
| A2A | Persist session state |

---

## Story Breakdown

1. **11-1 Fix datetime deprecation**
   - Replace deprecated datetime APIs across backend.

2. **11-2 Delete legacy modules**
   - Remove unused legacy ingestion/retrieval modules.

3. **11-3 Complete Graphiti migration**
   - Ensure Graphiti paths are the only supported ingestion/retrieval flows.

4. **11-4 Wire HITL validation**
   - Connect validation endpoints to persistence and workflow.

5. **11-5 Implement workspace persistence**
   - Persist save/share/export actions.

6. **11-6 Replace HTML markdown regex**
   - Use a reliable parser for HTML to Markdown conversion.

7. **11-7 Configure Neo4j pooling**
   - Set driver pooling settings and document defaults.

8. **11-8 Add LLM token monitoring**
   - Track token usage consistently for cost analysis.

9. **11-9 A2A session persistence**
   - Persist A2A session state across restarts.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Legacy removal breaks hidden dependencies | Medium | Search references and update callers first |
| Migration leaves partial paths | High | Enforce Graphiti-first flows in tests |
| Persistence changes require schema updates | Medium | Add safe migrations and backfill steps |
| Pooling misconfiguration | Medium | Use conservative defaults with env overrides |

---

## Testing Strategy

- Update unit tests affected by deprecated APIs or legacy removals.
- Add integration coverage for HITL and workspace persistence if missing.
- Validate A2A session persistence via protocol tests.

---

## Out of Scope

- New product features not tied to cleanup/migration
- Major UI redesign work
