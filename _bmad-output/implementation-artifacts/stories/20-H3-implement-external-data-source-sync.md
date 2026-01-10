# Story 20-H3: Implement External Data Source Sync

Status: done

## Story

As a developer building enterprise RAG systems,
I want to sync content from external data sources (Confluence, S3, Notion, etc.),
so that the knowledge base stays updated with organizational content.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group H: Competitive Features. It implements external data source connectors enabling:

- **Pluggable Connectors**: Base protocol for building new connectors
- **S3 Connector**: Sync documents from AWS S3 buckets
- **Confluence Connector**: Sync pages from Atlassian Confluence
- **Notion Connector**: Sync pages from Notion workspaces
- **Sync Manager**: Orchestrate connectors with incremental sync support

**Competitive Positioning**: Enterprise RAG systems need to connect to where data lives. This enables integration with common enterprise tools.

**Dependencies**:
- boto3 (already installed) for S3
- httpx (already installed) for API calls
- No new dependencies required

## Acceptance Criteria

1. Given EXTERNAL_SYNC_ENABLED=true, when the system starts, then sync connectors are available.
2. Given an S3 bucket configured, when sync runs, then documents are fetched and indexed.
3. Given a Confluence space configured, when sync runs, then pages are fetched and indexed.
4. Given a Notion database configured, when sync runs, then pages are fetched and indexed.
5. Given EXTERNAL_SYNC_ENABLED=false (default), when the system starts, then sync features are not active.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/sync/
├── __init__.py                # Exports
├── base.py                    # BaseConnector protocol
├── s3_connector.py            # S3 sync connector
├── confluence_connector.py    # Confluence sync connector
├── notion_connector.py        # Notion sync connector
├── manager.py                 # SyncManager orchestrator
└── models.py                  # SyncResult, SyncConfig models
```

### Core Components

1. **BaseConnector** - Protocol for all connectors:
   - `list_items()`: List available items to sync
   - `fetch_content(item_id)`: Fetch content for an item
   - `get_last_sync()`: Get last sync timestamp
   - `update_sync_state()`: Update sync state

2. **S3Connector** - AWS S3 document sync:
   - Uses boto3 for S3 access
   - Supports prefix filtering
   - Tracks ETags for incremental sync

3. **ConfluenceConnector** - Atlassian Confluence sync:
   - Uses REST API v2
   - Syncs pages from configured spaces
   - Tracks last modified for incremental

4. **NotionConnector** - Notion workspace sync:
   - Uses Notion API
   - Syncs pages from databases
   - Tracks last edited for incremental

5. **SyncManager** - Orchestrates sync:
   - Runs connectors based on config
   - Handles rate limiting
   - Reports sync status

### Configuration

```bash
EXTERNAL_SYNC_ENABLED=true|false            # Default: false
SYNC_SOURCES=s3,confluence,notion           # Comma-separated
S3_SYNC_BUCKET=my-docs-bucket
S3_SYNC_PREFIX=documents/
CONFLUENCE_URL=https://xxx.atlassian.net
CONFLUENCE_API_TOKEN=xxx
CONFLUENCE_SPACES=SPACE1,SPACE2
NOTION_API_KEY=secret_xxx
NOTION_DATABASE_IDS=db1,db2
```

## Tasks / Subtasks

- [x] Create sync/ module directory
- [x] Create models.py with SyncResult, SyncConfig, SyncItem
- [x] Create base.py with BaseConnector protocol
- [x] Implement S3Connector with boto3
- [x] Implement ConfluenceConnector with REST API
- [x] Implement NotionConnector with Notion API
- [x] Implement SyncManager orchestrator
- [x] Create __init__.py with exports
- [x] Add configuration variables to settings
- [x] Write unit tests for all components

## Testing Requirements

### Unit Tests
- BaseConnector protocol compliance
- S3Connector with mocked boto3
- ConfluenceConnector with mocked API
- NotionConnector with mocked API
- SyncManager orchestration
- Feature flag behavior

## Definition of Done

- [x] All acceptance criteria pass
- [x] All tasks completed
- [x] Unit test coverage >= 80%
- [x] Feature flag (EXTERNAL_SYNC_ENABLED) works correctly
- [x] Configuration documented
- [x] Code review approved
- [x] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-H3 section)
- Use httpx for async HTTP calls to APIs
- boto3 requires AWS credentials in environment
- Confluence uses API token auth with email
- Notion uses integration API key

---

## Dev Agent Record

### File List

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/sync/__init__.py` | NEW | Module exports |
| `backend/src/agentic_rag_backend/sync/models.py` | NEW | SyncResult, SyncConfig, SyncItem models |
| `backend/src/agentic_rag_backend/sync/base.py` | NEW | BaseConnector protocol |
| `backend/src/agentic_rag_backend/sync/s3_connector.py` | NEW | S3 sync connector |
| `backend/src/agentic_rag_backend/sync/confluence_connector.py` | NEW | Confluence sync connector |
| `backend/src/agentic_rag_backend/sync/notion_connector.py` | NEW | Notion sync connector |
| `backend/src/agentic_rag_backend/sync/manager.py` | NEW | SyncManager orchestrator |
| `backend/src/agentic_rag_backend/config.py` | MODIFIED | Add sync settings |
| `backend/tests/sync/test_sync.py` | NEW | Unit tests |

### Change Log

| Date | Change | Details |
|------|--------|---------|
| 2026-01-06 | Initial implementation | Created story file |
| 2026-01-06 | Full implementation | Created sync/ module with models (SyncItem, SyncResult, SyncState, SyncConfig), BaseConnector protocol, S3Connector (boto3), ConfluenceConnector (REST API v2), NotionConnector (Notion API), SyncManager orchestrator. Added 9 config settings. 36 unit tests passing. |
