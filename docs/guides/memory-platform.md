# Memory Platform Guide

This guide covers the Memory Platform feature for managing scoped, persistent memories across agent interactions. The Memory Platform provides hierarchical memory scopes, automatic consolidation, time-based decay, and importance scoring - enabling agents to maintain context across sessions while automatically managing memory lifecycle.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Memory Scopes](#memory-scopes)
- [Importance Scoring](#importance-scoring)
- [Memory Consolidation](#memory-consolidation)
- [Memory Decay and Cleanup](#memory-decay-and-cleanup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Memory Platform implements a hierarchical memory system similar to [Mem0](https://mem0.ai/), providing:

- **Scoped Memories**: Four distinct scopes (user, session, agent, global) with inheritance
- **Semantic Search**: Vector-based similarity search using pgvector
- **Automatic Consolidation**: Scheduled deduplication and cleanup
- **Importance Decay**: Time-based decay with access frequency boost
- **Multi-Tenancy**: Full tenant isolation via `tenant_id` filtering
- **Redis Caching**: Hot cache with circuit breaker pattern
- **Graphiti Integration**: Optional graph-based memory relationships

## Architecture

### Storage Layer

```
+------------------+     +------------------+     +------------------+
|   PostgreSQL     |     |      Redis       |     |    Graphiti      |
|   + pgvector     |     |    (optional)    |     |   (optional)     |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        +------------------------+------------------------+
                                 |
                    +---------------------------+
                    |   ScopedMemoryStore       |
                    |   - CRUD operations       |
                    |   - Embedding generation  |
                    |   - Cache management      |
                    +---------------------------+
                                 |
                    +---------------------------+
                    |   MemoryConsolidator      |
                    |   - Decay calculation     |
                    |   - Duplicate merging     |
                    |   - Low-importance cleanup|
                    +---------------------------+
                                 |
                    +---------------------------+
                    |   ConsolidationScheduler  |
                    |   - APScheduler cron jobs |
                    |   - Tenant-isolated runs  |
                    +---------------------------+
```

### Database Schema

Memories are stored in the `scoped_memories` table with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `tenant_id` | UUID | Tenant identifier (required) |
| `scope` | VARCHAR | Memory scope (user/session/agent/global) |
| `user_id` | UUID | User identifier (for user/session scope) |
| `session_id` | UUID | Session identifier (for session scope) |
| `agent_id` | VARCHAR | Agent identifier (for agent scope) |
| `content` | TEXT | Memory content (max 10,000 characters) |
| `importance` | FLOAT | Importance score (0.0-1.0) |
| `metadata` | JSONB | Additional metadata |
| `embedding` | vector(1536) | Embedding for similarity search |
| `created_at` | TIMESTAMP | Creation timestamp |
| `accessed_at` | TIMESTAMP | Last access timestamp |
| `access_count` | INTEGER | Number of times accessed |

## Memory Scopes

The Memory Platform supports four hierarchical scopes, each serving different persistence needs:

### Scope Hierarchy

```
+--------------------+
|      GLOBAL        |  Tenant-wide shared knowledge
+--------------------+
         ^
         |
+--------+--------+--------+
|        |        |        |
v        v        v        v
+------+ +------+ +------+
| USER | | AGENT| |      |
+------+ +------+ |      |
    ^             |      |
    |             |      |
+-------+         |      |
|SESSION|         |      |
+-------+---------+------+
```

### Scope Definitions

| Scope | Context Required | Persistence | Use Case |
|-------|------------------|-------------|----------|
| **GLOBAL** | `tenant_id` only | Tenant-wide | Shared knowledge, company policies, FAQs |
| **USER** | `tenant_id` + `user_id` | Cross-session | User preferences, personal facts, history |
| **SESSION** | `tenant_id` + `session_id` | Single session | Conversation context, working memory |
| **AGENT** | `tenant_id` + `agent_id` | Cross-invocation | Agent operational state, learned behaviors |

### Scope Inheritance (Search)

When searching memories with `include_parent_scopes=true` (default):

- **SESSION** scope includes: SESSION + USER + GLOBAL memories
- **USER** scope includes: USER + GLOBAL memories
- **AGENT** scope includes: AGENT + GLOBAL memories
- **GLOBAL** scope includes: GLOBAL only

This enables agents to access both specific context and shared knowledge in a single query.

### Scope Validation

Each scope requires specific context identifiers:

```python
# USER scope - requires user_id
POST /api/v1/memories
{
    "content": "Prefers dark mode",
    "scope": "user",
    "tenant_id": "uuid",
    "user_id": "uuid"  # Required
}

# SESSION scope - requires session_id
POST /api/v1/memories
{
    "content": "Working on billing integration",
    "scope": "session",
    "tenant_id": "uuid",
    "session_id": "uuid"  # Required
}

# AGENT scope - requires agent_id
POST /api/v1/memories
{
    "content": "Learned to check stock before recommending products",
    "scope": "agent",
    "tenant_id": "uuid",
    "agent_id": "product-assistant"  # Required
}

# GLOBAL scope - no additional context
POST /api/v1/memories
{
    "content": "Company returns policy: 30 days with receipt",
    "scope": "global",
    "tenant_id": "uuid"
}
```

## Importance Scoring

Importance scores (0.0-1.0) determine memory priority during consolidation and cleanup.

### Initial Importance

Set importance when creating a memory:

```python
POST /api/v1/memories
{
    "content": "Critical: API rate limit is 1000/hour",
    "scope": "global",
    "tenant_id": "uuid",
    "importance": 0.95  # High importance - will decay slower
}
```

### Importance Guidelines

| Score Range | Use Case | Examples |
|-------------|----------|----------|
| 0.9 - 1.0 | Critical | Security policies, critical configs |
| 0.7 - 0.9 | Important | User preferences, key decisions |
| 0.5 - 0.7 | Standard | General context, facts |
| 0.3 - 0.5 | Low priority | Temporary notes, minor details |
| 0.0 - 0.3 | Ephemeral | Working memory, soon to be cleaned |

### Access Frequency Boost

Memories that are accessed frequently receive an importance boost during consolidation:

```
access_boost = min(1.0, 0.5 + (access_count * 0.1))
```

A memory accessed 5 times has `access_boost = 1.0`, completely counteracting decay.

## Memory Consolidation

Consolidation is the automated process that maintains memory quality by:

1. **Applying importance decay** based on time and access frequency
2. **Merging duplicate memories** using embedding similarity
3. **Removing low-importance memories** below the threshold

### Consolidation Process

```
+----------------+     +------------------+     +------------------+
| 1. Decay       |---->| 2. Deduplicate   |---->| 3. Cleanup       |
| - Time-based   |     | - Similarity > θ |     | - importance < ε |
| - Access boost |     | - Merge metadata |     | - Delete memory  |
+----------------+     +------------------+     +------------------+
```

### Automatic Scheduling

Configure scheduled consolidation via APScheduler:

```bash
# Enable consolidation
MEMORY_CONSOLIDATION_ENABLED=true

# Run daily at 2 AM (cron format)
MEMORY_CONSOLIDATION_SCHEDULE="0 2 * * *"

# Alternative schedules:
# Every 6 hours: "0 */6 * * *"
# Weekly on Sunday: "0 3 * * 0"
# Every night at midnight: "0 0 * * *"
```

### Manual Consolidation

Trigger consolidation via API:

```bash
# Consolidate all scopes for a tenant
POST /api/v1/memories/consolidate
{
    "tenant_id": "uuid"
}

# Consolidate specific scope
POST /api/v1/memories/consolidate
{
    "tenant_id": "uuid",
    "scope": "user",
    "user_id": "uuid"
}
```

### Consolidation Status

Check consolidation status:

```bash
GET /api/v1/memories/consolidation/status

# Response
{
    "data": {
        "last_run_at": "2026-01-12T02:00:00Z",
        "last_result": {
            "memories_processed": 150,
            "duplicates_merged": 5,
            "memories_decayed": 45,
            "memories_removed": 3,
            "processing_time_ms": 1250.5
        },
        "scheduler_enabled": true,
        "next_scheduled_run": "2026-01-13T02:00:00Z"
    }
}
```

### Duplicate Detection

Memories with embedding similarity above the threshold are considered duplicates:

```python
# Default: 90% similarity threshold
MEMORY_SIMILARITY_THRESHOLD=0.9
```

When duplicates are found:
- **Primary memory** (highest importance) is kept
- **Secondary memories** are deleted
- **Metadata is merged** (primary takes precedence)
- **Access counts are summed**
- **Merge tracking is recorded** in metadata

## Memory Decay and Cleanup

### Decay Formula

Importance decays exponentially based on time since last access:

```python
decay_factor = 2 ** (-days_since_access / half_life_days)
access_boost = min(1.0, 0.5 + (access_count * 0.1))
new_importance = current_importance * decay_factor * access_boost
```

### Decay Examples

With default `half_life_days=30`:

| Days Since Access | Access Count | Decay | New Importance (starting at 1.0) |
|-------------------|--------------|-------|----------------------------------|
| 0 | 0 | 1.0 | 0.50 |
| 0 | 5 | 1.0 | 1.00 |
| 30 | 0 | 0.5 | 0.25 |
| 30 | 5 | 0.5 | 0.50 |
| 60 | 0 | 0.25 | 0.125 |
| 90 | 0 | 0.125 | 0.0625 |

### Cleanup Threshold

Memories with importance below `MEMORY_MIN_IMPORTANCE` are automatically removed:

```bash
# Default: remove memories below 0.1 importance
MEMORY_MIN_IMPORTANCE=0.1
```

### Manual Cleanup by Scope

Delete all memories in a scope:

```bash
# Clear session memories
DELETE /api/v1/memories/scope/session?tenant_id=uuid&session_id=uuid

# Clear user memories (e.g., account deletion)
DELETE /api/v1/memories/scope/user?tenant_id=uuid&user_id=uuid

# Clear agent operational memory (reset agent)
DELETE /api/v1/memories/scope/agent?tenant_id=uuid&agent_id=product-assistant
```

## Configuration

### Environment Variables

#### Basic Settings (Story 20-A1)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_SCOPES_ENABLED` | `false` | Enable/disable memory scopes feature |
| `MEMORY_DEFAULT_SCOPE` | `session` | Default scope for new memories |
| `MEMORY_INCLUDE_PARENT_SCOPES` | `true` | Include parent scopes in search |
| `MEMORY_CACHE_TTL_SECONDS` | `3600` | Redis cache TTL (seconds) |
| `MEMORY_MAX_PER_SCOPE` | `10000` | Maximum memories per scope |

#### Consolidation Settings (Story 20-A2)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_CONSOLIDATION_ENABLED` | `false` | Enable automatic consolidation |
| `MEMORY_CONSOLIDATION_SCHEDULE` | `0 2 * * *` | Cron schedule (default: 2 AM daily) |
| `MEMORY_SIMILARITY_THRESHOLD` | `0.9` | Threshold for duplicate detection (0.0-1.0) |
| `MEMORY_DECAY_HALF_LIFE_DAYS` | `30` | Days for importance to halve |
| `MEMORY_MIN_IMPORTANCE` | `0.1` | Minimum importance before removal |
| `MEMORY_CONSOLIDATION_BATCH_SIZE` | `100` | Batch size for consolidation processing |

### Example Configuration

```bash
# .env

# Enable memory platform
MEMORY_SCOPES_ENABLED=true
MEMORY_DEFAULT_SCOPE=session
MEMORY_INCLUDE_PARENT_SCOPES=true
MEMORY_CACHE_TTL_SECONDS=3600
MEMORY_MAX_PER_SCOPE=10000

# Enable consolidation
MEMORY_CONSOLIDATION_ENABLED=true
MEMORY_CONSOLIDATION_SCHEDULE="0 2 * * *"
MEMORY_SIMILARITY_THRESHOLD=0.9
MEMORY_DECAY_HALF_LIFE_DAYS=30
MEMORY_MIN_IMPORTANCE=0.1
MEMORY_CONSOLIDATION_BATCH_SIZE=100

# Embedding provider for semantic search
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
```

## API Reference

### Create Memory

```http
POST /api/v1/memories
Content-Type: application/json

{
    "content": "User prefers dark mode interface",
    "scope": "user",
    "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
    "user_id": "123e4567-e89b-12d3-a456-426614174002",
    "importance": 0.8,
    "metadata": {
        "source": "preferences",
        "category": "ui"
    }
}
```

**Response (201 Created):**
```json
{
    "data": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "content": "User prefers dark mode interface",
        "scope": "user",
        "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
        "user_id": "123e4567-e89b-12d3-a456-426614174002",
        "importance": 0.8,
        "metadata": {"source": "preferences", "category": "ui"},
        "created_at": "2026-01-12T12:00:00Z",
        "accessed_at": "2026-01-12T12:00:00Z",
        "access_count": 0
    },
    "meta": {
        "requestId": "uuid",
        "timestamp": "2026-01-12T12:00:00Z"
    }
}
```

### List Memories

```http
GET /api/v1/memories?tenant_id=uuid&scope=user&user_id=uuid&limit=50&offset=0
```

**Response:**
```json
{
    "data": {
        "memories": [...],
        "total": 42,
        "limit": 50,
        "offset": 0
    },
    "meta": {...}
}
```

### Search Memories

```http
POST /api/v1/memories/search
Content-Type: application/json

{
    "query": "user preferences dark mode",
    "scope": "session",
    "tenant_id": "uuid",
    "user_id": "uuid",
    "session_id": "uuid",
    "limit": 10,
    "include_parent_scopes": true
}
```

**Response:**
```json
{
    "data": {
        "memories": [...],
        "total": 3,
        "query": "user preferences dark mode",
        "scopes_searched": ["session", "user", "global"]
    },
    "meta": {...}
}
```

### Get Memory by ID

```http
GET /api/v1/memories/{memory_id}?tenant_id=uuid
```

### Update Memory

```http
PUT /api/v1/memories/{memory_id}?tenant_id=uuid
Content-Type: application/json

{
    "content": "Updated content",
    "importance": 0.9,
    "metadata": {"updated": true}
}
```

### Delete Memory

```http
DELETE /api/v1/memories/{memory_id}?tenant_id=uuid
```

### Delete by Scope

```http
DELETE /api/v1/memories/scope/session?tenant_id=uuid&session_id=uuid
```

### Trigger Consolidation

```http
POST /api/v1/memories/consolidate
Content-Type: application/json

{
    "tenant_id": "uuid",
    "scope": "user",
    "user_id": "uuid"
}
```

### Get Consolidation Status

```http
GET /api/v1/memories/consolidation/status
```

## Best Practices

### 1. Choose Appropriate Scopes

- **GLOBAL**: Shared knowledge that applies to all users
- **USER**: Persistent user preferences and history
- **SESSION**: Working context for current conversation
- **AGENT**: Learned behaviors and operational state

### 2. Set Meaningful Importance

```python
# Critical information - high importance
{
    "content": "User has severe nut allergy",
    "importance": 0.95
}

# Temporary context - low importance
{
    "content": "Currently discussing product A",
    "importance": 0.3
}
```

### 3. Clean Up Sessions

Clear session memories when conversations end:

```python
# On session end
DELETE /api/v1/memories/scope/session?tenant_id=uuid&session_id=uuid
```

### 4. Monitor Consolidation

Check consolidation status regularly:

```python
# Monitor merge rate and cleanup
GET /api/v1/memories/consolidation/status

# High merge rate may indicate:
# - Too many duplicate memories being created
# - Similarity threshold too low
```

### 5. Tune Decay Parameters

Adjust based on your application's memory retention needs:

```bash
# Long-lived memories (knowledge bases)
MEMORY_DECAY_HALF_LIFE_DAYS=90
MEMORY_MIN_IMPORTANCE=0.05

# Short-lived memories (chat applications)
MEMORY_DECAY_HALF_LIFE_DAYS=7
MEMORY_MIN_IMPORTANCE=0.2
```

## Troubleshooting

### Feature Not Enabled

```
HTTP 404: Memory scopes feature is not enabled
```

**Solution:** Set `MEMORY_SCOPES_ENABLED=true` in environment.

### Scope Context Missing

```
HTTP 400: user_id is required for USER scope
```

**Solution:** Provide required context identifiers for the scope.

### Limit Exceeded

```
HTTP 429: Memory limit exceeded for scope 'user': 10000/10000
```

**Solution:**
- Increase `MEMORY_MAX_PER_SCOPE`
- Run consolidation to remove low-importance memories
- Clean up old memories manually

### Consolidation Not Running

```
Scheduler not started, APScheduler not available
```

**Solution:** Install APScheduler:
```bash
uv add apscheduler
```

### Redis Circuit Breaker Open

```
redis_circuit_breaker_opened: failure_count=5
```

**Solution:** Redis is experiencing connection issues. The system will automatically retry after 30 seconds. Check Redis connectivity.

### Embedding Dimension Mismatch

```
Embedding dimension mismatch: got 768, expected 1536
```

**Solution:** Ensure your embedding model produces 1536-dimension vectors (OpenAI text-embedding-3-small) or update database schema.

## Related Documentation

- [Provider Configuration Guide](./provider-configuration.md) - Embedding provider setup
- [Advanced Retrieval Configuration](./advanced-retrieval-configuration.md) - Semantic search tuning
- [Observability Guide](./observability.md) - Memory metrics and alerts
