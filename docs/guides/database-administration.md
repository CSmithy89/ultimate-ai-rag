# Database Administration Guide

This guide covers the administration, configuration, and maintenance of the three database systems used by the Agentic RAG platform: PostgreSQL with pgvector, Neo4j, and Redis.

## Table of Contents

- [Database Architecture Overview](#database-architecture-overview)
- [PostgreSQL / pgvector Administration](#postgresql--pgvector-administration)
- [Neo4j Administration](#neo4j-administration)
- [Redis Administration](#redis-administration)
- [Health Checks and Monitoring](#health-checks-and-monitoring)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

---

## Database Architecture Overview

The platform uses a polyglot persistence architecture with three specialized databases:

| Database | Purpose | Data Stored |
|----------|---------|-------------|
| **PostgreSQL + pgvector** | Document storage, vector search | Documents, chunks, embeddings, jobs, workspace items, memories, LLM usage metrics |
| **Neo4j** | Knowledge graph | Entities, relationships, documents, chunks, communities |
| **Redis** | Job queues, caching | Ingestion job streams, rate limiting, session data |

### Data Flow

```
User Request
    |
    v
[Backend API] ---> [Redis Streams] ---> Job Workers
    |                                        |
    v                                        v
[PostgreSQL]                           [Neo4j Graph]
 - Documents                            - Entities
 - Chunks + Embeddings                  - Relationships
 - Memories                             - Communities
```

### Docker Compose Services

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]

  neo4j:
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

---

## PostgreSQL / pgvector Administration

### Connection Configuration

PostgreSQL connection is managed via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | Required | PostgreSQL connection URL |
| `DB_POOL_MIN` | `1` | Minimum pool size |
| `DB_POOL_MAX` | `50` | Maximum pool size |

**Example DATABASE_URL:**
```
postgresql://agentic_rag:password@localhost:5432/agentic_rag
```

### Connection Pool Configuration

The `PostgresClient` uses `asyncpg` with connection pooling:

```python
# Connection pool settings (from postgres.py)
self._pool = await asyncpg.create_pool(
    self.url,
    min_size=2,    # Minimum connections kept open
    max_size=10,   # Maximum connections allowed
)
```

**Production Recommendations:**

```bash
# For production workloads with moderate traffic
DB_POOL_MIN=5
DB_POOL_MAX=25

# For high-traffic production
DB_POOL_MIN=10
DB_POOL_MAX=50
```

### Schema and Tables

The platform automatically creates the following tables on startup:

#### Core Tables

| Table | Purpose |
|-------|---------|
| `documents` | Source document metadata |
| `ingestion_jobs` | Job tracking for crawl/parse/index |
| `chunks` | Document chunks with embeddings |
| `hierarchical_chunks` | Multi-level chunks for small-to-big retrieval |

#### Memory Tables (Epic 20)

| Table | Purpose |
|-------|---------|
| `scoped_memories` | User/session/agent memories with embeddings |

#### Workspace Tables (Epic 11)

| Table | Purpose |
|-------|---------|
| `workspace_items` | Saved workspace content |
| `workspace_shares` | Shareable workspace links |
| `workspace_bookmarks` | User bookmarks |

#### Operations Tables (Epic 8)

| Table | Purpose |
|-------|---------|
| `llm_usage_events` | LLM token and cost tracking |
| `llm_cost_alerts` | Cost threshold alerts |

### Index Management

#### Standard Indexes

The system creates B-tree indexes for efficient querying:

```sql
-- Tenant isolation (CRITICAL for multi-tenancy)
CREATE INDEX idx_documents_tenant_id ON documents(tenant_id);
CREATE INDEX idx_chunks_tenant_id ON chunks(tenant_id);
CREATE INDEX idx_scoped_memories_tenant_id ON scoped_memories(tenant_id);

-- Status filtering for job tracking
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);

-- Deduplication
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
```

#### pgvector Indexes (HNSW / IVFFlat)

Vector similarity search uses IVFFlat indexes:

```sql
-- Chunk embeddings (1536-dimension OpenAI vectors)
CREATE INDEX idx_chunks_embedding
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Hierarchical chunk embeddings
CREATE INDEX idx_hierarchical_chunks_embedding
ON hierarchical_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Memory embeddings
CREATE INDEX idx_scoped_memories_embedding
ON scoped_memories USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Index Tuning Guidelines:**

| Data Size | Recommended `lists` Value |
|-----------|---------------------------|
| < 10,000 vectors | 50-100 |
| 10,000 - 100,000 | 100-200 |
| 100,000 - 1,000,000 | 200-500 |
| > 1,000,000 | 500-1000 |

**Rebuild Index (after significant data changes):**

```sql
-- Drop and recreate for optimal performance
DROP INDEX IF EXISTS idx_chunks_embedding;
CREATE INDEX idx_chunks_embedding
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);

-- Analyze table statistics
ANALYZE chunks;
```

### Embedding Validation

Embeddings are validated before storage:

```python
def _validate_embedding(embedding: list[float], expected_dim: int = 1536) -> None:
    """Validates dimension and checks for NaN/Inf values."""
    if len(embedding) != expected_dim:
        raise ValueError(f"Embedding dimension mismatch")
    for i, val in enumerate(embedding):
        if not math.isfinite(val):
            raise ValueError(f"Invalid embedding value at index {i}")
```

### Maintenance Commands

```sql
-- Vacuum and analyze for performance
VACUUM ANALYZE documents;
VACUUM ANALYZE chunks;
VACUUM ANALYZE scoped_memories;

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;

-- Check index usage
SELECT
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

---

## Neo4j Administration

### Connection Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | Required | Bolt connection URI |
| `NEO4J_USER` | Required | Neo4j username |
| `NEO4J_PASSWORD` | Required | Neo4j password |
| `NEO4J_POOL_MIN` | `1` | Minimum pool size |
| `NEO4J_POOL_MAX` | `50` | Maximum pool size |
| `NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS` | `30` | Pool acquisition timeout |
| `NEO4J_CONNECTION_TIMEOUT_SECONDS` | `30` | Connection timeout |
| `NEO4J_MAX_CONNECTION_LIFETIME_SECONDS` | `3600` | Maximum connection age |
| `NEO4J_TRANSACTION_TIMEOUT_SECONDS` | `300` | Query timeout (for LazyRAG) |

**Example Configuration:**

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password
NEO4J_POOL_MIN=2
NEO4J_POOL_MAX=50
NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS=30
NEO4J_CONNECTION_TIMEOUT_SECONDS=30
NEO4J_MAX_CONNECTION_LIFETIME_SECONDS=3600
NEO4J_TRANSACTION_TIMEOUT_SECONDS=300
```

### Connection Pool Management

The `Neo4jClient` implements pool warming for optimal startup:

```python
class Neo4jClient:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        pool_min_size: int = 1,
        pool_max_size: int = 50,
        pool_acquire_timeout: float = 30.0,
        connection_timeout: float = 30.0,
        max_connection_lifetime: int = 3600,
    ) -> None:
        # ...

    async def _warm_pool(self, target_size: int) -> None:
        """Warm the pool by opening concurrent sessions."""
        async def _ping() -> None:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1")
                await result.consume()
        await asyncio.gather(*[_ping() for _ in range(target_size)])
```

### Graph Schema

#### Node Labels

| Label | Description | Key Properties |
|-------|-------------|----------------|
| `Entity` | Extracted entities | `id`, `tenant_id`, `name`, `type`, `description` |
| `Document` | Source documents | `id`, `tenant_id`, `title`, `source_url` |
| `Chunk` | Document chunks | `id`, `tenant_id`, `document_id`, `chunk_index` |
| `Community` | Entity communities | `id`, `tenant_id`, `level`, `summary` |

#### Relationship Types

| Relationship | Description | Properties |
|--------------|-------------|------------|
| `MENTIONS` | Chunk mentions entity | `confidence` |
| `AUTHORED_BY` | Entity authored by another | `confidence`, `description` |
| `PART_OF` | Entity is part of another | `confidence` |
| `USES` | Entity uses another | `confidence` |
| `RELATED_TO` | General relationship | `confidence`, `description` |
| `CONTAINS` | Document contains chunk | - |
| `CALLS` | Code symbol calls another | `confidence` |
| `IMPORTS` | Code imports module | `confidence` |
| `EXTENDS` | Class extends another | `confidence` |
| `IMPLEMENTS` | Class implements interface | `confidence` |
| `DEFINED_IN` | Symbol defined in file | `confidence` |
| `USES_TYPE` | Code uses type | `confidence` |

### Index Management

Indexes are automatically created on startup:

```cypher
-- Entity indexes
CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id);
CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);

-- Document indexes
CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id);
CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id);

-- Chunk indexes
CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id);
CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id);

-- Community indexes (Story 20-B1)
CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.id);
CREATE INDEX community_tenant IF NOT EXISTS FOR (c:Community) ON (c.tenant_id);
CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.tenant_id, c.level);
```

### Graph Maintenance

**Check Graph Statistics:**

```cypher
-- Node counts by label
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
RETURN label, value.count;

-- Relationship counts by type
CALL db.relationshipTypes() YIELD relationshipType
CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {}) YIELD value
RETURN relationshipType, value.count;

-- Check for orphan nodes (no relationships)
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE NOT (e)-[]-()
RETURN count(e) as orphan_count;
```

**Tenant-Scoped Cleanup:**

```cypher
-- Delete all data for a specific tenant
MATCH (n {tenant_id: $tenant_id})
DETACH DELETE n;
```

**Memory Settings (neo4j.conf):**

```properties
# For production workloads
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
dbms.memory.pagecache.size=2g
```

---

## Redis Administration

### Connection Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | Required | Redis connection URL |
| `RATE_LIMIT_BACKEND` | `memory` | Rate limit storage (`memory` or `redis`) |
| `RATE_LIMIT_REDIS_PREFIX` | `rate-limit` | Key prefix for rate limiting |

**Example Configuration:**

```bash
REDIS_URL=redis://localhost:6379
RATE_LIMIT_BACKEND=redis
RATE_LIMIT_REDIS_PREFIX=rag-rate-limit
```

### Redis Streams (Job Queues)

The platform uses Redis Streams for job processing:

| Stream | Consumer Group | Purpose |
|--------|----------------|---------|
| `crawl.jobs` | `crawl-workers` | URL crawling jobs |
| `parse.jobs` | `parse-workers` | Document parsing jobs |
| `index.jobs` | `index-workers` | Embedding/indexing jobs |

**Stream Operations:**

```bash
# Check stream length
redis-cli XLEN crawl.jobs

# Check pending messages
redis-cli XPENDING crawl.jobs crawl-workers

# View stream entries
redis-cli XRANGE crawl.jobs - + COUNT 10

# Monitor streams in real-time
redis-cli MONITOR
```

### Memory Management

**Check Memory Usage:**

```bash
redis-cli INFO memory
```

**Key Metrics:**
- `used_memory_human`: Current memory usage
- `used_memory_peak_human`: Peak memory usage
- `maxmemory`: Maximum configured memory

**Memory Configuration (redis.conf):**

```conf
# Set maximum memory limit
maxmemory 1gb

# Eviction policy for caching
maxmemory-policy allkeys-lru
```

### Caching Configuration

The platform uses Redis for:

1. **Rate Limiting** - Request rate limiting per tenant/IP
2. **Reranking Cache** (Story 19-G1) - Cache reranking results

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_CACHE_ENABLED` | `false` | Enable reranking cache |
| `RERANKER_CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `RERANKER_CACHE_MAX_SIZE` | `10000` | Maximum cache entries |

---

## Health Checks and Monitoring

### Application Health Endpoint

```bash
# Basic health check
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

### Docker Compose Health Checks

```yaml
services:
  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-agentic_rag}"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -a bolt://localhost:7687 -u neo4j -p password 'RETURN 1'"]
      interval: 15s
      timeout: 10s
      retries: 10

  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### Database Connectivity Checks

**PostgreSQL:**
```sql
SELECT 1;  -- Simple connectivity
SELECT pg_is_in_recovery();  -- Check if replica
```

**Neo4j:**
```cypher
RETURN 1;  -- Simple connectivity
CALL dbms.cluster.role() YIELD role;  -- Cluster status
```

**Redis:**
```bash
redis-cli PING  # Returns PONG
redis-cli INFO server  # Server information
```

### Prometheus Metrics

When `PROMETHEUS_ENABLED=true`, metrics are available at `/metrics`:

```yaml
# Database-related metrics
pg_pool_size
pg_pool_available
neo4j_pool_size
neo4j_pool_available
redis_connected_clients
```

---

## Backup and Recovery

### PostgreSQL Backup

**Full Backup with pg_dump:**

```bash
# Backup single database
pg_dump -h localhost -U agentic_rag -d agentic_rag -F c -f backup.dump

# Restore
pg_restore -h localhost -U agentic_rag -d agentic_rag backup.dump
```

**Continuous Archiving (Production):**

```bash
# Enable WAL archiving in postgresql.conf
archive_mode = on
archive_command = 'cp %p /path/to/archive/%f'

# Point-in-time recovery
recovery_target_time = '2026-01-12 12:00:00'
```

**Docker Volume Backup:**

```bash
# Stop container and backup volume
docker compose stop postgres
docker run --rm -v postgres-data:/data -v $(pwd):/backup alpine \
    tar czf /backup/postgres-backup.tar.gz /data
docker compose start postgres
```

### Neo4j Backup

**Online Backup (Enterprise):**

```bash
neo4j-admin backup --database=neo4j --backup-dir=/backups
```

**Docker Volume Backup:**

```bash
# Stop container and backup volume
docker compose stop neo4j
docker run --rm -v neo4j-data:/data -v $(pwd):/backup alpine \
    tar czf /backup/neo4j-backup.tar.gz /data
docker compose start neo4j
```

**Cypher Export (for smaller datasets):**

```cypher
CALL apoc.export.cypher.all('/exports/backup.cypher', {format: 'cypher-shell'});
```

### Redis Backup

**RDB Snapshot:**

```bash
# Trigger manual save
redis-cli BGSAVE

# Check save status
redis-cli LASTSAVE

# Backup file location (default)
/var/lib/redis/dump.rdb
```

**AOF Persistence:**

```conf
# Enable in redis.conf
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
```

---

## Troubleshooting

### PostgreSQL Issues

**Connection Pool Exhaustion:**

```
DatabaseError: connection pool exhausted
```

**Solution:** Increase `DB_POOL_MAX` or check for connection leaks.

```bash
# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'agentic_rag';
```

**pgvector Index Not Used:**

```sql
-- Check if index is being used
EXPLAIN ANALYZE
SELECT * FROM chunks
WHERE tenant_id = $1
ORDER BY embedding <=> $2::vector
LIMIT 10;
```

**Solution:** Ensure `embedding IS NOT NULL` filter is present, and run `ANALYZE`.

**Embedding Dimension Mismatch:**

```
ValueError: Embedding dimension mismatch: expected 1536, got 768
```

**Solution:** Ensure embedding model matches schema (default: OpenAI 1536 dimensions).

### Neo4j Issues

**Connection Timeout:**

```
Neo4jError: connection: Connection timeout
```

**Solution:** Increase `NEO4J_CONNECTION_TIMEOUT_SECONDS` or check network connectivity.

**Query Timeout (LazyRAG):**

```
TransactionTimeout: Transaction timed out
```

**Solution:** Increase `NEO4J_TRANSACTION_TIMEOUT_SECONDS` for expensive traversals.

```bash
NEO4J_TRANSACTION_TIMEOUT_SECONDS=600  # 10 minutes
```

**Memory Issues:**

```
OutOfMemoryError: Java heap space
```

**Solution:** Increase Neo4j heap in `neo4j.conf`:

```properties
dbms.memory.heap.max_size=4g
```

### Redis Issues

**Connection Refused:**

```
RedisError: connection: Redis client not connected
```

**Solution:** Check Redis is running and `REDIS_URL` is correct.

**Memory Limit Reached:**

```
OOM command not allowed when used memory > 'maxmemory'
```

**Solution:** Increase `maxmemory` or configure eviction policy.

**Consumer Group Lag:**

```bash
# Check consumer group lag
redis-cli XINFO GROUPS crawl.jobs
```

**Solution:** Scale workers or investigate slow consumers.

### Multi-Tenancy Issues

**Cross-Tenant Data Access:**

All database queries MUST include `tenant_id` filtering. If data leakage is suspected:

```sql
-- PostgreSQL: Check for missing tenant filters
SELECT * FROM chunks WHERE tenant_id IS NULL;

-- Neo4j: Check for untagged nodes
MATCH (n) WHERE n.tenant_id IS NULL RETURN n LIMIT 10;
```

### Performance Diagnostics

**PostgreSQL Slow Queries:**

```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = '1000';  -- Log queries > 1s
SELECT pg_reload_conf();

-- Check slow query log
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

**Neo4j Query Profiling:**

```cypher
PROFILE MATCH (e:Entity {tenant_id: $tenant_id})-[r*1..2]-(target)
RETURN e, r, target LIMIT 100;
```

**Redis Latency:**

```bash
redis-cli --latency
redis-cli --latency-history
```

---

## Environment Variable Reference

### PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | Required | Connection URL |
| `DB_POOL_MIN` | `1` | Min pool size |
| `DB_POOL_MAX` | `50` | Max pool size |

### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | Required | Bolt URI |
| `NEO4J_USER` | Required | Username |
| `NEO4J_PASSWORD` | Required | Password |
| `NEO4J_POOL_MIN` | `1` | Min pool size |
| `NEO4J_POOL_MAX` | `50` | Max pool size |
| `NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS` | `30` | Acquire timeout |
| `NEO4J_CONNECTION_TIMEOUT_SECONDS` | `30` | Connection timeout |
| `NEO4J_MAX_CONNECTION_LIFETIME_SECONDS` | `3600` | Max connection age |
| `NEO4J_TRANSACTION_TIMEOUT_SECONDS` | `300` | Query timeout |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | Required | Connection URL |
| `RATE_LIMIT_BACKEND` | `memory` | Rate limit storage |
| `RATE_LIMIT_REDIS_PREFIX` | `rate-limit` | Key prefix |
| `RERANKER_CACHE_ENABLED` | `false` | Enable rerank cache |
| `RERANKER_CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `RERANKER_CACHE_MAX_SIZE` | `10000` | Max cache entries |

---

## See Also

- [Provider Configuration Guide](./provider-configuration.md) - LLM and embedding provider setup
- [Observability Guide](./observability.md) - Prometheus metrics and alerts
- [Memory Platform Guide](./memory-platform.md) - Memory scopes and consolidation
- [Ingestion Pipeline Guide](./ingestion-pipeline.md) - Document processing configuration
