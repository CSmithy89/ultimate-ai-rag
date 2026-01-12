# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Agentic RAG + GraphRAG platform. For each problem, we provide symptoms, diagnostic commands, and solutions.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation and Setup Issues](#installation-and-setup-issues)
- [Database Connection Issues](#database-connection-issues)
  - [PostgreSQL Issues](#postgresql-issues)
  - [Neo4j Issues](#neo4j-issues)
  - [Redis Issues](#redis-issues)
- [API Errors](#api-errors)
- [Ingestion Problems](#ingestion-problems)
- [Retrieval Quality Issues](#retrieval-quality-issues)
- [Memory and Performance Issues](#memory-and-performance-issues)
- [Debug Mode Configuration](#debug-mode-configuration)
- [Log Analysis](#log-analysis)
- [Diagnostic Commands Reference](#diagnostic-commands-reference)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Run these commands first to quickly assess system health.

### Health Check Commands

```bash
# Full system diagnostic (CLI)
rag-cli doctor

# JSON output for scripting
rag-cli doctor --json

# Quick check (skip service health)
rag-cli doctor --quick

# Check specific service
rag-cli doctor --service backend
rag-cli doctor --service frontend
```

### Service Health Endpoints

```bash
# Backend health check
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Backend readiness (includes database checks)
curl http://localhost:8000/health/ready
# Expected: {"status": "ready", "databases": {"postgres": "ok", "neo4j": "ok", "redis": "ok"}}

# Frontend health check
curl http://localhost:3000
# Expected: HTTP 200

# API documentation
open http://localhost:8000/docs
```

### Docker Container Status

```bash
# Check all container status
docker compose ps

# View container logs (last 100 lines)
docker compose logs --tail=100

# View specific service logs
docker compose logs backend --tail=50
docker compose logs postgres --tail=50
docker compose logs neo4j --tail=50
docker compose logs redis --tail=50

# Real-time log streaming
docker compose logs -f backend

# Check container resource usage
docker stats
```

### Quick Database Connectivity Tests

```bash
# PostgreSQL connectivity
docker compose exec postgres pg_isready -U agentic_rag

# Neo4j connectivity
docker compose exec neo4j cypher-shell -a bolt://localhost:7687 -u neo4j -p neo4j_password "RETURN 1"

# Redis connectivity
docker compose exec redis redis-cli ping
# Expected: PONG
```

---

## Installation and Setup Issues

### Docker Not Running

**Symptom:**
```
Docker daemon not running. Start Docker Desktop
```

**Solutions:**

1. **macOS/Windows:** Start Docker Desktop application from the system tray or Applications folder

2. **Linux:** Start the Docker service:
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker  # Enable on boot
   ```

3. **Verify Docker is running:**
   ```bash
   docker info
   docker version
   ```

4. **Check Docker socket permissions (Linux):**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker  # Apply without logout
   ```

### Port Already in Use

**Symptom:**
```
Error: Port 8000 is already in use
Bind for 0.0.0.0:8000 failed: port is already allocated
```

**Diagnostic:**
```bash
# Find process using the port (Linux/macOS)
lsof -i :8000
lsof -i :3000
lsof -i :5432
lsof -i :7687
lsof -i :6379

# Windows
netstat -ano | findstr :8000
```

**Solutions:**

1. **Stop the conflicting process:**
   ```bash
   kill -9 <PID>
   ```

2. **Stop existing Docker containers:**
   ```bash
   docker compose down
   docker compose up -d
   ```

3. **Change ports in docker-compose.yml:**
   ```yaml
   services:
     backend:
       ports:
         - "8001:8000"  # Use alternative port
   ```

**Common port conflicts:**
| Port | Service | Common Conflicts |
|------|---------|-----------------|
| 8000 | Backend API | Django, Flask, other Python apps |
| 3000 | Frontend | Create React App, Next.js |
| 5432 | PostgreSQL | Other PostgreSQL instances |
| 7474/7687 | Neo4j | Other Neo4j instances |
| 6379 | Redis | Other Redis instances |

### Missing .env File

**Symptom:**
```
FileNotFoundError: .env file not found
Settings validation error: OPENAI_API_KEY required
```

**Diagnostic:**
```bash
ls -la .env
rag-cli doctor
```

**Solutions:**

1. **Create from example:**
   ```bash
   cp .env.example .env
   ```

2. **Auto-fix with CLI:**
   ```bash
   rag-cli doctor --fix
   ```

3. **Run installation wizard:**
   ```bash
   rag-install
   ```

### Invalid API Key

**Symptom:**
```
Invalid key format. Please try again.
401 Unauthorized: Invalid API key
```

**Diagnostic:**
```bash
# Check key format
echo $OPENAI_API_KEY | head -c 10
# OpenAI: sk-...
# Anthropic: sk-ant-...
# OpenRouter: sk-or-...
```

**Solutions:**

1. **Verify key prefix:**
   | Provider | Required Prefix |
   |----------|-----------------|
   | OpenAI | `sk-` |
   | Anthropic | `sk-ant-` |
   | OpenRouter | `sk-or-` |
   | Gemini | Any |
   | Ollama | None (local) |

2. **Check for whitespace:**
   ```bash
   # Remove leading/trailing whitespace
   OPENAI_API_KEY=$(echo "$OPENAI_API_KEY" | tr -d '[:space:]')
   ```

3. **Test API key directly:**
   ```bash
   # OpenAI
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Services Not Starting

**Symptom:**
```
Backend failed to become healthy. Check for port conflicts or docker logs.
Container exited with code 1
```

**Diagnostic:**
```bash
docker compose logs backend
docker compose ps
```

**Solutions:**

1. **Check disk space:**
   ```bash
   df -h
   docker system df
   ```

2. **Clean Docker resources:**
   ```bash
   docker system prune -f
   docker volume prune -f
   ```

3. **Recreate containers:**
   ```bash
   docker compose down -v
   docker compose up -d --build
   ```

4. **Check environment variables:**
   ```bash
   docker compose config  # Validate compose file
   ```

---

## Database Connection Issues

### PostgreSQL Issues

#### Connection Pool Exhausted

**Symptom:**
```
DatabaseError: connection pool exhausted
asyncpg.exceptions.TooManyConnectionsError
```

**Diagnostic:**
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'agentic_rag';

-- View connection details
SELECT pid, usename, application_name, client_addr, state, query_start
FROM pg_stat_activity
WHERE datname = 'agentic_rag'
ORDER BY query_start DESC;
```

**Solutions:**

1. **Increase pool size in .env:**
   ```bash
   DB_POOL_MIN=5
   DB_POOL_MAX=50
   ```

2. **Kill idle connections:**
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'agentic_rag'
   AND state = 'idle'
   AND query_start < NOW() - INTERVAL '5 minutes';
   ```

3. **Check for connection leaks in application code**

#### Connection Refused

**Symptom:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Diagnostic:**
```bash
docker compose ps postgres
docker compose logs postgres --tail=50
```

**Solutions:**

1. **Verify PostgreSQL is running:**
   ```bash
   docker compose up -d postgres
   docker compose exec postgres pg_isready
   ```

2. **Check DATABASE_URL format:**
   ```bash
   # Correct format
   DATABASE_URL=postgresql://agentic_rag:password@localhost:5432/agentic_rag
   ```

3. **Verify network connectivity:**
   ```bash
   docker network inspect agentic-rag-graphrag-copilot_default
   ```

#### Embedding Dimension Mismatch

**Symptom:**
```
ValueError: Embedding dimension mismatch: expected 1536, got 768
pgvector: different vector dimensions
```

**Solutions:**

1. **Ensure consistent embedding model:**
   ```bash
   # Check current model
   echo $EMBEDDING_MODEL

   # OpenAI models: 1536 dimensions
   EMBEDDING_MODEL=text-embedding-3-small

   # Some models: 768 dimensions
   # Ensure schema matches your model
   ```

2. **Reset and re-index (data loss warning):**
   ```sql
   -- Drop and recreate tables with correct dimensions
   DROP TABLE IF EXISTS chunks CASCADE;
   -- Restart backend to recreate tables
   ```

### Neo4j Issues

#### Connection Timeout

**Symptom:**
```
Neo4jError: connection: Connection timeout
ServiceUnavailable: Unable to retrieve routing information
```

**Diagnostic:**
```bash
docker compose ps neo4j
docker compose logs neo4j --tail=50
curl http://localhost:7474
```

**Solutions:**

1. **Increase timeout:**
   ```bash
   NEO4J_CONNECTION_TIMEOUT_SECONDS=60
   NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS=60
   ```

2. **Wait for Neo4j startup (can take 60-90 seconds):**
   ```bash
   # Neo4j health check
   until docker compose exec neo4j cypher-shell -a bolt://localhost:7687 -u neo4j -p neo4j_password "RETURN 1" 2>/dev/null; do
     echo "Waiting for Neo4j..."
     sleep 5
   done
   ```

3. **Check Neo4j credentials:**
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=neo4j_password
   ```

#### Query Timeout (LazyRAG)

**Symptom:**
```
TransactionTimeout: Transaction timed out
neo4j.exceptions.TransientError
```

**Solutions:**

1. **Increase transaction timeout:**
   ```bash
   NEO4J_TRANSACTION_TIMEOUT_SECONDS=600  # 10 minutes
   ```

2. **Optimize query or reduce traversal depth:**
   ```cypher
   -- Add LIMIT to expensive queries
   MATCH (e:Entity {tenant_id: $tenant_id})-[r*1..2]-(target)
   RETURN e, r, target LIMIT 100
   ```

#### Memory Issues

**Symptom:**
```
OutOfMemoryError: Java heap space
neo4j.exceptions.ServiceUnavailable
```

**Diagnostic:**
```bash
docker stats neo4j
docker compose logs neo4j | grep -i memory
```

**Solutions:**

1. **Increase Neo4j memory in docker-compose.yml:**
   ```yaml
   neo4j:
     environment:
       NEO4J_dbms_memory_heap_initial__size: 2g
       NEO4J_dbms_memory_heap_max__size: 4g
       NEO4J_dbms_memory_pagecache_size: 2g
   ```

2. **Increase Docker memory allocation:**
   - Docker Desktop > Settings > Resources > Memory
   - Recommended: 8GB minimum for Neo4j + other services

### Redis Issues

#### Connection Refused

**Symptom:**
```
RedisError: connection: Redis client not connected
redis.exceptions.ConnectionError
```

**Diagnostic:**
```bash
docker compose ps redis
docker compose exec redis redis-cli ping
```

**Solutions:**

1. **Verify Redis is running:**
   ```bash
   docker compose up -d redis
   docker compose exec redis redis-cli ping
   ```

2. **Check REDIS_URL:**
   ```bash
   REDIS_URL=redis://localhost:6379
   ```

#### Memory Limit Reached

**Symptom:**
```
OOM command not allowed when used memory > 'maxmemory'
redis.exceptions.ResponseError
```

**Diagnostic:**
```bash
docker compose exec redis redis-cli INFO memory
```

**Solutions:**

1. **Increase maxmemory:**
   ```bash
   docker compose exec redis redis-cli CONFIG SET maxmemory 2gb
   ```

2. **Configure eviction policy:**
   ```bash
   docker compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

3. **Clear unnecessary data:**
   ```bash
   docker compose exec redis redis-cli FLUSHDB
   ```

---

## API Errors

### Common HTTP Error Codes

| Code | Error | Common Causes |
|------|-------|---------------|
| 400 | Bad Request | Invalid parameters, malformed JSON |
| 401 | Unauthorized | Missing/invalid tenant_id or API key |
| 404 | Not Found | Resource doesn't exist |
| 413 | Payload Too Large | File exceeds size limit |
| 422 | Unprocessable Entity | Validation failed, hallucination detected |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side exception |
| 502 | Bad Gateway | Backend not responding |
| 503 | Service Unavailable | Database/service down |
| 504 | Gateway Timeout | Request timeout |

### Missing tenant_id

**Symptom:**
```json
{
  "type": "https://api.example.com/errors/tenant-required",
  "title": "Tenant Required",
  "status": 401,
  "detail": "tenant_id is required for this operation"
}
```

**Solution:**
Add `tenant_id` header or query parameter to all requests:
```bash
curl -H "X-Tenant-ID: my-tenant" http://localhost:8000/api/v1/query
```

### Rate Limit Exceeded

**Symptom:**
```json
{
  "type": "https://api.example.com/errors/rate-limit-exceeded",
  "status": 429,
  "detail": "Rate limit exceeded",
  "errors": {"retry_after": 60}
}
```

**Solution:**
1. Wait for the `Retry-After` header duration
2. Implement exponential backoff in client code
3. Request rate limit increase if needed

### Validation Errors

**Symptom:**
```json
{
  "type": "https://api.example.com/errors/validation-error",
  "status": 400,
  "detail": "Invalid request parameters",
  "errors": {"url": "URL must be valid HTTP(S)"}
}
```

**Solutions:**
1. Check request payload against API documentation at `/docs`
2. Validate URLs are properly formatted
3. Ensure required fields are present

---

## Ingestion Problems

### Crawl Failed

**Symptom:**
```json
{
  "type": "https://api.example.com/errors/crawl-failed",
  "status": 500,
  "detail": "Crawl failed: Connection timeout"
}
```

**Diagnostic:**
```bash
# Test URL accessibility
curl -I https://example.com

# Check crawler logs
docker compose logs backend | grep -i crawl
```

**Solutions:**

1. **Verify URL is accessible:**
   ```bash
   curl -v https://example.com
   ```

2. **Try different crawl profile:**
   ```bash
   # In .env
   CRAWL_PROFILE=stealth  # Options: fast, thorough, stealth
   ```

3. **Enable fallback providers:**
   ```bash
   CRAWL_FALLBACK_ENABLED=true
   APIFY_API_KEY=your-key
   BRIGHTDATA_API_KEY=your-key
   ```

4. **Check rate limiting:**
   - Some sites block rapid requests
   - Use `stealth` profile for aggressive sites

### PDF Parse Failed

**Symptom:**
```json
{
  "type": "https://api.example.com/errors/parse-failed",
  "status": 500,
  "detail": "Failed to parse document: Unsupported PDF format"
}
```

**Solutions:**

1. **Check file size:**
   ```bash
   # Default limit: 50MB
   MAX_PDF_SIZE_MB=100
   ```

2. **Password-protected PDFs are not supported:**
   - Remove password protection before uploading

3. **Try different PDF:**
   - Some PDFs with complex layouts may fail
   - Image-only PDFs require OCR (not currently supported)

### YouTube Ingestion Failed

**Symptom:**
```
Ingestion failed: No transcript available
```

**Solutions:**

1. **Verify video has captions:**
   - Check video on YouTube for CC button
   - Auto-generated captions may not be available for all videos

2. **Check YouTube API quota:**
   ```bash
   # If using YouTube Data API
   echo $YOUTUBE_API_KEY
   ```

### Entity Extraction Slow

**Symptom:**
Ingestion jobs taking very long (>5 minutes per document)

**Diagnostic:**
```bash
# Check job status
curl "http://localhost:8000/api/v1/ingestion/jobs?tenant_id=my-tenant"

# Monitor LLM usage
curl "http://localhost:8000/api/v1/ops/costs/summary?tenant_id=my-tenant"
```

**Solutions:**

1. **Use faster model for extraction:**
   ```bash
   LLM_MODEL_ID=gpt-4o-mini  # Faster, cheaper
   ```

2. **Reduce chunk size:**
   ```bash
   CHUNK_SIZE=500  # Default: 1000
   ```

3. **Disable contextual retrieval during ingestion:**
   ```bash
   CONTEXTUAL_RETRIEVAL_ENABLED=false
   ```

---

## Retrieval Quality Issues

### Empty or Irrelevant Results

**Symptom:**
Queries return no results or irrelevant documents

**Diagnostic:**
```bash
# Check document count
curl "http://localhost:8000/api/v1/documents?tenant_id=my-tenant"

# Check Prometheus metrics (if enabled)
curl http://localhost:8000/metrics | grep retrieval
```

**Solutions:**

1. **Verify documents are indexed:**
   ```sql
   -- PostgreSQL
   SELECT COUNT(*) FROM chunks WHERE tenant_id = 'my-tenant';
   ```

2. **Adjust retrieval strategy:**
   ```bash
   RETRIEVAL_STRATEGY=hybrid  # vector, graph, or hybrid
   ```

3. **Lower similarity threshold:**
   ```bash
   VECTOR_SIMILARITY_THRESHOLD=0.5  # Default: 0.7
   ```

4. **Enable reranking:**
   ```bash
   RERANKER_ENABLED=true
   RERANKER_PROVIDER=flashrank  # cohere or flashrank
   ```

### High Fallback Rate

**Symptom:**
```
Prometheus: retrieval_fallback_triggered_total rate > 10/min
```

**Diagnostic:**
```bash
# Check fallback reasons
curl http://localhost:8000/metrics | grep fallback
```

**Solutions by Reason:**

1. **`low_score` - Grader threshold too high:**
   ```bash
   GRADER_THRESHOLD=0.3  # Default: 0.5, lower = more permissive
   ```

2. **`empty_results` - Index issues:**
   - Verify documents are indexed
   - Check embedding model consistency

3. **`timeout` - Query too complex:**
   ```bash
   RETRIEVAL_TIMEOUT_SECONDS=30  # Increase timeout
   ```

### Slow Retrieval

**Symptom:**
p95 latency > 2 seconds

**Diagnostic:**
```bash
# Check phase breakdown
curl http://localhost:8000/metrics | grep retrieval_latency
```

**Solutions by Phase:**

1. **`embed` phase slow:**
   - Check embedding provider latency
   - Use local embeddings for faster response

2. **`search` phase slow:**
   ```sql
   -- Rebuild pgvector index
   REINDEX INDEX idx_chunks_embedding;
   ANALYZE chunks;
   ```

3. **`rerank` phase slow:**
   ```bash
   RERANKER_TOP_K=5  # Reduce from default 10
   RERANKER_CACHE_ENABLED=true
   ```

4. **`grade` phase slow:**
   ```bash
   GRADER_MODEL=gpt-4o-mini  # Use faster model
   ```

---

## Memory and Performance Issues

### Out of Memory (Backend)

**Symptom:**
```
Container killed: Out of memory
MemoryError: Unable to allocate
```

**Diagnostic:**
```bash
docker stats
free -h  # Linux
```

**Solutions:**

1. **Use minimal profile:**
   ```bash
   rag-install --profile minimal
   ```

2. **Increase Docker memory:**
   - Docker Desktop > Settings > Resources > Memory
   - Recommended: 8GB minimum, 16GB for standard

3. **Reduce batch sizes:**
   ```bash
   EMBEDDING_BATCH_SIZE=10  # Default: 100
   CHUNK_SIZE=500  # Default: 1000
   ```

4. **Disable unused features:**
   ```bash
   CONTEXTUAL_RETRIEVAL_ENABLED=false
   VOICE_IO_ENABLED=false
   COMMUNITY_DETECTION_ENABLED=false
   ```

### High CPU Usage

**Symptom:**
Backend using 100% CPU continuously

**Diagnostic:**
```bash
docker stats
docker compose logs backend | grep -i error
```

**Solutions:**

1. **Check for infinite loops in logs**

2. **Reduce concurrent operations:**
   ```bash
   CRAWL_MAX_CONCURRENT=5  # Default: 10
   ```

3. **Use rate limiting:**
   ```bash
   RATE_LIMIT_BACKEND=redis
   RATE_LIMIT_REQUESTS_PER_MINUTE=60
   ```

### Disk Space Issues

**Symptom:**
```
No space left on device
docker: write /var/lib/docker/...: no space left on device
```

**Diagnostic:**
```bash
df -h
docker system df
```

**Solutions:**

1. **Clean Docker resources:**
   ```bash
   docker system prune -af
   docker volume prune -f
   docker builder prune -af
   ```

2. **Remove old images:**
   ```bash
   docker image prune -af
   ```

3. **Clean PostgreSQL:**
   ```sql
   VACUUM FULL;
   ```

---

## Debug Mode Configuration

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG
DEBUG=true

# For specific modules
LOG_LEVEL_RETRIEVAL=DEBUG
LOG_LEVEL_INGESTION=DEBUG
LOG_LEVEL_AGENTS=DEBUG
```

### Enable Trajectory Debugging

```bash
# Enable trajectory logging
TRAJECTORY_LOGGING_ENABLED=true

# View trajectories
curl "http://localhost:8000/api/v1/trajectories?tenant_id=my-tenant"

# View specific trajectory events
curl "http://localhost:8000/api/v1/trajectories/{trajectory_id}/events"
```

### Enable Prometheus Dev Console

```bash
# In .env
PROMETHEUS_ENABLED=true
PROMETHEUS_PATH=/metrics

# View metrics
curl http://localhost:8000/metrics
```

### Frontend Debug Mode

```typescript
// In frontend/.env.local
NEXT_PUBLIC_DEBUG=true
NEXT_PUBLIC_SHOW_DEV_CONSOLE=true
```

### CopilotKit Development Console

Enable the development console to inspect AI interactions:

```tsx
<CopilotKit showDevConsole={true}>
  {/* ... */}
</CopilotKit>
```

---

## Log Analysis

### Structured Log Format

All logs use JSON format with consistent fields:

```json
{
  "timestamp": "2026-01-13T10:30:00.000000Z",
  "level": "info",
  "logger": "agentic_rag_backend.retrieval.pipeline",
  "event": "retrieval_completed",
  "tenant_id": "tenant-123",
  "request_id": "req-abc-123",
  "duration_ms": 245
}
```

### Common Log Searches

```bash
# Find errors
docker compose logs backend 2>&1 | jq 'select(.level == "error")'

# Find slow queries (>1s)
docker compose logs backend 2>&1 | jq 'select(.duration_ms > 1000)'

# Find by tenant
docker compose logs backend 2>&1 | jq 'select(.tenant_id == "my-tenant")'

# Find by request ID
docker compose logs backend 2>&1 | jq 'select(.request_id == "abc-123")'
```

### Log Level Reference

| Level | Description | Production |
|-------|-------------|------------|
| `debug` | Detailed diagnostic info | Disabled |
| `info` | Normal operations | Enabled |
| `warning` | Recoverable issues | Enabled |
| `error` | Errors requiring attention | Enabled |
| `exception` | Errors with stack traces | Enabled |

### Log Rotation (Production)

```bash
# Docker log rotation in daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}
```

---

## Diagnostic Commands Reference

### System Information

```bash
# System resources
free -h          # Memory (Linux)
df -h            # Disk space
nproc            # CPU cores
nvidia-smi       # GPU info (if available)

# Docker resources
docker system df
docker stats --no-stream

# Network
netstat -tlnp    # Listening ports
ss -tlnp         # Alternative to netstat
```

### Backend Diagnostics

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/health/ready

# API info
curl http://localhost:8000/openapi.json | jq '.info'

# Metrics
curl http://localhost:8000/metrics

# Configuration (development only)
curl http://localhost:8000/api/v1/debug/config
```

### Database Diagnostics

```bash
# PostgreSQL
docker compose exec postgres psql -U agentic_rag -c "SELECT version();"
docker compose exec postgres psql -U agentic_rag -c "SELECT count(*) FROM chunks;"
docker compose exec postgres psql -U agentic_rag -c "\dt"  # List tables

# Neo4j
docker compose exec neo4j cypher-shell -u neo4j -p neo4j_password "CALL db.stats.retrieve('GRAPH COUNTS');"
docker compose exec neo4j cypher-shell -u neo4j -p neo4j_password "MATCH (n) RETURN labels(n), count(*) GROUP BY labels(n);"

# Redis
docker compose exec redis redis-cli INFO
docker compose exec redis redis-cli DBSIZE
docker compose exec redis redis-cli KEYS "*"
```

### CLI Diagnostics

```bash
# Full diagnostic
rag-cli doctor

# Check update status
rag-cli update check

# Validate configuration
rag-cli migrate analyze
```

---

## Getting Help

### Self-Service Resources

1. **API Documentation:** http://localhost:8000/docs (when running)
2. **Configuration Guide:** [docs/guides/provider-configuration.md](./provider-configuration.md)
3. **Database Administration:** [docs/guides/database-administration.md](./database-administration.md)
4. **Observability Guide:** [docs/guides/observability.md](./observability.md)

### Diagnostic Information to Collect

When reporting issues, include:

1. **System information:**
   ```bash
   uname -a
   docker version
   docker compose version
   ```

2. **Configuration:**
   ```bash
   rag-cli doctor --json
   ```

3. **Relevant logs:**
   ```bash
   docker compose logs --tail=200 > logs.txt
   ```

4. **Error messages (full RFC 7807 response)**

5. **Steps to reproduce**

### Support Channels

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** Check guides in `docs/guides/`
- **CLI Help:** `rag-cli --help` or `rag-install --help`

### Common RFC 7807 Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `validation_error` | Invalid request data | Check request format |
| `invalid_url` | URL not accessible | Verify URL works |
| `tenant_required` | Missing tenant_id | Add tenant header |
| `crawl_failed` | Crawl error | Check URL, try stealth profile |
| `database_error` | PostgreSQL error | Check connection |
| `neo4j_error` | Neo4j error | Check connection |
| `redis_error` | Redis error | Check connection |
| `embedding_failed` | Embedding error | Check API key |
| `ingestion_failed` | Ingestion error | Check logs |
| `rate_limit_exceeded` | Too many requests | Wait and retry |

---

## See Also

- [CLI Installation Manual](./cli-installation.md) - CLI setup and commands
- [Provider Configuration Guide](./provider-configuration.md) - LLM and embedding setup
- [Database Administration Guide](./database-administration.md) - Database management
- [Observability Guide](./observability.md) - Metrics and monitoring
- [Deployment Guide](./deployment-production.md) - Production deployment
