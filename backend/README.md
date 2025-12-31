# Backend

## Setup

```bash
uv sync
```

## Run (local)

```bash
uv run agentic-rag-backend
```

The service starts on `http://localhost:8000`.

## Neo4j pooling settings

Configure the Neo4j connection pool with optional environment variables:

```
NEO4J_POOL_MIN=1
NEO4J_POOL_MAX=50
NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS=30
NEO4J_CONNECTION_TIMEOUT_SECONDS=30
NEO4J_MAX_CONNECTION_LIFETIME_SECONDS=3600
```

These defaults are conservative for local development; production deployments can
increase pool sizes based on expected concurrency.
