# Integration Tests

Integration tests exercise real services (PostgreSQL, Neo4j, Redis).
They are gated behind `INTEGRATION_TESTS=1` and will skip if services
are unavailable.

## Prerequisites

1. Start services:

```bash
docker compose up -d postgres neo4j redis
```

2. Export environment variables:

```bash
export INTEGRATION_TESTS=1
export DATABASE_URL=postgresql://agentic_rag:agentic_rag@localhost:5432/agentic_rag
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4j_password
export REDIS_URL=redis://localhost:6379
```

## Running Integration Tests

```bash
cd backend
uv run pytest -m integration
```

## Notes

- Integration tests use isolated tenant IDs and clean up after each test.
- If services are not reachable, tests skip with explicit reasons.
