# Quickstart Guide

Get from zero to your first RAG query in under 10 minutes.

## Prerequisites

Before you begin, ensure you have:

- **Docker Desktop** (or Docker Engine + Docker Compose)
- **An OpenAI API key** (or another supported LLM provider key)

That's it. No Python installation required for basic usage.

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/agentic-rag-graphrag.git
cd agentic-rag-graphrag
```

---

## 2. Configure Environment

Create your environment file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
# Required: Add your API key
OPENAI_API_KEY=sk-your-api-key-here
```

> **Using a different LLM provider?** See the [Provider Configuration Guide](provider-configuration.md) for Anthropic, Gemini, OpenRouter, or Ollama setup.

---

## 3. Start Services

Launch all services with Docker Compose:

```bash
docker compose up -d
```

This starts:
- Backend API (port 8000)
- Frontend UI (port 3000)
- PostgreSQL + pgvector (port 5432)
- Neo4j graph database (port 7687)
- Redis cache (port 6379)

Wait approximately 30-60 seconds for all services to initialize.

---

## 4. Verify Setup

Check that all services are healthy:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

You can also view the API documentation at: http://localhost:8000/docs

---

## 5. Ingest Your First Document

Let's ingest a webpage into the knowledge graph. We'll use a simple curl command:

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.python.org/3/tutorial/classes.html",
    "tenant_id": "quickstart-demo",
    "max_depth": 0
  }'
```

Expected response:
```json
{
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued"
  },
  "meta": {
    "requestId": "...",
    "timestamp": "..."
  }
}
```

The document is now being processed in the background. For this quickstart, we'll give it a moment to complete.

```bash
# Wait for processing (typically 10-30 seconds)
sleep 30
```

---

## 6. Run Your First Query

Now query the knowledge you just ingested:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a Python class and how do I define one?",
    "tenant_id": "quickstart-demo"
  }'
```

Example response:
```json
{
  "data": {
    "answer": "A Python class is a blueprint for creating objects...",
    "retrieval_strategy": "hybrid",
    "plan": [...],
    "evidence": {...}
  },
  "meta": {
    "requestId": "...",
    "timestamp": "..."
  }
}
```

Congratulations! You've successfully:
1. Set up the Agentic RAG + GraphRAG system
2. Ingested documentation into the knowledge graph
3. Queried your knowledge base with hybrid retrieval

---

## 7. View Results in the UI

Open your browser to http://localhost:3000 to access the CopilotKit-powered chat interface.

You can:
- Ask questions in natural language
- View the knowledge graph visualization
- See retrieval evidence and sources
- Validate sources with human-in-the-loop controls

---

## Next Steps

Now that you have a working system, explore these guides:

| Guide | Description |
|-------|-------------|
| [CLI Installation](cli-installation.md) | Use the interactive CLI for advanced setup |
| [Ingestion Pipeline](ingestion-pipeline.md) | Ingest PDFs, YouTube videos, and more |
| [Advanced Retrieval](advanced-retrieval-configuration.md) | Enable reranking, contextual retrieval, and CRAG |
| [Provider Configuration](provider-configuration.md) | Configure Anthropic, Gemini, or local models |
| [Memory Platform](memory-platform.md) | Set up memory scopes and consolidation |
| [Graph Intelligence](graph-intelligence.md) | Enable community detection and LazyRAG |
| [Deployment Guide](deployment-production.md) | Production deployment with Kubernetes |

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/api/v1/query` | POST | Query the knowledge base |
| `/api/v1/ingest/url` | POST | Ingest a webpage |
| `/api/v1/ingest/document` | POST | Upload a PDF document |
| `/api/v1/knowledge/graph` | GET | Get knowledge graph data |
| `/api/v1/knowledge/stats` | GET | Get graph statistics |

### Common Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f backend

# Restart a service
docker compose restart backend

# Check service status
docker compose ps
```

### Useful URLs

| URL | Description |
|-----|-------------|
| http://localhost:3000 | Frontend UI |
| http://localhost:8000/docs | API Documentation |
| http://localhost:8000/health | Health Check |
| http://localhost:7474 | Neo4j Browser |

---

## Troubleshooting

If you encounter issues, run the diagnostic command:

```bash
# If you have the CLI installed
rag-cli doctor

# Or check Docker logs directly
docker compose logs backend
```

For detailed troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).

---

**Need help?** Check the [full documentation](../README.md) or open an issue on GitHub.
