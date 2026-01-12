# Feature Discovery Matrix

This document provides a comprehensive overview of all platform features, organized by category. Each feature includes profile availability, configuration environment variables, and documentation links.

**Last Updated:** 2026-01-13

---

## Profile Legend

| Profile | Description | Use Case |
|---------|-------------|----------|
| **Minimal** | Core features only, lowest resource requirements | Development, prototyping, resource-constrained environments |
| **Standard** | Balanced feature set for most deployments | Production deployments, typical enterprise use |
| **Enterprise** | All features enabled, maximum capability | High-scale deployments, advanced AI workloads |

---

## 1. Retrieval Features

### Vector Search & Embedding

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Vector Semantic Search | Y | Y | Y | `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` | [Provider Configuration](guides/provider-configuration.md) |
| Multi-Provider Embeddings | Y | Y | Y | `EMBEDDING_PROVIDER` (openai/openrouter/ollama/gemini/voyage) | [Provider Configuration](guides/provider-configuration.md) |
| Embedding Dimension Control | Y | Y | Y | `EMBEDDING_DIMENSION` (default: 1536) | [Provider Configuration](guides/provider-configuration.md) |
| Sparse Vector Search (BM42) | - | - | Y | `SPARSE_VECTORS_ENABLED`, `SPARSE_MODEL`, `HYBRID_DENSE_WEIGHT`, `HYBRID_SPARSE_WEIGHT` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |
| Cross-Language Query | - | - | Y | `CROSS_LANGUAGE_ENABLED`, `CROSS_LANGUAGE_EMBEDDING`, `CROSS_LANGUAGE_TRANSLATION` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |

### Graph Search

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Graph Relationship Traversal | Y | Y | Y | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | [Database Administration](guides/database-administration.md) |
| Graphiti Temporal Knowledge | Y | Y | Y | `GRAPHITI_EMBEDDING_MODEL`, `GRAPHITI_LLM_MODEL` | [Graph Intelligence](guides/graph-intelligence.md) |
| Temporal Query Capabilities | Y | Y | Y | (built-in with Graphiti) | [Graph Intelligence](guides/graph-intelligence.md) |

### Hybrid Retrieval

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Hybrid Answer Synthesis | Y | Y | Y | (enabled by default) | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Dual-Level Retrieval | - | Y | Y | `DUAL_LEVEL_RETRIEVAL_ENABLED`, `DUAL_LEVEL_LOW_WEIGHT`, `DUAL_LEVEL_HIGH_WEIGHT`, `DUAL_LEVEL_LOW_LIMIT`, `DUAL_LEVEL_HIGH_LIMIT`, `DUAL_LEVEL_SYNTHESIS_MODEL` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |
| Small-to-Big Retrieval | - | Y | Y | `HIERARCHICAL_CHUNKS_ENABLED`, `HIERARCHICAL_CHUNK_LEVELS`, `SMALL_TO_BIG_RETURN_LEVEL` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |

### Reranking

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Cross-Encoder Reranking | - | Y | Y | `RERANKER_ENABLED`, `RERANKER_PROVIDER` (cohere/flashrank), `RERANKER_MODEL`, `RERANKER_TOP_K` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Cohere Reranker | - | Y | Y | `RERANKER_PROVIDER=cohere`, `COHERE_API_KEY` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| FlashRank Reranker (Local) | - | Y | Y | `RERANKER_PROVIDER=flashrank` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| ColBERT Reranking | - | - | Y | `COLBERT_ENABLED`, `COLBERT_MODEL`, `COLBERT_MAX_LENGTH` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |
| Graph-Based Rerankers | - | Y | Y | `GRAPH_RERANKER_ENABLED`, `GRAPH_RERANKER_TYPE` (episode/distance/hybrid), `GRAPH_RERANKER_*_WEIGHT` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |
| Reranking Cache | - | Y | Y | `RERANKER_CACHE_ENABLED`, `RERANKER_CACHE_TTL_SECONDS`, `RERANKER_CACHE_MAX_SIZE` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |

### Corrective RAG

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| CRAG Grader Agent | - | Y | Y | `GRADER_ENABLED`, `GRADER_MODEL`, `GRADER_THRESHOLD` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Heuristic Grader | - | Y | Y | `GRADER_MODEL=heuristic`, `GRADER_HEURISTIC_LENGTH_WEIGHT` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Cross-Encoder Grader | - | Y | Y | `GRADER_MODEL=cross-encoder/*` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Fallback Web Search | - | Y | Y | `GRADER_FALLBACK_ENABLED`, `GRADER_FALLBACK_STRATEGY`, `TAVILY_API_KEY` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Score Normalization | - | Y | Y | `GRADER_NORMALIZATION_STRATEGY` (min_max/z_score/softmax/percentile) | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |

### Contextual Retrieval

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Contextual Chunk Enrichment | - | Y | Y | `CONTEXTUAL_RETRIEVAL_ENABLED`, `CONTEXTUAL_MODEL`, `CONTEXTUAL_PROMPT_CACHING` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |
| Custom Context Prompts | - | Y | Y | `CONTEXTUAL_RETRIEVAL_PROMPT_PATH` | [Advanced Retrieval](guides/advanced-retrieval-configuration.md) |

### Caching

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Redis Result Cache | Y | Y | Y | `REDIS_URL` | [Database Administration](guides/database-administration.md) |
| Retrieval Cache | Y | Y | Y | (built-in with Redis) | [Database Administration](guides/database-administration.md) |

---

## 2. Ingestion Features

### URL & Web Crawling

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| URL Documentation Crawling | Y | Y | Y | `CRAWL4AI_*` settings | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Crawl4AI Integration | Y | Y | Y | `CRAWL4AI_HEADLESS`, `CRAWL4AI_MAX_CONCURRENT`, `CRAWL4AI_CACHE_ENABLED` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Crawl Profiles | Y | Y | Y | `CRAWL4AI_PROFILE` (fast/thorough/stealth) | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| JavaScript Rendering | Y | Y | Y | `CRAWL4AI_JS_WAIT_SECONDS`, `CRAWL4AI_PAGE_TIMEOUT_MS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Stealth Mode | - | Y | Y | `CRAWL4AI_PROFILE=stealth`, `CRAWL4AI_STEALTH_PROXY` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Apify Fallback | - | Y | Y | `CRAWL_FALLBACK_ENABLED`, `APIFY_API_TOKEN` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| BrightData Fallback | - | Y | Y | `BRIGHTDATA_USERNAME`, `BRIGHTDATA_PASSWORD`, `BRIGHTDATA_ZONE` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Dynamic User-Agent Rotation | - | Y | Y | (built-in with profiles) | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Bloom Filter Deduplication | - | Y | Y | (automatic for large crawls) | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### PDF Processing

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| PDF Document Parsing | Y | Y | Y | `MAX_UPLOAD_SIZE_MB`, `TEMP_UPLOAD_DIR` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Docling Integration | Y | Y | Y | `DOCLING_TABLE_MODE`, `DOCLING_SERVICE_URL` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Enhanced Table Extraction | - | Y | Y | `ENHANCED_DOCLING_ENABLED`, `DOCLING_TABLE_EXTRACTION`, `DOCLING_TABLE_AS_MARKDOWN` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Layout Preservation | - | Y | Y | `DOCLING_PRESERVE_LAYOUT` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### YouTube Ingestion

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| YouTube Transcript API | Y | Y | Y | `YOUTUBE_PREFERRED_LANGUAGES`, `YOUTUBE_CHUNK_DURATION_SECONDS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Multi-Language Transcripts | Y | Y | Y | `YOUTUBE_PREFERRED_LANGUAGES` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### Codebase Indexing

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Codebase RAG Context | - | Y | Y | `CODEBASE_RAG_ENABLED`, `CODEBASE_LANGUAGES`, `CODEBASE_EXCLUDE_PATTERNS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Symbol Table Extraction | - | Y | Y | `CODEBASE_SYMBOL_TABLE_MAX_SYMBOLS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Incremental Indexing | - | Y | Y | `CODEBASE_INCREMENTAL_INDEXING`, `CODEBASE_INDEX_CACHE_TTL_SECONDS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Class Context Inclusion | - | Y | Y | `CODEBASE_INCLUDE_CLASS_CONTEXT` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### External Data Sync

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| External Sync Framework | - | - | Y | `EXTERNAL_SYNC_ENABLED`, `SYNC_SOURCES` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Confluence Connector | - | - | Y | `CONFLUENCE_URL`, `CONFLUENCE_API_TOKEN`, `CONFLUENCE_SPACES` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| S3 Connector | - | - | Y | `S3_SYNC_BUCKET`, `S3_SYNC_PREFIX` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Notion Connector | - | - | Y | `NOTION_API_KEY`, `NOTION_DATABASE_IDS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### Multimodal

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Multimodal Ingestion | - | - | Y | `MULTIMODAL_INGESTION_ENABLED` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Office Documents | - | Y | Y | `OFFICE_DOCS_ENABLED` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### Chunking

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Basic Chunking | Y | Y | Y | `CHUNK_SIZE`, `CHUNK_OVERLAP` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| Hierarchical Chunking | - | Y | Y | `HIERARCHICAL_CHUNKS_ENABLED`, `HIERARCHICAL_CHUNK_LEVELS`, `HIERARCHICAL_OVERLAP_RATIO`, `HIERARCHICAL_EMBEDDING_LEVEL` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |

---

## 3. Agent Features

### Orchestration

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Orchestrator Agent | Y | Y | Y | `LLM_PROVIDER`, `LLM_MODEL_ID` | [Provider Configuration](guides/provider-configuration.md) |
| Multi-Step Query Planning | Y | Y | Y | (built-in) | [Architecture](../CLAUDE.md) |
| Dynamic Retrieval Selection | Y | Y | Y | (built-in) | [Architecture](../CLAUDE.md) |

### Protocol Support

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| MCP (Model Context Protocol) | Y | Y | Y | `MCP_TOOL_TIMEOUT_SECONDS`, `MCP_TOOL_TIMEOUT_OVERRIDES` | [Protocol Integration](guides/protocol-integration/mcp.md) |
| A2A (Agent-to-Agent) | Y | Y | Y | `A2A_ENABLED`, `A2A_AGENT_ID`, `A2A_ENDPOINT_URL` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| AG-UI Protocol | Y | Y | Y | (built-in with CopilotKit) | [Protocol Integration](guides/protocol-integration/ag-ui.md) |

### Trajectory Logging

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Persistent Trajectory Logging | Y | Y | Y | (built-in) | [Observability Metrics](guides/observability-metrics.md) |
| Trajectory Debugging Interface | Y | Y | Y | (built-in) | [Observability Metrics](guides/observability-metrics.md) |
| Encrypted Trace Storage | - | Y | Y | `TRACE_ENCRYPTION_KEY` | [Observability Metrics](guides/observability-metrics.md) |

### Agent Delegation

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| A2A Agent Collaboration | - | Y | Y | `A2A_ENABLED` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| A2A Middleware Agent | - | Y | Y | (built-in with A2A) | [Protocol Integration](guides/protocol-integration/a2a.md) |
| A2A Session Management | - | Y | Y | `A2A_SESSION_TTL_SECONDS`, `A2A_CLEANUP_INTERVAL_SECONDS` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| A2A Request Signing | - | Y | Y | `A2A_SIGNING_SECRET`, `A2A_SIGNING_TTL_SECONDS` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| A2A Heartbeat | - | Y | Y | `A2A_HEARTBEAT_INTERVAL_SECONDS`, `A2A_HEARTBEAT_TIMEOUT_SECONDS` | [Protocol Integration](guides/protocol-integration/a2a.md) |

---

## 4. Memory Features

### Memory Scopes

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Memory Scopes | - | Y | Y | `MEMORY_SCOPES_ENABLED`, `MEMORY_DEFAULT_SCOPE` (user/session/agent/global) | [Memory Platform](guides/memory-platform.md) |
| Parent Scope Inheritance | - | Y | Y | `MEMORY_INCLUDE_PARENT_SCOPES` | [Memory Platform](guides/memory-platform.md) |
| Memory Cache | - | Y | Y | `MEMORY_CACHE_TTL_SECONDS`, `MEMORY_MAX_PER_SCOPE` | [Memory Platform](guides/memory-platform.md) |

### Memory Consolidation

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Memory Consolidation | - | - | Y | `MEMORY_CONSOLIDATION_ENABLED`, `MEMORY_CONSOLIDATION_SCHEDULE` | [Memory Platform](guides/memory-platform.md) |
| Similarity Deduplication | - | - | Y | `MEMORY_SIMILARITY_THRESHOLD` | [Memory Platform](guides/memory-platform.md) |
| Memory Decay | - | - | Y | `MEMORY_DECAY_HALF_LIFE_DAYS`, `MEMORY_MIN_IMPORTANCE` | [Memory Platform](guides/memory-platform.md) |
| Consolidation Batching | - | - | Y | `MEMORY_CONSOLIDATION_BATCH_SIZE` | [Memory Platform](guides/memory-platform.md) |

---

## 5. Graph Features

### Community Detection

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Community Detection | - | Y | Y | `COMMUNITY_DETECTION_ENABLED`, `COMMUNITY_ALGORITHM` (louvain/leiden) | [Graph Intelligence](guides/graph-intelligence.md) |
| Community Size Control | - | Y | Y | `COMMUNITY_MIN_SIZE`, `COMMUNITY_MAX_LEVELS` | [Graph Intelligence](guides/graph-intelligence.md) |
| Community Summarization | - | Y | Y | `COMMUNITY_SUMMARY_MODEL`, `COMMUNITY_REFRESH_SCHEDULE` | [Graph Intelligence](guides/graph-intelligence.md) |

### LazyRAG

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| LazyRAG Pattern | - | Y | Y | `LAZY_RAG_ENABLED`, `LAZY_RAG_MAX_ENTITIES`, `LAZY_RAG_MAX_HOPS` | [Graph Intelligence](guides/graph-intelligence.md) |
| Query-Time Summarization | - | Y | Y | `LAZY_RAG_SUMMARY_MODEL` | [Graph Intelligence](guides/graph-intelligence.md) |
| Community Integration | - | Y | Y | `LAZY_RAG_USE_COMMUNITIES` | [Graph Intelligence](guides/graph-intelligence.md) |

### Query Routing

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Global/Local Query Routing | - | Y | Y | `QUERY_ROUTING_ENABLED`, `QUERY_ROUTING_CONFIDENCE_THRESHOLD` | [Graph Intelligence](guides/graph-intelligence.md) |
| LLM-Based Routing | - | - | Y | `QUERY_ROUTING_USE_LLM`, `QUERY_ROUTING_LLM_MODEL` | [Graph Intelligence](guides/graph-intelligence.md) |

### Ontology Support

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Ontology Support | - | - | Y | `ONTOLOGY_SUPPORT_ENABLED`, `ONTOLOGY_PATH`, `ONTOLOGY_AUTO_TYPE` | [Graph Intelligence](guides/graph-intelligence.md) |

---

## 6. Observability Features

### Metrics & Monitoring

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Prometheus Metrics | - | Y | Y | `PROMETHEUS_ENABLED`, `PROMETHEUS_PATH` | [Observability Metrics](guides/observability-metrics.md) |
| LLM Cost Monitoring | - | Y | Y | `MODEL_PRICING_JSON` | [Observability Metrics](guides/observability-metrics.md) |
| Retrieval Quality Benchmarks | - | Y | Y | (built-in) | [Observability Metrics](guides/observability-metrics.md) |
| AG-UI Stream Metrics | - | Y | Y | (built-in with Prometheus) | [Observability Metrics](guides/observability-metrics.md) |

### Cost Tracking

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Token Usage Tracking | - | Y | Y | (built-in) | [Observability Metrics](guides/observability-metrics.md) |
| Cost Attribution | - | Y | Y | `MODEL_PRICING_JSON` | [Observability Metrics](guides/observability-metrics.md) |

### Intelligent Model Routing

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Model Router | - | Y | Y | `ROUTING_SIMPLE_MODEL`, `ROUTING_MEDIUM_MODEL`, `ROUTING_COMPLEX_MODEL` | [Observability Metrics](guides/observability-metrics.md) |
| Complexity-Based Routing | - | Y | Y | `ROUTING_SIMPLE_MAX_SCORE`, `ROUTING_COMPLEX_MIN_SCORE` | [Observability Metrics](guides/observability-metrics.md) |

### Logging

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Structured Logging | Y | Y | Y | (built-in with structlog) | [Observability Metrics](guides/observability-metrics.md) |
| Contextual Retrieval Cost Logging | - | Y | Y | (built-in when enabled) | [Observability Metrics](guides/observability-metrics.md) |

---

## 7. Security Features

### Multi-Tenancy

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Tenant Isolation | Y | Y | Y | (built-in, requires `tenant_id`) | [Security Best Practices](guides/security-best-practices.md) |
| Per-Tenant Rate Limiting | Y | Y | Y | `RATE_LIMIT_PER_MINUTE` | [Security Best Practices](guides/security-best-practices.md) |
| Tenant-Scoped Sessions | Y | Y | Y | (built-in) | [Security Best Practices](guides/security-best-practices.md) |

### Authentication & Authorization

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| API Key Authentication | Y | Y | Y | (via headers) | [Security Best Practices](guides/security-best-practices.md) |
| MCP Server Auth | Y | Y | Y | (built-in) | [MCP Wrapper Architecture](guides/mcp-wrapper-architecture.md) |
| Share Link Signing | - | Y | Y | `SHARE_SECRET` | [Security Best Practices](guides/security-best-practices.md) |

### Rate Limiting

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Request Rate Limiting | Y | Y | Y | `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_BACKEND` (memory/redis) | [Security Best Practices](guides/security-best-practices.md) |
| Redis-Backed Rate Limiting | - | Y | Y | `RATE_LIMIT_BACKEND=redis`, `RATE_LIMIT_REDIS_PREFIX` | [Security Best Practices](guides/security-best-practices.md) |
| A2A Resource Limits | - | Y | Y | `A2A_LIMITS_BACKEND`, `A2A_SESSION_LIMIT_PER_TENANT`, `A2A_MESSAGE_LIMIT_PER_SESSION`, `A2A_MESSAGE_RATE_LIMIT` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| Codebase Index Rate Limiting | - | Y | Y | `CODEBASE_INDEX_RATE_LIMIT_MAX`, `CODEBASE_INDEX_RATE_LIMIT_WINDOW_SECONDS` | [Ingestion Pipeline](guides/ingestion-pipeline.md) |

### Encryption & Secrets

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Trace Encryption | - | Y | Y | `TRACE_ENCRYPTION_KEY` | [Security Best Practices](guides/security-best-practices.md) |
| MCP-UI Signing | - | Y | Y | `MCP_UI_SIGNING_SECRET` | [Protocol Integration](guides/protocol-integration/mcp.md) |

### Validation & Compliance

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Codebase Path Validation | - | Y | Y | `CODEBASE_ALLOWED_BASE_PATH` | [Security Best Practices](guides/security-best-practices.md) |
| Hallucination Detection | - | Y | Y | `CODEBASE_HALLUCINATION_THRESHOLD`, `CODEBASE_DETECTOR_MODE` (warn/block) | [Ingestion Pipeline](guides/ingestion-pipeline.md) |
| RFC 7807 Error Compliance | Y | Y | Y | (built-in) | [API Reference](guides/api-reference.md) |

---

## 8. Frontend Features

### CopilotKit Integration

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| CopilotKit React Integration | Y | Y | Y | `FRONTEND_URL` | [Quickstart](guides/quickstart.md) |
| Chat Sidebar Interface | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| CopilotPopup Component | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| CopilotChat Embedded | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| CopilotTextarea (Autocomplete) | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |

### Generative UI

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Generative UI Components | Y | Y | Y | (built-in) | [Protocol Integration](guides/protocol-integration/a2ui.md) |
| A2UI Widget Rendering | - | Y | Y | (built-in with AG-UI) | [Protocol Integration](guides/protocol-integration/a2ui.md) |
| MCP-UI Renderer | - | Y | Y | `MCP_UI_ENABLED`, `MCP_UI_ALLOWED_ORIGINS` | [Protocol Integration](guides/protocol-integration/mcp.md) |
| Open-JSON-UI Renderer | - | Y | Y | (built-in) | [Protocol Integration](guides/protocol-integration/open-json-ui.md) |
| Tool Call Visualization | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |

### Human-in-the-Loop

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Source Validation (HITL) | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| useHumanInTheLoop Hook | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |

### Voice I/O

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Voice Input (STT) | - | - | Y | `VOICE_IO_ENABLED`, `WHISPER_MODEL` | [Voice I/O Configuration](guides/voice-io-configuration.md) |
| Voice Output (TTS) | - | - | Y | `TTS_PROVIDER`, `TTS_VOICE`, `TTS_SPEED` | [Voice I/O Configuration](guides/voice-io-configuration.md) |

### Context & Suggestions

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| useCopilotReadable Context | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| useCopilotChatSuggestions | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| useCopilotAdditionalInstructions | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |

### Visualization

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Knowledge Graph Visualization | Y | Y | Y | (built-in) | [Quickstart](guides/quickstart.md) |
| Graph Explainability | Y | Y | Y | (built-in) | [Graph Intelligence](guides/graph-intelligence.md) |
| Visual Workflow Editor | - | - | Y | (built-in) | [Quickstart](guides/quickstart.md) |

---

## 9. LLM Provider Features

### Provider Support

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| OpenAI Provider | Y | Y | Y | `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_MODEL_ID` | [Provider Configuration](guides/provider-configuration.md) |
| Anthropic Provider | Y | Y | Y | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` | [Provider Configuration](guides/provider-configuration.md) |
| Google Gemini Provider | Y | Y | Y | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY` | [Provider Configuration](guides/provider-configuration.md) |
| OpenRouter Provider | Y | Y | Y | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | [Provider Configuration](guides/provider-configuration.md) |
| Ollama Provider (Local) | Y | Y | Y | `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL` | [Provider Configuration](guides/provider-configuration.md) |
| Voyage Embeddings | Y | Y | Y | `EMBEDDING_PROVIDER=voyage`, `VOYAGE_API_KEY` | [Provider Configuration](guides/provider-configuration.md) |

### Model Configuration

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Custom Base URL | Y | Y | Y | `OPENAI_BASE_URL`, `OPENROUTER_BASE_URL`, `OLLAMA_BASE_URL` | [Provider Configuration](guides/provider-configuration.md) |
| Separate Embedding Provider | Y | Y | Y | `EMBEDDING_PROVIDER` (independent of LLM_PROVIDER) | [Provider Configuration](guides/provider-configuration.md) |

---

## 10. Database Features

### PostgreSQL/pgvector

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| PostgreSQL 16+ | Y | Y | Y | `DATABASE_URL` | [Database Administration](guides/database-administration.md) |
| pgvector Extension | Y | Y | Y | (automatic) | [Database Administration](guides/database-administration.md) |
| Connection Pooling | Y | Y | Y | `DB_POOL_MIN`, `DB_POOL_MAX` | [Database Administration](guides/database-administration.md) |

### Neo4j

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Neo4j 5 Community | Y | Y | Y | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | [Database Administration](guides/database-administration.md) |
| Connection Pooling | Y | Y | Y | `NEO4J_POOL_MIN`, `NEO4J_POOL_MAX` | [Database Administration](guides/database-administration.md) |
| Transaction Timeout | Y | Y | Y | `NEO4J_TRANSACTION_TIMEOUT_SECONDS` | [Database Administration](guides/database-administration.md) |

### Redis

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Redis 7 | Y | Y | Y | `REDIS_URL` | [Database Administration](guides/database-administration.md) |
| Session Storage | Y | Y | Y | (automatic) | [Database Administration](guides/database-administration.md) |
| Rate Limit Storage | - | Y | Y | `RATE_LIMIT_BACKEND=redis` | [Database Administration](guides/database-administration.md) |

---

## 11. Protocol Features

### MCP (Model Context Protocol)

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| MCP Tool Server | Y | Y | Y | `MCP_TOOL_TIMEOUT_SECONDS` | [MCP Wrapper Architecture](guides/mcp-wrapper-architecture.md) |
| MCP stdio Transport | Y | Y | Y | (built-in) | [MCP Wrapper Architecture](guides/mcp-wrapper-architecture.md) |
| Per-Tool Timeout Overrides | - | Y | Y | `MCP_TOOL_TIMEOUT_OVERRIDES` | [MCP Wrapper Architecture](guides/mcp-wrapper-architecture.md) |
| MCP Client Integration | - | Y | Y | (built-in) | [Protocol Integration](guides/protocol-integration/mcp.md) |

### A2A (Agent-to-Agent)

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| A2A Protocol | Y | Y | Y | `A2A_ENABLED`, `A2A_AGENT_ID` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| Task Delegation | - | Y | Y | `A2A_TASK_DEFAULT_TIMEOUT_SECONDS`, `A2A_TASK_MAX_RETRIES` | [Protocol Integration](guides/protocol-integration/a2a.md) |
| Agent Registry | - | Y | Y | (built-in) | [Protocol Integration](guides/protocol-integration/a2a.md) |
| Resource Limits | - | Y | Y | `A2A_LIMITS_BACKEND`, `A2A_SESSION_LIMIT_PER_TENANT` | [Protocol Integration](guides/protocol-integration/a2a.md) |

### AG-UI (Agent-User Interface)

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| AG-UI Bridge | Y | Y | Y | (built-in with CopilotKit) | [Protocol Integration](guides/protocol-integration/ag-ui.md) |
| Stream Metrics | - | Y | Y | `PROMETHEUS_ENABLED` | [Protocol Integration](guides/protocol-integration/ag-ui.md) |
| Extended Error Events | - | Y | Y | (built-in) | [Protocol Integration](guides/protocol-integration/ag-ui.md) |

---

## 12. Feedback & Self-Improvement

| Feature | Minimal | Standard | Enterprise | Config Env Var(s) | Documentation |
|---------|:-------:|:--------:|:----------:|-------------------|---------------|
| Self-Improving Feedback Loop | - | - | Y | `FEEDBACK_LOOP_ENABLED`, `FEEDBACK_MIN_SAMPLES` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |
| Query Boost Adaptation | - | - | Y | `FEEDBACK_BOOST_MAX`, `FEEDBACK_BOOST_MIN` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |
| Feedback Decay | - | - | Y | `FEEDBACK_DECAY_DAYS` | [Advanced Retrieval](guides/advanced-retrieval-deep-dive.md) |

---

## Feature Count Summary

| Category | Minimal | Standard | Enterprise |
|----------|:-------:|:--------:|:----------:|
| Retrieval Features | 9 | 27 | 32 |
| Ingestion Features | 11 | 23 | 28 |
| Agent Features | 9 | 18 | 18 |
| Memory Features | 0 | 3 | 8 |
| Graph Features | 2 | 9 | 11 |
| Observability Features | 2 | 10 | 10 |
| Security Features | 6 | 14 | 16 |
| Frontend Features | 14 | 21 | 24 |
| LLM Provider Features | 7 | 7 | 7 |
| Database Features | 6 | 7 | 7 |
| Protocol Features | 4 | 10 | 10 |
| Feedback Features | 0 | 0 | 3 |
| **Total** | **70** | **149** | **174** |

---

## Quick Reference: Configuration Profiles

### Minimal Profile
Best for: Development, prototyping, resource-constrained environments
```yaml
# Core features only
CONFIG_PROFILE=minimal
LLM_PROVIDER=ollama  # Local models
EMBEDDING_PROVIDER=ollama
```

### Standard Profile
Best for: Production deployments, typical enterprise use
```yaml
# Balanced feature set
CONFIG_PROFILE=standard
RERANKER_ENABLED=true
GRADER_ENABLED=true
MEMORY_SCOPES_ENABLED=true
COMMUNITY_DETECTION_ENABLED=true
LAZY_RAG_ENABLED=true
```

### Enterprise Profile
Best for: High-scale deployments, advanced AI workloads
```yaml
# All features enabled
CONFIG_PROFILE=enterprise
SPARSE_VECTORS_ENABLED=true
CROSS_LANGUAGE_ENABLED=true
MEMORY_CONSOLIDATION_ENABLED=true
EXTERNAL_SYNC_ENABLED=true
VOICE_IO_ENABLED=true
FEEDBACK_LOOP_ENABLED=true
```

---

## See Also

- [Configuration Profiles Guide](guides/configuration-profiles.md)
- [Quickstart Tutorial](guides/quickstart.md)
- [Provider Configuration](guides/provider-configuration.md)
- [Advanced Retrieval Configuration](guides/advanced-retrieval-configuration.md)
- [Glossary](glossary.md)
