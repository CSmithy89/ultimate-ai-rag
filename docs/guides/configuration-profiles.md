# Configuration Profiles Guide

This guide documents the configuration profile system for Agentic RAG, which provides pre-configured settings optimized for different deployment scenarios.

## Table of Contents

- [Overview](#overview)
- [Available Profiles](#available-profiles)
- [Profile Feature Matrix](#profile-feature-matrix)
- [Switching Profiles](#switching-profiles)
- [Profile Configuration Details](#profile-configuration-details)
- [Custom Profile Creation](#custom-profile-creation)
- [Environment Variable Overrides](#environment-variable-overrides)
- [Profile Migration](#profile-migration)
- [Best Practices](#best-practices)

## Overview

The profile system provides a layered configuration approach:

1. **Base Profile**: YAML files in `config/profiles/` define default settings
2. **Environment Overrides**: Environment variables override profile defaults
3. **Runtime Configuration**: Application settings are built from both sources

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Settings                        │
├─────────────────────────────────────────────────────────────────┤
│  Environment Variables (highest priority - always override)     │
├─────────────────────────────────────────────────────────────────┤
│  Profile YAML (provides sensible defaults if env var not set)   │
├─────────────────────────────────────────────────────────────────┤
│  Code Defaults (fallback when neither profile nor env defines)  │
└─────────────────────────────────────────────────────────────────┘
```

## Available Profiles

### Minimal Profile

**Target**: Development, testing, resource-constrained environments

```yaml
# config/profiles/minimal.yaml
llm:
  provider: openai
  model: gpt-4o-mini

embedding:
  provider: openai
  model: text-embedding-3-small
  dimension: 1536

retrieval:
  strategy: vector
  reranker:
    enabled: false
    provider: flashrank
  contextual_retrieval:
    enabled: false
  grader:
    enabled: false

memory:
  scopes_enabled: false
  default_scope: session
  consolidation_enabled: false

community:
  detection_enabled: false

ingestion:
  crawl_profile: fast
  fallback_enabled: false
  codebase_enabled: false
  external_sync_enabled: false

voice:
  enabled: false

graph_intelligence:
  lazy_rag_enabled: false
  query_routing_enabled: false
  graph_reranker_enabled: false

observability:
  prometheus_enabled: false

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 10
    max_messages_per_session: 100
```

**Use Cases**:
- Local development workstations
- CI/CD pipeline testing
- Low-resource Docker containers
- Quick prototyping

### Standard Profile

**Target**: Production deployments, small-medium teams

```yaml
# config/profiles/standard.yaml
llm:
  provider: openai
  model: gpt-4o

embedding:
  provider: openai
  model: text-embedding-3-small
  dimension: 1536

retrieval:
  strategy: hybrid
  reranker:
    enabled: true
    provider: flashrank
  contextual_retrieval:
    enabled: false
  grader:
    enabled: false

memory:
  scopes_enabled: true
  default_scope: session
  consolidation_enabled: false

community:
  detection_enabled: false

ingestion:
  crawl_profile: thorough
  fallback_enabled: false
  codebase_enabled: false
  external_sync_enabled: false

voice:
  enabled: false

graph_intelligence:
  lazy_rag_enabled: false
  query_routing_enabled: true
  graph_reranker_enabled: false

observability:
  prometheus_enabled: true

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 100
    max_messages_per_session: 1000
```

**Use Cases**:
- Production deployments
- Teams of 5-50 users
- SaaS applications
- General-purpose RAG systems

### Enterprise Profile

**Target**: Large teams, advanced retrieval, enterprise ingestion

```yaml
# config/profiles/enterprise.yaml
llm:
  provider: openrouter
  model: claude-3.5-sonnet

embedding:
  provider: voyage
  model: voyage-code-3
  dimension: 1024

retrieval:
  strategy: hybrid
  reranker:
    enabled: true
    provider: cohere
  contextual_retrieval:
    enabled: true
  grader:
    enabled: true

memory:
  scopes_enabled: true
  default_scope: user
  consolidation_enabled: true

community:
  detection_enabled: true

ingestion:
  crawl_profile: stealth
  fallback_enabled: true
  codebase_enabled: true
  external_sync_enabled: true

voice:
  enabled: true

graph_intelligence:
  lazy_rag_enabled: true
  query_routing_enabled: true
  graph_reranker_enabled: true

observability:
  prometheus_enabled: true

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 500
    max_messages_per_session: 5000
```

**Use Cases**:
- Large enterprise deployments
- Teams of 50+ users
- Multi-tenant SaaS platforms
- Mission-critical RAG applications
- Advanced code intelligence workloads

## Profile Feature Matrix

| Feature Category | Minimal | Standard | Enterprise |
|-----------------|---------|----------|------------|
| **LLM** | | | |
| Provider | openai | openai | openrouter |
| Model | gpt-4o-mini | gpt-4o | claude-3.5-sonnet |
| **Embedding** | | | |
| Provider | openai | openai | voyage |
| Model | text-embedding-3-small | text-embedding-3-small | voyage-code-3 |
| **Retrieval** | | | |
| Strategy | vector | hybrid | hybrid |
| Reranker | disabled | flashrank | cohere |
| Contextual Retrieval | disabled | disabled | enabled |
| CRAG Grader | disabled | disabled | enabled |
| **Memory** | | | |
| Scopes | disabled | enabled | enabled |
| Default Scope | session | session | user |
| Consolidation | disabled | disabled | enabled |
| **Community** | | | |
| Detection | disabled | disabled | enabled |
| **Ingestion** | | | |
| Crawl Profile | fast | thorough | stealth |
| Fallback Providers | disabled | disabled | enabled |
| Codebase RAG | disabled | disabled | enabled |
| External Sync | disabled | disabled | enabled |
| **Voice** | disabled | disabled | enabled |
| **Graph Intelligence** | | | |
| LazyRAG | disabled | disabled | enabled |
| Query Routing | disabled | enabled | enabled |
| Graph Reranker | disabled | disabled | enabled |
| **Observability** | | | |
| Prometheus | disabled | enabled | enabled |
| **A2A Protocol** | | | |
| Sessions/Tenant | 10 | 100 | 500 |
| Messages/Session | 100 | 1,000 | 5,000 |

## Switching Profiles

### Method 1: Environment Variable

Set `CONFIG_PROFILE` before starting the application:

```bash
# Use minimal profile
export CONFIG_PROFILE=minimal

# Use standard profile (default)
export CONFIG_PROFILE=standard

# Use enterprise profile
export CONFIG_PROFILE=enterprise

# Start the backend
cd backend && uv run uvicorn agentic_rag_backend.main:app --reload
```

### Method 2: Docker Compose

In your `docker-compose.yml` or `docker-compose.override.yml`:

```yaml
services:
  backend:
    environment:
      - CONFIG_PROFILE=enterprise
```

### Method 3: .env File

Add to your `.env` file:

```bash
CONFIG_PROFILE=standard
```

### Method 4: CLI Tool

Using the Agentic RAG CLI:

```bash
# Interactive profile selection
agentic-rag configure --profile

# Direct profile selection
agentic-rag configure --profile enterprise
```

## Profile Configuration Details

### How Profiles Work

The profile system uses a mapping from YAML paths to environment variables:

```python
_PROFILE_ENV_MAPPING = {
    ("llm", "provider"): "LLM_PROVIDER",
    ("llm", "model"): "LLM_MODEL_ID",
    ("embedding", "provider"): "EMBEDDING_PROVIDER",
    ("embedding", "model"): "EMBEDDING_MODEL",
    ("embedding", "dimension"): "EMBEDDING_DIMENSION",
    ("retrieval", "reranker", "enabled"): "RERANKER_ENABLED",
    ("retrieval", "reranker", "provider"): "RERANKER_PROVIDER",
    ("retrieval", "contextual_retrieval", "enabled"): "CONTEXTUAL_RETRIEVAL_ENABLED",
    ("retrieval", "grader", "enabled"): "GRADER_ENABLED",
    ("memory", "scopes_enabled"): "MEMORY_SCOPES_ENABLED",
    ("memory", "default_scope"): "MEMORY_DEFAULT_SCOPE",
    ("memory", "consolidation_enabled"): "MEMORY_CONSOLIDATION_ENABLED",
    ("community", "detection_enabled"): "COMMUNITY_DETECTION_ENABLED",
    ("ingestion", "crawl_profile"): "CRAWL4AI_PROFILE",
    ("ingestion", "fallback_enabled"): "CRAWL_FALLBACK_ENABLED",
    ("ingestion", "codebase_enabled"): "CODEBASE_RAG_ENABLED",
    ("ingestion", "external_sync_enabled"): "EXTERNAL_SYNC_ENABLED",
    ("voice", "enabled"): "VOICE_IO_ENABLED",
    ("graph_intelligence", "lazy_rag_enabled"): "LAZY_RAG_ENABLED",
    ("graph_intelligence", "query_routing_enabled"): "QUERY_ROUTING_ENABLED",
    ("graph_intelligence", "graph_reranker_enabled"): "GRAPH_RERANKER_ENABLED",
    ("observability", "prometheus_enabled"): "PROMETHEUS_ENABLED",
    ("protocols", "a2a", "enabled"): "A2A_ENABLED",
    ("protocols", "a2a", "max_sessions_per_tenant"): "A2A_MAX_SESSIONS_PER_TENANT",
    ("protocols", "a2a", "max_messages_per_session"): "A2A_MAX_MESSAGES_PER_SESSION",
}
```

### Loading Sequence

1. Application reads `CONFIG_PROFILE` env var (defaults to "standard")
2. Loads `config/profiles/{profile}.yaml`
3. Validates profile schema
4. For each mapping, if env var is NOT already set, applies profile value
5. Environment variables always have priority over profile values

```python
# Example: Profile provides default, env var overrides
# Profile: reranker.enabled = true
# Env: RERANKER_ENABLED=false
# Result: reranker disabled (env var wins)
```

## Custom Profile Creation

### Step 1: Copy the Template

```bash
cp config/profiles/custom.yaml.template config/profiles/myprofile.yaml
```

### Step 2: Configure Your Profile

```yaml
# config/profiles/myprofile.yaml
# Custom Profile - Based on standard with modifications

llm:
  provider: anthropic
  model: claude-3-haiku-20240307

embedding:
  provider: openai
  model: text-embedding-3-large
  dimension: 3072

retrieval:
  strategy: hybrid
  reranker:
    enabled: true
    provider: cohere
  contextual_retrieval:
    enabled: true
  grader:
    enabled: true

memory:
  scopes_enabled: true
  default_scope: user
  consolidation_enabled: true

community:
  detection_enabled: false

ingestion:
  crawl_profile: thorough
  fallback_enabled: true
  codebase_enabled: true
  external_sync_enabled: false

voice:
  enabled: false

graph_intelligence:
  lazy_rag_enabled: true
  query_routing_enabled: true
  graph_reranker_enabled: false

observability:
  prometheus_enabled: true

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 200
    max_messages_per_session: 2000
```

### Step 3: Use Your Custom Profile

```bash
export CONFIG_PROFILE=myprofile
```

### Profile Schema Validation

Profiles are validated against Pydantic models:

```python
class ProfileConfig(BaseModel):
    llm: LLMProfile
    embedding: EmbeddingProfile
    retrieval: RetrievalConfig
    memory: MemoryConfig | None = None
    community: CommunityConfig | None = None
    ingestion: IngestionConfig | None = None
    voice: VoiceConfig | None = None
    graph_intelligence: GraphIntelligenceConfig | None = None
    observability: ObservabilityConfig | None = None
    protocols: ProtocolsConfig | None = None
```

Invalid profiles will fail fast at application startup with descriptive error messages.

## Environment Variable Overrides

You can override any profile setting with environment variables. Environment variables always take precedence.

### Common Override Patterns

**Override LLM while keeping other profile settings**:

```bash
CONFIG_PROFILE=standard
LLM_PROVIDER=anthropic
LLM_MODEL_ID=claude-3-opus-20240229
```

**Enable advanced retrieval on standard profile**:

```bash
CONFIG_PROFILE=standard
RERANKER_ENABLED=true
RERANKER_PROVIDER=cohere
COHERE_API_KEY=your-key
CONTEXTUAL_RETRIEVAL_ENABLED=true
GRADER_ENABLED=true
```

**Enterprise features with minimal base**:

```bash
CONFIG_PROFILE=minimal
MEMORY_SCOPES_ENABLED=true
PROMETHEUS_ENABLED=true
```

### Override Reference

| Profile Setting | Environment Variable | Type | Valid Values |
|----------------|---------------------|------|--------------|
| `llm.provider` | `LLM_PROVIDER` | string | openai, openrouter, ollama, anthropic, gemini |
| `llm.model` | `LLM_MODEL_ID` | string | Model identifier |
| `embedding.provider` | `EMBEDDING_PROVIDER` | string | openai, openrouter, ollama, gemini, voyage |
| `embedding.model` | `EMBEDDING_MODEL` | string | Model identifier |
| `embedding.dimension` | `EMBEDDING_DIMENSION` | int | 1-4096 |
| `retrieval.reranker.enabled` | `RERANKER_ENABLED` | bool | true, false |
| `retrieval.reranker.provider` | `RERANKER_PROVIDER` | string | cohere, flashrank |
| `retrieval.contextual_retrieval.enabled` | `CONTEXTUAL_RETRIEVAL_ENABLED` | bool | true, false |
| `retrieval.grader.enabled` | `GRADER_ENABLED` | bool | true, false |
| `memory.scopes_enabled` | `MEMORY_SCOPES_ENABLED` | bool | true, false |
| `memory.default_scope` | `MEMORY_DEFAULT_SCOPE` | string | user, session, agent, global |
| `memory.consolidation_enabled` | `MEMORY_CONSOLIDATION_ENABLED` | bool | true, false |
| `community.detection_enabled` | `COMMUNITY_DETECTION_ENABLED` | bool | true, false |
| `ingestion.crawl_profile` | `CRAWL4AI_PROFILE` | string | fast, thorough, stealth |
| `ingestion.fallback_enabled` | `CRAWL_FALLBACK_ENABLED` | bool | true, false |
| `ingestion.codebase_enabled` | `CODEBASE_RAG_ENABLED` | bool | true, false |
| `ingestion.external_sync_enabled` | `EXTERNAL_SYNC_ENABLED` | bool | true, false |
| `voice.enabled` | `VOICE_IO_ENABLED` | bool | true, false |
| `graph_intelligence.lazy_rag_enabled` | `LAZY_RAG_ENABLED` | bool | true, false |
| `graph_intelligence.query_routing_enabled` | `QUERY_ROUTING_ENABLED` | bool | true, false |
| `graph_intelligence.graph_reranker_enabled` | `GRAPH_RERANKER_ENABLED` | bool | true, false |
| `observability.prometheus_enabled` | `PROMETHEUS_ENABLED` | bool | true, false |
| `protocols.a2a.enabled` | `A2A_ENABLED` | bool | true, false |
| `protocols.a2a.max_sessions_per_tenant` | `A2A_MAX_SESSIONS_PER_TENANT` | int | 1-10000 |
| `protocols.a2a.max_messages_per_session` | `A2A_MAX_MESSAGES_PER_SESSION` | int | 1-100000 |

## Profile Migration

### Migrating from Minimal to Standard

The CLI provides automated migration assistance:

```bash
agentic-rag migrate-profile --from minimal --to standard
```

**Manual migration checklist**:

1. Set `CONFIG_PROFILE=standard`
2. Verify API keys for new features:
   - No additional keys required (uses existing OpenAI)
3. Update resource allocations:
   - Memory: +200MB (reranker models)
   - CPU: +10% (hybrid retrieval)
4. Test retrieval quality improvements

### Migrating from Standard to Enterprise

```bash
agentic-rag migrate-profile --from standard --to enterprise
```

**Manual migration checklist**:

1. Set `CONFIG_PROFILE=enterprise`
2. Configure required API keys:
   ```bash
   OPENROUTER_API_KEY=your-key      # For claude-3.5-sonnet
   VOYAGE_API_KEY=your-key          # For voyage-code-3 embeddings
   COHERE_API_KEY=your-key          # For Cohere reranking
   ```
3. Configure optional enterprise features:
   ```bash
   # Crawl fallback (if using stealth profile)
   APIFY_API_TOKEN=your-token
   BRIGHTDATA_USERNAME=your-username
   BRIGHTDATA_PASSWORD=your-password

   # External sync
   CONFLUENCE_URL=your-url
   CONFLUENCE_API_TOKEN=your-token
   S3_SYNC_BUCKET=your-bucket
   ```
4. Update resource allocations:
   - Memory: +500MB (community detection, LazyRAG)
   - Neo4j: Increase transaction timeout for graph operations
5. Re-index existing documents for contextual retrieval
6. Run community detection on existing graph

### Migration Script

The CLI includes a migration validation script:

```bash
# Validate migration readiness
agentic-rag doctor --profile enterprise

# Output:
# [PASS] LLM API key configured
# [PASS] Embedding API key configured
# [PASS] Reranker API key configured
# [WARN] APIFY_API_TOKEN not set (required for crawl fallback)
# [WARN] External sync sources not configured
# [PASS] Database connections healthy
# [PASS] Neo4j community algorithms available
```

## Best Practices

### Profile Selection Guidelines

| Scenario | Recommended Profile |
|----------|---------------------|
| Local development | minimal |
| CI/CD testing | minimal |
| Single developer prototype | minimal |
| Small team (< 10 users) | standard |
| Production SaaS | standard |
| Enterprise deployment | enterprise |
| Multi-tenant platform | enterprise |
| Code intelligence focus | enterprise |

### Performance Considerations

**Minimal Profile**:
- Fastest startup time (~2s)
- Lowest memory usage (~512MB)
- Best for quick iterations

**Standard Profile**:
- Moderate startup time (~5s)
- Medium memory usage (~1GB)
- Good balance of features and resources

**Enterprise Profile**:
- Longer startup time (~10s with model preloading)
- Higher memory usage (~2GB+)
- Maximum feature set

### Cost Optimization

**Development**: Use minimal profile to reduce API costs

```bash
# .env.development
CONFIG_PROFILE=minimal
LLM_MODEL_ID=gpt-4o-mini
```

**Staging**: Use standard profile with cost monitoring

```bash
# .env.staging
CONFIG_PROFILE=standard
PROMETHEUS_ENABLED=true
```

**Production**: Use enterprise with full observability

```bash
# .env.production
CONFIG_PROFILE=enterprise
PROMETHEUS_ENABLED=true
```

### Security Considerations

1. **API Key Management**: Use different API keys per environment
2. **Profile Storage**: Profile files should not contain secrets
3. **Environment Isolation**: Each environment should have its own profile or overrides
4. **Audit Logging**: Enable Prometheus for enterprise deployments

### Troubleshooting

**Profile not loading**:
```bash
# Check active profile
curl http://localhost:8000/api/health | jq '.config_profile'

# Verify profile file exists
ls config/profiles/
```

**Environment override not working**:
```bash
# Environment variables must be set BEFORE app start
export RERANKER_ENABLED=true
# Then start the app
```

**Profile validation errors**:
```bash
# Check profile syntax
python -c "import yaml; yaml.safe_load(open('config/profiles/myprofile.yaml'))"
```

## Related Documentation

- [Provider Configuration Guide](./provider-configuration.md) - LLM and embedding provider setup
- [Advanced Retrieval Configuration](./advanced-retrieval-configuration.md) - Reranking and grading details
- [Memory Platform Guide](./memory-platform.md) - Memory scope configuration
- [Graph Intelligence Guide](./graph-intelligence.md) - Community detection and LazyRAG
- [CLI Installation Manual](./cli-installation.md) - CLI tool usage
- [Deployment Production Guide](./deployment-production.md) - Production deployment patterns
