# Provider Configuration Guide

**Date:** 2026-01-12
**Version:** 1.0
**Related Epic:** Epic 18 - Enhanced Documentation & DevOps

---

## Overview

This guide documents how to configure LLM and embedding providers for the Agentic RAG platform. The system supports multiple providers through a unified configuration interface, allowing you to switch between cloud and local models based on your requirements.

### Provider Architecture

```
+------------------------------------------------------------------+
|                    PROVIDER ABSTRACTION LAYER                     |
+------------------------------------------------------------------+
|                                                                    |
|  LLM PROVIDERS              EMBEDDING PROVIDERS                    |
|  +-------------+            +----------------+                     |
|  | OpenAI      |            | OpenAI         |                     |
|  | Anthropic   |            | Voyage AI      |                     |
|  | Google      |            | Google Gemini  |                     |
|  | OpenRouter  |            | OpenRouter     |                     |
|  | Ollama      |            | Ollama         |                     |
|  +-------------+            +----------------+                     |
|        |                           |                               |
|        v                           v                               |
|  +-------------------------------------------+                     |
|  |        Unified Configuration              |                     |
|  |  - Environment Variables                  |                     |
|  |  - Configuration Profiles                 |                     |
|  |  - Runtime Overrides                      |                     |
|  +-------------------------------------------+                     |
|                                                                    |
+------------------------------------------------------------------+
```

### Quick Start

1. **Choose a configuration profile** (minimal, standard, or enterprise)
2. **Set your API keys** in `.env`
3. **Optionally override** provider settings via environment variables

```bash
# Minimal configuration (OpenAI only)
CONFIG_PROFILE=minimal
OPENAI_API_KEY=sk-...

# Start the services
docker compose up -d
```

---

## Configuration Profiles

The system uses YAML-based configuration profiles located in `config/profiles/`. Each profile pre-configures LLM and embedding providers along with feature flags.

| Profile | Target Use Case | Default LLM | Default Embedding |
|---------|-----------------|-------------|-------------------|
| `minimal` | Development, testing | gpt-4o-mini | text-embedding-3-small |
| `standard` | Production, small teams | gpt-4o | text-embedding-3-small |
| `enterprise` | Large teams, advanced features | claude-3.5-sonnet (via OpenRouter) | voyage-code-3 |

### Setting the Profile

```bash
# In .env file
CONFIG_PROFILE=standard  # Options: minimal | standard | enterprise
```

Profile settings serve as defaults. Any environment variable you set explicitly will override the profile value.

---

## LLM Providers

The system supports five LLM providers. Set `LLM_PROVIDER` to choose your provider.

### Supported Providers

| Provider | Value | API Key Required | Base URL Configurable |
|----------|-------|------------------|----------------------|
| OpenAI | `openai` | Yes | Optional |
| Anthropic | `anthropic` | Yes | No |
| Google Gemini | `gemini` | Yes | No |
| OpenRouter | `openrouter` | Yes | Optional |
| Ollama | `ollama` | No | Yes |

---

### OpenAI

**Best for:** Production deployments, latest GPT models, broad ecosystem support

#### Setup

```bash
# Required
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key

# Optional
LLM_MODEL_ID=gpt-4o          # Default: gpt-4o-mini
OPENAI_BASE_URL=             # Custom endpoint (e.g., Azure OpenAI)
```

#### Available Models

| Model | Context | Input Cost | Output Cost | Use Case |
|-------|---------|------------|-------------|----------|
| `gpt-4o` | 128K | $2.50/1M | $10.00/1M | Production, complex reasoning |
| `gpt-4o-mini` | 128K | $0.15/1M | $0.60/1M | Development, cost-sensitive |
| `gpt-4-turbo` | 128K | $10.00/1M | $30.00/1M | Legacy compatibility |
| `o1-preview` | 128K | $15.00/1M | $60.00/1M | Advanced reasoning |
| `o1-mini` | 128K | $3.00/1M | $12.00/1M | Reasoning, cost-optimized |

#### Custom Base URL (Azure OpenAI)

To use Azure OpenAI or other OpenAI-compatible endpoints:

```bash
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_API_KEY=your-azure-key
LLM_MODEL_ID=your-deployment-name
```

---

### Anthropic

**Best for:** Long context, complex reasoning, reduced hallucination

#### Setup

```bash
# Required
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-api-key

# Optional
LLM_MODEL_ID=claude-3-5-sonnet-20241022  # Default: gpt-4o-mini (falls back)
```

#### Available Models

| Model | Context | Input Cost | Output Cost | Use Case |
|-------|---------|------------|-------------|----------|
| `claude-3-5-sonnet-20241022` | 200K | $3.00/1M | $15.00/1M | Best overall performance |
| `claude-3-5-haiku-20241022` | 200K | $0.80/1M | $4.00/1M | Fast, cost-effective |
| `claude-3-opus-20240229` | 200K | $15.00/1M | $75.00/1M | Most capable |

#### Important Note: Embeddings

Anthropic does not provide native embedding models. When using `LLM_PROVIDER=anthropic`, you must set a separate embedding provider:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Recommended: Voyage AI (endorsed by Anthropic)
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=your-voyage-key

# Alternative: OpenAI
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

---

### Google Gemini

**Best for:** Multimodal capabilities, generous free tier, long context

#### Setup

```bash
# Required
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key

# Optional
LLM_MODEL_ID=gemini-1.5-pro  # Default: gpt-4o-mini (falls back)
```

#### Available Models

| Model | Context | Input Cost | Output Cost | Use Case |
|-------|---------|------------|-------------|----------|
| `gemini-1.5-pro` | 2M | $1.25/1M | $5.00/1M | Long context, multimodal |
| `gemini-1.5-flash` | 1M | $0.075/1M | $0.30/1M | Fast, cost-effective |
| `gemini-1.0-pro` | 32K | $0.50/1M | $1.50/1M | Legacy |

#### Getting an API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create a new key or select existing project

---

### OpenRouter

**Best for:** Model flexibility, unified billing, access to multiple providers

OpenRouter provides access to 100+ models from different providers through a single API.

#### Setup

```bash
# Required
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-your-api-key

# Optional
LLM_MODEL_ID=anthropic/claude-3.5-sonnet  # Use provider/model format
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # Default
```

#### Available Models (Selection)

| Model ID | Provider | Context | ~Cost/1M Tokens |
|----------|----------|---------|-----------------|
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K | ~$18 |
| `openai/gpt-4o` | OpenAI | 128K | ~$12 |
| `google/gemini-pro-1.5` | Google | 2M | ~$6 |
| `meta-llama/llama-3.1-70b-instruct` | Meta | 128K | ~$0.50 |
| `mistralai/mistral-large` | Mistral | 128K | ~$4 |
| `deepseek/deepseek-chat` | DeepSeek | 128K | ~$0.14 |

#### Getting an API Key

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign in and navigate to "Keys"
3. Create a new API key

#### Cost Optimization

OpenRouter offers credit-based pricing. Use the routing features for cost control:

```bash
# Route simple queries to cheaper models
ROUTING_SIMPLE_MODEL=meta-llama/llama-3.1-8b-instruct
ROUTING_MEDIUM_MODEL=anthropic/claude-3-5-haiku
ROUTING_COMPLEX_MODEL=anthropic/claude-3.5-sonnet
```

---

### Ollama (Local)

**Best for:** Privacy, offline operation, no API costs, custom models

#### Setup

```bash
# Required
LLM_PROVIDER=ollama

# Optional
OLLAMA_BASE_URL=http://localhost:11434/v1  # Default
LLM_MODEL_ID=llama3.1                       # Must be pulled first
```

#### Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull your desired model:

```bash
# Recommended models
ollama pull llama3.1          # 8B, good balance
ollama pull llama3.1:70b      # 70B, highest quality
ollama pull mistral           # 7B, fast
ollama pull codellama         # 7B, code-focused
ollama pull deepseek-coder    # Code generation
```

#### Available Models

| Model | Size | RAM Required | Use Case |
|-------|------|--------------|----------|
| `llama3.1` | 8B | ~8GB | General purpose |
| `llama3.1:70b` | 70B | ~48GB | High quality |
| `mistral` | 7B | ~6GB | Fast inference |
| `codellama` | 7B | ~6GB | Code generation |
| `mixtral` | 8x7B | ~32GB | MoE, versatile |
| `phi3` | 3.8B | ~4GB | Lightweight |

#### Hardware Detection

The CLI's `rag install` command automatically detects your hardware:

```bash
rag install
# Detects: GPU type, VRAM, system RAM
# Recommends: Appropriate model size
```

#### Estimated Costs

Ollama is free to run locally. Hardware costs depend on your setup:

| Setup | Hardware | Power Cost (24/7) |
|-------|----------|-------------------|
| CPU-only | Any modern CPU | ~$5-15/month |
| Consumer GPU | RTX 3080 (10GB) | ~$10-20/month |
| Pro GPU | RTX 4090 (24GB) | ~$15-30/month |
| Apple Silicon | M2/M3 Pro/Max | ~$5-10/month |

---

## Embedding Providers

Embeddings are essential for semantic search. The platform supports five embedding providers.

### Provider Selection Logic

1. If `EMBEDDING_PROVIDER` is set explicitly, use that provider
2. If `LLM_PROVIDER` supports embeddings, use the same provider
3. If `LLM_PROVIDER=anthropic`, fall back to OpenAI with a warning

```bash
# Explicit embedding provider (recommended)
EMBEDDING_PROVIDER=voyage

# Implicit (uses LLM provider if it supports embeddings)
LLM_PROVIDER=openai  # Embeddings automatically use OpenAI
```

### Supported Providers

| Provider | Value | Dimension Options | API Key |
|----------|-------|-------------------|---------|
| OpenAI | `openai` | 1536, 3072 | `OPENAI_API_KEY` |
| Voyage AI | `voyage` | 1024, 1536 | `VOYAGE_API_KEY` |
| Google Gemini | `gemini` | 768 | `GEMINI_API_KEY` |
| OpenRouter | `openrouter` | Varies | `OPENROUTER_API_KEY` |
| Ollama | `ollama` | Varies | None |

---

### OpenAI Embeddings

**Best for:** General purpose, wide language support, stable API

```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small  # Default
EMBEDDING_DIMENSION=1536                # Default
```

#### Available Models

| Model | Dimensions | Cost/1M Tokens | Use Case |
|-------|------------|----------------|----------|
| `text-embedding-3-small` | 1536 | $0.02 | Cost-effective, recommended |
| `text-embedding-3-large` | 3072 | $0.13 | Higher quality |
| `text-embedding-ada-002` | 1536 | $0.10 | Legacy |

---

### Voyage AI Embeddings

**Best for:** Code embeddings, Anthropic integration, specialized domains

Voyage AI is [recommended by Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/embeddings) for users of Claude models.

```bash
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=your-voyage-key
EMBEDDING_MODEL=voyage-3         # Default
EMBEDDING_DIMENSION=1024
```

#### Available Models

| Model | Dimensions | Cost/1M Tokens | Use Case |
|-------|------------|----------------|----------|
| `voyage-3` | 1024 | $0.06 | General purpose, latest |
| `voyage-3-lite` | 512 | $0.02 | Cost-optimized |
| `voyage-code-3` | 1024 | $0.06 | Code and documentation |
| `voyage-finance-2` | 1024 | $0.12 | Financial documents |
| `voyage-law-2` | 1024 | $0.12 | Legal documents |

#### Getting an API Key

1. Go to [Voyage AI](https://www.voyageai.com/)
2. Sign up and create an API key
3. Set `VOYAGE_API_KEY` in your `.env`

---

### Google Gemini Embeddings

**Best for:** Multimodal embeddings, Gemini LLM users

```bash
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
EMBEDDING_MODEL=text-embedding-004  # Default
EMBEDDING_DIMENSION=768
```

#### Available Models

| Model | Dimensions | Cost/1M Tokens | Use Case |
|-------|------------|----------------|----------|
| `text-embedding-004` | 768 | Free tier available | General purpose |
| `embedding-001` | 768 | Free tier available | Legacy |

---

### OpenRouter Embeddings

**Best for:** Model flexibility, trying different providers

```bash
EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Note: OpenRouter embedding support varies. Check their [models page](https://openrouter.ai/models) for current availability.

---

### Ollama Embeddings (Local)

**Best for:** Privacy, offline operation, no API costs

```bash
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text  # Must be pulled first
```

#### Available Models

```bash
# Pull embedding models
ollama pull nomic-embed-text   # 137M params, 768 dim
ollama pull mxbai-embed-large  # 335M params, 1024 dim
ollama pull all-minilm         # 22M params, 384 dim
```

| Model | Dimensions | Size | Use Case |
|-------|------------|------|----------|
| `nomic-embed-text` | 768 | 274MB | Recommended default |
| `mxbai-embed-large` | 1024 | 670MB | Higher quality |
| `all-minilm` | 384 | 46MB | Lightweight |

---

## Common Configuration Patterns

### Pattern 1: Development (Cost-Optimized)

```bash
CONFIG_PROFILE=minimal
LLM_PROVIDER=openai
LLM_MODEL_ID=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Uses OpenAI for embeddings by default
# Estimated cost: ~$5-20/month for development
```

### Pattern 2: Production (Balanced)

```bash
CONFIG_PROFILE=standard
LLM_PROVIDER=openai
LLM_MODEL_ID=gpt-4o
OPENAI_API_KEY=sk-...

# Enable reranking for better quality
RERANKER_ENABLED=true
RERANKER_PROVIDER=flashrank  # Free, local

# Estimated cost: ~$50-200/month depending on usage
```

### Pattern 3: Enterprise (Best Quality)

```bash
CONFIG_PROFILE=enterprise
LLM_PROVIDER=openrouter
LLM_MODEL_ID=anthropic/claude-3.5-sonnet
OPENROUTER_API_KEY=sk-or-...

# Use Voyage for embeddings (Anthropic recommended)
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=...

# Enable all advanced features
RERANKER_ENABLED=true
RERANKER_PROVIDER=cohere
COHERE_API_KEY=...

# Estimated cost: ~$200-1000/month depending on usage
```

### Pattern 4: Privacy-First (Local)

```bash
CONFIG_PROFILE=minimal
LLM_PROVIDER=ollama
LLM_MODEL_ID=llama3.1
OLLAMA_BASE_URL=http://localhost:11434/v1

# Local embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text

# No API costs, requires local hardware
```

### Pattern 5: Hybrid (Cloud LLM, Local Embeddings)

```bash
LLM_PROVIDER=openai
LLM_MODEL_ID=gpt-4o
OPENAI_API_KEY=sk-...

# Local embeddings for cost savings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434/v1
```

---

## Cost Estimation

### Monthly Cost Calculator

| Component | Low Usage | Medium Usage | High Usage |
|-----------|-----------|--------------|------------|
| LLM (gpt-4o-mini) | ~$5 | ~$25 | ~$100 |
| LLM (gpt-4o) | ~$20 | ~$100 | ~$500 |
| LLM (claude-3.5-sonnet) | ~$30 | ~$150 | ~$750 |
| Embeddings (OpenAI) | ~$2 | ~$10 | ~$50 |
| Embeddings (Voyage) | ~$3 | ~$15 | ~$75 |
| Reranking (Cohere) | ~$5 | ~$25 | ~$125 |
| Reranking (FlashRank) | $0 | $0 | $0 |

**Usage Definitions:**
- **Low:** ~10K queries/month, ~1M tokens/month
- **Medium:** ~50K queries/month, ~5M tokens/month
- **High:** ~200K queries/month, ~20M tokens/month

### Cost Optimization Strategies

1. **Use intelligent model routing** - Route simple queries to cheaper models
2. **Enable FlashRank** - Free local reranking instead of Cohere
3. **Cache aggressively** - Enable reranking cache to avoid duplicate work
4. **Use contextual prompt caching** - 90% cost reduction for contextual retrieval
5. **Local embeddings** - Zero API cost with Ollama embeddings

---

## Troubleshooting

### Common Issues

#### "Missing required environment variable: OPENAI_API_KEY"

**Cause:** LLM provider requires an API key that is not set.

**Solution:**
```bash
# Check your LLM_PROVIDER setting
LLM_PROVIDER=openai  # Requires OPENAI_API_KEY

# Set the appropriate key
OPENAI_API_KEY=sk-your-key-here
```

#### "EMBEDDING_PROVIDER must be one of: openai, openrouter, ollama, gemini, voyage"

**Cause:** Invalid embedding provider value.

**Solution:**
```bash
# Use a valid provider
EMBEDDING_PROVIDER=openai  # Not "open-ai" or "OPENAI"
```

#### "Anthropic doesn't support embeddings"

**Cause:** Using Anthropic as LLM provider without setting an embedding provider.

**Solution:**
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Must set an embedding provider
EMBEDDING_PROVIDER=voyage  # Recommended by Anthropic
VOYAGE_API_KEY=...
```

#### "Connection refused" with Ollama

**Cause:** Ollama service is not running or wrong URL.

**Solution:**
```bash
# Start Ollama
ollama serve

# Verify the URL (default port is 11434)
OLLAMA_BASE_URL=http://localhost:11434/v1

# Test connectivity
curl http://localhost:11434/api/tags
```

#### "Model not found" with Ollama

**Cause:** Requested model not pulled locally.

**Solution:**
```bash
# Pull the model first
ollama pull llama3.1

# Then set it in config
LLM_MODEL_ID=llama3.1
```

#### Rate Limit Errors

**Cause:** Too many API requests in a short period.

**Solution:**
```bash
# Configure rate limiting
RATE_LIMIT_PER_MINUTE=30  # Reduce from default 60
RATE_LIMIT_BACKEND=redis   # Use Redis for distributed limiting
```

### Validation Commands

Use the CLI to validate your configuration:

```bash
# Check configuration
rag doctor

# Validate provider connectivity
rag validate --providers

# Test a simple query
rag query "Hello, world"
```

---

## Environment Variable Reference

### Core Provider Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PROFILE` | Configuration profile | `standard` |
| `LLM_PROVIDER` | LLM provider | `openai` |
| `LLM_MODEL_ID` | Model identifier | `gpt-4o-mini` |
| `EMBEDDING_PROVIDER` | Embedding provider | (from LLM_PROVIDER) |
| `EMBEDDING_MODEL` | Embedding model | (provider default) |
| `EMBEDDING_DIMENSION` | Vector dimensions | `1536` |

### Provider-Specific API Keys

| Variable | Required When |
|----------|--------------|
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | `LLM_PROVIDER=anthropic` |
| `GEMINI_API_KEY` | `LLM_PROVIDER=gemini` or `EMBEDDING_PROVIDER=gemini` |
| `OPENROUTER_API_KEY` | `LLM_PROVIDER=openrouter` or `EMBEDDING_PROVIDER=openrouter` |
| `VOYAGE_API_KEY` | `EMBEDDING_PROVIDER=voyage` |

### Provider Base URLs

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | (none) | Custom OpenAI-compatible endpoint |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server URL |

### Model Routing (Cost Optimization)

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTING_SIMPLE_MODEL` | `gpt-4o-mini` | Model for simple queries |
| `ROUTING_MEDIUM_MODEL` | `gpt-4o` | Model for medium complexity |
| `ROUTING_COMPLEX_MODEL` | `gpt-4o` | Model for complex queries |
| `ROUTING_BASELINE_MODEL` | `gpt-4o` | Default fallback model |

---

## Related Documentation

- [Advanced Retrieval Configuration](./advanced-retrieval-configuration.md) - Reranking, CRAG, contextual retrieval
- [Voice I/O Configuration](./voice-io-configuration.md) - Speech-to-text and text-to-speech setup
- [Observability Guide](./observability.md) - Metrics and monitoring
- [MCP Wrapper Architecture](./mcp-wrapper-architecture.md) - Tool server integration
