# Graph Intelligence Guide

This guide covers the Graph Intelligence features for advanced knowledge graph retrieval and reasoning. The system implements competitive features inspired by Microsoft GraphRAG, LightRAG, and Zep memory systems - enabling sophisticated query routing, community detection, and graph-based reranking.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Graphiti Integration](#graphiti-integration)
- [Community Detection](#community-detection)
- [LazyRAG Pattern](#lazyrag-pattern)
- [Query Routing](#query-routing)
- [Graph-Based Rerankers](#graph-based-rerankers)
- [Dual-Level Retrieval](#dual-level-retrieval)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Use Cases](#use-cases)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

The Graph Intelligence system provides advanced retrieval capabilities through:

- **Graphiti Integration**: Temporal knowledge graph with hybrid search (semantic + BM25 + graph)
- **Community Detection**: Louvain/Leiden algorithms for entity clustering and thematic abstraction
- **LazyRAG Pattern**: Query-time summarization achieving 99% reduction in indexing costs
- **Query Routing**: Global/Local/Hybrid routing for optimal retrieval strategy selection
- **Graph Rerankers**: Episode-mentions, node-distance, and hybrid scoring
- **Dual-Level Retrieval**: Combines entity-level facts with community-level themes

These features work together to answer both specific entity questions and abstract thematic queries effectively.

## Architecture

### Component Overview

```
+------------------+     +------------------+     +------------------+
|    Graphiti      |     |      Neo4j       |     |   PostgreSQL     |
|   (Temporal KG)  |     |  (Graph Store)   |     |   + pgvector     |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        +------------------------+------------------------+
                                 |
                    +---------------------------+
                    |    Query Router (20-B3)   |
                    |   - Global query patterns |
                    |   - Local query patterns  |
                    |   - LLM classification    |
                    +---------------------------+
                           |          |
              +------------+          +-------------+
              |                                     |
    +-----------------+                 +-------------------+
    | Community-Level |                 |   Entity-Level    |
    |   (GLOBAL)      |                 |    (LOCAL)        |
    +-----------------+                 +-------------------+
              |                                     |
    +-----------------+                 +-------------------+
    |  CommunityDetector (20-B1)  |     |  LazyRAGRetriever (20-B2)  |
    |  - Louvain/Leiden           |     |  - Seed entity search      |
    |  - Hierarchical summaries   |     |  - N-hop traversal         |
    +-----------------+                 +-------------------+
              |                                     |
              +-------------------------------------+
                                 |
                    +---------------------------+
                    |  DualLevelRetriever (20-C2) |
                    |   - Parallel execution      |
                    |   - LLM synthesis           |
                    +---------------------------+
                                 |
                    +---------------------------+
                    |   Graph Rerankers (20-C1)  |
                    |   - Episode mentions       |
                    |   - Node distance          |
                    |   - Hybrid scoring         |
                    +---------------------------+
```

### Data Flow

1. **Query Intake**: User query enters the system
2. **Query Classification**: QueryRouter determines GLOBAL/LOCAL/HYBRID
3. **Retrieval Execution**: Appropriate retriever(s) execute in parallel
4. **Graph Reranking**: Results are reranked using graph signals
5. **Synthesis**: DualLevelRetriever combines perspectives via LLM
6. **Response**: Unified answer with confidence score

## Graphiti Integration

### Overview

Graphiti provides the temporal knowledge graph foundation with hybrid search capabilities. It combines semantic similarity, BM25 text matching, and graph traversal in a single search operation.

### GraphitiClient

```python
from agentic_rag_backend.db.graphiti import GraphitiClient, create_graphiti_client

# Factory function creates and connects
client = await create_graphiti_client(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_provider="openai",
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
)

# Execute hybrid search
search_result = await client.client.search(
    query="What are the main features of the product?",
    group_ids=["tenant-123"],  # Multi-tenancy
    num_results=10,
    center_node_uuid=None,  # Optional: center search around a node
)

# Access results
for node in search_result.nodes:
    print(f"Entity: {node.name} ({node.uuid})")
    print(f"Summary: {node.summary}")

for edge in search_result.edges:
    print(f"Relationship: {edge.source_node_uuid} --{edge.name}--> {edge.target_node_uuid}")
    print(f"Fact: {edge.fact}")
```

### Supported LLM Providers

Graphiti supports multiple LLM providers for entity extraction:

| Provider | Environment Variables | Notes |
|----------|----------------------|-------|
| OpenAI | `OPENAI_API_KEY` | Default provider |
| OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | OpenAI-compatible |
| Ollama | `OLLAMA_BASE_URL` | Local models |
| Anthropic | `ANTHROPIC_API_KEY` | Claude models |
| Gemini | `GEMINI_API_KEY` | Google models |

### Supported Embedding Providers

| Provider | Environment Variables | Default Model |
|----------|----------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `text-embedding-3-small` |
| OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | Provider-specific |
| Ollama | `OLLAMA_BASE_URL` | Provider-specific |
| Gemini | `GEMINI_API_KEY` | `embedding-001` |
| Voyage | `VOYAGE_API_KEY` | `voyage-2` |

### Connection Lifecycle

```python
# State machine: NEW -> CONNECTING -> CONNECTED -> DISCONNECTED
client = GraphitiClient(...)

# Connect (transitions to CONNECTED)
await client.connect()

# Use client
if client.is_connected:
    result = await client.client.search(...)

# Disconnect (cannot reconnect - create new instance)
await client.disconnect(timeout=5.0)
```

## Community Detection

Community detection identifies clusters of densely connected entities, enabling abstract "global" queries about themes and patterns across the knowledge base.

### Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Louvain** | Fast, greedy modularity optimization | General purpose, large graphs |
| **Leiden** | Improved Louvain with guaranteed connectivity | Higher quality, smaller graphs |

### CommunityDetector Usage

```python
from agentic_rag_backend.graph.community import CommunityDetector, CommunityAlgorithm

detector = CommunityDetector(
    neo4j_client=neo4j_client,
    llm_client=openai_client,  # For summary generation
    algorithm=CommunityAlgorithm.LOUVAIN,
    min_community_size=3,
    max_hierarchy_levels=3,
    summary_model="gpt-4o-mini",
)

# Detect communities for a tenant
communities = await detector.detect_communities(
    tenant_id="tenant-123",
    generate_summaries=True,
    algorithm=CommunityAlgorithm.LOUVAIN,
    min_size=3,
    max_levels=3,
)

# Each community includes:
for community in communities:
    print(f"Community: {community.name}")
    print(f"Level: {community.level}")
    print(f"Entities: {community.entity_count}")
    print(f"Summary: {community.summary}")
    print(f"Keywords: {community.keywords}")
```

### Hierarchical Structure

Communities are organized in levels:

```
Level 2 (Highest Abstraction)
    +-- "Finance, Technology, Operations Group"
        |
Level 1 (Mid-Level)
        +-- "Finance, Accounting Community"
        +-- "Technology, Engineering Community"
        |
Level 0 (Most Granular)
            +-- "Finance Team Community" (5 entities)
            +-- "Accounting Team Community" (4 entities)
            +-- "Backend Engineering Community" (8 entities)
            +-- "Frontend Engineering Community" (6 entities)
```

### Neo4j Schema

Communities are stored with the following structure:

```cypher
// Community node
(:Community {
    id: "uuid",
    tenant_id: "uuid",
    name: "Finance Team Community",
    level: 0,
    summary: "A group of entities related to...",
    keywords: ["finance", "accounting", "budget"],
    entity_count: 5,
    parent_id: "parent-uuid",
    child_ids: [],
    created_at: datetime(),
    updated_at: datetime()
})

// Relationships
(entity:Entity)-[:BELONGS_TO]->(community:Community)
(parent:Community)-[:PARENT_OF]->(child:Community)
(child:Community)-[:CHILD_OF]->(parent:Community)
```

### Search Communities

```python
# Search by keyword/summary
results = await detector.search_communities(
    query="financial processes",
    tenant_id="tenant-123",
    level=None,  # All levels
    limit=10,
)

# List communities by level
communities, total = await detector.list_communities(
    tenant_id="tenant-123",
    level=0,  # Most granular
    limit=50,
    offset=0,
)
```

## LazyRAG Pattern

LazyRAG defers graph summarization to query time, achieving up to 99% reduction in indexing costs compared to Microsoft GraphRAG's eager summarization approach.

### How It Works

1. **Find Seed Entities**: Graphiti hybrid search finds relevant starting points
2. **Expand Subgraph**: N-hop traversal from seeds via Neo4j
3. **Get Community Context**: Optional community summaries from CommunityDetector
4. **Generate Summary**: LLM synthesizes answer at query time
5. **Estimate Confidence**: Based on entity coverage and relationship density

### LazyRAGRetriever Usage

```python
from agentic_rag_backend.retrieval.lazy_rag import LazyRAGRetriever

retriever = LazyRAGRetriever(
    graphiti_client=graphiti_client,
    neo4j_client=neo4j_client,
    settings=settings,
    community_detector=community_detector,
)

# Full query with summary
result = await retriever.query(
    query="How does the authentication system work?",
    tenant_id="tenant-123",
    max_entities=50,
    max_hops=2,
    use_communities=True,
    include_summary=True,
)

print(f"Summary: {result.summary}")
print(f"Confidence: {result.confidence}")
print(f"Entities found: {result.expanded_entity_count}")
print(f"Processing time: {result.processing_time_ms}ms")

# Debug: expand subgraph without summary
expansion = await retriever.expand_only(
    query="authentication",
    tenant_id="tenant-123",
    max_entities=50,
    max_hops=2,
)
```

### Confidence Scoring

Confidence is calculated from three factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Entity Coverage | 0.4 | Ratio of found entities to max_entities |
| Query Term Coverage | 0.4 | Percentage of query terms found in entity names/descriptions |
| Relationship Density | 0.2 | Ratio of relationships to possible connections |

### Circuit Breaker Pattern

LazyRAG implements per-tenant circuit breakers for Graphiti failures:

```python
# Circuit breaker configuration
_graphiti_circuit_threshold = 3  # failures before opening
_graphiti_circuit_timeout = 30   # seconds to keep open

# When Graphiti fails 3+ times for a tenant:
# - Circuit opens for 30 seconds
# - Fallback to direct Neo4j text search
# - After timeout, circuit closes and retries Graphiti
```

## Query Routing

Query routing classifies queries to determine the optimal retrieval strategy:

- **GLOBAL**: Community-level retrieval for abstract questions
- **LOCAL**: Entity-level retrieval for specific questions
- **HYBRID**: Weighted combination of both approaches

### Classification Methods

#### Rule-Based (Fast Path < 10ms)

Pattern matching against predefined regex patterns:

**Global Patterns** (indicate need for corpus-wide understanding):
- "What are the main themes..."
- "Summarize/overview/synopsis..."
- "All types/kinds/categories of..."
- "Trends/patterns across..."
- "Compare X across Y..."

**Local Patterns** (indicate need for specific entity details):
- "What is X..."
- "Who is/was..."
- "Where/when did..."
- "How does X work..."
- "Find/locate X named..."
- "Definition of..."

#### LLM Classification (Fallback)

When rule-based confidence is below threshold:

```python
from agentic_rag_backend.retrieval.query_router import QueryRouter

router = QueryRouter(
    settings=settings,
    use_llm=True,
    llm_model="gpt-4o-mini",
    confidence_threshold=0.7,
)

decision = await router.route(
    query="What are the main security concerns?",
    tenant_id="tenant-123",
)

print(f"Type: {decision.query_type}")  # GLOBAL, LOCAL, or HYBRID
print(f"Confidence: {decision.confidence}")
print(f"Method: {decision.classification_method}")
print(f"Global weight: {decision.global_weight}")
print(f"Local weight: {decision.local_weight}")
```

### Routing Decision

```python
from agentic_rag_backend.retrieval.query_router_models import QueryType, RoutingDecision

# RoutingDecision includes:
decision = RoutingDecision(
    query_type=QueryType.HYBRID,
    confidence=0.8,
    reasoning="Mixed patterns detected",
    global_weight=0.4,
    local_weight=0.6,
    classification_method="combined",
    processing_time_ms=5,
    global_matches=2,
    local_matches=3,
)
```

### Caching

Routing decisions are cached with LRU eviction:

- **Cache Size**: 1000 entries
- **TTL**: 300 seconds (5 minutes)
- **Key**: SHA256 hash of query + use_llm flag

## Graph-Based Rerankers

Graph rerankers use temporal and structural graph signals to improve retrieval ranking beyond semantic similarity.

### Reranker Types

| Type | Signal | Description |
|------|--------|-------------|
| **Episode** | Temporal | Boost entities mentioned in recent episodes |
| **Distance** | Structural | Boost entities closer to query concepts in graph |
| **Hybrid** | Combined | Weighted combination of both signals |

### EpisodeMentionsReranker

Entities mentioned in more recent episodes are considered more contextually relevant:

```python
from agentic_rag_backend.retrieval.graph_rerankers import (
    EpisodeMentionsReranker,
    NodeDistanceReranker,
    HybridGraphReranker,
)

reranker = EpisodeMentionsReranker(
    neo4j_client=neo4j_client,
    episode_window_days=30,  # Look-back window
    original_weight=0.7,     # Weight for original score
)

reranked = await reranker.rerank(
    query="authentication",
    results=initial_results,
    tenant_id="tenant-123",
)

for result in reranked:
    print(f"Score: {result.combined_score}")
    print(f"Episode mentions: {result.graph_context.episode_mentions}")
```

**Scoring Formula**:
```
episode_score = min(1.0, mentions / 10)  # 10+ mentions = 1.0
combined_score = original_weight * original_score + (1 - original_weight) * episode_score
```

### NodeDistanceReranker

Entities closer to query concepts in the knowledge graph receive higher scores:

```python
reranker = NodeDistanceReranker(
    neo4j_client=neo4j_client,
    graphiti_client=graphiti_client,  # For entity extraction
    max_distance=3,       # Max hops (beyond = 0 score)
    original_weight=0.7,
)

reranked = await reranker.rerank(
    query="authentication security",
    results=initial_results,
    tenant_id="tenant-123",
)

for result in reranked:
    print(f"Score: {result.combined_score}")
    print(f"Min distance: {result.graph_context.min_distance}")
    print(f"Query entities: {result.graph_context.query_entities}")
```

**Scoring Formula**:
```
distance_score = max(0.0, 1.0 - (distance / max_distance))
combined_score = original_weight * original_score + (1 - original_weight) * distance_score
```

### HybridGraphReranker

Combines both episode and distance signals:

```python
reranker = HybridGraphReranker(
    neo4j_client=neo4j_client,
    graphiti_client=graphiti_client,
    episode_weight=0.3,
    distance_weight=0.3,
    original_weight=0.4,
    episode_window_days=30,
    max_distance=3,
)

# Runs both sub-rerankers in parallel
reranked = await reranker.rerank(
    query="authentication security",
    results=initial_results,
    tenant_id="tenant-123",
)
```

**Scoring Formula**:
```
combined_score = original_weight * original_score
               + episode_weight * episode_score
               + distance_weight * distance_score
```

### Factory Functions

```python
from agentic_rag_backend.retrieval.graph_rerankers import (
    get_graph_reranker_adapter,
    create_graph_reranker,
)

# Get adapter from settings
adapter = get_graph_reranker_adapter(settings)

# Create appropriate reranker
reranker = create_graph_reranker(
    adapter=adapter,
    neo4j_client=neo4j_client,
    graphiti_client=graphiti_client,
)
```

## Dual-Level Retrieval

Dual-level retrieval combines entity-level facts with community-level themes for comprehensive answers, inspired by LightRAG's architecture.

### DualLevelRetriever Usage

```python
from agentic_rag_backend.retrieval.dual_level import DualLevelRetriever

retriever = DualLevelRetriever(
    graphiti_client=graphiti_client,
    neo4j_client=neo4j_client,
    settings=settings,
    community_detector=community_detector,
)

result = await retriever.retrieve(
    query="How does user authentication integrate with security policies?",
    tenant_id="tenant-123",
    low_level_limit=10,
    high_level_limit=5,
    include_synthesis=True,
    low_weight=0.6,
    high_weight=0.4,
)

print(f"Synthesis: {result.synthesis}")
print(f"Confidence: {result.confidence}")
print(f"Low-level results: {len(result.low_level_results)}")
print(f"High-level results: {len(result.high_level_results)}")
print(f"Fallback used: {result.fallback_used}")
```

### Result Types

```python
# Low-level: specific facts from entities
LowLevelResult(
    id="entity-uuid",
    name="AuthenticationService",
    type="Service",
    content="Handles user authentication...",
    score=0.85,
    source="source-doc-id",
    labels=["Entity", "Service"],
)

# High-level: thematic context from communities
HighLevelResult(
    id="community-uuid",
    name="Security Infrastructure Community",
    summary="A group of entities handling security...",
    keywords=["authentication", "authorization", "security"],
    level=1,
    entity_count=12,
    score=0.72,
    entity_ids=["entity-1", "entity-2", ...],
)
```

### Synthesis Prompt

The LLM synthesis combines both perspectives:

```
=== LOW-LEVEL CONTEXT (Specific Facts & Entities) ===
- AuthenticationService (Service): Handles user authentication...
- SecurityPolicy (Policy): Defines access control rules...

=== HIGH-LEVEL CONTEXT (Themes & Patterns) ===
- Security Infrastructure Community (Level 1, 12 entities): A group of entities...
  Keywords: authentication, authorization, security

Instructions:
1. Synthesize BOTH perspectives into a coherent answer
2. Use specific facts from low-level context for precision
3. Frame the answer within the broader themes from high-level context
4. If contexts conflict, prefer low-level facts but acknowledge the broader pattern
5. Indicate confidence: HIGH (both levels agree), MEDIUM (partial overlap), LOW (one level only)
```

## Configuration Reference

### Environment Variables

#### Community Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `COMMUNITY_DETECTION_ENABLED` | `false` | Enable community detection feature |
| `COMMUNITY_ALGORITHM` | `louvain` | Algorithm: `louvain` or `leiden` |
| `COMMUNITY_MIN_SIZE` | `3` | Minimum entities per community |
| `COMMUNITY_MAX_HIERARCHY_LEVELS` | `3` | Maximum hierarchy depth |
| `COMMUNITY_SUMMARY_MODEL` | `gpt-4o-mini` | LLM model for summaries |
| `COMMUNITY_REFRESH_SCHEDULE` | `0 2 * * *` | Cron schedule for refresh |

#### LazyRAG

| Variable | Default | Description |
|----------|---------|-------------|
| `LAZY_RAG_ENABLED` | `false` | Enable LazyRAG feature |
| `LAZY_RAG_MAX_ENTITIES` | `50` | Maximum entities in context |
| `LAZY_RAG_MAX_HOPS` | `2` | Relationship expansion depth (1-5) |
| `LAZY_RAG_SUMMARY_MODEL` | `gpt-4o-mini` | LLM model for summaries |
| `LAZY_RAG_USE_COMMUNITIES` | `true` | Include community context |

#### Query Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `QUERY_ROUTING_ENABLED` | `false` | Enable query routing feature |
| `QUERY_ROUTING_USE_LLM` | `false` | Use LLM for uncertain queries |
| `QUERY_ROUTING_LLM_MODEL` | `gpt-4o-mini` | LLM model for classification |
| `QUERY_ROUTING_CONFIDENCE_THRESHOLD` | `0.7` | Threshold for LLM fallback |

#### Graph Rerankers

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_RERANKER_ENABLED` | `false` | Enable graph reranking |
| `GRAPH_RERANKER_TYPE` | `hybrid` | Type: `episode`, `distance`, `hybrid` |
| `GRAPH_RERANKER_EPISODE_WEIGHT` | `0.3` | Weight for episode signal |
| `GRAPH_RERANKER_DISTANCE_WEIGHT` | `0.3` | Weight for distance signal |
| `GRAPH_RERANKER_ORIGINAL_WEIGHT` | `0.4` | Weight for original score |
| `GRAPH_RERANKER_EPISODE_WINDOW_DAYS` | `30` | Look-back window for episodes |
| `GRAPH_RERANKER_MAX_DISTANCE` | `3` | Maximum hops for distance scoring |

#### Dual-Level Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `DUAL_LEVEL_RETRIEVAL_ENABLED` | `false` | Enable dual-level retrieval |
| `DUAL_LEVEL_LOW_WEIGHT` | `0.6` | Weight for low-level (entities) |
| `DUAL_LEVEL_HIGH_WEIGHT` | `0.4` | Weight for high-level (communities) |
| `DUAL_LEVEL_LOW_LIMIT` | `10` | Max low-level results |
| `DUAL_LEVEL_HIGH_LIMIT` | `5` | Max high-level results |
| `DUAL_LEVEL_SYNTHESIS_MODEL` | `gpt-4o-mini` | LLM model for synthesis |
| `DUAL_LEVEL_SYNTHESIS_TEMPERATURE` | `0.3` | Temperature for synthesis |

### Feature Flags

Enable graph intelligence features in `.env`:

```bash
# Minimal setup
LAZY_RAG_ENABLED=true

# Standard setup
COMMUNITY_DETECTION_ENABLED=true
LAZY_RAG_ENABLED=true
QUERY_ROUTING_ENABLED=true
GRAPH_RERANKER_ENABLED=true

# Enterprise setup (all features)
COMMUNITY_DETECTION_ENABLED=true
COMMUNITY_ALGORITHM=leiden
LAZY_RAG_ENABLED=true
LAZY_RAG_MAX_ENTITIES=100
LAZY_RAG_MAX_HOPS=3
QUERY_ROUTING_ENABLED=true
QUERY_ROUTING_USE_LLM=true
GRAPH_RERANKER_ENABLED=true
GRAPH_RERANKER_TYPE=hybrid
DUAL_LEVEL_RETRIEVAL_ENABLED=true
```

## API Reference

### LazyRAG Endpoints

#### POST /api/v1/lazy-rag/query

Execute LazyRAG query with summarization.

**Request**:
```json
{
    "query": "How does authentication work?",
    "tenant_id": "uuid",
    "max_entities": 50,
    "max_hops": 2,
    "use_communities": true,
    "include_summary": true
}
```

**Response**:
```json
{
    "query": "How does authentication work?",
    "tenant_id": "uuid",
    "summary": "Authentication is handled by...",
    "confidence": 0.85,
    "entities": [...],
    "relationships": [...],
    "communities": [...],
    "seed_entity_count": 5,
    "expanded_entity_count": 23,
    "processing_time_ms": 1234,
    "missing_info": null
}
```

#### POST /api/v1/lazy-rag/expand

Debug endpoint to expand subgraph without summary.

#### GET /api/v1/lazy-rag/status

Get LazyRAG feature status and configuration.

### Query Router Endpoints

#### POST /api/v1/query-router/route

Classify a query for routing.

**Request**:
```json
{
    "query": "What are the main security themes?",
    "tenant_id": "uuid",
    "use_llm": false
}
```

**Response**:
```json
{
    "query_type": "GLOBAL",
    "confidence": 0.85,
    "reasoning": "Global patterns matched: 2 (ratio: 0.80)",
    "global_weight": 1.0,
    "local_weight": 0.0,
    "classification_method": "rule_based",
    "processing_time_ms": 3,
    "global_matches": 2,
    "local_matches": 0
}
```

### Community Endpoints

#### POST /api/v1/communities/detect

Trigger community detection for a tenant.

#### GET /api/v1/communities

List communities for a tenant.

#### GET /api/v1/communities/{id}

Get a specific community with details.

#### GET /api/v1/communities/search

Search communities by keyword.

### Dual-Level Endpoints

#### POST /api/v1/dual-level/retrieve

Execute dual-level retrieval with synthesis.

## Use Cases

### Use Case 1: Technical Documentation Q&A

**Scenario**: Users ask both specific ("How do I configure X?") and abstract ("What are the main features?") questions.

**Configuration**:
```bash
QUERY_ROUTING_ENABLED=true
LAZY_RAG_ENABLED=true
COMMUNITY_DETECTION_ENABLED=true
```

**Flow**:
1. QueryRouter classifies question
2. LOCAL queries -> LazyRAG for specific entity retrieval
3. GLOBAL queries -> Community summaries for themes
4. HYBRID queries -> Both, with synthesis

### Use Case 2: Customer Support Knowledge Base

**Scenario**: Support agents need quick access to both specific procedures and broader policy context.

**Configuration**:
```bash
DUAL_LEVEL_RETRIEVAL_ENABLED=true
DUAL_LEVEL_LOW_WEIGHT=0.7  # Prefer specific procedures
DUAL_LEVEL_HIGH_WEIGHT=0.3  # Include policy context
```

**Flow**:
1. DualLevelRetriever executes parallel retrieval
2. Low-level: Specific procedures and steps
3. High-level: Policy communities and guidelines
4. LLM synthesizes actionable answer

### Use Case 3: Conversational Memory Enhancement

**Scenario**: Long conversations need context from recent interactions (episodes) to stay coherent.

**Configuration**:
```bash
GRAPH_RERANKER_ENABLED=true
GRAPH_RERANKER_TYPE=episode
GRAPH_RERANKER_EPISODE_WINDOW_DAYS=7  # Short-term memory
```

**Flow**:
1. Standard retrieval finds relevant entities
2. EpisodeMentionsReranker boosts recently discussed entities
3. Conversation maintains coherent context

### Use Case 4: Large-Scale Knowledge Discovery

**Scenario**: Exploring a large codebase or documentation corpus.

**Configuration**:
```bash
COMMUNITY_DETECTION_ENABLED=true
COMMUNITY_ALGORITHM=leiden
COMMUNITY_MAX_HIERARCHY_LEVELS=4
LAZY_RAG_MAX_ENTITIES=100
LAZY_RAG_MAX_HOPS=3
```

**Flow**:
1. Community detection builds hierarchical structure
2. Users browse communities to understand high-level themes
3. Drill down to specific entities as needed
4. LazyRAG provides detailed answers on demand

## Performance Tuning

### Query Routing Performance

Target: < 10ms for rule-based classification

**Optimizations**:
- Compiled regex patterns at module load
- Non-greedy quantifiers to prevent ReDoS
- MAX_QUERY_LENGTH validation (10,000 chars)
- LRU cache for routing decisions (1000 entries, 5 min TTL)

### LazyRAG Performance

Target: < 3 seconds for typical queries (< 50 entities)

**Tuning Parameters**:
- `LAZY_RAG_MAX_ENTITIES`: Reduce for faster queries
- `LAZY_RAG_MAX_HOPS`: Lower = faster, higher = more context
- `LAZY_RAG_USE_COMMUNITIES`: Disable if not needed

### Community Detection Performance

| Graph Size | Algorithm | Typical Time |
|------------|-----------|--------------|
| < 1,000 entities | Louvain | < 5 seconds |
| 1,000-10,000 entities | Louvain | 10-60 seconds |
| > 10,000 entities | Louvain/Leiden | Minutes |

**Limits**:
- `MAX_COMMUNITY_ENTITIES`: 100,000 (safety limit)
- `MAX_COMMUNITY_RELATIONSHIPS`: 500,000 (safety limit)
- `MAX_META_GRAPH_EDGES`: 50,000 (hierarchy building)

### Graph Reranker Performance

Target: < 300ms additional latency

**Optimizations**:
- Batch Neo4j queries (UNWIND pattern)
- Parallel execution in HybridGraphReranker
- Entity ID limits (1000 per query)

## Troubleshooting

### Common Issues

#### Community Detection Fails

**Symptom**: `GraphTooSmallError` or no communities detected

**Causes**:
1. Graph has fewer nodes than `min_community_size`
2. Entities are disconnected (no relationships)
3. Graph exceeds size limits

**Solutions**:
```bash
# Check entity count
MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e)

# Check relationship count
MATCH (e:Entity {tenant_id: $tenant_id})-[r]-() RETURN count(r)

# Reduce minimum size
COMMUNITY_MIN_SIZE=2
```

#### LazyRAG Returns Empty Results

**Symptom**: Zero entities in response

**Causes**:
1. Graphiti search finds no matches
2. Entities not linked to tenant
3. Query terms don't match entity names/descriptions

**Solutions**:
```python
# Debug: use expand_only to see raw subgraph
expansion = await retriever.expand_only(query, tenant_id)
print(f"Seeds: {expansion.seed_count}, Expanded: {expansion.expanded_count}")

# Check Graphiti search directly
result = await graphiti_client.client.search(
    query=query,
    group_ids=[tenant_id],
    num_results=10,
)
```

#### Query Routing Incorrect Classification

**Symptom**: GLOBAL queries routed as LOCAL or vice versa

**Causes**:
1. Rule-based patterns don't match query style
2. Confidence threshold too high/low

**Solutions**:
```python
# Debug: check pattern matches
decision = await router.route(query, tenant_id)
print(f"Global matches: {decision.global_matches}")
print(f"Local matches: {decision.local_matches}")
print(f"Confidence: {decision.confidence}")

# Adjust threshold or enable LLM
QUERY_ROUTING_USE_LLM=true
QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.6
```

#### Graph Reranker No Effect

**Symptom**: Results unchanged after reranking

**Causes**:
1. No entities linked in results
2. No episodes in time window
3. No graph paths between query and result entities

**Solutions**:
```python
# Debug: check graph context
for result in reranked:
    ctx = result.graph_context
    print(f"Episode mentions: {ctx.episode_mentions}")
    print(f"Min distance: {ctx.min_distance}")
    print(f"Query entities: {ctx.query_entities}")
    print(f"Result entities: {ctx.result_entities}")
```

### Logging

Enable debug logging for graph intelligence:

```python
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)
```

Key log events:
- `community_detection_started/completed`
- `lazy_rag_query_started/completed`
- `query_routed`
- `episode_rerank_complete`
- `distance_rerank_complete`
- `hybrid_rerank_complete`
- `dual_level_retrieval_started/completed`
