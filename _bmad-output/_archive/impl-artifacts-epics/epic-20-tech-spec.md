# Epic 20 Tech Spec: Advanced Retrieval Intelligence

**Date:** 2026-01-05
**Status:** Backlog
**Epic Owner:** Product and Engineering
**Origin:** Party Mode Competitive Analysis (2026-01-04) + Original Epic 19 Competitive Features

---

## Overview

Epic 20 delivers advanced retrieval intelligence features identified through competitive analysis. These features establish competitive parity and differentiation against Mem0, Zep, MS GraphRAG, LightRAG, Cognee, RAGFlow, and Qdrant.

### Strategic Context

This epic was split from the original Epic 19 to separate competitive features from quality foundation work:
- **Epic 19:** Quality Foundation & Tech Debt (26 stories) - Must complete first
- **Epic 20:** Advanced Retrieval Intelligence (this document) - 18 stories

### Split Rationale (2026-01-05)

1. **Dependencies:** Epic 19 establishes testing infrastructure, observability, and quality gates that Epic 20 features depend on
2. **Measurement:** Epic 19's benchmarks (19-C4) and Prometheus metrics (19-C5) enable measuring Epic 20 improvements
3. **Risk Reduction:** Quality foundation reduces risk when adding complex features

### Goals

- Achieve competitive parity with leading memory and graph platforms
- Implement differentiated retrieval strategies
- Enable enterprise document intelligence
- Support multi-modal and multi-language use cases

### Competitors Addressed

| Competitor | Key Features | Stories Addressing |
|------------|-------------|-------------------|
| Mem0 | Memory scopes, consolidation | 20-A1, 20-A2 |
| MS GraphRAG | Community detection, global/local queries | 20-B1, 20-B3 |
| LightRAG | Lazy summarization, dual-level retrieval | 20-B2, 20-C2 |
| Zep | Graph-based reranking | 20-C1 |
| RAGFlow | Table extraction, multimodal | 20-D1, 20-D2 |
| Cognee | Ontologies, feedback loops | 20-E1, 20-E2 |
| Qdrant | Sparse vectors, ColBERT | 20-H1, 20-H5 |

---

## Story Groups

### Group A: Memory Platform (Compete with Mem0)

*Origin: Mem0 competitive analysis*
*Focus: Hierarchical memory scopes and consolidation*

#### Story 20-A1: Implement Memory Scopes

**Priority:** HIGH
**Competitor:** Mem0

**Objective:** Enable hierarchical memory scopes for different persistence levels.

**Memory Scope Hierarchy:**
```
┌─────────────────────────────────────────────────────────────┐
│                      ORGANIZATION                            │
│  Long-term facts, policies, shared knowledge                 │
│  TTL: Permanent until deleted                                │
├─────────────────────────────────────────────────────────────┤
│                         USER                                 │
│  User preferences, history, personal facts                   │
│  TTL: Account lifetime                                       │
├─────────────────────────────────────────────────────────────┤
│                        AGENT                                 │
│  Agent-specific learned patterns, tool preferences           │
│  TTL: Configurable (default: 30 days)                        │
├─────────────────────────────────────────────────────────────┤
│                       SESSION                                │
│  Conversation context, working memory                        │
│  TTL: Session end + grace period (default: 24 hours)         │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
MEMORY_SCOPES_ENABLED=true|false  # Default: false
MEMORY_DEFAULT_SCOPE=user|session|agent  # Default: session
MEMORY_USER_SCOPE_TTL=0  # 0 = no expiry
MEMORY_AGENT_SCOPE_TTL=2592000  # 30 days in seconds
MEMORY_SESSION_SCOPE_TTL=86400  # 24 hours in seconds
```

**Data Model:**
```python
class Memory(BaseModel):
    id: UUID
    tenant_id: str
    scope: MemoryScope  # organization | user | agent | session
    scope_id: str  # org_id | user_id | agent_id | session_id
    content: str
    embedding: list[float]
    metadata: dict
    created_at: datetime
    expires_at: datetime | None
    importance: float  # 0-1, for consolidation priority
```

**API Endpoints:**
```
POST /memories
  Body: { scope, content, metadata }

GET /memories
  Query: scope, scope_id, limit, before

DELETE /memories/{id}

POST /memories/search
  Body: { query, scopes: ["user", "session"], limit }
```

**Acceptance Criteria:**
- Memory scopes persist to PostgreSQL with scope isolation
- Queries can filter by scope or search across scopes
- TTL enforcement runs on configurable schedule
- Scope inheritance: session can read user, user can read org
- Metrics track memory count per scope per tenant
- API endpoints follow RFC 7807 error format

---

#### Story 20-A2: Implement Memory Consolidation

**Priority:** MEDIUM
**Competitor:** Mem0

**Objective:** Automatically consolidate similar memories and extract lasting facts.

**Consolidation Pipeline:**
```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Raw Memories │────▶│ Cluster Similar│────▶│  Extract Facts │
│   (session)    │     │  (embeddings)  │     │    (LLM)       │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
                       ┌────────────────┐             │
                       │ Promote to     │◀────────────┘
                       │ higher scope   │
                       └────────────────┘
```

**Configuration:**
```bash
MEMORY_CONSOLIDATION_ENABLED=true|false  # Default: false
MEMORY_CONSOLIDATION_SCHEDULE=0 2 * * *  # Cron: 2 AM daily
MEMORY_CONSOLIDATION_THRESHOLD=0.85  # Similarity threshold
MEMORY_CONSOLIDATION_MIN_OCCURRENCES=3  # Minimum to consolidate
MEMORY_CONSOLIDATION_MODEL=claude-3-haiku  # Extraction model
```

**Consolidation Rules:**
```python
class ConsolidationRule(BaseModel):
    """Rules for automatic memory consolidation."""

    min_occurrences: int = 3  # Minimum similar memories
    similarity_threshold: float = 0.85  # Embedding similarity
    time_window_hours: int = 168  # 1 week
    promotion_confidence: float = 0.9  # LLM confidence to promote

    # Scope promotion rules
    session_to_user: bool = True  # Promote recurring session facts
    user_to_org: bool = False  # Requires admin approval
```

**LLM Extraction Prompt:**
```
Given these similar memories:
{memories}

Extract a single consolidated fact. Rules:
1. Keep only information mentioned in 2+ memories
2. Remove temporal context (yesterday, last week)
3. Prefer specific over general
4. Output structured format

Consolidated fact:
```

**Acceptance Criteria:**
- Similar memories are clustered using embedding similarity
- LLM extracts lasting facts from clusters
- Facts are promoted to appropriate scope based on rules
- Original memories are archived (not deleted)
- Consolidation logs track decisions for debugging
- Manual consolidation trigger available via API
- Metrics track consolidation rate and quality

---

### Group B: Graph Intelligence (Compete with MS GraphRAG)

*Origin: MS GraphRAG, LightRAG competitive analysis*
*Focus: Community detection and intelligent query routing*

#### Story 20-B1: Implement Community Detection

**Priority:** HIGH
**Competitor:** MS GraphRAG

**Objective:** Detect communities in knowledge graph for high-level summaries.

**Community Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH                           │
│                                                              │
│    ┌─────────────────┐         ┌─────────────────┐          │
│    │  Community A    │         │  Community B    │          │
│    │  (AI/ML Topic)  │         │  (DevOps Topic) │          │
│    │                 │         │                 │          │
│    │  [Entity1]─────[Entity2]──[Entity3]─────[Entity4]      │
│    │      │              │          │              │        │
│    │  [Entity5]      [Entity6]  [Entity7]      [Entity8]    │
│    └─────────────────┘         └─────────────────┘          │
│                                                              │
│              Global Summary: "This graph covers..."          │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
COMMUNITY_DETECTION_ENABLED=true|false  # Default: false
COMMUNITY_ALGORITHM=louvain|leiden|infomap  # Default: leiden
COMMUNITY_MIN_SIZE=3  # Minimum nodes per community
COMMUNITY_RESOLUTION=1.0  # Higher = more communities
COMMUNITY_REBUILD_SCHEDULE=0 3 * * *  # Cron: 3 AM daily
```

**Data Model:**
```python
class Community(BaseModel):
    id: UUID
    tenant_id: str
    name: str  # LLM-generated community name
    summary: str  # LLM-generated summary
    keywords: list[str]
    member_count: int
    centroid_embedding: list[float]
    created_at: datetime
    updated_at: datetime

class CommunityMembership(BaseModel):
    entity_id: str
    community_id: UUID
    membership_score: float  # 0-1
```

**Community Summary Generation:**
```python
async def generate_community_summary(community: Community) -> str:
    """Generate LLM summary for community."""
    members = await get_community_members(community.id)
    relationships = await get_internal_relationships(community.id)

    prompt = f"""
    Community contains {len(members)} entities:
    {[m.name for m in members[:20]]}  # Top 20

    Key relationships:
    {relationships[:30]}  # Top 30

    Generate a 2-3 sentence summary of this community's theme and key topics.
    """
    return await llm.generate(prompt)
```

**Acceptance Criteria:**
- Leiden algorithm detects communities in Neo4j graph
- Each community has LLM-generated name and summary
- Community summaries are stored in PostgreSQL
- Incremental update when graph changes (not full rebuild)
- API endpoint lists communities with summaries
- Metrics track community count and stability
- Community visualization data available for frontend

---

#### Story 20-B2: Implement LazyRAG Pattern

**Priority:** HIGH
**Competitor:** LightRAG

**Objective:** Defer graph summarization to query time for 99% indexing cost reduction.

**LazyRAG vs Traditional Flow:**
```
Traditional:
  Ingest → Summarize All Nodes → Store Summaries → Query → Retrieve

LazyRAG:
  Ingest → Store Raw Nodes → Query → Summarize Relevant Only → Response
```

**Configuration:**
```bash
LAZYRAG_ENABLED=true|false  # Default: false
LAZYRAG_CACHE_ENABLED=true|false  # Cache computed summaries
LAZYRAG_CACHE_TTL=3600  # 1 hour cache
LAZYRAG_MAX_NODES_PER_QUERY=50  # Limit nodes to summarize
LAZYRAG_SUMMARY_MODEL=claude-3-haiku  # Fast model for JIT summaries
```

**Implementation Architecture:**
```python
class LazyRAGRetriever:
    """Just-in-time summarization for graph retrieval."""

    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        # 1. Find relevant nodes (fast, no LLM)
        relevant_nodes = await self.graph_search(query, limit=top_k * 2)

        # 2. Check summary cache
        cached_summaries = await self.cache.get_many(
            [n.id for n in relevant_nodes]
        )

        # 3. Generate summaries for cache misses only
        uncached_nodes = [n for n in relevant_nodes if n.id not in cached_summaries]
        if uncached_nodes:
            new_summaries = await self.batch_summarize(uncached_nodes)
            await self.cache.set_many(new_summaries)
            cached_summaries.update(new_summaries)

        # 4. Rerank with summaries and return
        return await self.rerank(query, cached_summaries, limit=top_k)

    async def batch_summarize(self, nodes: list[Node]) -> dict[str, str]:
        """Batch summarize nodes with single LLM call."""
        # Use structured output for efficiency
        ...
```

**Cost Comparison:**
| Approach | Indexing Cost | Query Cost | Total (1K docs, 100 queries) |
|----------|---------------|------------|------------------------------|
| Traditional | $50 (summarize all) | $0.10/query | $60 |
| LazyRAG | $0 | $0.50/query (cache miss) | $5 (assuming 10% unique) |

**Acceptance Criteria:**
- Graph nodes stored without pre-computed summaries
- Summaries generated on-demand during retrieval
- Summary cache reduces redundant LLM calls
- Cache hit rate tracked via Prometheus
- Fallback to traditional mode if LazyRAG disabled
- Query latency overhead documented (target: <500ms first query)
- Indexing cost reduction validated (target: >90%)

---

#### Story 20-B3: Implement Global/Local Query Routing

**Priority:** MEDIUM
**Competitor:** MS GraphRAG

**Objective:** Route queries to appropriate retrieval strategy based on query type.

**Query Classification:**
```
┌─────────────────────────────────────────────────────────────┐
│                      QUERY ROUTER                            │
│                                                              │
│  Query: "What are the main themes?"                         │
│    → Classification: GLOBAL                                  │
│    → Strategy: Community summaries + global patterns         │
│                                                              │
│  Query: "What did John say about API design?"               │
│    → Classification: LOCAL                                   │
│    → Strategy: Entity search + relationship traversal        │
│                                                              │
│  Query: "How does our auth system relate to industry best   │
│          practices?"                                         │
│    → Classification: HYBRID                                  │
│    → Strategy: Local entities + global context               │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
QUERY_ROUTING_ENABLED=true|false  # Default: false
QUERY_ROUTING_MODEL=claude-3-haiku  # Classification model
QUERY_ROUTING_CACHE_ENABLED=true  # Cache classifications
QUERY_ROUTING_DEFAULT_STRATEGY=hybrid  # Fallback strategy
```

**Query Type Indicators:**
```python
class QueryClassifier:
    GLOBAL_INDICATORS = [
        "main themes", "overall", "summary", "big picture",
        "what are the trends", "common patterns", "key topics"
    ]

    LOCAL_INDICATORS = [
        "specific", "exactly", "who said", "when did",
        "what is the", "find the", "show me"
    ]

    async def classify(self, query: str) -> QueryType:
        # Fast heuristic first
        if any(ind in query.lower() for ind in self.GLOBAL_INDICATORS):
            return QueryType.GLOBAL
        if any(ind in query.lower() for ind in self.LOCAL_INDICATORS):
            return QueryType.LOCAL

        # LLM classification for ambiguous queries
        return await self.llm_classify(query)
```

**Retrieval Strategies:**
```python
class RetrievalStrategies:
    async def global_retrieval(self, query: str) -> list[Document]:
        """Use community summaries and high-level patterns."""
        communities = await self.get_relevant_communities(query)
        global_context = await self.synthesize_community_context(communities)
        return global_context

    async def local_retrieval(self, query: str) -> list[Document]:
        """Use entity search and relationship traversal."""
        entities = await self.entity_search(query)
        relationships = await self.expand_relationships(entities)
        return entities + relationships

    async def hybrid_retrieval(self, query: str) -> list[Document]:
        """Combine global context with local specifics."""
        global_ctx = await self.global_retrieval(query)
        local_ctx = await self.local_retrieval(query)
        return self.merge_and_dedupe(global_ctx, local_ctx)
```

**Acceptance Criteria:**
- Queries classified into global, local, or hybrid types
- Classification uses fast heuristics with LLM fallback
- Each strategy type has distinct retrieval path
- Routing decision logged for debugging
- A/B comparison shows improved relevance
- Override parameter allows forcing specific strategy

---

### Group C: Retrieval Excellence (Differentiation)

*Origin: Zep, LightRAG competitive analysis + internal optimization*
*Note: Stories C4 (Benchmarks) and C5 (Prometheus) are in Epic 19*

#### Story 20-C1: Implement Graph-Based Rerankers

**Priority:** HIGH
**Competitor:** Zep

**Objective:** Rerank results using graph structure in addition to semantic similarity.

**Reranking Signals:**
```python
class GraphRerankerSignals:
    """Graph-based signals for reranking."""

    # 1. Episode Mentions
    # Entities mentioned in more episodes are more important
    episode_mentions: int

    # 2. Node Distance
    # Entities closer to query entities are more relevant
    node_distance: float  # 0 = direct relationship, 1 = 2-hop, etc.

    # 3. Centrality
    # More connected entities are more authoritative
    pagerank_score: float

    # 4. Recency
    # Recently mentioned entities may be more relevant
    last_mention_recency: float  # 0-1, higher = more recent

    # 5. Relationship Strength
    # Edge weight between query entity and result
    relationship_strength: float
```

**Configuration:**
```bash
GRAPH_RERANKER_ENABLED=true|false  # Default: false
GRAPH_RERANKER_WEIGHTS='{"episode_mentions": 0.2, "node_distance": 0.3, "pagerank": 0.2, "recency": 0.15, "relationship": 0.15}'
```

**Combined Scoring:**
```python
class GraphReranker:
    def compute_score(
        self,
        semantic_score: float,
        graph_signals: GraphRerankerSignals,
        weights: dict[str, float]
    ) -> float:
        """Combine semantic and graph signals."""
        graph_score = (
            weights["episode_mentions"] * normalize(graph_signals.episode_mentions) +
            weights["node_distance"] * (1 - graph_signals.node_distance) +
            weights["pagerank"] * graph_signals.pagerank_score +
            weights["recency"] * graph_signals.last_mention_recency +
            weights["relationship"] * graph_signals.relationship_strength
        )

        # Blend semantic and graph scores
        # Default: 60% semantic, 40% graph
        return 0.6 * semantic_score + 0.4 * graph_score
```

**Acceptance Criteria:**
- Graph signals computed for each retrieval result
- Weights are configurable via environment
- Reranking improves MRR@10 by measurable amount
- Graph queries are optimized (single Cypher query)
- Metrics track graph vs semantic contribution
- Fallback to semantic-only if graph unavailable

---

#### Story 20-C2: Implement Dual-Level Retrieval

**Priority:** MEDIUM
**Competitor:** LightRAG

**Objective:** Retrieve at both entity (low-level) and theme (high-level) granularity.

**Dual-Level Architecture:**
```
                         ┌─────────────────┐
                         │     QUERY       │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
           ┌────────▼────────┐        ┌────────▼────────┐
           │   LOW-LEVEL     │        │   HIGH-LEVEL    │
           │   (Entities)    │        │    (Themes)     │
           └────────┬────────┘        └────────┬────────┘
                    │                          │
           ┌────────▼────────┐        ┌────────▼────────┐
           │ Specific facts  │        │ Conceptual      │
           │ Named entities  │        │ summaries       │
           │ Direct answers  │        │ Context         │
           └────────┬────────┘        └────────┬────────┘
                    │                          │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   MERGED RESPONSE   │
                    └─────────────────────┘
```

**Configuration:**
```bash
DUAL_LEVEL_RETRIEVAL_ENABLED=true|false  # Default: false
DUAL_LEVEL_LOW_K=5  # Low-level results
DUAL_LEVEL_HIGH_K=3  # High-level results
DUAL_LEVEL_MERGE_STRATEGY=interleave|low_first|high_first  # Default: interleave
```

**Implementation:**
```python
class DualLevelRetriever:
    async def retrieve(self, query: str) -> list[Document]:
        # Parallel retrieval at both levels
        low_level, high_level = await asyncio.gather(
            self.low_level_search(query),
            self.high_level_search(query)
        )

        return self.merge(low_level, high_level)

    async def low_level_search(self, query: str) -> list[Document]:
        """Search for specific entities and facts."""
        # Entity extraction from query
        query_entities = await self.extract_entities(query)

        # Direct entity lookup + semantic search
        entity_results = await self.entity_lookup(query_entities)
        semantic_results = await self.vector_search(query)

        return self.dedupe(entity_results + semantic_results)

    async def high_level_search(self, query: str) -> list[Document]:
        """Search for themes and conceptual context."""
        # Community-level search
        communities = await self.get_relevant_communities(query)

        # Theme extraction
        themes = await self.extract_themes(query)
        theme_docs = await self.theme_search(themes)

        return communities + theme_docs
```

**Acceptance Criteria:**
- Low-level retrieves specific entities and direct facts
- High-level retrieves community summaries and themes
- Merge strategy is configurable
- Both levels run in parallel for performance
- Metrics track contribution of each level
- Query response includes level annotations

---

#### Story 20-C3: Implement Parent-Child Chunk Hierarchy

**Priority:** MEDIUM
**Origin:** Small-to-big retrieval pattern

**Objective:** Retrieve small chunks for precision, expand to parent for context.

**Chunk Hierarchy:**
```
┌─────────────────────────────────────────────────────────────┐
│                     DOCUMENT                                 │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 PARENT CHUNK 1                        │   │
│  │                 (512 tokens)                          │   │
│  │                                                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Child 1  │  │ Child 2  │  │ Child 3  │            │   │
│  │  │(128 tok) │  │(128 tok) │  │(128 tok) │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 PARENT CHUNK 2                        │   │
│  │                 ...                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
PARENT_CHILD_CHUNKING_ENABLED=true|false  # Default: false
PARENT_CHUNK_SIZE=512  # tokens
CHILD_CHUNK_SIZE=128  # tokens
PARENT_CHUNK_OVERLAP=50  # tokens
CHILD_CHUNK_OVERLAP=20  # tokens
EXPAND_TO_PARENT=true|false  # Return parent when child matches
```

**Data Model:**
```python
class ChunkHierarchy(BaseModel):
    document_id: UUID
    parent_chunk_id: UUID
    child_chunk_ids: list[UUID]
    parent_content: str
    parent_embedding: list[float]

class ChildChunk(BaseModel):
    id: UUID
    parent_chunk_id: UUID
    content: str
    embedding: list[float]
    position: int  # Order within parent
```

**Retrieval Flow:**
```python
class ParentChildRetriever:
    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # 1. Search child chunks (high precision)
        child_matches = await self.child_search(query, limit=top_k * 3)

        # 2. Group by parent
        parent_groups = self.group_by_parent(child_matches)

        # 3. Score parents by child match quality
        scored_parents = self.score_parents(parent_groups)

        # 4. Return parent chunks with context
        return [
            Document(
                content=parent.content,
                metadata={
                    "matched_children": [c.content for c in children],
                    "child_scores": [c.score for c in children]
                }
            )
            for parent, children in scored_parents[:top_k]
        ]
```

**Acceptance Criteria:**
- Documents chunked into parent-child hierarchy at ingestion
- Child chunks indexed for precise matching
- Parent chunks returned for full context
- Metadata tracks which children triggered match
- Storage overhead documented (target: <30% increase)
- Retrieval quality improved vs flat chunking

---

### Group D: Document Intelligence (RAGFlow Approach)

*Origin: RAGFlow competitive analysis*
*Focus: Structure-aware document parsing*

#### Story 20-D1: Enhance Table/Layout Extraction

**Priority:** MEDIUM
**Competitor:** RAGFlow

**Objective:** Extract tables and structured layouts with semantic preservation.

**Table Extraction Pipeline:**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │────▶│  Layout      │────▶│   Table      │
│   (PDF/HTML) │     │  Detection   │     │   Parser     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                     ┌──────────────┐            │
                     │  Structured  │◀───────────┘
                     │  Output      │
                     └──────────────┘
```

**Supported Formats:**
- PDF tables (with Docling)
- HTML tables
- Markdown tables
- CSV/Excel embedded in documents

**Configuration:**
```bash
TABLE_EXTRACTION_ENABLED=true|false  # Default: false
TABLE_EXTRACTION_MODEL=docling|tabula|camelot  # Default: docling
TABLE_MARKDOWN_OUTPUT=true  # Convert to markdown for LLM
TABLE_SEMANTIC_HEADERS=true  # Infer header meanings
```

**Table Output Format:**
```python
class ExtractedTable(BaseModel):
    document_id: UUID
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    markdown: str  # Markdown representation
    caption: str | None
    semantic_summary: str  # LLM-generated summary
    bbox: tuple[float, float, float, float]  # Bounding box
```

**Semantic Processing:**
```python
async def process_table(table: ExtractedTable) -> str:
    """Generate searchable content from table."""
    prompt = f"""
    Table from document: {table.caption or 'No caption'}

    Headers: {table.headers}
    Sample rows: {table.rows[:3]}

    Generate a semantic summary that includes:
    1. What the table represents
    2. Key data points
    3. Searchable keywords

    Summary:
    """
    return await llm.generate(prompt)
```

**Acceptance Criteria:**
- PDF tables extracted with structure preserved
- HTML tables parsed correctly
- Tables stored as both structured data and markdown
- Semantic summaries enable table search
- Table chunks linked to source document
- Metrics track extraction accuracy

---

#### Story 20-D2: Implement Multimodal Ingestion

**Priority:** LOW
**Competitor:** RAGFlow

**Objective:** Ingest images and Office documents with content extraction.

**Supported Formats:**
| Format | Extraction Method | Output |
|--------|-------------------|--------|
| Images (PNG/JPG) | Vision LLM | Text description |
| Office docs (DOCX) | python-docx | Structured text |
| Presentations (PPTX) | python-pptx | Slide text + notes |
| Spreadsheets (XLSX) | openpyxl | Tables + formulas |
| Diagrams | Vision LLM | Description + entities |

**Configuration:**
```bash
MULTIMODAL_INGESTION_ENABLED=true|false  # Default: false
MULTIMODAL_VISION_MODEL=claude-3-haiku  # For image understanding
MULTIMODAL_MAX_IMAGE_SIZE=4096  # Max dimension in pixels
MULTIMODAL_EXTRACT_OCR=true  # Run OCR on images
```

**Image Processing:**
```python
class ImageProcessor:
    async def process(self, image_path: str) -> Document:
        # 1. Resize if needed
        image = await self.resize_if_needed(image_path)

        # 2. OCR for text extraction
        ocr_text = await self.run_ocr(image) if settings.MULTIMODAL_EXTRACT_OCR else ""

        # 3. Vision LLM for understanding
        description = await self.vision_describe(image)

        # 4. Entity extraction
        entities = await self.extract_entities(description)

        return Document(
            content=f"{description}\n\nText in image:\n{ocr_text}",
            metadata={
                "type": "image",
                "entities": entities,
                "dimensions": image.size
            }
        )
```

**Acceptance Criteria:**
- Images processed with vision LLM + OCR
- Office documents parsed to structured text
- Embedded images in documents are processed
- Content indexed for semantic search
- Original files preserved with references
- Processing cost tracked per format

---

### Group E: Advanced Features (Cognee-Inspired)

*Origin: Cognee competitive analysis*
*Focus: Ontologies and self-improvement*

#### Story 20-E1: Implement Ontology Support

**Priority:** MEDIUM
**Competitor:** Cognee

**Objective:** Support domain-specific ontologies for consistent entity typing.

**Ontology Structure:**
```yaml
# ontologies/software-engineering.yaml
name: Software Engineering
version: 1.0
entities:
  - name: Component
    description: A software component or module
    properties:
      - language: string
      - framework: string
    relationships:
      - imports: Component
      - depends_on: Library

  - name: API
    description: An API endpoint
    properties:
      - method: enum[GET, POST, PUT, DELETE]
      - path: string
    relationships:
      - belongs_to: Service
      - calls: API

  - name: Developer
    description: A person who writes code
    relationships:
      - maintains: Component
      - authored: Commit
```

**Configuration:**
```bash
ONTOLOGY_ENABLED=true|false  # Default: false
ONTOLOGY_PATH=ontologies/  # Directory with .yaml ontologies
ONTOLOGY_STRICT_MODE=false  # Reject entities not in ontology
ONTOLOGY_AUTO_SUGGEST=true  # Suggest ontology additions
```

**Entity Extraction with Ontology:**
```python
class OntologyEntityExtractor:
    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    async def extract(self, text: str) -> list[Entity]:
        prompt = f"""
        Using this ontology:
        {self.ontology.to_prompt()}

        Extract entities from the following text.
        Only use entity types defined in the ontology.

        Text: {text}

        Entities (JSON):
        """

        raw_entities = await llm.generate(prompt, structured=True)
        return self.validate_against_ontology(raw_entities)
```

**Acceptance Criteria:**
- Ontologies defined in YAML format
- Entity extraction respects ontology constraints
- Relationships validated against ontology
- Ontology violations logged (or rejected in strict mode)
- Multiple ontologies can be loaded simultaneously
- API endpoint to query available ontologies

---

#### Story 20-E2: Implement Self-Improving Feedback Loop

**Priority:** LOW
**Competitor:** Cognee

**Objective:** Learn from user feedback to improve retrieval quality over time.

**Feedback Loop Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Retrieve   │────▶│  Response   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐            │
                    │  Store      │◀───────────┤
                    │  Feedback   │      User rates
                    └──────┬──────┘      (thumbs up/down)
                           │
                    ┌──────▼──────┐
                    │  Analyze    │
                    │  Patterns   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Adjust    │
                    │  Weights    │
                    └─────────────┘
```

**Configuration:**
```bash
FEEDBACK_LOOP_ENABLED=true|false  # Default: false
FEEDBACK_MIN_SAMPLES=100  # Minimum feedback before adjusting
FEEDBACK_ANALYSIS_SCHEDULE=0 4 * * 0  # Weekly analysis
FEEDBACK_ADJUSTMENT_MAX=0.2  # Max weight change per cycle
```

**Feedback Data Model:**
```python
class QueryFeedback(BaseModel):
    query_id: UUID
    query_text: str
    retrieved_doc_ids: list[UUID]
    response_text: str
    rating: int  # 1-5 or -1/+1
    timestamp: datetime
    user_id: str | None
    tenant_id: str

class FeedbackAnalysis(BaseModel):
    period_start: datetime
    period_end: datetime
    total_feedback: int
    positive_rate: float
    low_rated_patterns: list[str]  # Query patterns with low ratings
    high_rated_patterns: list[str]  # Query patterns with high ratings
    recommended_adjustments: dict[str, float]  # Config key -> adjustment
```

**Improvement Actions:**
```python
class FeedbackAnalyzer:
    async def analyze_and_improve(self):
        feedback = await self.get_recent_feedback()

        # 1. Identify poorly performing query types
        low_performing = self.identify_patterns(
            feedback,
            rating_threshold=2.5
        )

        # 2. Analyze retrieval strategy used
        for pattern in low_performing:
            strategy_analysis = await self.analyze_strategy(pattern)
            if strategy_analysis.suggests_change:
                await self.queue_adjustment(strategy_analysis)

        # 3. Tune parameters based on successful queries
        high_performing = self.identify_patterns(
            feedback,
            rating_threshold=4.0
        )
        await self.learn_from_success(high_performing)
```

**Acceptance Criteria:**
- Feedback collected on query responses
- Analysis runs on schedule with sufficient data
- Weight adjustments are bounded and gradual
- Improvement metrics tracked over time
- Manual override available for adjustments
- Privacy: feedback is anonymized for analysis

---

### Group H: Additional Competitive Features

*Origin: Qdrant, RAGFlow, general competitive analysis*
*Focus: Advanced search and integration capabilities*

#### Story 20-H1: Implement Sparse Vector Search (BM42)

**Priority:** MEDIUM
**Competitor:** Qdrant

**Objective:** Add sparse vector search for keyword-matching precision.

**Hybrid Vector Architecture:**
```
                    ┌──────────────────┐
                    │      QUERY       │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐        ┌─────────▼─────────┐
    │   DENSE VECTOR    │        │  SPARSE VECTOR    │
    │   (Embeddings)    │        │   (BM25/BM42)     │
    │                   │        │                   │
    │  Semantic match   │        │  Keyword match    │
    └─────────┬─────────┘        └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  RRF/Weighted   │
                    │     Merge       │
                    └─────────────────┘
```

**Configuration:**
```bash
SPARSE_VECTOR_ENABLED=true|false  # Default: false
SPARSE_VECTOR_MODEL=bm42|bm25|splade  # Default: bm42
SPARSE_DENSE_WEIGHT=0.5  # 0=sparse only, 1=dense only
SPARSE_INDEX_PATH=./sparse_index/  # Sparse index storage
```

**Implementation:**
```python
class SparseVectorSearch:
    """BM42 sparse vector search implementation."""

    def __init__(self, model: str = "bm42"):
        if model == "bm42":
            # BM42 = BM25 with learned importance weights
            self.encoder = BM42Encoder()
        elif model == "splade":
            self.encoder = SpladeEncoder()
        else:
            self.encoder = BM25Encoder()

    async def encode(self, text: str) -> SparseVector:
        """Encode text to sparse vector."""
        return self.encoder.encode(text)

    async def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search using sparse vectors."""
        query_vector = await self.encode(query)
        return self.index.search(query_vector, limit=top_k)
```

**Acceptance Criteria:**
- Sparse vectors computed during ingestion
- BM42 or SPLADE model supported
- Hybrid search combines sparse + dense scores
- Index stored efficiently (sparse representation)
- Search latency comparable to dense-only
- Keyword queries show improved precision

---

#### Story 20-H2: Implement Cross-Language Query

**Priority:** LOW
**Competitor:** RAGFlow

**Objective:** Query in one language, retrieve documents in another.

**Cross-Language Flow:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query (EN)     │────▶│  Multilingual   │────▶│  Retrieve ALL   │
│  "AI regulations"│    │  Embedding      │     │   languages     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                       │
                        ┌─────────────────┐            │
                        │  Translate      │◀───────────┘
                        │  Results (opt)  │  French, German,
                        └─────────────────┘  Spanish docs
```

**Configuration:**
```bash
CROSS_LANGUAGE_ENABLED=true|false  # Default: false
CROSS_LANGUAGE_MODEL=multilingual-e5|mxbai-embed  # Embedding model
CROSS_LANGUAGE_TRANSLATE_RESULTS=true  # Translate to query language
CROSS_LANGUAGE_DETECT_QUERY=true  # Auto-detect query language
```

**Implementation:**
```python
class CrossLanguageRetriever:
    def __init__(self):
        # Use multilingual embedding model
        self.embedder = MultilingualEmbedder("multilingual-e5-large")
        self.translator = Translator()

    async def retrieve(
        self,
        query: str,
        translate_results: bool = True
    ) -> list[Document]:
        # 1. Detect query language
        query_lang = await self.detect_language(query)

        # 2. Embed with multilingual model (works for any language)
        query_embedding = await self.embedder.embed(query)

        # 3. Search across all documents (any language)
        results = await self.vector_search(query_embedding)

        # 4. Optionally translate results
        if translate_results:
            results = await self.translate_to(results, query_lang)

        return results
```

**Acceptance Criteria:**
- Multilingual embedding model deployed
- Documents in any language retrievable
- Query language auto-detected
- Results optionally translated
- Language metadata tracked on documents
- Retrieval quality validated across 5+ languages

---

#### Story 20-H3: Implement External Data Source Sync

**Priority:** MEDIUM
**Competitor:** RAGFlow

**Objective:** Automatically sync knowledge from external data sources.

**Supported Sources:**
| Source | Sync Method | Auth |
|--------|-------------|------|
| Confluence | REST API | OAuth 2.0 |
| Notion | API | Integration token |
| Google Drive | Drive API | Service account |
| S3/R2 | S3 API | Access keys |
| Discord | Bot | Bot token |
| Slack | Bolt | OAuth 2.0 |

**Configuration:**
```bash
# Example: Confluence sync
DATA_SOURCE_CONFLUENCE_ENABLED=true
DATA_SOURCE_CONFLUENCE_URL=https://company.atlassian.net
DATA_SOURCE_CONFLUENCE_SPACE_KEYS=DEV,DOCS
DATA_SOURCE_CONFLUENCE_SYNC_SCHEDULE=0 */6 * * *  # Every 6 hours

# Example: S3 sync
DATA_SOURCE_S3_ENABLED=true
DATA_SOURCE_S3_BUCKET=company-docs
DATA_SOURCE_S3_PREFIX=knowledge/
DATA_SOURCE_S3_SYNC_SCHEDULE=0 */4 * * *  # Every 4 hours
```

**Sync Architecture:**
```python
class DataSourceSync:
    async def sync_source(self, source: DataSource):
        # 1. Get last sync checkpoint
        checkpoint = await self.get_checkpoint(source.id)

        # 2. Fetch changes since checkpoint
        changes = await source.get_changes_since(checkpoint)

        # 3. Process changes
        for change in changes:
            if change.type == "created" or change.type == "updated":
                await self.ingest_document(change.document)
            elif change.type == "deleted":
                await self.delete_document(change.document_id)

        # 4. Update checkpoint
        await self.set_checkpoint(source.id, datetime.utcnow())
```

**Acceptance Criteria:**
- At least 3 data sources implemented (Confluence, S3, Notion)
- Incremental sync (only changes, not full re-index)
- Sync status visible via API
- Conflict resolution documented
- Rate limiting respected for external APIs
- Authentication credentials securely stored

---

#### Story 20-H4: Implement Voice I/O

**Priority:** LOW
**Competitor:** General AI assistants

**Objective:** Enable voice input for queries and voice output for responses.

**Voice Pipeline:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Audio Input    │────▶│  Speech-to-     │────▶│  RAG Query      │
│  (microphone)   │     │  Text (Whisper) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                       │
┌─────────────────┐     ┌─────────────────┐            │
│  Audio Output   │◀────│  Text-to-       │◀───────────┘
│  (speaker)      │     │  Speech (TTS)   │  Response text
└─────────────────┘     └─────────────────┘
```

**Configuration:**
```bash
VOICE_IO_ENABLED=true|false  # Default: false
VOICE_STT_MODEL=whisper-large-v3|whisper-small  # Default: whisper-small
VOICE_TTS_MODEL=eleven-labs|openai-tts|local  # Default: openai-tts
VOICE_TTS_VOICE=alloy|echo|fable  # For OpenAI TTS
VOICE_STREAM_RESPONSE=true  # Stream TTS output
```

**API Endpoints:**
```
POST /voice/query
  Content-Type: audio/wav
  Body: <audio data>
  Response: { query_text, response_text, audio_url }

WebSocket /voice/stream
  Bidirectional audio streaming
```

**Acceptance Criteria:**
- Whisper STT converts audio to text
- TTS converts response to audio
- WebSocket enables real-time conversation
- Latency: <2s for short queries
- Audio formats: WAV, MP3, WebM
- Frontend audio component provided

---

#### Story 20-H5: Implement ColBERT Reranking

**Priority:** LOW
**Competitor:** Qdrant

**Objective:** Add ColBERT late-interaction reranking for improved precision.

**ColBERT Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    ColBERT Reranking                         │
│                                                              │
│  Query: "best practices for API design"                     │
│         ↓                                                    │
│  Query Tokens: [best] [practices] [for] [API] [design]      │
│         ↓                                                    │
│  Token Embeddings: [e1] [e2] [e3] [e4] [e5]                 │
│                                                              │
│  For each document:                                          │
│    Doc Tokens: [REST] [API] [design] [patterns] [guide]     │
│    Doc Embeddings: [d1] [d2] [d3] [d4] [d5]                 │
│                                                              │
│    MaxSim: max(sim(e1,d1..d5)) + max(sim(e2,d1..d5)) + ...  │
│                                                              │
│  Score: Sum of MaxSim for all query tokens                   │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
COLBERT_RERANKER_ENABLED=true|false  # Default: false
COLBERT_MODEL=colbertv2.0|answerai-colbert  # Default: colbertv2.0
COLBERT_MAX_QUERY_LENGTH=128  # tokens
COLBERT_MAX_DOC_LENGTH=512  # tokens
COLBERT_TOP_K=10  # Candidates to rerank
```

**Implementation:**
```python
class ColBERTReranker:
    def __init__(self, model: str = "colbertv2.0"):
        self.model = ColBERT.load(model)

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 10
    ) -> list[Document]:
        # Encode query tokens
        query_embeddings = self.model.encode_query(query)

        # Score each document using MaxSim
        scored = []
        for doc in documents:
            doc_embeddings = self.model.encode_document(doc.content)
            score = self.max_sim(query_embeddings, doc_embeddings)
            scored.append((doc, score))

        # Sort by score and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_k]]

    def max_sim(self, q_emb: Tensor, d_emb: Tensor) -> float:
        """Compute MaxSim score."""
        # For each query token, find max similarity with any doc token
        sim_matrix = torch.matmul(q_emb, d_emb.T)
        max_sims = sim_matrix.max(dim=1).values
        return max_sims.sum().item()
```

**Acceptance Criteria:**
- ColBERT model loaded and functional
- MaxSim scoring implemented correctly
- Reranking improves MRR@10 vs cross-encoder baseline
- GPU acceleration supported
- Latency: <200ms for top-10 reranking
- Memory usage documented

---

#### Story 20-H6: Implement Visual Workflow Editor

**Priority:** LOW
**Competitor:** RAGFlow

**Objective:** Visual editor for designing RAG pipelines.

**Workflow Editor Components:**
```
┌─────────────────────────────────────────────────────────────┐
│  VISUAL WORKFLOW EDITOR                                      │
│                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │ INGEST  │────▶│ CHUNK   │────▶│ EMBED   │               │
│  └─────────┘     └─────────┘     └─────────┘               │
│       │                               │                      │
│       │              ┌────────────────┘                      │
│       ▼              ▼                                       │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │ EXTRACT │────▶│ INDEX   │────▶│ RETRIEVE│               │
│  │ ENTITIES│     │         │     │         │               │
│  └─────────┘     └─────────┘     └─────────┘               │
│                                       │                      │
│                                       ▼                      │
│                                  ┌─────────┐                │
│                                  │ RERANK  │                │
│                                  └─────────┘                │
│                                       │                      │
│                                       ▼                      │
│                                  ┌─────────┐                │
│                                  │ RESPOND │                │
│                                  └─────────┘                │
│                                                              │
│  [Save] [Load] [Run] [Debug]                                 │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
WORKFLOW_EDITOR_ENABLED=true|false  # Default: false
WORKFLOW_STORAGE_PATH=workflows/  # Workflow definitions
WORKFLOW_AUTO_SAVE=true  # Auto-save changes
```

**Workflow Definition (JSON):**
```json
{
  "name": "Custom RAG Pipeline",
  "version": "1.0",
  "nodes": [
    {
      "id": "ingest",
      "type": "ingestion",
      "config": { "source": "url" },
      "outputs": ["content"]
    },
    {
      "id": "chunk",
      "type": "chunking",
      "config": { "strategy": "parent_child", "size": 512 },
      "inputs": ["ingest.content"],
      "outputs": ["chunks"]
    },
    {
      "id": "embed",
      "type": "embedding",
      "config": { "model": "text-embedding-3-small" },
      "inputs": ["chunk.chunks"],
      "outputs": ["embeddings"]
    }
  ],
  "edges": [
    { "from": "ingest", "to": "chunk" },
    { "from": "chunk", "to": "embed" }
  ]
}
```

**Acceptance Criteria:**
- React-based visual editor component
- Drag-and-drop node placement
- Node configuration panels
- Workflow validation before save
- Import/export workflow JSON
- Debug mode shows data flow
- Integration with CopilotKit UI

---

## Technical Notes

### Dependencies

- **Epic 19 (Required):** Quality Foundation must complete first
  - 19-C4: Benchmarks needed to measure improvements
  - 19-C5: Prometheus metrics for monitoring
  - 19-F1, 19-F2: Integration tests as safety net

### Implementation Order

**Phase 1 (Core Intelligence):**
1. 20-B1: Community detection (enables 20-B3)
2. 20-B2: LazyRAG pattern
3. 20-C1: Graph-based rerankers
4. 20-A1: Memory scopes

**Phase 2 (Retrieval Enhancement):**
5. 20-B3: Global/local query routing
6. 20-C2: Dual-level retrieval
7. 20-C3: Parent-child chunking
8. 20-A2: Memory consolidation

**Phase 3 (Document Intelligence):**
9. 20-D1: Table extraction
10. 20-E1: Ontology support

**Phase 4 (Advanced Features):**
11. 20-H1: Sparse vectors
12. 20-H3: External data sync
13. 20-H5: ColBERT reranking

**Phase 5 (Nice-to-Have):**
14. 20-D2: Multimodal ingestion
15. 20-E2: Feedback loop
16. 20-H2: Cross-language
17. 20-H4: Voice I/O
18. 20-H6: Visual workflow editor

---

## Risks

- Complex features may introduce performance regressions
- Some features require additional infrastructure (GPU, external APIs)
- Competitive features may be duplicating existing capabilities

**Mitigation:**
- All features behind feature flags
- Performance benchmarks before/after each feature
- A/B testing framework for measuring improvement
- Clear dependency on Epic 19 quality foundation

---

## Success Metrics

- Retrieval quality: MRR@10 improvement >15% vs baseline
- Memory platform: 80% user satisfaction with memory features
- Community detection: <5 minute rebuild time for 100K nodes
- External sync: 99% sync reliability for connected sources
- Feature adoption: >50% of tenants enable 2+ Epic 20 features

---

## References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Quality Foundation)
- `docs/guides/advanced-retrieval-configuration.md`
- `docs/roadmap-decisions-2026-01-03.md` (Party Mode analysis)
- [Mem0 Documentation](https://docs.mem0.ai)
- [MS GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [LightRAG Paper](https://arxiv.org/abs/2406.12456)
- [ColBERT v2 Paper](https://arxiv.org/abs/2112.01488)
