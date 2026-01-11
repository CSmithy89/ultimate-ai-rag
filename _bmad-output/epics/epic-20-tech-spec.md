# Epic 20 Tech Spec: Advanced Retrieval Intelligence

**Date:** 2026-01-05
**Status:** Complete
**Epic Owner:** Product and Platform Engineering
**Related Documents:**
- `_bmad-output/architecture.md`
- `_bmad-output/project-context.md`
- `_bmad-output/epics/epic-14-tech-spec.md` (MCP Wrapper reference)
- `_bmad-output/epics/epic-15-tech-spec.md` (Codebase Intelligence reference)

---

## Executive Summary

Epic 20 delivers advanced retrieval intelligence features that position this platform competitively against leading RAG solutions. This epic was split from Epic 19 (Quality Foundation) to separate competitive features from quality/tech-debt work.

### Competitive Positioning

| Competitor | Key Feature | Our Response |
|-----------|-------------|--------------|
| **Mem0** | Memory scopes (user/session/agent) | Group A: Memory Platform |
| **Zep** | Graph-based rerankers, temporal edges | Group B: Graph Intelligence, Group C: Rerankers |
| **MS GraphRAG** | Community detection, global/local queries | Group B: Community Detection, Query Routing |
| **LightRAG** | Dual-level retrieval (entities + themes) | Group C: Dual-Level Retrieval |
| **Cognee** | Ontology support, feedback loops | Group E: Advanced Features |
| **RAGFlow** | Table extraction, multimodal | Group D: Document Intelligence |
| **Qdrant** | BM42 sparse vectors, ColBERT | Group H: Competitive Features |

### Strategic Context

**Key Decision (2026-01-05):** Epic 19 was split to create Epic 20 for competitive features.

**Rationale:**
- Epic 19 focuses on quality foundation and tech debt resolution (must complete first)
- Epic 20 contains competitive features that differentiate against Mem0, Zep, MS GraphRAG, etc.
- Clear separation allows independent prioritization of quality vs. features
- All Epic 20 features are **OPT-IN** via configuration flags

**Dependencies:**
- **Epic 19 MUST complete before Epic 20 begins** (quality foundation required)
- Epic 12 (Advanced Retrieval) provides base reranking/grading infrastructure
- Epic 5 (Graphiti) provides temporal graph capabilities
- Epic 14 (MCP/A2A) provides protocol infrastructure for external integrations

---

## Technical Architecture

### High-Level Architecture

```
+---------------------------------------------------------------------------------+
|                    ADVANCED RETRIEVAL INTELLIGENCE (Epic 20)                      |
+---------------------------------------------------------------------------------+
|                                                                                   |
|  +---------------------------+  +---------------------------+  +---------------+ |
|  |    MEMORY PLATFORM        |  |    GRAPH INTELLIGENCE     |  |   DOCUMENT    | |
|  |    (Group A)              |  |    (Group B)              |  |   INTEL       | |
|  +---------------------------+  +---------------------------+  |   (Group D)   | |
|  |                           |  |                           |  +---------------+ |
|  |  +---------------------+  |  |  +---------------------+  |  | Table Extract | |
|  |  | Memory Scopes       |  |  |  | Community Detection |  |  | Multimodal    | |
|  |  | - user scope        |  |  |  | (Louvain/Leiden)    |  |  | Office docs   | |
|  |  | - session scope     |  |  |  +---------------------+  |  +---------------+ |
|  |  | - agent scope       |  |  |                           |                    |
|  |  +---------------------+  |  |  +---------------------+  |  +---------------+ |
|  |                           |  |  | LazyRAG Pattern     |  |  |   ADVANCED    | |
|  |  +---------------------+  |  |  | (Query-time summ.)  |  |  |   FEATURES    | |
|  |  | Memory Consolidation|  |  |  +---------------------+  |  |   (Group E)   | |
|  |  | - dedup & merge     |  |  |                           |  +---------------+ |
|  |  | - importance decay  |  |  |  +---------------------+  |  | Ontology      | |
|  |  +---------------------+  |  |  | Global/Local Router |  |  | Feedback Loop | |
|  |                           |  |  | (Query classifier)  |  |  +---------------+ |
|  +---------------------------+  |  +---------------------+  |                    |
|                                 +---------------------------+                    |
|                                                                                   |
|  +---------------------------------------+  +----------------------------------+ |
|  |    RETRIEVAL EXCELLENCE (Group C)     |  |    COMPETITIVE FEATURES (H)      | |
|  +---------------------------------------+  +----------------------------------+ |
|  |                                       |  |                                  | |
|  |  +-------------------------------+    |  |  +----------------------------+  | |
|  |  | Graph-Based Rerankers         |    |  |  | BM42 Sparse Vectors        |  | |
|  |  | - episode-mentions            |    |  |  | Cross-Language Query       |  | |
|  |  | - node-distance               |    |  |  | External Data Sync         |  | |
|  |  | - hybrid scoring              |    |  |  | Voice I/O                  |  | |
|  |  +-------------------------------+    |  |  | ColBERT Reranking          |  | |
|  |                                       |  |  | Visual Workflow Editor     |  | |
|  |  +-------------------------------+    |  |  +----------------------------+  | |
|  |  | Dual-Level Retrieval          |    |  |                                  | |
|  |  | - low-level (entities)        |    |  +----------------------------------+ |
|  |  | - high-level (themes)         |    |                                       |
|  |  +-------------------------------+    |                                       |
|  |                                       |                                       |
|  |  +-------------------------------+    |                                       |
|  |  | Parent-Child Chunks           |    |                                       |
|  |  | - small-to-big retrieval      |    |                                       |
|  |  +-------------------------------+    |                                       |
|  +---------------------------------------+                                       |
|                                                                                   |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                         EXISTING INFRASTRUCTURE                                   |
+---------------------------------------------------------------------------------+
|  Graphiti (Epic 5) | Reranking (Epic 12) | MCP/A2A (Epic 14) | pgvector (Epic 3) |
+---------------------------------------------------------------------------------+
```

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Community Detection | Neo4j GDS / networkx | Latest | Louvain/Leiden algorithms |
| Sparse Vectors | fastembed / Qdrant | Latest | BM42 implementation |
| ColBERT | colbert-ai | Latest | Late interaction reranking |
| Voice I/O | Whisper (OpenAI) / TTS | Latest | Speech-to-text, text-to-speech |
| Ontology | owlready2 | Latest | OWL ontology parsing |
| Table Extraction | Docling (existing) | 2.66.0 | Enhanced table parsing |
| External Sync | aioboto3, notion-client | Latest | S3, Notion, etc. |

### New Dependencies

```toml
# backend/pyproject.toml additions for Epic 20
dependencies = [
  # ... existing deps ...

  # Epic 20 - Group A: Memory Platform
  # (uses existing Graphiti + Redis)

  # Epic 20 - Group B: Graph Intelligence
  "networkx>=3.0",                    # Community detection algorithms
  # Neo4j GDS via existing driver

  # Epic 20 - Group C: Retrieval Excellence
  # (extends existing reranking infrastructure)

  # Epic 20 - Group D: Document Intelligence
  # (extends existing Docling)
  "openpyxl>=3.1.0",                  # Excel parsing
  "python-pptx>=0.6.0",               # PowerPoint parsing

  # Epic 20 - Group E: Advanced Features
  "owlready2>=0.45",                  # OWL ontology support

  # Epic 20 - Group H: Competitive Features (required)
  "fastembed>=0.3.0",                 # BM42 sparse vectors
]

[project.optional-dependencies]
# Optional features - install with: pip install .[voice,colbert,sync]
voice = ["openai-whisper>=20231117", "pyttsx3>=2.90"]
colbert = ["colbert-ai>=0.2.0"]
sync = ["notion-client>=2.0.0", "aioboto3>=12.0.0", "google-api-python-client>=2.0.0"]
```

---

## Group A: Memory Platform (Compete with Mem0)

### Story 20-A1: Implement Memory Scopes

#### Objective

Add hierarchical memory scopes (user, session, agent) that allow memories to be isolated and managed at different levels, similar to Mem0's memory management approach.

#### Why This Matters

- **Mem0 Parity:** Mem0's key differentiator is flexible memory scopes
- **Use Cases:** Personal assistants need user-level memory; chatbots need session memory; agents need operational memory
- **Multi-tenant:** Different scope levels enable fine-grained tenant isolation

#### Technical Design

```
backend/src/agentic_rag_backend/
+-- memory/                              # NEW: Memory platform module
|   +-- __init__.py
|   +-- scopes.py                        # MemoryScope enum and management
|   +-- store.py                         # ScopedMemoryStore class
|   +-- models.py                        # Pydantic models for memories
|   +-- consolidation.py                 # Memory consolidation (Story 20-A2)
```

#### Core Classes

```python
# backend/src/agentic_rag_backend/memory/scopes.py
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
import structlog

logger = structlog.get_logger(__name__)


class MemoryScope(str, Enum):
    """Hierarchical memory scopes."""
    USER = "user"         # Persists across all sessions for a user
    SESSION = "session"   # Persists within a single conversation session
    AGENT = "agent"       # Persists across agent invocations (operational memory)
    GLOBAL = "global"     # Tenant-wide shared memory


@dataclass
class ScopedMemory:
    """A memory entry with scope context."""
    id: str
    content: str
    scope: MemoryScope
    tenant_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    importance: float = 1.0  # 0.0-1.0, used for consolidation
    metadata: dict[str, Any] = None
    created_at: datetime = None
    accessed_at: datetime = None
    access_count: int = 0
    embedding: Optional[list[float]] = None


class ScopedMemoryStore:
    """Store and retrieve memories with scope-aware queries."""

    def __init__(
        self,
        graphiti_client,
        redis_client,
        embedding_provider: str = "openai",
    ):
        self._graphiti = graphiti_client
        self._redis = redis_client
        self._embedding_provider = embedding_provider

    async def add_memory(
        self,
        content: str,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        importance: float = 1.0,
        metadata: dict = None,
    ) -> ScopedMemory:
        """Add a memory with specified scope.

        Args:
            content: The memory content
            scope: Memory scope level
            tenant_id: Tenant identifier (always required)
            user_id: User identifier (required for USER scope)
            session_id: Session identifier (required for SESSION scope)
            agent_id: Agent identifier (required for AGENT scope)
            importance: Importance score for consolidation
            metadata: Additional metadata

        Returns:
            The created ScopedMemory
        """
        # Validate scope requirements
        self._validate_scope_context(scope, user_id, session_id, agent_id)

        # Generate embedding
        embedding = await self._generate_embedding(content)

        # Create memory
        memory = ScopedMemory(
            id=str(uuid4()),
            content=content,
            scope=scope,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            importance=importance,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
            access_count=0,
            embedding=embedding,
        )

        # Store in Graphiti as episode with scope metadata
        await self._store_in_graphiti(memory)

        # Cache hot path in Redis
        await self._cache_memory(memory)

        logger.info(
            "memory_added",
            memory_id=memory.id,
            scope=scope.value,
            tenant_id=tenant_id,
        )

        return memory

    async def search_memories(
        self,
        query: str,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10,
        include_parent_scopes: bool = True,
    ) -> list[ScopedMemory]:
        """Search memories within scope hierarchy.

        If include_parent_scopes is True, searches up the hierarchy:
        - SESSION scope includes USER and GLOBAL memories
        - USER scope includes GLOBAL memories
        - AGENT scope includes GLOBAL memories
        """
        scopes_to_search = [scope]

        if include_parent_scopes:
            if scope == MemoryScope.SESSION:
                scopes_to_search.extend([MemoryScope.USER, MemoryScope.GLOBAL])
            elif scope == MemoryScope.USER:
                scopes_to_search.append(MemoryScope.GLOBAL)
            elif scope == MemoryScope.AGENT:
                scopes_to_search.append(MemoryScope.GLOBAL)

        # Search Graphiti with scope filters
        results = await self._search_graphiti(
            query=query,
            scopes=scopes_to_search,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
        )

        # Update access counts
        for memory in results:
            await self._update_access_stats(memory.id)

        return results

    async def delete_memories_by_scope(
        self,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Delete all memories in a scope.

        Returns count of deleted memories.
        """
        # Build filter based on scope
        filters = {"tenant_id": tenant_id, "scope": scope.value}

        if scope == MemoryScope.USER:
            filters["user_id"] = user_id
        elif scope == MemoryScope.SESSION:
            filters["session_id"] = session_id
        elif scope == MemoryScope.AGENT:
            filters["agent_id"] = agent_id

        count = await self._delete_from_graphiti(filters)
        await self._invalidate_cache(filters)

        logger.info(
            "memories_deleted",
            scope=scope.value,
            count=count,
            tenant_id=tenant_id,
        )

        return count

    def _validate_scope_context(
        self,
        scope: MemoryScope,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> None:
        """Validate that required context is provided for scope."""
        if scope == MemoryScope.USER and not user_id:
            raise ValueError("user_id required for USER scope")
        if scope == MemoryScope.SESSION and not session_id:
            raise ValueError("session_id required for SESSION scope")
        if scope == MemoryScope.AGENT and not agent_id:
            raise ValueError("agent_id required for AGENT scope")
```

#### Database Schema

```sql
-- PostgreSQL: Memory metadata and embeddings
CREATE TABLE scoped_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('user', 'session', 'agent', 'global')),
    user_id UUID,
    session_id UUID,
    agent_id TEXT,
    content TEXT NOT NULL,
    importance FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,

    CONSTRAINT fk_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for scope-based queries
CREATE INDEX idx_memories_tenant_scope ON scoped_memories(tenant_id, scope);
CREATE INDEX idx_memories_user ON scoped_memories(tenant_id, user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_memories_session ON scoped_memories(tenant_id, session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_memories_agent ON scoped_memories(tenant_id, agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_memories_embedding ON scoped_memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

```cypher
// Neo4j: Memory nodes with scope relationships
(:Memory {
    id: String,
    content: String,
    scope: String,
    tenantId: String,
    userId: String,
    sessionId: String,
    agentId: String,
    importance: Float,
    createdAt: DateTime,
    accessedAt: DateTime,
    accessCount: Integer
})

// Scope hierarchy relationships
(:User)-[:HAS_MEMORY]->(:Memory)
(:Session)-[:HAS_MEMORY]->(:Memory)
(:Agent)-[:HAS_MEMORY]->(:Memory)
(:Tenant)-[:HAS_GLOBAL_MEMORY]->(:Memory)
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memories` | POST | Add a scoped memory |
| `/api/v1/memories/search` | POST | Search memories within scope |
| `/api/v1/memories/scope/{scope}` | DELETE | Delete memories by scope |
| `/api/v1/memories/{id}` | GET | Get memory by ID |
| `/api/v1/memories/{id}` | DELETE | Delete specific memory |

#### Configuration

```bash
# Epic 20 - Memory Platform
MEMORY_SCOPES_ENABLED=true|false           # Default: false
MEMORY_DEFAULT_SCOPE=user|session|agent    # Default: session
MEMORY_INCLUDE_PARENT_SCOPES=true|false    # Default: true
MEMORY_CACHE_TTL_SECONDS=3600              # Hot cache TTL
MEMORY_MAX_PER_SCOPE=10000                 # Max memories per scope
```

#### Acceptance Criteria

- [ ] Given a memory with USER scope, when searched from SESSION scope with parent inclusion, then it is found
- [ ] Given a memory with SESSION scope, when the session ends, then it can be deleted via scope-based deletion
- [ ] Given multiple scopes, when memories are added, then they are isolated by scope context
- [ ] All memory operations enforce tenant isolation via `tenant_id` filtering
- [ ] Memory search latency < 100ms for typical queries

---

### Story 20-A2: Implement Memory Consolidation

#### Objective

Implement memory consolidation that deduplicates, merges, and applies importance decay to memories over time, preventing unbounded memory growth.

#### Technical Design

```python
# backend/src/agentic_rag_backend/memory/consolidation.py
from datetime import datetime, timezone, timedelta
from typing import Optional
import structlog

from .models import ScopedMemory
from .store import ScopedMemoryStore

logger = structlog.get_logger(__name__)


class MemoryConsolidator:
    """Consolidate and manage memory lifecycle."""

    def __init__(
        self,
        store: ScopedMemoryStore,
        similarity_threshold: float = 0.9,
        decay_half_life_days: int = 30,
        min_importance: float = 0.1,
        consolidation_batch_size: int = 100,
    ):
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.decay_half_life_days = decay_half_life_days
        self.min_importance = min_importance
        self.batch_size = consolidation_batch_size

    async def consolidate_scope(
        self,
        tenant_id: str,
        scope: MemoryScope,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> ConsolidationResult:
        """Consolidate memories in a scope.

        Consolidation steps:
        1. Apply importance decay based on time and access
        2. Identify duplicate/similar memories
        3. Merge similar memories into consolidated entries
        4. Remove memories below importance threshold
        """
        start_time = time.perf_counter()

        # Get all memories in scope
        memories = await self._get_scope_memories(
            tenant_id, scope, user_id, session_id, agent_id
        )

        if not memories:
            return ConsolidationResult(
                memories_processed=0,
                duplicates_merged=0,
                memories_decayed=0,
                memories_removed=0,
            )

        # Step 1: Apply importance decay
        decayed_count = await self._apply_importance_decay(memories)

        # Step 2: Find and merge duplicates
        merged_count = await self._merge_similar_memories(memories)

        # Step 3: Remove low-importance memories
        removed_count = await self._remove_low_importance(memories)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = ConsolidationResult(
            memories_processed=len(memories),
            duplicates_merged=merged_count,
            memories_decayed=decayed_count,
            memories_removed=removed_count,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "memory_consolidation_complete",
            tenant_id=tenant_id,
            scope=scope.value,
            **result.__dict__,
        )

        return result

    async def _apply_importance_decay(
        self,
        memories: list[ScopedMemory],
    ) -> int:
        """Apply time-based importance decay.

        Uses exponential decay: importance *= 2^(-days / half_life)
        Boosted by access frequency.
        """
        decayed_count = 0
        now = datetime.now(timezone.utc)

        for memory in memories:
            days_since_access = (now - memory.accessed_at).days

            # Exponential decay
            decay_factor = 2 ** (-days_since_access / self.decay_half_life_days)

            # Access frequency boost (more accesses = slower decay)
            access_boost = min(1.0, 0.5 + (memory.access_count * 0.1))

            new_importance = memory.importance * decay_factor * access_boost

            if new_importance != memory.importance:
                memory.importance = max(self.min_importance, new_importance)
                await self.store.update_memory(memory.id, importance=memory.importance)
                decayed_count += 1

        return decayed_count

    async def _merge_similar_memories(
        self,
        memories: list[ScopedMemory],
    ) -> int:
        """Find and merge similar memories.

        Uses embedding similarity to find duplicates.
        """
        merged_count = 0
        processed_ids = set()

        for i, memory in enumerate(memories):
            if memory.id in processed_ids:
                continue

            # Find similar memories
            similar = []
            for j, other in enumerate(memories[i+1:], i+1):
                if other.id in processed_ids:
                    continue

                similarity = self._cosine_similarity(
                    memory.embedding,
                    other.embedding,
                )

                if similarity >= self.similarity_threshold:
                    similar.append(other)

            if similar:
                # Merge into primary memory
                merged = await self._merge_memories(memory, similar)
                await self.store.update_memory(memory.id, **merged)

                # Mark merged memories for deletion
                for s in similar:
                    await self.store.delete_memory(s.id)
                    processed_ids.add(s.id)

                merged_count += len(similar)

        return merged_count

    async def _merge_memories(
        self,
        primary: ScopedMemory,
        similar: list[ScopedMemory],
    ) -> dict:
        """Merge similar memories into primary.

        Returns updated fields for primary memory.
        """
        # Combine importance (max of all)
        all_importance = [primary.importance] + [s.importance for s in similar]
        combined_importance = max(all_importance)

        # Combine access counts
        combined_access = primary.access_count + sum(s.access_count for s in similar)

        # Use LLM to synthesize merged content (optional)
        # For now, keep primary content and note merged count
        merged_metadata = primary.metadata.copy()
        merged_metadata["merged_count"] = len(similar)
        merged_metadata["merged_at"] = datetime.now(timezone.utc).isoformat()

        return {
            "importance": combined_importance,
            "access_count": combined_access,
            "metadata": merged_metadata,
        }
```

#### Configuration

```bash
# Epic 20 - Memory Consolidation
MEMORY_CONSOLIDATION_ENABLED=true|false      # Default: true (if scopes enabled)
MEMORY_SIMILARITY_THRESHOLD=0.9              # Duplicate detection threshold
MEMORY_DECAY_HALF_LIFE_DAYS=30               # Importance decay rate
MEMORY_MIN_IMPORTANCE=0.1                    # Below this, memory is removed
MEMORY_CONSOLIDATION_SCHEDULE=0 2 * * *      # Cron schedule (2 AM daily)
```

#### Acceptance Criteria

- [ ] Given similar memories (>0.9 similarity), when consolidation runs, then they are merged into one
- [ ] Given old, unaccessed memories, when decay runs, then importance decreases exponentially
- [ ] Given memories below min importance threshold, when consolidation runs, then they are removed
- [ ] Consolidation can run on schedule or be triggered manually
- [ ] Access frequency boosts importance to retain frequently-used memories

---

## Group B: Graph Intelligence (Compete with MS GraphRAG)

### Story 20-B1: Implement Community Detection

#### Objective

Add community detection algorithms (Louvain/Leiden) to identify clusters of related entities in the knowledge graph, enabling community-level summaries for global queries.

#### Why This Matters

- **MS GraphRAG Core Feature:** Microsoft's GraphRAG uses community detection for "global" queries
- **Answer Quality:** Community summaries provide high-level context for abstract questions
- **Efficiency:** Pre-computed communities reduce query-time graph traversal

#### Technical Design

```python
# backend/src/agentic_rag_backend/graph/community.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import networkx as nx
import structlog

logger = structlog.get_logger(__name__)


class CommunityAlgorithm(str, Enum):
    LOUVAIN = "louvain"   # Good for general use
    LEIDEN = "leiden"     # Better quality, more expensive


@dataclass
class Community:
    """A community of related entities."""
    id: str
    name: str
    level: int  # Hierarchy level (0 = most granular)
    entity_ids: list[str]
    entity_count: int
    summary: Optional[str] = None
    keywords: list[str] = None
    parent_id: Optional[str] = None
    child_ids: list[str] = None


class CommunityDetector:
    """Detect and manage entity communities in knowledge graph."""

    def __init__(
        self,
        neo4j_client,
        graphiti_client,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LOUVAIN,
        min_community_size: int = 3,
        max_hierarchy_levels: int = 3,
    ):
        self._neo4j = neo4j_client
        self._graphiti = graphiti_client
        self.algorithm = algorithm
        self.min_community_size = min_community_size
        self.max_hierarchy_levels = max_hierarchy_levels

    async def detect_communities(
        self,
        tenant_id: str,
        generate_summaries: bool = True,
    ) -> list[Community]:
        """Detect communities in the knowledge graph.

        Uses hierarchical community detection to create
        multi-level community structure.
        """
        # Export graph to NetworkX for community detection
        G = await self._export_to_networkx(tenant_id)

        if len(G.nodes) < self.min_community_size:
            logger.info("graph_too_small_for_communities", node_count=len(G.nodes))
            return []

        # Run community detection
        if self.algorithm == CommunityAlgorithm.LOUVAIN:
            partition = self._run_louvain(G)
        else:
            partition = self._run_leiden(G)

        # Build community objects
        communities = self._build_communities(partition, G)

        # Generate hierarchical structure
        hierarchical = self._build_hierarchy(communities)

        # Generate summaries using LLM
        if generate_summaries:
            await self._generate_community_summaries(hierarchical, tenant_id)

        # Store communities in graph
        await self._store_communities(hierarchical, tenant_id)

        logger.info(
            "communities_detected",
            tenant_id=tenant_id,
            community_count=len(hierarchical),
            algorithm=self.algorithm.value,
        )

        return hierarchical

    def _run_louvain(self, G: nx.Graph) -> dict:
        """Run Louvain community detection."""
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(G, resolution=1.0)

        # Convert to partition dict
        partition = {}
        for i, community in enumerate(communities):
            for node in community:
                partition[node] = i

        return partition

    def _run_leiden(self, G: nx.Graph) -> dict:
        """Run Leiden community detection (if available)."""
        try:
            import leidenalg
            import igraph as ig

            # Convert to igraph
            ig_graph = ig.Graph.from_networkx(G)

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
            )

            return {node: membership for node, membership
                    in zip(G.nodes(), partition.membership)}
        except ImportError:
            logger.warning("leiden_not_available_falling_back_to_louvain")
            return self._run_louvain(G)

    async def _generate_community_summaries(
        self,
        communities: list[Community],
        tenant_id: str,
    ) -> None:
        """Generate LLM summaries for each community."""
        for community in communities:
            # Get entity details
            entities = await self._get_entity_details(community.entity_ids, tenant_id)

            # Generate summary using LLM
            summary = await self._summarize_community(entities, community.level)

            community.summary = summary.text
            community.keywords = summary.keywords

    async def get_community_for_query(
        self,
        query: str,
        tenant_id: str,
        level: int = 1,
    ) -> list[Community]:
        """Find relevant communities for a query.

        Used by global/local query router.
        """
        # Embed query
        query_embedding = await self._embed_query(query)

        # Find communities with similar summaries
        communities = await self._search_communities_by_summary(
            query_embedding,
            tenant_id,
            level,
        )

        return communities
```

#### Neo4j Schema

```cypher
// Community nodes
(:Community {
    id: String,
    name: String,
    level: Integer,
    tenantId: String,
    summary: String,
    keywords: List<String>,
    entityCount: Integer,
    createdAt: DateTime
})

// Relationships
(:Entity)-[:BELONGS_TO]->(:Community)
(:Community)-[:PARENT_OF]->(:Community)
(:Community)-[:CHILD_OF]->(:Community)

// Indexes
CREATE INDEX community_tenant FOR (c:Community) ON (c.tenantId);
CREATE INDEX community_level FOR (c:Community) ON (c.tenantId, c.level);
```

#### Configuration

```bash
# Epic 20 - Community Detection
COMMUNITY_DETECTION_ENABLED=true|false       # Default: false
COMMUNITY_ALGORITHM=louvain|leiden           # Default: louvain
COMMUNITY_MIN_SIZE=3                         # Min entities per community
COMMUNITY_MAX_LEVELS=3                       # Hierarchy depth
COMMUNITY_SUMMARY_MODEL=gpt-4o-mini          # Model for summaries
COMMUNITY_REFRESH_SCHEDULE=0 3 * * 0         # Weekly at 3 AM Sunday
```

#### Acceptance Criteria

- [ ] Given a knowledge graph with >10 entities, when detection runs, then communities are identified
- [ ] Community hierarchy has configurable levels
- [ ] Each community has an LLM-generated summary
- [ ] Communities are stored in Neo4j with proper relationships
- [ ] Detection completes in <5 minutes for graphs with <10K entities

---

### Story 20-B2: Implement LazyRAG Pattern

#### Objective

Implement the LazyRAG pattern that defers summarization to query time, achieving 99% reduction in indexing costs compared to eager summarization.

#### Why This Matters

- **Cost Reduction:** Eager summarization is expensive (MS GraphRAG indexing costs)
- **Freshness:** Query-time summaries always use latest entities
- **Flexibility:** Can adjust summary style per query type

#### Technical Design

```python
# backend/src/agentic_rag_backend/retrieval/lazy_rag.py
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LazyRAGResult:
    """Result from LazyRAG retrieval."""
    query: str
    entities: list[dict]
    relationships: list[dict]
    summary: str
    confidence: float
    generation_time_ms: int


class LazyRAGRetriever:
    """Query-time summarization retriever.

    Instead of pre-computing summaries at index time,
    generates summaries on-demand based on retrieved
    graph subsets.
    """

    def __init__(
        self,
        graphiti_client,
        llm_client,
        max_entities_for_summary: int = 50,
        summary_model: str = "gpt-4o-mini",
    ):
        self._graphiti = graphiti_client
        self._llm = llm_client
        self.max_entities = max_entities_for_summary
        self.summary_model = summary_model

    async def retrieve_and_summarize(
        self,
        query: str,
        tenant_id: str,
        use_communities: bool = True,
        max_hops: int = 2,
    ) -> LazyRAGResult:
        """Retrieve relevant graph subset and summarize.

        Steps:
        1. Extract entities from query
        2. Find matching entities in graph
        3. Expand via relationships (up to max_hops)
        4. If communities enabled, include community context
        5. Generate summary using LLM
        """
        import time
        start_time = time.perf_counter()

        # Step 1-2: Find seed entities
        seed_entities = await self._find_seed_entities(query, tenant_id)

        # Step 3: Expand via relationships
        expanded = await self._expand_subgraph(
            seed_entities,
            tenant_id,
            max_hops,
        )

        # Step 4: Add community context
        community_context = None
        if use_communities:
            community_context = await self._get_community_context(
                expanded["entities"],
                tenant_id,
            )

        # Step 5: Generate summary
        summary = await self._generate_summary(
            query,
            expanded["entities"][:self.max_entities],
            expanded["relationships"],
            community_context,
        )

        generation_time_ms = int((time.perf_counter() - start_time) * 1000)

        return LazyRAGResult(
            query=query,
            entities=expanded["entities"],
            relationships=expanded["relationships"],
            summary=summary.text,
            confidence=summary.confidence,
            generation_time_ms=generation_time_ms,
        )

    async def _generate_summary(
        self,
        query: str,
        entities: list[dict],
        relationships: list[dict],
        community_context: Optional[str],
    ) -> SummaryResult:
        """Generate query-focused summary of graph subset."""

        # Build context from entities
        entity_context = self._format_entities(entities)
        relationship_context = self._format_relationships(relationships)

        prompt = f"""Based on the following knowledge graph subset, answer the query.

Query: {query}

Entities:
{entity_context}

Relationships:
{relationship_context}

{f"Community Context: {community_context}" if community_context else ""}

Provide a comprehensive answer based only on the information above.
If the information is insufficient, indicate what's missing.
"""

        response = await self._llm.generate(
            prompt,
            model=self.summary_model,
            temperature=0.3,
        )

        return SummaryResult(
            text=response.text,
            confidence=self._estimate_confidence(response, entities),
        )
```

#### Configuration

```bash
# Epic 20 - LazyRAG
LAZY_RAG_ENABLED=true|false                  # Default: false
LAZY_RAG_MAX_ENTITIES=50                     # Max entities in summary context
LAZY_RAG_MAX_HOPS=2                          # Relationship expansion depth
LAZY_RAG_SUMMARY_MODEL=gpt-4o-mini           # Model for query-time summaries
LAZY_RAG_USE_COMMUNITIES=true                # Include community context
```

#### Acceptance Criteria

- [ ] Given a query, when LazyRAG runs, then graph subset is retrieved and summarized
- [ ] No pre-computed summaries are required (lazy generation)
- [ ] Community context is optionally included
- [ ] Summary generation completes in <3 seconds for typical queries

---

### Story 20-B3: Implement Global/Local Query Routing

#### Objective

Implement a query classifier that routes queries to either global (community-level) or local (entity-level) retrieval based on query characteristics.

#### Technical Design

```python
# backend/src/agentic_rag_backend/retrieval/query_router.py
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    GLOBAL = "global"     # High-level, abstract queries
    LOCAL = "local"       # Specific entity queries
    HYBRID = "hybrid"     # Both global and local


@dataclass
class RoutingDecision:
    """Query routing decision."""
    query_type: QueryType
    confidence: float
    reasoning: str
    global_weight: float  # 0.0-1.0, for hybrid
    local_weight: float   # 0.0-1.0, for hybrid


class QueryRouter:
    """Route queries to appropriate retrieval strategy."""

    # Patterns indicating global queries
    GLOBAL_PATTERNS = [
        r"what (are|is) the (main|primary|key|overall)",
        r"summarize|summary|overview",
        r"(all|every|each) .*(types?|kinds?|categories?)",
        r"how (many|much) .* (total|overall|in general)",
        r"what themes|main topics",
        r"general (understanding|overview|summary)",
    ]

    # Patterns indicating local queries
    LOCAL_PATTERNS = [
        r"what is (\w+)",
        r"who (is|was) (\w+)",
        r"where (is|was|does)",
        r"when (did|was|is)",
        r"how (do|does|did) (\w+)",
        r"specific|particular|exact",
        r"(this|that|the) (\w+)",
    ]

    def __init__(
        self,
        llm_client=None,
        use_llm_classification: bool = False,
    ):
        self._llm = llm_client
        self.use_llm = use_llm_classification

    async def route_query(self, query: str) -> RoutingDecision:
        """Determine optimal retrieval strategy for query."""

        # First try rule-based classification
        rule_decision = self._rule_based_classification(query)

        if rule_decision.confidence >= 0.8:
            return rule_decision

        # Fall back to LLM classification if enabled
        if self.use_llm and self._llm:
            return await self._llm_classification(query)

        # Default to hybrid
        return RoutingDecision(
            query_type=QueryType.HYBRID,
            confidence=0.5,
            reasoning="Uncertain, using hybrid approach",
            global_weight=0.5,
            local_weight=0.5,
        )

    def _rule_based_classification(self, query: str) -> RoutingDecision:
        """Classify using regex patterns."""
        import re

        query_lower = query.lower()

        global_matches = sum(
            1 for pattern in self.GLOBAL_PATTERNS
            if re.search(pattern, query_lower)
        )

        local_matches = sum(
            1 for pattern in self.LOCAL_PATTERNS
            if re.search(pattern, query_lower)
        )

        total = global_matches + local_matches

        if total == 0:
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                confidence=0.3,
                reasoning="No pattern matches",
                global_weight=0.5,
                local_weight=0.5,
            )

        global_ratio = global_matches / total

        if global_ratio >= 0.7:
            return RoutingDecision(
                query_type=QueryType.GLOBAL,
                confidence=0.8,
                reasoning=f"Global patterns: {global_matches}",
                global_weight=1.0,
                local_weight=0.0,
            )
        elif global_ratio <= 0.3:
            return RoutingDecision(
                query_type=QueryType.LOCAL,
                confidence=0.8,
                reasoning=f"Local patterns: {local_matches}",
                global_weight=0.0,
                local_weight=1.0,
            )
        else:
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                confidence=0.6,
                reasoning=f"Mixed patterns: global={global_matches}, local={local_matches}",
                global_weight=global_ratio,
                local_weight=1 - global_ratio,
            )
```

#### Configuration

```bash
# Epic 20 - Query Routing
QUERY_ROUTING_ENABLED=true|false             # Default: true (if community detection enabled)
QUERY_ROUTING_USE_LLM=true|false             # Use LLM for uncertain queries
QUERY_ROUTING_LLM_MODEL=gpt-4o-mini          # Classification model
QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.7       # Below this, use hybrid
```

#### Acceptance Criteria

- [ ] Given an abstract query like "What are the main themes?", when routed, then GLOBAL strategy is selected
- [ ] Given a specific query like "What is function X?", when routed, then LOCAL strategy is selected
- [ ] Uncertain queries default to HYBRID with weighted combination
- [ ] Routing adds <50ms latency

---

## Group C: Retrieval Excellence

### Story 20-C1: Implement Graph-Based Rerankers

#### Objective

Implement graph-aware reranking strategies inspired by Zep: episode-mentions scoring, node-distance weighting, and hybrid scoring.

#### Technical Design

```python
# backend/src/agentic_rag_backend/retrieval/graph_rerankers.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GraphRerankedResult:
    """A result with graph-based reranking score."""
    original_result: dict
    original_score: float
    graph_score: float
    combined_score: float
    graph_context: dict  # Node distances, episode counts, etc.


class GraphReranker(ABC):
    """Base class for graph-aware rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[dict],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        pass


class EpisodeMentionsReranker(GraphReranker):
    """Rerank based on how often entities appear in recent episodes.

    Entities mentioned in more episodes are considered more relevant.
    """

    def __init__(self, graphiti_client, episode_window_days: int = 30):
        self._graphiti = graphiti_client
        self.window_days = episode_window_days

    async def rerank(
        self,
        query: str,
        results: list[dict],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Rerank by episode mention frequency."""
        reranked = []

        for result in results:
            # Get entities mentioned in result
            entities = self._extract_entities(result)

            # Count episode mentions for each entity
            total_mentions = 0
            for entity in entities:
                mentions = await self._count_episode_mentions(
                    entity, tenant_id, self.window_days
                )
                total_mentions += mentions

            # Normalize to 0-1 score
            graph_score = min(1.0, total_mentions / 10.0)

            # Combine with original score
            combined = 0.6 * result["score"] + 0.4 * graph_score

            reranked.append(GraphRerankedResult(
                original_result=result,
                original_score=result["score"],
                graph_score=graph_score,
                combined_score=combined,
                graph_context={"episode_mentions": total_mentions},
            ))

        # Sort by combined score
        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        return reranked


class NodeDistanceReranker(GraphReranker):
    """Rerank based on graph distance from query entities.

    Results closer to query entities in the graph are ranked higher.
    """

    def __init__(self, graphiti_client, max_distance: int = 3):
        self._graphiti = graphiti_client
        self.max_distance = max_distance

    async def rerank(
        self,
        query: str,
        results: list[dict],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Rerank by graph distance."""
        # Extract query entities
        query_entities = await self._extract_query_entities(query, tenant_id)

        if not query_entities:
            # No graph context, return original order
            return [
                GraphRerankedResult(
                    original_result=r,
                    original_score=r["score"],
                    graph_score=0.5,
                    combined_score=r["score"],
                    graph_context={},
                )
                for r in results
            ]

        reranked = []

        for result in results:
            result_entities = self._extract_entities(result)

            # Calculate minimum distance to any query entity
            min_distance = float("inf")
            for r_entity in result_entities:
                for q_entity in query_entities:
                    distance = await self._get_graph_distance(
                        r_entity, q_entity, tenant_id
                    )
                    min_distance = min(min_distance, distance)

            # Convert distance to score (closer = higher)
            if min_distance == float("inf"):
                graph_score = 0.0
            else:
                graph_score = max(0, 1 - (min_distance / self.max_distance))

            combined = 0.5 * result["score"] + 0.5 * graph_score

            reranked.append(GraphRerankedResult(
                original_result=result,
                original_score=result["score"],
                graph_score=graph_score,
                combined_score=combined,
                graph_context={"min_distance": min_distance},
            ))

        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        return reranked


class HybridGraphReranker(GraphReranker):
    """Combine multiple graph-based signals."""

    def __init__(
        self,
        graphiti_client,
        episode_weight: float = 0.3,
        distance_weight: float = 0.3,
        original_weight: float = 0.4,
    ):
        self._episode_reranker = EpisodeMentionsReranker(graphiti_client)
        self._distance_reranker = NodeDistanceReranker(graphiti_client)
        self.episode_weight = episode_weight
        self.distance_weight = distance_weight
        self.original_weight = original_weight

    async def rerank(
        self,
        query: str,
        results: list[dict],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Combine episode and distance signals."""
        # Get both reranking signals
        episode_results = await self._episode_reranker.rerank(query, results, tenant_id)
        distance_results = await self._distance_reranker.rerank(query, results, tenant_id)

        # Build lookup by result ID
        episode_scores = {r.original_result["id"]: r.graph_score for r in episode_results}
        distance_scores = {r.original_result["id"]: r.graph_score for r in distance_results}

        # Combine signals
        reranked = []
        for result in results:
            result_id = result["id"]

            combined = (
                self.original_weight * result["score"] +
                self.episode_weight * episode_scores.get(result_id, 0) +
                self.distance_weight * distance_scores.get(result_id, 0)
            )

            reranked.append(GraphRerankedResult(
                original_result=result,
                original_score=result["score"],
                graph_score=episode_scores.get(result_id, 0) + distance_scores.get(result_id, 0),
                combined_score=combined,
                graph_context={
                    "episode_score": episode_scores.get(result_id, 0),
                    "distance_score": distance_scores.get(result_id, 0),
                },
            ))

        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        return reranked
```

#### Configuration

```bash
# Epic 20 - Graph-Based Rerankers
GRAPH_RERANKER_ENABLED=true|false            # Default: false
GRAPH_RERANKER_TYPE=episode|distance|hybrid  # Default: hybrid
GRAPH_RERANKER_EPISODE_WEIGHT=0.3
GRAPH_RERANKER_DISTANCE_WEIGHT=0.3
GRAPH_RERANKER_ORIGINAL_WEIGHT=0.4
GRAPH_RERANKER_EPISODE_WINDOW_DAYS=30
GRAPH_RERANKER_MAX_DISTANCE=3
```

#### Acceptance Criteria

- [ ] Given results, when episode reranking runs, then frequently-mentioned entities score higher
- [ ] Given results, when distance reranking runs, then closer entities score higher
- [ ] Hybrid reranker combines signals with configurable weights
- [ ] Reranking adds <200ms latency

---

### Story 20-C2: Implement Dual-Level Retrieval

#### Objective

Implement LightRAG-style dual-level retrieval that combines low-level (entity) and high-level (theme/community) retrieval for comprehensive results.

#### Technical Design

```python
# backend/src/agentic_rag_backend/retrieval/dual_level.py
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DualLevelResult:
    """Combined low-level and high-level retrieval result."""
    query: str
    low_level_results: list[dict]   # Entity-level matches
    high_level_results: list[dict]  # Theme/community matches
    synthesized_answer: str
    confidence: float


class DualLevelRetriever:
    """LightRAG-style dual-level retrieval.

    Low-level: Direct entity matching (specific facts)
    High-level: Theme/community matching (abstract concepts)
    """

    def __init__(
        self,
        vector_search,
        graphiti_client,
        community_detector,
        llm_client,
        low_level_weight: float = 0.6,
        high_level_weight: float = 0.4,
    ):
        self._vector = vector_search
        self._graphiti = graphiti_client
        self._communities = community_detector
        self._llm = llm_client
        self.low_weight = low_level_weight
        self.high_weight = high_level_weight

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        low_level_limit: int = 10,
        high_level_limit: int = 5,
    ) -> DualLevelResult:
        """Perform dual-level retrieval.

        1. Low-level: Entity/chunk retrieval via vector + graph
        2. High-level: Community/theme retrieval
        3. Synthesize combined answer
        """
        # Low-level retrieval (entities/chunks)
        low_level = await self._low_level_retrieve(
            query, tenant_id, low_level_limit
        )

        # High-level retrieval (communities/themes)
        high_level = await self._high_level_retrieve(
            query, tenant_id, high_level_limit
        )

        # Synthesize answer from both levels
        answer = await self._synthesize_answer(
            query, low_level, high_level
        )

        return DualLevelResult(
            query=query,
            low_level_results=low_level,
            high_level_results=high_level,
            synthesized_answer=answer.text,
            confidence=answer.confidence,
        )

    async def _low_level_retrieve(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[dict]:
        """Retrieve specific entities and chunks."""
        # Hybrid vector + graph search
        results = await self._vector.search(query, tenant_id, limit=limit * 2)

        # Enhance with Graphiti entity data
        enhanced = []
        for result in results:
            entities = await self._graphiti.search(
                result["content"][:200],
                center_node_uuid=None,
                group_ids=[tenant_id],
            )
            enhanced.append({
                **result,
                "entities": entities[:3],
            })

        return enhanced[:limit]

    async def _high_level_retrieve(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[dict]:
        """Retrieve themes and communities."""
        communities = await self._communities.get_community_for_query(
            query, tenant_id, level=1  # Higher level communities
        )

        results = []
        for community in communities[:limit]:
            results.append({
                "type": "community",
                "id": community.id,
                "name": community.name,
                "summary": community.summary,
                "keywords": community.keywords,
                "entity_count": community.entity_count,
            })

        return results

    async def _synthesize_answer(
        self,
        query: str,
        low_level: list[dict],
        high_level: list[dict],
    ):
        """Synthesize answer from both levels."""
        low_context = self._format_low_level(low_level)
        high_context = self._format_high_level(high_level)

        prompt = f"""Answer the query using both specific facts and high-level context.

Query: {query}

Specific Facts (Low-Level):
{low_context}

Themes and Context (High-Level):
{high_context}

Provide a comprehensive answer that integrates both specific details and broader context.
"""

        return await self._llm.generate(prompt, temperature=0.3)
```

#### Configuration

```bash
# Epic 20 - Dual-Level Retrieval
DUAL_LEVEL_RETRIEVAL_ENABLED=true|false      # Default: false
DUAL_LEVEL_LOW_WEIGHT=0.6                    # Weight for entity-level
DUAL_LEVEL_HIGH_WEIGHT=0.4                   # Weight for theme-level
DUAL_LEVEL_LOW_LIMIT=10                      # Max low-level results
DUAL_LEVEL_HIGH_LIMIT=5                      # Max high-level results
```

#### Acceptance Criteria

- [ ] Given a query, when dual-level retrieval runs, then both entity and community results are returned
- [ ] Synthesis combines both levels into comprehensive answer
- [ ] Weights are configurable for different use cases

---

### Story 20-C3: Implement Parent-Child Chunk Hierarchy

#### Objective

Implement small-to-big retrieval pattern where small chunks are used for matching but parent chunks provide full context.

#### Technical Design

```python
# backend/src/agentic_rag_backend/indexing/hierarchical_chunker.py
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HierarchicalChunk:
    """A chunk with parent/child relationships."""
    id: str
    content: str
    level: int  # 0 = leaf (smallest), higher = larger context
    parent_id: Optional[str]
    child_ids: list[str]
    metadata: dict


class HierarchicalChunker:
    """Create parent-child chunk hierarchies.

    Level 0: Small chunks (256 tokens) - for precise matching
    Level 1: Medium chunks (512 tokens) - combining level 0
    Level 2: Large chunks (1024 tokens) - combining level 1
    Level 3: Document sections - combining level 2
    """

    def __init__(
        self,
        level_sizes: list[int] = None,
        overlap_ratio: float = 0.1,
    ):
        self.level_sizes = level_sizes or [256, 512, 1024, 2048]
        self.overlap_ratio = overlap_ratio

    def chunk_document(
        self,
        content: str,
        metadata: dict = None,
    ) -> list[HierarchicalChunk]:
        """Create hierarchical chunks from document."""
        all_chunks = []

        # Level 0: Smallest chunks
        level_0_chunks = self._create_level_chunks(content, 0)
        all_chunks.extend(level_0_chunks)

        # Build higher levels by combining lower
        for level in range(1, len(self.level_sizes)):
            previous_level = [c for c in all_chunks if c.level == level - 1]
            level_chunks = self._combine_to_level(previous_level, level)
            all_chunks.extend(level_chunks)

        return all_chunks

    def _create_level_chunks(
        self,
        content: str,
        level: int,
    ) -> list[HierarchicalChunk]:
        """Create chunks at a specific level."""
        target_size = self.level_sizes[level]
        overlap = int(target_size * self.overlap_ratio)

        chunks = []
        start = 0

        while start < len(content):
            end = min(start + target_size, len(content))
            chunk_content = content[start:end]

            chunk = HierarchicalChunk(
                id=f"chunk_{level}_{len(chunks)}",
                content=chunk_content,
                level=level,
                parent_id=None,  # Set later
                child_ids=[],    # Set later
                metadata={},
            )
            chunks.append(chunk)

            start = end - overlap
            if start >= len(content) - overlap:
                break

        return chunks

    def _combine_to_level(
        self,
        lower_chunks: list[HierarchicalChunk],
        level: int,
    ) -> list[HierarchicalChunk]:
        """Combine lower level chunks into higher level."""
        target_size = self.level_sizes[level]

        combined = []
        current_content = ""
        current_children = []

        for chunk in lower_chunks:
            if len(current_content) + len(chunk.content) > target_size:
                # Create combined chunk
                if current_content:
                    parent = HierarchicalChunk(
                        id=f"chunk_{level}_{len(combined)}",
                        content=current_content,
                        level=level,
                        parent_id=None,
                        child_ids=[c.id for c in current_children],
                        metadata={},
                    )
                    combined.append(parent)

                    # Update children's parent references
                    for child in current_children:
                        child.parent_id = parent.id

                current_content = chunk.content
                current_children = [chunk]
            else:
                current_content += "\n" + chunk.content
                current_children.append(chunk)

        # Handle remaining
        if current_content:
            parent = HierarchicalChunk(
                id=f"chunk_{level}_{len(combined)}",
                content=current_content,
                level=level,
                parent_id=None,
                child_ids=[c.id for c in current_children],
                metadata={},
            )
            combined.append(parent)
            for child in current_children:
                child.parent_id = parent.id

        return combined


class SmallToBigRetriever:
    """Retrieve small chunks but return parent context."""

    def __init__(
        self,
        vector_search,
        chunk_store,
        return_level: int = 2,  # Return level 2 chunks
    ):
        self._vector = vector_search
        self._chunks = chunk_store
        self.return_level = return_level

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        limit: int = 5,
    ) -> list[HierarchicalChunk]:
        """Retrieve using small chunks, return large context.

        1. Search level 0 (small) chunks for precise matching
        2. Deduplicate by parent at return_level
        3. Return parent chunks for full context
        """
        # Search small chunks
        small_matches = await self._vector.search(
            query, tenant_id,
            filter={"level": 0},
            limit=limit * 3,
        )

        # Get unique parents at return level
        parent_ids = set()
        for match in small_matches:
            parent = await self._get_ancestor_at_level(
                match["id"], self.return_level
            )
            if parent:
                parent_ids.add(parent.id)

        # Fetch parent chunks
        parents = []
        for parent_id in list(parent_ids)[:limit]:
            chunk = await self._chunks.get(parent_id)
            if chunk:
                parents.append(chunk)

        return parents

    async def _get_ancestor_at_level(
        self,
        chunk_id: str,
        target_level: int,
    ) -> Optional[HierarchicalChunk]:
        """Traverse up to find ancestor at target level."""
        chunk = await self._chunks.get(chunk_id)

        while chunk and chunk.level < target_level:
            if not chunk.parent_id:
                break
            chunk = await self._chunks.get(chunk.parent_id)

        return chunk if chunk and chunk.level == target_level else None
```

#### Configuration

```bash
# Epic 20 - Parent-Child Chunks
HIERARCHICAL_CHUNKS_ENABLED=true|false       # Default: false
HIERARCHICAL_CHUNK_LEVELS=256,512,1024,2048  # Token sizes per level
HIERARCHICAL_OVERLAP_RATIO=0.1               # Overlap between chunks
SMALL_TO_BIG_RETURN_LEVEL=2                  # Level to return (larger context)
```

#### Acceptance Criteria

- [ ] Given a document, when chunked hierarchically, then 4 levels are created
- [ ] Child chunks reference parents, parents reference children
- [ ] Small-to-big retrieval matches small chunks but returns parent context
- [ ] Deduplication prevents returning overlapping parents

---

## Group D: Document Intelligence (RAGFlow Approach)

### Story 20-D1: Enhance Table/Layout Extraction

#### Objective

Enhance Docling integration to better extract tables, preserve layout, and handle complex document structures.

#### Technical Design

```python
# backend/src/agentic_rag_backend/indexing/enhanced_docling.py
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedTable:
    """A table extracted from a document."""
    id: str
    page_number: int
    position: dict  # {x, y, width, height}
    headers: list[str]
    rows: list[list[str]]
    caption: Optional[str]
    markdown: str
    structured_data: list[dict]  # Row dicts with headers as keys


@dataclass
class DocumentLayout:
    """Structured document layout."""
    sections: list[dict]
    tables: list[ExtractedTable]
    figures: list[dict]
    footnotes: list[dict]
    headers: list[dict]
    page_count: int


class EnhancedDoclingParser:
    """Enhanced document parsing with better table/layout extraction."""

    def __init__(
        self,
        docling_client,
        table_extraction_model: str = "docling-table",
        preserve_layout: bool = True,
    ):
        self._docling = docling_client
        self.table_model = table_extraction_model
        self.preserve_layout = preserve_layout

    async def parse_document(
        self,
        file_path: str,
        extract_tables: bool = True,
        extract_layout: bool = True,
    ) -> DocumentLayout:
        """Parse document with enhanced extraction."""
        # Basic Docling parsing
        result = await self._docling.parse(file_path)

        # Enhanced table extraction
        tables = []
        if extract_tables:
            tables = await self._extract_tables(result, file_path)

        # Layout analysis
        layout = DocumentLayout(
            sections=result.sections,
            tables=tables,
            figures=result.figures if hasattr(result, 'figures') else [],
            footnotes=result.footnotes if hasattr(result, 'footnotes') else [],
            headers=result.headers if hasattr(result, 'headers') else [],
            page_count=result.page_count,
        )

        return layout

    async def _extract_tables(
        self,
        docling_result,
        file_path: str,
    ) -> list[ExtractedTable]:
        """Extract tables with structure preservation."""
        tables = []

        for i, raw_table in enumerate(docling_result.tables):
            # Parse table structure
            headers = raw_table.headers if hasattr(raw_table, 'headers') else []
            rows = raw_table.rows if hasattr(raw_table, 'rows') else []

            # Generate markdown representation
            markdown = self._table_to_markdown(headers, rows)

            # Generate structured data (list of dicts)
            structured = []
            for row in rows:
                row_dict = {}
                for j, cell in enumerate(row):
                    header = headers[j] if j < len(headers) else f"col_{j}"
                    row_dict[header] = cell
                structured.append(row_dict)

            table = ExtractedTable(
                id=f"table_{i}",
                page_number=raw_table.page if hasattr(raw_table, 'page') else 0,
                position=raw_table.bbox if hasattr(raw_table, 'bbox') else {},
                headers=headers,
                rows=rows,
                caption=raw_table.caption if hasattr(raw_table, 'caption') else None,
                markdown=markdown,
                structured_data=structured,
            )
            tables.append(table)

        return tables

    def _table_to_markdown(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> str:
        """Convert table to markdown format."""
        lines = []

        # Header row
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Data rows
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)
```

#### Configuration

```bash
# Epic 20 - Enhanced Document Intelligence
ENHANCED_DOCLING_ENABLED=true|false          # Default: true (uses existing Docling)
DOCLING_TABLE_EXTRACTION=true|false          # Default: true
DOCLING_PRESERVE_LAYOUT=true|false           # Default: true
DOCLING_TABLE_AS_MARKDOWN=true|false         # Store tables as markdown chunks
```

#### Acceptance Criteria

- [ ] Given a PDF with tables, when parsed, then tables are extracted with headers and rows
- [ ] Tables are converted to markdown for embedding
- [ ] Structured data (list of dicts) is available for programmatic access
- [ ] Layout preservation maintains section hierarchy

---

### Story 20-D2: Implement Multimodal Ingestion

#### Objective

Add support for additional document types: Office documents (Word, Excel, PowerPoint) and formulas.

#### Technical Design

```python
# backend/src/agentic_rag_backend/indexing/multimodal.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class DocumentType(str, Enum):
    PDF = "pdf"
    WORD = "docx"
    EXCEL = "xlsx"
    POWERPOINT = "pptx"
    IMAGE = "image"
    MARKDOWN = "md"
    TEXT = "txt"


class MultimodalIngester:
    """Ingest multiple document types."""

    def __init__(
        self,
        docling_parser,
        office_parser,
        image_processor,
    ):
        self._docling = docling_parser
        self._office = office_parser
        self._image = image_processor

    async def ingest(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
    ):
        """Ingest document based on type."""
        if document_type is None:
            document_type = self._detect_type(file_path)

        if document_type == DocumentType.PDF:
            return await self._docling.parse_document(file_path)

        elif document_type in (DocumentType.WORD, DocumentType.EXCEL, DocumentType.POWERPOINT):
            return await self._office.parse(file_path, document_type)

        elif document_type == DocumentType.IMAGE:
            return await self._image.process(file_path)

        else:
            # Plain text/markdown
            return await self._text_parse(file_path)


class OfficeDocumentParser:
    """Parse Office documents (Word, Excel, PowerPoint)."""

    async def parse(self, file_path: str, doc_type: DocumentType):
        """Parse Office document."""
        if doc_type == DocumentType.WORD:
            return await self._parse_word(file_path)
        elif doc_type == DocumentType.EXCEL:
            return await self._parse_excel(file_path)
        elif doc_type == DocumentType.POWERPOINT:
            return await self._parse_powerpoint(file_path)

    async def _parse_word(self, file_path: str):
        """Parse Word document."""
        from docx import Document

        doc = Document(file_path)

        content = []
        for para in doc.paragraphs:
            content.append(para.text)

        tables = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text for cell in row.cells]
                rows.append(cells)
            tables.append(rows)

        return {
            "content": "\n".join(content),
            "tables": tables,
            "paragraphs": len(doc.paragraphs),
        }

    async def _parse_excel(self, file_path: str):
        """Parse Excel spreadsheet."""
        import openpyxl

        wb = openpyxl.load_workbook(file_path)
        sheets = {}

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            data = []
            for row in ws.iter_rows(values_only=True):
                data.append(list(row))
            sheets[sheet_name] = data

        return {
            "sheets": sheets,
            "sheet_count": len(sheets),
        }

    async def _parse_powerpoint(self, file_path: str):
        """Parse PowerPoint presentation."""
        from pptx import Presentation

        prs = Presentation(file_path)
        slides = []

        for slide in prs.slides:
            slide_content = {
                "texts": [],
                "notes": "",
            }

            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_content["texts"].append(shape.text)

            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                slide_content["notes"] = notes_frame.text

            slides.append(slide_content)

        return {
            "slides": slides,
            "slide_count": len(slides),
        }
```

#### Configuration

```bash
# Epic 20 - Multimodal Ingestion
MULTIMODAL_INGESTION_ENABLED=true|false      # Default: false
OFFICE_DOCS_ENABLED=true|false               # Word, Excel, PowerPoint
IMAGE_INGESTION_ENABLED=true|false           # Image processing
FORMULA_EXTRACTION_ENABLED=true|false        # LaTeX/MathML formulas
```

#### Acceptance Criteria

- [ ] Given a Word document, when ingested, then text and tables are extracted
- [ ] Given an Excel file, when ingested, then all sheets are parsed
- [ ] Given a PowerPoint, when ingested, then slides and notes are extracted
- [ ] Document type is auto-detected from file extension

---

## Group E: Advanced Features (Cognee-Inspired)

### Story 20-E1: Implement Ontology Support

#### Objective

Add support for OWL ontologies to provide domain-specific entity types and relationships.

#### Technical Design

```python
# backend/src/agentic_rag_backend/ontology/loader.py
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OntologyClass:
    """A class from an ontology."""
    uri: str
    name: str
    description: Optional[str]
    parent_uris: list[str]
    properties: list[str]


@dataclass
class OntologyProperty:
    """A property from an ontology."""
    uri: str
    name: str
    domain: str  # Class URI
    range: str   # Class URI or datatype
    description: Optional[str]


class OntologyLoader:
    """Load and manage OWL ontologies."""

    def __init__(self, graphiti_client):
        self._graphiti = graphiti_client
        self._loaded_ontologies: dict[str, dict] = {}

    async def load_ontology(
        self,
        ontology_path: str,
        tenant_id: str,
    ) -> dict:
        """Load an OWL ontology and register types.

        This allows domain-specific entity typing during ingestion.
        """
        from owlready2 import get_ontology

        onto = get_ontology(ontology_path).load()

        classes = []
        for cls in onto.classes():
            ont_class = OntologyClass(
                uri=cls.iri,
                name=cls.name,
                description=cls.comment.first() if cls.comment else None,
                parent_uris=[p.iri for p in cls.is_a if hasattr(p, 'iri')],
                properties=[p.name for p in cls.get_properties()],
            )
            classes.append(ont_class)

        properties = []
        for prop in onto.properties():
            ont_prop = OntologyProperty(
                uri=prop.iri,
                name=prop.name,
                domain=prop.domain[0].iri if prop.domain else None,
                range=prop.range[0].iri if prop.range else None,
                description=prop.comment.first() if prop.comment else None,
            )
            properties.append(ont_prop)

        ontology_data = {
            "name": onto.name,
            "classes": classes,
            "properties": properties,
            "class_count": len(classes),
            "property_count": len(properties),
        }

        self._loaded_ontologies[tenant_id] = ontology_data

        # Register as custom entity types in Graphiti
        await self._register_graphiti_types(ontology_data, tenant_id)

        logger.info(
            "ontology_loaded",
            name=onto.name,
            classes=len(classes),
            properties=len(properties),
            tenant_id=tenant_id,
        )

        return ontology_data

    async def _register_graphiti_types(
        self,
        ontology: dict,
        tenant_id: str,
    ) -> None:
        """Register ontology classes as Graphiti entity types."""
        # This integrates with Graphiti's custom entity type system
        pass

    def get_entity_type(
        self,
        entity_name: str,
        tenant_id: str,
    ) -> Optional[OntologyClass]:
        """Find matching ontology class for an entity."""
        if tenant_id not in self._loaded_ontologies:
            return None

        onto = self._loaded_ontologies[tenant_id]

        # Simple name matching (could be enhanced with NLP)
        for cls in onto["classes"]:
            if cls.name.lower() in entity_name.lower():
                return cls

        return None
```

#### Configuration

```bash
# Epic 20 - Ontology Support
ONTOLOGY_SUPPORT_ENABLED=true|false          # Default: false
ONTOLOGY_PATH=/path/to/domain.owl            # Default ontology
ONTOLOGY_AUTO_TYPE=true|false                # Auto-type entities
```

#### Acceptance Criteria

- [ ] Given an OWL ontology, when loaded, then classes and properties are extracted
- [ ] Ontology classes can be used as entity types during ingestion
- [ ] Entity extraction uses ontology for domain-specific typing

---

### Story 20-E2: Implement Self-Improving Feedback Loop

#### Objective

Implement a feedback mechanism that uses user corrections and preferences to improve retrieval quality over time.

#### Technical Design

```python
# backend/src/agentic_rag_backend/feedback/loop.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class FeedbackType(str, Enum):
    RELEVANCE = "relevance"          # Was result relevant?
    ACCURACY = "accuracy"            # Was answer accurate?
    COMPLETENESS = "completeness"    # Was answer complete?
    PREFERENCE = "preference"        # User preference between options


@dataclass
class UserFeedback:
    """Feedback on a retrieval/response."""
    id: str
    query_id: str
    result_id: Optional[str]
    feedback_type: FeedbackType
    score: float  # -1.0 to 1.0
    correction: Optional[str]  # User-provided correction
    tenant_id: str
    user_id: str
    created_at: datetime


class FeedbackLoop:
    """Self-improving feedback system."""

    def __init__(
        self,
        postgres_client,
        embedding_provider,
        reranker=None,
    ):
        self._postgres = postgres_client
        self._embeddings = embedding_provider
        self._reranker = reranker

        # Feedback aggregations
        self._query_feedback: dict[str, list[UserFeedback]] = {}

    async def record_feedback(
        self,
        feedback: UserFeedback,
    ) -> None:
        """Record user feedback."""
        # Store in database
        await self._store_feedback(feedback)

        # Update query-level aggregations
        if feedback.query_id not in self._query_feedback:
            self._query_feedback[feedback.query_id] = []
        self._query_feedback[feedback.query_id].append(feedback)

        # If correction provided, learn from it
        if feedback.correction:
            await self._learn_from_correction(feedback)

        logger.info(
            "feedback_recorded",
            feedback_type=feedback.feedback_type.value,
            score=feedback.score,
            has_correction=bool(feedback.correction),
        )

    async def _learn_from_correction(
        self,
        feedback: UserFeedback,
    ) -> None:
        """Learn from user correction.

        This can:
        1. Add correction to knowledge base
        2. Adjust embedding weights
        3. Update reranker training data
        """
        # Add correction as a high-quality example
        correction_embedding = await self._embeddings.embed(feedback.correction)

        # Store as training signal for reranking
        if self._reranker:
            await self._reranker.add_training_example(
                query_id=feedback.query_id,
                positive_text=feedback.correction,
                embedding=correction_embedding,
            )

    async def get_query_boost(
        self,
        query: str,
        tenant_id: str,
    ) -> dict:
        """Get boost factors based on similar query feedback.

        Returns adjustment factors for retrieval based on
        feedback from similar past queries.
        """
        # Find similar past queries with feedback
        similar_queries = await self._find_similar_queries(query, tenant_id)

        if not similar_queries:
            return {"boost": 1.0}

        # Aggregate feedback scores
        total_score = 0
        count = 0

        for similar in similar_queries:
            feedback_list = self._query_feedback.get(similar["id"], [])
            for fb in feedback_list:
                total_score += fb.score
                count += 1

        if count == 0:
            return {"boost": 1.0}

        avg_score = total_score / count

        # Convert to boost factor (0.5 to 1.5 range)
        boost = 1.0 + (avg_score * 0.5)

        return {
            "boost": boost,
            "based_on_queries": len(similar_queries),
            "feedback_count": count,
        }

    async def _find_similar_queries(
        self,
        query: str,
        tenant_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """Find similar past queries."""
        query_embedding = await self._embeddings.embed(query)

        # Search for similar queries in feedback history
        similar = await self._postgres.search_similar_queries(
            embedding=query_embedding,
            tenant_id=tenant_id,
            limit=limit,
        )

        return similar
```

#### Configuration

```bash
# Epic 20 - Feedback Loop
FEEDBACK_LOOP_ENABLED=true|false             # Default: false
FEEDBACK_MIN_SAMPLES=10                      # Min feedback before using
FEEDBACK_DECAY_DAYS=90                       # Feedback relevance decay
FEEDBACK_BOOST_MAX=1.5                       # Max boost factor
```

#### Acceptance Criteria

- [ ] Given user feedback, when recorded, then it is stored and aggregated
- [ ] User corrections are learned from and influence future results
- [ ] Similar queries benefit from past feedback (boost factors)
- [ ] Feedback decays over time to prioritize recent signals

---

## Group H: Additional Competitive Features

### Story 20-H1: Implement Sparse Vector Search (BM42)

#### Objective

Add sparse vector (BM42) search capability for improved lexical matching alongside dense vectors.

#### Technical Design

```python
# backend/src/agentic_rag_backend/retrieval/sparse_vectors.py
from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SparseVector:
    """A sparse vector representation."""
    indices: list[int]
    values: list[float]

    def to_dict(self) -> dict:
        return {"indices": self.indices, "values": self.values}


class BM42Encoder:
    """BM42 sparse vector encoder using fastembed."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
    ):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts to sparse vectors."""
        embeddings = list(self.model.embed(texts))

        return [
            SparseVector(
                indices=list(emb.indices),
                values=list(emb.values),
            )
            for emb in embeddings
        ]

    def encode_query(self, query: str) -> SparseVector:
        """Encode a single query."""
        return self.encode([query])[0]


class HybridVectorSearch:
    """Combine dense and sparse vector search."""

    def __init__(
        self,
        dense_search,
        sparse_encoder: BM42Encoder,
        postgres_client,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self._dense = dense_search
        self._sparse = sparse_encoder
        self._postgres = postgres_client
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    async def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search combining dense and sparse."""
        # Dense search
        dense_results = await self._dense.search(query, tenant_id, limit=limit * 2)

        # Sparse search
        sparse_vector = self._sparse.encode_query(query)
        sparse_results = await self._sparse_search(
            sparse_vector, tenant_id, limit=limit * 2
        )

        # Combine with RRF (Reciprocal Rank Fusion)
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            self.dense_weight,
            self.sparse_weight,
        )

        return combined[:limit]

    def _reciprocal_rank_fusion(
        self,
        dense: list[dict],
        sparse: list[dict],
        dense_weight: float,
        sparse_weight: float,
        k: int = 60,
    ) -> list[dict]:
        """Combine results using RRF."""
        scores = {}

        # Score dense results
        for i, result in enumerate(dense):
            doc_id = result["id"]
            rrf_score = dense_weight / (k + i + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            scores[f"{doc_id}_data"] = result

        # Score sparse results
        for i, result in enumerate(sparse):
            doc_id = result["id"]
            rrf_score = sparse_weight / (k + i + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if f"{doc_id}_data" not in scores:
                scores[f"{doc_id}_data"] = result

        # Sort by combined score
        doc_scores = [(k, v) for k, v in scores.items() if not k.endswith("_data")]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [scores[f"{doc_id}_data"] for doc_id, _ in doc_scores]
```

#### Configuration

```bash
# Epic 20 - Sparse Vectors (BM42)
SPARSE_VECTORS_ENABLED=true|false            # Default: false
SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions
HYBRID_DENSE_WEIGHT=0.7
HYBRID_SPARSE_WEIGHT=0.3
```

#### Acceptance Criteria

- [ ] Given text, when encoded with BM42, then sparse vectors are generated
- [ ] Hybrid search combines dense and sparse with configurable weights
- [ ] RRF fusion produces better results than either alone

---

### Story 20-H2: Implement Cross-Language Query

#### Objective

Support queries in languages different from the indexed content, using translation or multilingual embeddings.

#### Configuration

```bash
# Epic 20 - Cross-Language
CROSS_LANGUAGE_ENABLED=true|false            # Default: false
CROSS_LANGUAGE_EMBEDDING=multilingual-e5     # Multilingual model
CROSS_LANGUAGE_TRANSLATION=true|false        # Translate queries
```

---

### Story 20-H3: Implement External Data Source Sync

#### Objective

Add connectors to sync content from external sources: Confluence, S3, Notion, Google Drive, Discord.

#### Configuration

```bash
# Epic 20 - External Sync
EXTERNAL_SYNC_ENABLED=true|false             # Default: false
SYNC_SOURCES=notion,s3,confluence            # Enabled sources
NOTION_API_KEY=secret_xxx
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
CONFLUENCE_URL=https://xxx.atlassian.net
```

---

### Story 20-H4: Implement Voice I/O

#### Objective

Add speech-to-text (Whisper) and text-to-speech for voice interaction with the RAG system.

#### Configuration

```bash
# Epic 20 - Voice I/O
VOICE_IO_ENABLED=true|false                  # Default: false
WHISPER_MODEL=base|small|medium|large        # Default: base
TTS_PROVIDER=openai|pyttsx3                  # Default: openai
```

---

### Story 20-H5: Implement ColBERT Reranking

#### Objective

Add ColBERT (late interaction) reranking as an alternative to cross-encoder reranking.

#### Configuration

```bash
# Epic 20 - ColBERT
COLBERT_ENABLED=true|false                   # Default: false
COLBERT_MODEL=colbert-ir/colbertv2.0
COLBERT_MAX_LENGTH=512
```

---

### Story 20-H6: Implement Visual Workflow Editor

#### Objective

Add a visual workflow editor (React Flow based) for building and debugging retrieval pipelines.

#### Configuration

```bash
# Epic 20 - Visual Workflow
VISUAL_WORKFLOW_ENABLED=true|false           # Default: false
```

---

## Dependencies Between Stories

```
Epic 20 Story Dependencies
===========================

Group A (Memory):
  20-A1 (Memory Scopes) → 20-A2 (Memory Consolidation)

Group B (Graph Intelligence):
  20-B1 (Community Detection) → 20-B2 (LazyRAG)
  20-B1 (Community Detection) → 20-B3 (Query Routing)

Group C (Retrieval Excellence):
  20-C1 (Graph Rerankers) - Independent
  20-C2 (Dual-Level) - Requires 20-B1 (Communities)
  20-C3 (Parent-Child) - Independent

Group D (Document Intelligence):
  20-D1 (Table Extraction) - Independent (enhances existing)
  20-D2 (Multimodal) - Independent

Group E (Advanced):
  20-E1 (Ontology) - Independent
  20-E2 (Feedback Loop) - Independent

Group H (Competitive):
  All stories are independent

Recommended Implementation Order:
1. 20-A1, 20-A2 (Memory Platform - core differentiation)
2. 20-B1 (Community Detection - enables multiple features)
3. 20-B2, 20-B3 (LazyRAG, Query Routing - depends on B1)
4. 20-C1 (Graph Rerankers - enhances retrieval)
5. 20-C2 (Dual-Level - depends on B1)
6. 20-C3 (Parent-Child - independent enhancement)
7. 20-D1, 20-D2 (Document Intelligence)
8. 20-E1, 20-E2 (Advanced Features)
9. Group H (Competitive Features - as needed)
```

---

## Testing Strategy

### Unit Tests

Each module should have comprehensive unit tests covering:
- Memory scope isolation
- Community detection algorithm correctness
- Reranker scoring accuracy
- Chunker hierarchy building
- Document parsing accuracy

### Integration Tests

```python
# backend/tests/integration/test_epic_20.py

@pytest.mark.integration
async def test_memory_scope_isolation():
    """Test that memory scopes are properly isolated."""
    store = ScopedMemoryStore(...)

    # Add user memory
    await store.add_memory(
        content="User preference",
        scope=MemoryScope.USER,
        tenant_id="t1",
        user_id="u1",
    )

    # Add session memory
    await store.add_memory(
        content="Session context",
        scope=MemoryScope.SESSION,
        tenant_id="t1",
        session_id="s1",
    )

    # Search from session should include user
    results = await store.search_memories(
        query="preference",
        scope=MemoryScope.SESSION,
        tenant_id="t1",
        session_id="s1",
        user_id="u1",
        include_parent_scopes=True,
    )
    assert len(results) == 2


@pytest.mark.integration
async def test_community_detection():
    """Test community detection on sample graph."""
    detector = CommunityDetector(...)

    communities = await detector.detect_communities(
        tenant_id="test",
        generate_summaries=False,
    )

    assert len(communities) > 0
    assert all(c.entity_count >= 3 for c in communities)


@pytest.mark.integration
async def test_dual_level_retrieval():
    """Test dual-level retrieval returns both levels."""
    retriever = DualLevelRetriever(...)

    result = await retriever.retrieve(
        query="What are the main themes?",
        tenant_id="test",
    )

    assert len(result.low_level_results) > 0
    assert len(result.high_level_results) > 0
    assert result.synthesized_answer
```

---

## Migration and Deployment

### Phase 1: Foundation (Group A)
1. Add memory scope models and store
2. Implement memory consolidation
3. Add API endpoints
4. Feature flag everything

### Phase 2: Graph Intelligence (Group B)
1. Add community detection
2. Implement LazyRAG
3. Implement query routing
4. Connect to existing retrieval

### Phase 3: Retrieval Excellence (Group C)
1. Add graph-based rerankers
2. Implement dual-level retrieval
3. Implement parent-child chunks

### Phase 4: Document Intelligence (Group D)
1. Enhance table extraction
2. Add multimodal ingestion

### Phase 5: Advanced Features (Group E + H)
1. Add ontology support
2. Implement feedback loop
3. Add competitive features as prioritized

---

## Success Metrics

| Feature | Metric | Target |
|---------|--------|--------|
| Memory Scopes | Scope isolation accuracy | 100% |
| Community Detection | Detection time (10K entities) | <30 seconds |
| LazyRAG | Query-time summarization latency | <3 seconds |
| Graph Rerankers | MRR improvement | +10% |
| Dual-Level | Answer completeness (eval) | +15% |
| Parent-Child | Context relevance | +20% |
| Feedback Loop | Retrieval improvement over time | +5% per month |

---

## References

- [Mem0 Documentation](https://docs.mem0.ai/)
- [Zep Memory Architecture](https://arxiv.org/abs/2501.13956)
- [MS GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [LightRAG Paper](https://arxiv.org/abs/2410.05779)
- [Cognee Documentation](https://docs.cognee.ai/)
- [RAGFlow Documentation](https://ragflow.io/docs/)
- [Qdrant BM42](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors)
- [ColBERT Paper](https://arxiv.org/abs/2004.12832)
- `_bmad-output/architecture.md`
- `_bmad-output/project-context.md`
- `docs/roadmap-decisions-2026-01-03.md`
