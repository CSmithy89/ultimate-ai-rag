"""ScopedMemoryStore for Epic 20 Memory Platform.

Provides CRUD operations for scoped memories with:
- PostgreSQL + pgvector for persistent storage and similarity search
- Redis for hot cache optimization
- Optional Graphiti integration for graph-based relationships
"""

import inspect
import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.db.redis import RedisClient

from .errors import MemoryLimitExceededError, MemoryScopeError
from .models import MemoryScope, ScopedMemory
from .scopes import get_scopes_to_search, validate_scope_context

logger = structlog.get_logger(__name__)

# Expected embedding dimension for pgvector column (matches alembic migration)
# Must match the vector(1536) column definition in PostgreSQL
EXPECTED_EMBEDDING_DIMENSION = 1536


class ScopedMemoryStore:
    """Store and retrieve memories with scope-aware queries.

    Implements hierarchical memory scopes (user, session, agent, global)
    with PostgreSQL for persistent storage and Redis for caching.

    Attributes:
        _postgres: PostgreSQL client for persistent storage
        _redis: Redis client for caching
        _cache_ttl_seconds: TTL for cached memories
        _max_per_scope: Maximum memories per scope
        _embedding_generator: Optional embedding generator for similarity search
    """

    def __init__(
        self,
        postgres_client: PostgresClient,
        redis_client: Optional[RedisClient] = None,
        graphiti_client: Optional[Any] = None,  # GraphitiClient type
        embedding_provider: str = "openai",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        cache_ttl_seconds: int = 3600,
        max_per_scope: int = 10000,
        embedding_dimension: int = 1536,
    ) -> None:
        """Initialize the scoped memory store.

        Args:
            postgres_client: PostgreSQL client for persistent storage
            redis_client: Optional Redis client for caching
            graphiti_client: Optional Graphiti client for graph storage
            embedding_provider: Embedding provider name
            embedding_api_key: API key for embedding provider
            embedding_base_url: Base URL for embedding provider
            embedding_model: Embedding model name
            cache_ttl_seconds: TTL for Redis cache entries
            max_per_scope: Maximum memories allowed per scope
            embedding_dimension: Dimension of embedding vectors (default: 1536)
        """
        self._postgres = postgres_client
        self._redis = redis_client
        self._graphiti = graphiti_client
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_per_scope = max_per_scope
        self._embedding_dimension = embedding_dimension

        # Initialize embedding generator if credentials provided
        self._embedding_generator = None
        if embedding_api_key:
            try:
                from agentic_rag_backend.embeddings import EmbeddingGenerator
                from agentic_rag_backend.llm.providers import get_embedding_adapter

                adapter = get_embedding_adapter(
                    provider=embedding_provider,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    model=embedding_model,
                )
                self._embedding_generator = EmbeddingGenerator.from_adapter(adapter)
                logger.info(
                    "memory_store_embedding_initialized",
                    provider=embedding_provider,
                    model=embedding_model,
                )
            except Exception as e:
                logger.warning(
                    "memory_store_embedding_init_failed",
                    error=str(e),
                    hint="Memories will be stored without embeddings",
                )

    async def add_memory(
        self,
        content: str,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        importance: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ScopedMemory:
        """Add a memory with specified scope.

        Args:
            content: The memory content
            scope: Memory scope level
            tenant_id: Tenant identifier (always required)
            user_id: User identifier (required for USER scope)
            session_id: Session identifier (required for SESSION scope)
            agent_id: Agent identifier (required for AGENT scope)
            importance: Importance score for consolidation (0.0-1.0)
            metadata: Additional metadata

        Returns:
            The created ScopedMemory

        Raises:
            MemoryScopeError: If required context is missing for scope
            MemoryLimitExceededError: If scope limit is exceeded
        """
        # Validate scope requirements
        is_valid, error_msg = validate_scope_context(
            scope, user_id, session_id, agent_id
        )
        if not is_valid:
            raise MemoryScopeError(scope.value, error_msg or "Invalid scope context")

        # Check scope limit
        current_count = await self._get_scope_count(
            tenant_id=tenant_id,
            scope=scope,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        if current_count >= self._max_per_scope:
            raise MemoryLimitExceededError(scope.value, self._max_per_scope, current_count)

        # Generate embedding if available
        embedding = None
        if self._embedding_generator:
            try:
                embedding = await self._embedding_generator.generate_embedding(content)
            except Exception as e:
                logger.warning(
                    "memory_embedding_generation_failed",
                    error=str(e),
                    content_length=len(content),
                )

        now = datetime.now(timezone.utc)
        memory_id = uuid4()

        memory_metadata = dict(metadata or {})
        graphiti_episode_id = await self._maybe_add_graphiti_episode(
            tenant_id=tenant_id,
            memory_id=str(memory_id),
            content=content,
            scope=scope,
            created_at=now,
        )
        if graphiti_episode_id:
            memory_metadata.setdefault("graphiti_episode_id", graphiti_episode_id)

        # Store in PostgreSQL
        memory = await self._store_in_postgres(
            memory_id=memory_id,
            content=content,
            scope=scope,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            importance=importance,
            metadata=memory_metadata,
            embedding=embedding,
            created_at=now,
            accessed_at=now,
        )

        # Cache in Redis
        if self._redis:
            await self._cache_memory(memory)

        logger.info(
            "memory_added",
            memory_id=str(memory.id),
            scope=scope.value,
            tenant_id=tenant_id,
            has_embedding=embedding is not None,
        )

        return memory

    async def get_memory(
        self,
        memory_id: str,
        tenant_id: str,
    ) -> Optional[ScopedMemory]:
        """Get a memory by ID with tenant filtering.

        Args:
            memory_id: Memory UUID
            tenant_id: Tenant identifier for access control

        Returns:
            ScopedMemory if found, None otherwise
        """
        # Try cache first
        if self._redis:
            cached = await self._get_cached_memory(tenant_id, memory_id)
            if cached:
                # Update access stats asynchronously
                await self._update_access_stats(memory_id, tenant_id)
                return cached

        # Query PostgreSQL
        memory = await self._get_from_postgres(memory_id, tenant_id)
        if memory:
            # Update access stats
            await self._update_access_stats(memory_id, tenant_id)
            # Update cache
            if self._redis:
                await self._cache_memory(memory)

        return memory

    async def list_memories(
        self,
        tenant_id: str,
        scope: Optional[MemoryScope] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ScopedMemory], int]:
        """List memories with optional filtering.

        Args:
            tenant_id: Tenant identifier
            scope: Optional scope filter
            user_id: Optional user filter
            session_id: Optional session filter
            agent_id: Optional agent filter
            limit: Maximum results to return (capped at 100)
            offset: Pagination offset (capped at 10000)

        Returns:
            Tuple of (list of memories, total count)
        """
        # Defense-in-depth: enforce limits regardless of caller
        # API layer validates via Pydantic, but internal callers may bypass
        MAX_LIMIT = 100
        MAX_OFFSET = 10000

        if limit > MAX_LIMIT:
            logger.warning(
                "list_memories_limit_exceeded",
                requested=limit,
                max_allowed=MAX_LIMIT,
                tenant_id=tenant_id,
            )
            limit = MAX_LIMIT

        if offset > MAX_OFFSET:
            logger.warning(
                "list_memories_offset_exceeded",
                requested=offset,
                max_allowed=MAX_OFFSET,
                tenant_id=tenant_id,
            )
            offset = MAX_OFFSET

        if scope:
            is_valid, error_msg = validate_scope_context(
                scope, user_id, session_id, agent_id
            )
            if not is_valid:
                raise MemoryScopeError(scope.value, error_msg or "Invalid scope context")

        return await self._list_from_postgres(
            tenant_id=tenant_id,
            scope=scope,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
            offset=offset,
        )

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
    ) -> tuple[list[ScopedMemory], list[MemoryScope]]:
        """Search memories within scope hierarchy.

        If include_parent_scopes is True, searches up the hierarchy:
        - SESSION scope includes USER and GLOBAL memories
        - USER scope includes GLOBAL memories
        - AGENT scope includes GLOBAL memories

        Args:
            query: Search query text (1-5000 characters)
            scope: Starting scope level
            tenant_id: Tenant identifier
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            limit: Maximum results to return
            include_parent_scopes: Whether to include parent scopes

        Returns:
            Tuple of (list of matching memories, list of scopes searched)

        Raises:
            ValueError: If query is empty or exceeds 5000 characters
        """
        # Validate query length to prevent DoS
        MAX_QUERY_LENGTH = 5000
        if not query or len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query must be 1-{MAX_QUERY_LENGTH} characters")

        is_valid, error_msg = validate_scope_context(
            scope, user_id, session_id, agent_id
        )
        if not is_valid:
            raise MemoryScopeError(scope.value, error_msg or "Invalid scope context")

        scopes_to_search = get_scopes_to_search(scope, include_parent_scopes)

        # Generate query embedding for similarity search
        query_embedding = None
        if self._embedding_generator:
            try:
                query_embedding = await self._embedding_generator.generate_embedding(query)
            except Exception as e:
                logger.warning(
                    "memory_search_embedding_failed",
                    error=str(e),
                    query_length=len(query),
                )

        # Search PostgreSQL with embedding similarity
        memories = await self._search_in_postgres(
            query=query,
            query_embedding=query_embedding,
            scopes=scopes_to_search,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
        )

        # Update access stats for returned memories (batch to avoid N+1 updates)
        if memories:
            await self._update_access_stats_bulk(
                [str(memory.id) for memory in memories],
                tenant_id,
            )

        return memories, scopes_to_search

    async def update_memory(
        self,
        memory_id: str,
        tenant_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        access_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ScopedMemory]:
        """Update a memory's content, importance, or metadata.

        Args:
            memory_id: Memory UUID
            tenant_id: Tenant identifier
            content: New content (if updating)
            importance: New importance score (if updating)
            access_count: New access count (if updating)
            metadata: New metadata (replaces existing if provided)

        Returns:
            Updated ScopedMemory if found, None otherwise
        """
        # Get existing memory
        existing = await self._get_from_postgres(memory_id, tenant_id)
        if not existing:
            return None

        # Generate new embedding if content changed
        embedding = None
        if content and self._embedding_generator:
            try:
                embedding = await self._embedding_generator.generate_embedding(content)
            except Exception as e:
                logger.warning(
                    "memory_update_embedding_failed",
                    error=str(e),
                )

        # Update in PostgreSQL
        memory = await self._update_in_postgres(
            memory_id=memory_id,
            tenant_id=tenant_id,
            content=content,
            importance=importance,
            access_count=access_count,
            metadata=metadata,
            embedding=embedding,
        )

        # Invalidate and update cache
        if self._redis and memory:
            await self._invalidate_cache_entry(tenant_id, memory_id)
            await self._cache_memory(memory)

        logger.info(
            "memory_updated",
            memory_id=memory_id,
            tenant_id=tenant_id,
            content_updated=content is not None,
            importance_updated=importance is not None,
            access_count_updated=access_count is not None,
            metadata_updated=metadata is not None,
        )

        return memory

    async def delete_memory(
        self,
        memory_id: str,
        tenant_id: str,
    ) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: Memory UUID
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        graphiti_episode_id = None
        if self._graphiti and getattr(self._graphiti, "is_connected", False):
            existing = await self._get_from_postgres(memory_id, tenant_id)
            if existing and isinstance(existing.metadata, dict):
                graphiti_episode_id = existing.metadata.get("graphiti_episode_id")

        deleted = await self._delete_from_postgres(memory_id, tenant_id)

        if deleted and self._redis:
            await self._invalidate_cache_entry(tenant_id, memory_id)

        if deleted and graphiti_episode_id:
            await self._maybe_delete_graphiti_episode(
                tenant_id=tenant_id,
                episode_id=graphiti_episode_id,
            )

        if deleted:
            logger.info("memory_deleted", memory_id=memory_id, tenant_id=tenant_id)

        return deleted

    async def delete_memories_by_scope(
        self,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Delete all memories in a scope.

        Useful for cleaning up session memories when a session ends,
        or clearing user memories on account deletion.

        Args:
            scope: Scope to clear
            tenant_id: Tenant identifier
            user_id: User identifier (for USER/SESSION scope)
            session_id: Session identifier (for SESSION scope)
            agent_id: Agent identifier (for AGENT scope)

        Returns:
            Count of deleted memories
        """
        is_valid, error_msg = validate_scope_context(
            scope, user_id, session_id, agent_id
        )
        if not is_valid:
            raise MemoryScopeError(scope.value, error_msg or "Invalid scope context")

        graphiti_episode_ids: list[str] = []
        if self._graphiti and getattr(self._graphiti, "is_connected", False):
            graphiti_episode_ids = await self._collect_graphiti_episode_ids(
                tenant_id=tenant_id,
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
            )

        count = await self._delete_scope_from_postgres(
            scope=scope,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )

        # Invalidate cache for scope
        if self._redis:
            await self._invalidate_scope_cache(
                tenant_id=tenant_id,
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
            )

        if count and graphiti_episode_ids:
            for episode_id in graphiti_episode_ids:
                await self._maybe_delete_graphiti_episode(
                    tenant_id=tenant_id,
                    episode_id=episode_id,
                )

        logger.info(
            "memories_deleted_by_scope",
            scope=scope.value,
            count=count,
            tenant_id=tenant_id,
        )

        return count

    # ------------ PostgreSQL Operations ------------

    async def _store_in_postgres(
        self,
        memory_id: UUID,
        content: str,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        importance: float,
        metadata: dict[str, Any],
        embedding: Optional[list[float]],
        created_at: datetime,
        accessed_at: datetime,
    ) -> ScopedMemory:
        """Store a memory in PostgreSQL."""
        async with self._postgres.pool.acquire() as conn:
            # Convert embedding to pgvector format with dimension validation
            embedding_str = None
            if embedding:
                # Validate embedding dimension matches pgvector column
                if len(embedding) != self._embedding_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, "
                        f"expected {self._embedding_dimension}. "
                        "Ensure embedding model matches database schema."
                    )
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

            await conn.execute(
                """
                INSERT INTO scoped_memories (
                    id, tenant_id, scope, user_id, session_id, agent_id,
                    content, importance, metadata, embedding, created_at, accessed_at
                )
                VALUES ($1, $2::uuid, $3, $4::uuid, $5::uuid, $6, $7, $8, $9, $10::vector, $11, $12)
                """,
                memory_id,
                UUID(tenant_id),
                scope.value,
                UUID(user_id) if user_id else None,
                UUID(session_id) if session_id else None,
                agent_id,
                content,
                importance,
                json.dumps(metadata),
                embedding_str,
                created_at,
                accessed_at,
            )

        return ScopedMemory(
            id=memory_id,
            content=content,
            scope=scope,
            tenant_id=UUID(tenant_id),
            user_id=UUID(user_id) if user_id else None,
            session_id=UUID(session_id) if session_id else None,
            agent_id=agent_id,
            importance=importance,
            metadata=metadata,
            created_at=created_at,
            accessed_at=accessed_at,
            access_count=0,
            embedding=embedding,
        )

    async def _get_from_postgres(
        self,
        memory_id: str,
        tenant_id: str,
    ) -> Optional[ScopedMemory]:
        """Get a memory from PostgreSQL."""
        async with self._postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, tenant_id, scope, user_id, session_id, agent_id,
                       content, importance, metadata, created_at, accessed_at, access_count
                FROM scoped_memories
                WHERE id = $1 AND tenant_id = $2
                """,
                UUID(memory_id),
                UUID(tenant_id),
            )
            if not row:
                return None

            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            return ScopedMemory(
                id=row["id"],
                content=row["content"],
                scope=MemoryScope(row["scope"]),
                tenant_id=row["tenant_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                agent_id=row["agent_id"],
                importance=row["importance"],
                metadata=metadata or {},
                created_at=row["created_at"],
                accessed_at=row["accessed_at"],
                access_count=row["access_count"],
            )

    async def get_memory_embeddings(
        self,
        tenant_id: str,
        memory_ids: list[str],
    ) -> dict[str, list[float]]:
        """Fetch embeddings for a set of memories in a single query."""
        if not memory_ids:
            return {}
        
        # Prevent memory exhaustion with large batches
        if len(memory_ids) > 1000:
            raise ValueError(f"Batch size {len(memory_ids)} exceeds maximum of 1000")

        async with self._postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, embedding
                FROM scoped_memories
                WHERE tenant_id = $1 AND id = ANY($2::uuid[])
                """,
                UUID(tenant_id),
                [UUID(memory_id) for memory_id in memory_ids],
            )

        embeddings: dict[str, list[float]] = {}
        for row in rows:
            embedding = row["embedding"]
            if embedding is None:
                continue
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            elif isinstance(embedding, tuple):
                embedding = list(embedding)
            elif not isinstance(embedding, list):
                try:
                    embedding = list(embedding)
                except TypeError:
                    continue

            embeddings[str(row["id"])] = embedding

        return embeddings

    async def _list_from_postgres(
        self,
        tenant_id: str,
        scope: Optional[MemoryScope],
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        limit: int,
        offset: int,
    ) -> tuple[list[ScopedMemory], int]:
        """List memories from PostgreSQL with filtering."""
        async with self._postgres.pool.acquire() as conn:
            # Build WHERE clause using parameterized queries
            # SECURITY NOTE: Column names are hardcoded (whitelisted) below,
            # and all values use parameterized placeholders ($N) - this is safe.
            # Allowed filter columns (whitelist for explicit security):
            _ALLOWED_COLUMNS = frozenset({"scope", "user_id", "session_id", "agent_id"})

            conditions = ["tenant_id = $1"]
            params: list[Any] = [UUID(tenant_id)]
            param_idx = 2

            # Build filter conditions using whitelisted columns only
            filter_map = {
                "scope": (scope, lambda v: v.value) if scope else None,
                "user_id": (user_id, lambda v: UUID(v)) if user_id else None,
                "session_id": (session_id, lambda v: UUID(v)) if session_id else None,
                "agent_id": (agent_id, lambda v: v) if agent_id else None,
            }

            for column, filter_data in filter_map.items():
                if filter_data is not None and column in _ALLOWED_COLUMNS:
                    value, transform = filter_data
                    conditions.append(f"{column} = ${param_idx}")
                    params.append(transform(value))
                    param_idx += 1

            where_clause = " AND ".join(conditions)

            # Get total count
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) as total FROM scoped_memories WHERE {where_clause}",
                *params,
            )
            total = count_row["total"] if count_row else 0

            # Get paginated results
            params.extend([limit, offset])
            rows = await conn.fetch(
                f"""
                SELECT id, tenant_id, scope, user_id, session_id, agent_id,
                       content, importance, metadata, created_at, accessed_at, access_count
                FROM scoped_memories
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params,
            )

            memories = []
            for row in rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                memories.append(
                    ScopedMemory(
                        id=row["id"],
                        content=row["content"],
                        scope=MemoryScope(row["scope"]),
                        tenant_id=row["tenant_id"],
                        user_id=row["user_id"],
                        session_id=row["session_id"],
                        agent_id=row["agent_id"],
                        importance=row["importance"],
                        metadata=metadata or {},
                        created_at=row["created_at"],
                        accessed_at=row["accessed_at"],
                        access_count=row["access_count"],
                    )
                )

            return memories, total

    async def _search_in_postgres(
        self,
        query: str,
        query_embedding: Optional[list[float]],
        scopes: list[MemoryScope],
        tenant_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        limit: int,
    ) -> list[ScopedMemory]:
        """Search memories in PostgreSQL using embedding similarity."""
        async with self._postgres.pool.acquire() as conn:
            scope_values = [s.value for s in scopes]

            if query_embedding:
                # Vector similarity search
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

                # Build scope conditions for context IDs
                # For each scope, we need to match the appropriate context
                scope_conditions = []
                params: list[Any] = [UUID(tenant_id), embedding_str]
                param_idx = 3

                for scope_val in scope_values:
                    if scope_val == "session" and session_id:
                        scope_conditions.append(
                            f"(scope = 'session' AND session_id = ${param_idx})"
                        )
                        params.append(UUID(session_id))
                        param_idx += 1
                    elif scope_val == "user" and user_id:
                        scope_conditions.append(
                            f"(scope = 'user' AND user_id = ${param_idx})"
                        )
                        params.append(UUID(user_id))
                        param_idx += 1
                    elif scope_val == "agent" and agent_id:
                        scope_conditions.append(
                            f"(scope = 'agent' AND agent_id = ${param_idx})"
                        )
                        params.append(agent_id)
                        param_idx += 1
                    elif scope_val == "global":
                        scope_conditions.append("scope = 'global'")

                if not scope_conditions:
                    return []

                scope_clause = " OR ".join(scope_conditions)
                params.append(limit)

                rows = await conn.fetch(
                    f"""
                    SELECT id, tenant_id, scope, user_id, session_id, agent_id,
                           content, importance, metadata, created_at, accessed_at, access_count,
                           1 - (embedding <=> $2::vector) as similarity
                    FROM scoped_memories
                    WHERE tenant_id = $1
                      AND embedding IS NOT NULL
                      AND ({scope_clause})
                    ORDER BY embedding <=> $2::vector
                    LIMIT ${param_idx}
                    """,
                    *params,
                )
            else:
                # Fallback to text search (ILIKE)
                scope_conditions = []
                params = [UUID(tenant_id), f"%{query}%"]
                param_idx = 3

                for scope_val in scope_values:
                    if scope_val == "session" and session_id:
                        scope_conditions.append(
                            f"(scope = 'session' AND session_id = ${param_idx})"
                        )
                        params.append(UUID(session_id))
                        param_idx += 1
                    elif scope_val == "user" and user_id:
                        scope_conditions.append(
                            f"(scope = 'user' AND user_id = ${param_idx})"
                        )
                        params.append(UUID(user_id))
                        param_idx += 1
                    elif scope_val == "agent" and agent_id:
                        scope_conditions.append(
                            f"(scope = 'agent' AND agent_id = ${param_idx})"
                        )
                        params.append(agent_id)
                        param_idx += 1
                    elif scope_val == "global":
                        scope_conditions.append("scope = 'global'")

                if not scope_conditions:
                    return []

                scope_clause = " OR ".join(scope_conditions)
                params.append(limit)

                rows = await conn.fetch(
                    f"""
                    SELECT id, tenant_id, scope, user_id, session_id, agent_id,
                           content, importance, metadata, created_at, accessed_at, access_count
                    FROM scoped_memories
                    WHERE tenant_id = $1
                      AND content ILIKE $2
                      AND ({scope_clause})
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ${param_idx}
                    """,
                    *params,
                )

            memories = []
            for row in rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                memories.append(
                    ScopedMemory(
                        id=row["id"],
                        content=row["content"],
                        scope=MemoryScope(row["scope"]),
                        tenant_id=row["tenant_id"],
                        user_id=row["user_id"],
                        session_id=row["session_id"],
                        agent_id=row["agent_id"],
                        importance=row["importance"],
                        metadata=metadata or {},
                        created_at=row["created_at"],
                        accessed_at=row["accessed_at"],
                        access_count=row["access_count"],
                    )
                )

            return memories

    async def _update_in_postgres(
        self,
        memory_id: str,
        tenant_id: str,
        content: Optional[str],
        importance: Optional[float],
        access_count: Optional[int],
        metadata: Optional[dict[str, Any]],
        embedding: Optional[list[float]],
    ) -> Optional[ScopedMemory]:
        """Update a memory in PostgreSQL."""
        async with self._postgres.pool.acquire() as conn:
            # Build SET clause dynamically
            updates = ["accessed_at = NOW()"]
            params: list[Any] = []
            param_idx = 1

            if content is not None:
                updates.append(f"content = ${param_idx}")
                params.append(content)
                param_idx += 1

            if importance is not None:
                updates.append(f"importance = ${param_idx}")
                params.append(importance)
                param_idx += 1

            if access_count is not None:
                updates.append(f"access_count = ${param_idx}")
                params.append(access_count)
                param_idx += 1

            if metadata is not None:
                updates.append(f"metadata = ${param_idx}")
                params.append(json.dumps(metadata))
                param_idx += 1

            if embedding is not None:
                # Validate embedding dimension matches pgvector column
                if len(embedding) != self._embedding_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, "
                        f"expected {self._embedding_dimension}. "
                        "Ensure embedding model matches database schema."
                    )
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                updates.append(f"embedding = ${param_idx}::vector")
                params.append(embedding_str)
                param_idx += 1

            params.extend([UUID(memory_id), UUID(tenant_id)])
            set_clause = ", ".join(updates)

            row = await conn.fetchrow(
                f"""
                UPDATE scoped_memories
                SET {set_clause}
                WHERE id = ${param_idx} AND tenant_id = ${param_idx + 1}
                RETURNING id, tenant_id, scope, user_id, session_id, agent_id,
                          content, importance, metadata, created_at, accessed_at, access_count
                """,
                *params,
            )

            if not row:
                return None

            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)

            return ScopedMemory(
                id=row["id"],
                content=row["content"],
                scope=MemoryScope(row["scope"]),
                tenant_id=row["tenant_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                agent_id=row["agent_id"],
                importance=row["importance"],
                metadata=meta or {},
                created_at=row["created_at"],
                accessed_at=row["accessed_at"],
                access_count=row["access_count"],
            )

    async def _delete_from_postgres(
        self,
        memory_id: str,
        tenant_id: str,
    ) -> bool:
        """Delete a memory from PostgreSQL."""
        async with self._postgres.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM scoped_memories
                WHERE id = $1 AND tenant_id = $2
                """,
                UUID(memory_id),
                UUID(tenant_id),
            )
            return result == "DELETE 1"

    async def _delete_scope_from_postgres(
        self,
        scope: MemoryScope,
        tenant_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> int:
        """Delete memories by scope from PostgreSQL."""
        async with self._postgres.pool.acquire() as conn:
            conditions = ["tenant_id = $1", "scope = $2"]
            params: list[Any] = [UUID(tenant_id), scope.value]
            param_idx = 3

            if scope == MemoryScope.USER and user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(UUID(user_id))
            elif scope == MemoryScope.SESSION and session_id:
                conditions.append(f"session_id = ${param_idx}")
                params.append(UUID(session_id))
            elif scope == MemoryScope.AGENT and agent_id:
                conditions.append(f"agent_id = ${param_idx}")
                params.append(agent_id)

            where_clause = " AND ".join(conditions)
            result = await conn.execute(
                f"DELETE FROM scoped_memories WHERE {where_clause}",
                *params,
            )

            # Parse "DELETE N" result
            count = int(result.split(" ")[1]) if result else 0
            return count

    async def _get_scope_count(
        self,
        tenant_id: str,
        scope: MemoryScope,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> int:
        """Get count of memories in a scope."""
        async with self._postgres.pool.acquire() as conn:
            conditions = ["tenant_id = $1", "scope = $2"]
            params: list[Any] = [UUID(tenant_id), scope.value]
            param_idx = 3

            if scope == MemoryScope.USER and user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(UUID(user_id))
            elif scope == MemoryScope.SESSION and session_id:
                conditions.append(f"session_id = ${param_idx}")
                params.append(UUID(session_id))
            elif scope == MemoryScope.AGENT and agent_id:
                conditions.append(f"agent_id = ${param_idx}")
                params.append(agent_id)

            where_clause = " AND ".join(conditions)
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM scoped_memories WHERE {where_clause}",
                *params,
            )
            return row["count"] if row else 0

    async def _update_access_stats(
        self,
        memory_id: str,
        tenant_id: str,
    ) -> None:
        """Update access timestamp and count for a memory."""
        try:
            async with self._postgres.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE scoped_memories
                    SET accessed_at = NOW(), access_count = access_count + 1
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    UUID(memory_id),
                    UUID(tenant_id),
                )
        except Exception as e:
            logger.warning(
                "memory_access_stats_update_failed",
                memory_id=memory_id,
                error=str(e),
            )

    async def _update_access_stats_bulk(
        self,
        memory_ids: list[str],
        tenant_id: str,
    ) -> None:
        """Batch update access stats for a list of memories."""
        if not memory_ids:
            return

        try:
            async with self._postgres.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE scoped_memories
                    SET accessed_at = NOW(), access_count = access_count + 1
                    WHERE tenant_id = $1 AND id = ANY($2::uuid[])
                    """,
                    UUID(tenant_id),
                    [UUID(memory_id) for memory_id in memory_ids],
                )
        except Exception as e:
            logger.warning(
                "memory_access_stats_bulk_update_failed",
                memory_ids=len(memory_ids),
                error=str(e),
            )

    async def _maybe_add_graphiti_episode(
        self,
        tenant_id: str,
        memory_id: str,
        content: str,
        scope: MemoryScope,
        created_at: datetime,
    ) -> Optional[str]:
        """Best-effort Graphiti ingestion for a memory entry."""
        if not self._graphiti or not getattr(self._graphiti, "is_connected", False):
            return None

        content_stripped = content.strip()
        if len(content_stripped) < 10:
            return None

        try:
            episode = await self._graphiti.client.add_episode(
                name=f"Memory {memory_id}",
                episode_body=content,
                source_description=f"scoped_memory:{scope.value}",
                reference_time=created_at,
                group_id=tenant_id,
            )
            episode_id = str(getattr(episode, "uuid", "")) or None
            if episode_id:
                logger.info(
                    "memory_graphiti_episode_created",
                    memory_id=memory_id,
                    episode_id=episode_id,
                    tenant_id=tenant_id,
                )
            return episode_id
        except Exception as e:
            logger.warning(
                "memory_graphiti_ingestion_failed",
                memory_id=memory_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return None

    async def _maybe_delete_graphiti_episode(
        self,
        tenant_id: str,
        episode_id: str,
    ) -> None:
        """Best-effort Graphiti cleanup for a memory entry."""
        if not self._graphiti or not getattr(self._graphiti, "is_connected", False):
            return

        try:
            await self._graphiti.client.delete_episode(episode_id)
            logger.info(
                "memory_graphiti_episode_deleted",
                episode_id=episode_id,
                tenant_id=tenant_id,
            )
        except Exception as e:
            logger.warning(
                "memory_graphiti_delete_failed",
                episode_id=episode_id,
                tenant_id=tenant_id,
                error=str(e),
            )

    async def _collect_graphiti_episode_ids(
        self,
        tenant_id: str,
        scope: MemoryScope,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> list[str]:
        """Collect Graphiti episode IDs for a scoped delete."""
        episode_ids: list[str] = []
        limit = 200
        offset = 0

        while True:
            memories, _ = await self._list_from_postgres(
                tenant_id=tenant_id,
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                limit=limit,
                offset=offset,
            )
            if not memories:
                break

            for memory in memories:
                if isinstance(memory.metadata, dict):
                    episode_id = memory.metadata.get("graphiti_episode_id")
                    if episode_id:
                        episode_ids.append(episode_id)

            offset += len(memories)
            if len(memories) < limit:
                break

        return episode_ids

    # ------------ Redis Cache Operations ------------

    async def _cache_memory(self, memory: ScopedMemory) -> None:
        """Cache a memory in Redis."""
        if not self._redis:
            return

        cache_key = f"memory:{memory.tenant_id}:{memory.id}"
        try:
            # Serialize without embedding for cache efficiency
            cache_data = memory.model_dump(mode="json", exclude={"embedding"})
            await self._redis.client.setex(
                cache_key,
                self._cache_ttl_seconds,
                json.dumps(cache_data),
            )
            logger.debug("memory_cached", memory_id=str(memory.id), key=cache_key)
        except Exception as e:
            logger.warning("memory_cache_failed", error=str(e))

    async def _get_cached_memory(
        self, tenant_id: str, memory_id: str
    ) -> Optional[ScopedMemory]:
        """Get a memory from Redis cache."""
        if not self._redis:
            return None

        cache_key = f"memory:{tenant_id}:{memory_id}"
        try:
            data = await self._redis.client.get(cache_key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                parsed = json.loads(data)
                return ScopedMemory.model_validate(parsed)
        except Exception as e:
            logger.warning("memory_cache_get_failed", error=str(e))
        return None

    async def _invalidate_cache_entry(self, tenant_id: str, memory_id: str) -> None:
        """Invalidate a single cache entry."""
        if not self._redis:
            return

        cache_key = f"memory:{tenant_id}:{memory_id}"
        try:
            await self._redis.client.delete(cache_key)
        except Exception as e:
            logger.warning("memory_cache_invalidate_failed", error=str(e))

    async def _invalidate_scope_cache(
        self,
        tenant_id: str,
        scope: MemoryScope,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> None:
        """Invalidate all cached memories for a scope.

        Uses Redis pipeline for efficient batch deletion instead of deleting
        keys one at a time.
        """
        if not self._redis:
            return

        # Use pattern matching to find relevant keys
        pattern = f"memory:{tenant_id}:*"
        try:
            # Collect keys in batches and delete using pipeline
            keys_to_delete: list[str] = []
            batch_size = 100

            scan_iter = self._redis.client.scan_iter(match=pattern)
            if inspect.isawaitable(scan_iter):
                scan_iter = await scan_iter

            if hasattr(scan_iter, "__aiter__"):
                async for key in scan_iter:
                    keys_to_delete.append(key)

                    # Process in batches to avoid memory issues with large key sets
                    if len(keys_to_delete) >= batch_size:
                        async with self._redis.client.pipeline(transaction=False) as pipe:
                            for k in keys_to_delete:
                                pipe.delete(k)
                            await pipe.execute()
                        keys_to_delete = []
            else:
                for key in scan_iter:
                    keys_to_delete.append(key)

                    # Process in batches to avoid memory issues with large key sets
                    if len(keys_to_delete) >= batch_size:
                        async with self._redis.client.pipeline(transaction=False) as pipe:
                            for k in keys_to_delete:
                                pipe.delete(k)
                            await pipe.execute()
                        keys_to_delete = []

            # Delete remaining keys
            if keys_to_delete:
                async with self._redis.client.pipeline(transaction=False) as pipe:
                    for k in keys_to_delete:
                        pipe.delete(k)
                    await pipe.execute()

            logger.debug(
                "memory_scope_cache_invalidated",
                scope=scope.value,
                tenant_id=tenant_id,
            )
        except Exception as e:
            logger.warning("memory_scope_cache_invalidate_failed", error=str(e))
