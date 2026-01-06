"""Unit tests for ScopedMemoryStore (Story 20-A1).

Tests the memory store operations:
- Create, read, update, delete memories
- Scope validation and enforcement
- Limit checks
- Cache operations (mocked)
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agentic_rag_backend.memory.errors import (
    MemoryLimitExceededError,
    MemoryScopeError,
)
from agentic_rag_backend.memory.models import MemoryScope
from agentic_rag_backend.memory.store import ScopedMemoryStore


@pytest.fixture
def mock_postgres():
    """Create a mock PostgreSQL client."""
    postgres = MagicMock()
    postgres.pool = MagicMock()

    # Create async context manager mock for pool.acquire()
    conn_mock = AsyncMock()
    conn_mock.execute = AsyncMock()
    conn_mock.fetchrow = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=[])

    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=conn_mock)
    async_cm.__aexit__ = AsyncMock(return_value=None)

    postgres.pool.acquire = MagicMock(return_value=async_cm)
    postgres._conn_mock = conn_mock  # Store for test access

    return postgres


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.client = AsyncMock()
    redis.client.get = AsyncMock(return_value=None)
    redis.client.setex = AsyncMock()
    redis.client.delete = AsyncMock()
    redis.client.scan_iter = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def store(mock_postgres, mock_redis):
    """Create a ScopedMemoryStore with mocked dependencies."""
    return ScopedMemoryStore(
        postgres_client=mock_postgres,
        redis_client=mock_redis,
        cache_ttl_seconds=3600,
        max_per_scope=100,
    )


@pytest.fixture
def sample_tenant_id():
    """Sample tenant UUID."""
    return str(uuid4())


@pytest.fixture
def sample_user_id():
    """Sample user UUID."""
    return str(uuid4())


@pytest.fixture
def sample_session_id():
    """Sample session UUID."""
    return str(uuid4())


class TestAddMemory:
    """Test add_memory functionality."""

    @pytest.mark.asyncio
    async def test_add_user_scope_memory(self, store, mock_postgres, sample_tenant_id, sample_user_id):
        """Can add a USER scope memory with valid context."""
        # Mock scope count
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})

        memory = await store.add_memory(
            content="User preferences test",
            scope=MemoryScope.USER,
            tenant_id=sample_tenant_id,
            user_id=sample_user_id,
        )

        assert memory.content == "User preferences test"
        assert memory.scope == MemoryScope.USER
        assert memory.tenant_id == UUID(sample_tenant_id)
        assert memory.user_id == UUID(sample_user_id)
        assert memory.importance == 1.0

    @pytest.mark.asyncio
    async def test_add_session_scope_memory(self, store, mock_postgres, sample_tenant_id, sample_session_id):
        """Can add a SESSION scope memory with valid context."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})

        memory = await store.add_memory(
            content="Session context test",
            scope=MemoryScope.SESSION,
            tenant_id=sample_tenant_id,
            session_id=sample_session_id,
        )

        assert memory.scope == MemoryScope.SESSION
        assert memory.session_id == UUID(sample_session_id)

    @pytest.mark.asyncio
    async def test_add_agent_scope_memory(self, store, mock_postgres, sample_tenant_id):
        """Can add an AGENT scope memory with valid context."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})

        memory = await store.add_memory(
            content="Agent operational memory",
            scope=MemoryScope.AGENT,
            tenant_id=sample_tenant_id,
            agent_id="orchestrator-agent",
        )

        assert memory.scope == MemoryScope.AGENT
        assert memory.agent_id == "orchestrator-agent"

    @pytest.mark.asyncio
    async def test_add_global_scope_memory(self, store, mock_postgres, sample_tenant_id):
        """Can add a GLOBAL scope memory without user/session context."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})

        memory = await store.add_memory(
            content="Global shared knowledge",
            scope=MemoryScope.GLOBAL,
            tenant_id=sample_tenant_id,
        )

        assert memory.scope == MemoryScope.GLOBAL
        assert memory.user_id is None
        assert memory.session_id is None
        assert memory.agent_id is None

    @pytest.mark.asyncio
    async def test_add_memory_with_custom_importance(self, store, mock_postgres, sample_tenant_id):
        """Can add a memory with custom importance score."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})

        memory = await store.add_memory(
            content="High importance memory",
            scope=MemoryScope.GLOBAL,
            tenant_id=sample_tenant_id,
            importance=0.9,
        )

        assert memory.importance == 0.9

    @pytest.mark.asyncio
    async def test_add_memory_with_metadata(self, store, mock_postgres, sample_tenant_id):
        """Can add a memory with metadata."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})
        metadata = {"source": "user_profile", "category": "preferences"}

        memory = await store.add_memory(
            content="Memory with metadata",
            scope=MemoryScope.GLOBAL,
            tenant_id=sample_tenant_id,
            metadata=metadata,
        )

        assert memory.metadata == metadata

    @pytest.mark.asyncio
    async def test_add_user_scope_without_user_id_raises(self, store, sample_tenant_id):
        """Adding USER scope memory without user_id raises MemoryScopeError."""
        with pytest.raises(MemoryScopeError) as exc_info:
            await store.add_memory(
                content="Test",
                scope=MemoryScope.USER,
                tenant_id=sample_tenant_id,
            )

        assert "user_id is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_session_scope_without_session_id_raises(self, store, sample_tenant_id):
        """Adding SESSION scope memory without session_id raises MemoryScopeError."""
        with pytest.raises(MemoryScopeError):
            await store.add_memory(
                content="Test",
                scope=MemoryScope.SESSION,
                tenant_id=sample_tenant_id,
            )

    @pytest.mark.asyncio
    async def test_add_agent_scope_without_agent_id_raises(self, store, sample_tenant_id):
        """Adding AGENT scope memory without agent_id raises MemoryScopeError."""
        with pytest.raises(MemoryScopeError):
            await store.add_memory(
                content="Test",
                scope=MemoryScope.AGENT,
                tenant_id=sample_tenant_id,
            )

    @pytest.mark.asyncio
    async def test_add_memory_exceeds_limit_raises(self, store, mock_postgres, sample_tenant_id):
        """Adding memory when scope limit exceeded raises MemoryLimitExceededError."""
        # Mock scope count at max limit
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 100})

        with pytest.raises(MemoryLimitExceededError) as exc_info:
            await store.add_memory(
                content="Test",
                scope=MemoryScope.GLOBAL,
                tenant_id=sample_tenant_id,
            )

        assert "limit exceeded" in str(exc_info.value).lower()


class TestGetMemory:
    """Test get_memory functionality."""

    @pytest.mark.asyncio
    async def test_get_memory_returns_memory(self, store, mock_postgres, sample_tenant_id):
        """Can retrieve a memory by ID."""
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # Mock database response
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={
            "id": UUID(memory_id),
            "tenant_id": UUID(sample_tenant_id),
            "scope": "global",
            "user_id": None,
            "session_id": None,
            "agent_id": None,
            "content": "Test content",
            "importance": 0.8,
            "metadata": "{}",
            "created_at": now,
            "accessed_at": now,
            "access_count": 5,
        })

        memory = await store.get_memory(
            memory_id=memory_id,
            tenant_id=sample_tenant_id,
        )

        assert memory is not None
        assert memory.id == UUID(memory_id)
        assert memory.content == "Test content"
        assert memory.importance == 0.8

    @pytest.mark.asyncio
    async def test_get_memory_returns_none_when_not_found(self, store, mock_postgres, sample_tenant_id):
        """Returns None when memory not found."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value=None)

        memory = await store.get_memory(
            memory_id=str(uuid4()),
            tenant_id=sample_tenant_id,
        )

        assert memory is None

    @pytest.mark.asyncio
    async def test_get_memory_from_cache(self, store, mock_redis, mock_postgres, sample_tenant_id):
        """Can retrieve memory from cache."""
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # Mock cache hit
        cached_data = {
            "id": memory_id,
            "tenant_id": sample_tenant_id,
            "scope": "global",
            "user_id": None,
            "session_id": None,
            "agent_id": None,
            "content": "Cached content",
            "importance": 0.7,
            "metadata": {},
            "created_at": now.isoformat(),
            "accessed_at": now.isoformat(),
            "access_count": 3,
        }
        mock_redis.client.get = AsyncMock(return_value=json.dumps(cached_data).encode())

        memory = await store.get_memory(
            memory_id=memory_id,
            tenant_id=sample_tenant_id,
        )

        assert memory is not None
        assert memory.content == "Cached content"
        # Database should not be queried when cache hit
        # (fetchrow is only called for access stats update)


class TestListMemories:
    """Test list_memories functionality."""

    @pytest.mark.asyncio
    async def test_list_memories_returns_results(self, store, mock_postgres, sample_tenant_id):
        """Can list memories with pagination."""
        now = datetime.now(timezone.utc)
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"total": 2})
        mock_postgres._conn_mock.fetch = AsyncMock(return_value=[
            {
                "id": uuid4(),
                "tenant_id": UUID(sample_tenant_id),
                "scope": "global",
                "user_id": None,
                "session_id": None,
                "agent_id": None,
                "content": "Memory 1",
                "importance": 0.5,
                "metadata": "{}",
                "created_at": now,
                "accessed_at": now,
                "access_count": 0,
            },
            {
                "id": uuid4(),
                "tenant_id": UUID(sample_tenant_id),
                "scope": "global",
                "user_id": None,
                "session_id": None,
                "agent_id": None,
                "content": "Memory 2",
                "importance": 0.6,
                "metadata": "{}",
                "created_at": now,
                "accessed_at": now,
                "access_count": 1,
            },
        ])

        memories, total = await store.list_memories(
            tenant_id=sample_tenant_id,
            limit=10,
            offset=0,
        )

        assert total == 2
        assert len(memories) == 2
        assert memories[0].content == "Memory 1"
        assert memories[1].content == "Memory 2"

    @pytest.mark.asyncio
    async def test_list_memories_with_scope_filter(
        self, store, mock_postgres, sample_tenant_id, sample_user_id
    ):
        """Can filter memories by scope."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"total": 0})
        mock_postgres._conn_mock.fetch = AsyncMock(return_value=[])

        await store.list_memories(
            tenant_id=sample_tenant_id,
            scope=MemoryScope.USER,
            user_id=sample_user_id,
        )

        # Verify scope was included in query (check execute call)
        assert mock_postgres._conn_mock.fetch.called

    @pytest.mark.asyncio
    async def test_list_memories_missing_scope_context_raises(
        self, store, sample_tenant_id
    ):
        """Missing scope context should raise MemoryScopeError."""
        with pytest.raises(MemoryScopeError):
            await store.list_memories(
                tenant_id=sample_tenant_id,
                scope=MemoryScope.USER,
            )


class TestDeleteMemory:
    """Test delete_memory functionality."""

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, store, mock_postgres, sample_tenant_id):
        """Can delete a memory successfully."""
        mock_postgres._conn_mock.execute = AsyncMock(return_value="DELETE 1")

        result = await store.delete_memory(
            memory_id=str(uuid4()),
            tenant_id=sample_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, store, mock_postgres, sample_tenant_id):
        """Returns False when memory not found for deletion."""
        mock_postgres._conn_mock.execute = AsyncMock(return_value="DELETE 0")

        result = await store.delete_memory(
            memory_id=str(uuid4()),
            tenant_id=sample_tenant_id,
        )

        assert result is False


class TestDeleteMemoriesByScope:
    """Test delete_memories_by_scope functionality."""

    @pytest.mark.asyncio
    async def test_delete_by_scope_success(self, store, mock_postgres, sample_tenant_id):
        """Can delete all memories in a scope."""
        mock_postgres._conn_mock.execute = AsyncMock(return_value="DELETE 5")

        count = await store.delete_memories_by_scope(
            scope=MemoryScope.SESSION,
            tenant_id=sample_tenant_id,
            session_id=str(uuid4()),
        )

        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_by_scope_no_matches(self, store, mock_postgres, sample_tenant_id):
        """Returns 0 when no memories match scope."""
        mock_postgres._conn_mock.execute = AsyncMock(return_value="DELETE 0")

        count = await store.delete_memories_by_scope(
            scope=MemoryScope.GLOBAL,
            tenant_id=sample_tenant_id,
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_by_scope_missing_context_raises(
        self, store, sample_tenant_id
    ):
        """Missing scope context should raise MemoryScopeError."""
        with pytest.raises(MemoryScopeError):
            await store.delete_memories_by_scope(
                scope=MemoryScope.SESSION,
                tenant_id=sample_tenant_id,
            )


class TestSearchMemories:
    """Test search_memories functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_matching_memories(self, store, mock_postgres, sample_tenant_id):
        """Can search memories by text query."""
        now = datetime.now(timezone.utc)
        mock_postgres._conn_mock.fetch = AsyncMock(return_value=[
            {
                "id": uuid4(),
                "tenant_id": UUID(sample_tenant_id),
                "scope": "global",
                "user_id": None,
                "session_id": None,
                "agent_id": None,
                "content": "User prefers dark mode",
                "importance": 0.8,
                "metadata": "{}",
                "created_at": now,
                "accessed_at": now,
                "access_count": 0,
            },
        ])

        memories, scopes_searched = await store.search_memories(
            query="dark mode",
            scope=MemoryScope.GLOBAL,
            tenant_id=sample_tenant_id,
        )

        assert len(memories) == 1
        assert "dark mode" in memories[0].content
        assert MemoryScope.GLOBAL in scopes_searched

    @pytest.mark.asyncio
    async def test_search_with_parent_scopes(self, store, mock_postgres, sample_tenant_id, sample_session_id):
        """Search includes parent scopes when enabled."""
        mock_postgres._conn_mock.fetch = AsyncMock(return_value=[])

        memories, scopes_searched = await store.search_memories(
            query="test",
            scope=MemoryScope.SESSION,
            tenant_id=sample_tenant_id,
            session_id=sample_session_id,
            include_parent_scopes=True,
        )

        # SESSION search should include SESSION, USER, GLOBAL
        assert MemoryScope.SESSION in scopes_searched
        assert MemoryScope.USER in scopes_searched
        assert MemoryScope.GLOBAL in scopes_searched

    @pytest.mark.asyncio
    async def test_search_without_parent_scopes(self, store, mock_postgres, sample_tenant_id, sample_session_id):
        """Search excludes parent scopes when disabled."""
        mock_postgres._conn_mock.fetch = AsyncMock(return_value=[])

        memories, scopes_searched = await store.search_memories(
            query="test",
            scope=MemoryScope.SESSION,
            tenant_id=sample_tenant_id,
            session_id=sample_session_id,
            include_parent_scopes=False,
        )

        # Should only include SESSION
        assert scopes_searched == [MemoryScope.SESSION]

    @pytest.mark.asyncio
    async def test_search_missing_scope_context_raises(
        self, store, sample_tenant_id
    ):
        """Missing scope context should raise MemoryScopeError."""
        with pytest.raises(MemoryScopeError):
            await store.search_memories(
                query="test",
                scope=MemoryScope.SESSION,
                tenant_id=sample_tenant_id,
            )


class TestUpdateMemory:
    """Test update_memory functionality."""

    @pytest.mark.asyncio
    async def test_update_memory_content(self, store, mock_postgres, sample_tenant_id):
        """Can update memory content."""
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # Mock get existing memory
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={
            "id": UUID(memory_id),
            "tenant_id": UUID(sample_tenant_id),
            "scope": "global",
            "user_id": None,
            "session_id": None,
            "agent_id": None,
            "content": "Updated content",
            "importance": 0.5,
            "metadata": "{}",
            "created_at": now,
            "accessed_at": now,
            "access_count": 0,
        })

        memory = await store.update_memory(
            memory_id=memory_id,
            tenant_id=sample_tenant_id,
            content="Updated content",
        )

        assert memory is not None
        assert memory.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, store, mock_postgres, sample_tenant_id):
        """Returns None when memory to update not found."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value=None)

        memory = await store.update_memory(
            memory_id=str(uuid4()),
            tenant_id=sample_tenant_id,
            content="Updated content",
        )

        assert memory is None


class TestCacheOperations:
    """Test Redis cache operations."""

    @pytest.mark.asyncio
    async def test_memory_is_cached_after_add(self, store, mock_redis, mock_postgres, sample_tenant_id):
        """Memory is cached in Redis after add."""
        mock_postgres._conn_mock.fetchrow = AsyncMock(return_value={"count": 0})

        await store.add_memory(
            content="Test",
            scope=MemoryScope.GLOBAL,
            tenant_id=sample_tenant_id,
        )

        # Verify setex was called (cache write)
        mock_redis.client.setex.assert_called()

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_delete(self, store, mock_redis, mock_postgres, sample_tenant_id):
        """Cache is invalidated when memory deleted."""
        mock_postgres._conn_mock.execute = AsyncMock(return_value="DELETE 1")

        await store.delete_memory(
            memory_id=str(uuid4()),
            tenant_id=sample_tenant_id,
        )

        # Verify delete was called (cache invalidation)
        mock_redis.client.delete.assert_called()
