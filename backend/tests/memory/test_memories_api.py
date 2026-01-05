"""Integration tests for Memory API endpoints (Story 20-A1).

Tests the REST API for scoped memories:
- POST /api/v1/memories - Create memory
- GET /api/v1/memories - List memories
- GET /api/v1/memories/{memory_id} - Get single memory
- PUT /api/v1/memories/{memory_id} - Update memory
- DELETE /api/v1/memories/{memory_id} - Delete memory
- DELETE /api/v1/memories/scope/{scope} - Delete by scope
- POST /api/v1/memories/search - Search memories

Feature is gated by MEMORY_SCOPES_ENABLED=true.
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Set environment before imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


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


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store."""
    from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

    store = MagicMock()
    memory_id = uuid4()
    now = datetime.now(timezone.utc)

    # Default mock return for add_memory
    store.add_memory = AsyncMock(return_value=ScopedMemory(
        id=memory_id,
        content="Test content",
        scope=MemoryScope.USER,
        tenant_id=uuid4(),
        user_id=uuid4(),
        session_id=None,
        agent_id=None,
        importance=1.0,
        metadata={},
        created_at=now,
        accessed_at=now,
        access_count=0,
    ))

    store.get_memory = AsyncMock(return_value=None)
    store.list_memories = AsyncMock(return_value=([], 0))
    store.search_memories = AsyncMock(return_value=([], [MemoryScope.GLOBAL]))
    store.update_memory = AsyncMock(return_value=None)
    store.delete_memory = AsyncMock(return_value=False)
    store.delete_memories_by_scope = AsyncMock(return_value=0)

    return store


@pytest.fixture
def mock_settings_enabled():
    """Create mock settings with memory enabled."""
    settings = MagicMock()
    settings.memory_scopes_enabled = True
    settings.memory_default_scope = "session"
    settings.memory_include_parent_scopes = True
    settings.memory_cache_ttl_seconds = 3600
    settings.memory_max_per_scope = 10000
    settings.embedding_provider = "openai"
    settings.embedding_api_key = "test-key"
    settings.embedding_base_url = None
    settings.embedding_model = "text-embedding-3-small"
    return settings


@pytest.fixture
def mock_settings_disabled():
    """Create mock settings with memory disabled."""
    settings = MagicMock()
    settings.memory_scopes_enabled = False
    return settings


@pytest.fixture
def test_app_enabled(mock_memory_store, mock_settings_enabled):
    """Create a minimal test app with memory feature enabled."""
    from agentic_rag_backend.api.routes.memories import (
        router,
        get_settings,
        get_memory_store,
    )

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Override dependencies
    app.dependency_overrides[get_settings] = lambda: mock_settings_enabled
    app.dependency_overrides[get_memory_store] = lambda: mock_memory_store

    return app


@pytest.fixture
def test_app_disabled(mock_settings_disabled, mock_memory_store):
    """Create a minimal test app with memory feature disabled."""
    from agentic_rag_backend.api.routes.memories import (
        router,
        get_settings,
        get_memory_store,
    )

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    app.dependency_overrides[get_settings] = lambda: mock_settings_disabled
    app.dependency_overrides[get_memory_store] = lambda: mock_memory_store

    return app


@pytest.fixture
def client_enabled(test_app_enabled):
    """Test client with memory enabled."""
    with TestClient(test_app_enabled) as client:
        yield client


@pytest.fixture
def client_disabled(test_app_disabled):
    """Test client with memory disabled."""
    with TestClient(test_app_disabled) as client:
        yield client


class TestFeatureGating:
    """Test feature flag gating."""

    def test_memory_endpoint_returns_404_when_disabled(
        self, client_disabled, sample_tenant_id
    ):
        """Memory endpoint returns 404 when feature disabled."""
        response = client_disabled.get(
            f"/api/v1/memories?tenant_id={sample_tenant_id}"
        )
        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"]


class TestCreateMemory:
    """Test POST /api/v1/memories endpoint."""

    def test_create_user_scope_memory(
        self, client_enabled, mock_memory_store, sample_tenant_id, sample_user_id
    ):
        """Can create a USER scope memory."""
        from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

        memory_id = uuid4()
        now = datetime.now(timezone.utc)
        mock_memory_store.add_memory = AsyncMock(return_value=ScopedMemory(
            id=memory_id,
            content="User preference test",
            scope=MemoryScope.USER,
            tenant_id=uuid4(),
            user_id=uuid4(),
            session_id=None,
            agent_id=None,
            importance=0.8,
            metadata={"source": "test"},
            created_at=now,
            accessed_at=now,
            access_count=0,
        ))

        response = client_enabled.post(
            "/api/v1/memories",
            json={
                "content": "User preference test",
                "scope": "user",
                "tenant_id": sample_tenant_id,
                "user_id": sample_user_id,
                "importance": 0.8,
                "metadata": {"source": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["content"] == "User preference test"
        assert data["scope"] == "user"
        assert data["importance"] == 0.8
        assert data["metadata"] == {"source": "test"}

    def test_create_global_scope_memory(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can create a GLOBAL scope memory without user/session context."""
        from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

        mock_memory_store.add_memory = AsyncMock(return_value=ScopedMemory(
            id=uuid4(),
            content="Global knowledge",
            scope=MemoryScope.GLOBAL,
            tenant_id=uuid4(),
            user_id=None,
            session_id=None,
            agent_id=None,
            importance=1.0,
            metadata={},
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
            access_count=0,
        ))

        response = client_enabled.post(
            "/api/v1/memories",
            json={
                "content": "Global knowledge",
                "scope": "global",
                "tenant_id": sample_tenant_id,
            },
        )

        assert response.status_code == 200
        assert response.json()["data"]["scope"] == "global"

    def test_create_memory_missing_scope_context_returns_400(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Creating memory without required scope context returns 400."""
        from agentic_rag_backend.memory.errors import MemoryScopeError

        mock_memory_store.add_memory = AsyncMock(
            side_effect=MemoryScopeError("user", "user_id is required")
        )

        response = client_enabled.post(
            "/api/v1/memories",
            json={
                "content": "Test",
                "scope": "user",
                "tenant_id": sample_tenant_id,
            },
        )

        assert response.status_code == 400
        assert "user_id is required" in response.json()["detail"]

    def test_create_memory_limit_exceeded_returns_429(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Creating memory when limit exceeded returns 429."""
        from agentic_rag_backend.memory.errors import MemoryLimitExceededError

        mock_memory_store.add_memory = AsyncMock(
            side_effect=MemoryLimitExceededError("global", 10000, 10000)
        )

        response = client_enabled.post(
            "/api/v1/memories",
            json={
                "content": "Test",
                "scope": "global",
                "tenant_id": sample_tenant_id,
            },
        )

        assert response.status_code == 429
        assert "limit exceeded" in response.json()["detail"].lower()


class TestListMemories:
    """Test GET /api/v1/memories endpoint."""

    def test_list_memories_empty(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can list memories (empty result)."""
        mock_memory_store.list_memories = AsyncMock(return_value=([], 0))

        response = client_enabled.get(
            f"/api/v1/memories?tenant_id={sample_tenant_id}"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["memories"] == []
        assert data["total"] == 0

    def test_list_memories_with_results(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can list memories with results."""
        from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

        now = datetime.now(timezone.utc)
        memories = [
            ScopedMemory(
                id=uuid4(),
                content="Memory 1",
                scope=MemoryScope.GLOBAL,
                tenant_id=uuid4(),
                user_id=None,
                session_id=None,
                agent_id=None,
                importance=0.5,
                metadata={},
                created_at=now,
                accessed_at=now,
                access_count=0,
            ),
            ScopedMemory(
                id=uuid4(),
                content="Memory 2",
                scope=MemoryScope.GLOBAL,
                tenant_id=uuid4(),
                user_id=None,
                session_id=None,
                agent_id=None,
                importance=0.6,
                metadata={},
                created_at=now,
                accessed_at=now,
                access_count=1,
            ),
        ]
        mock_memory_store.list_memories = AsyncMock(return_value=(memories, 2))

        response = client_enabled.get(
            f"/api/v1/memories?tenant_id={sample_tenant_id}&limit=10&offset=0"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["memories"]) == 2
        assert data["total"] == 2


class TestGetMemory:
    """Test GET /api/v1/memories/{memory_id} endpoint."""

    def test_get_memory_found(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can get a specific memory by ID."""
        from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

        memory_id = uuid4()
        now = datetime.now(timezone.utc)
        mock_memory_store.get_memory = AsyncMock(return_value=ScopedMemory(
            id=memory_id,
            content="Test content",
            scope=MemoryScope.GLOBAL,
            tenant_id=uuid4(),
            user_id=None,
            session_id=None,
            agent_id=None,
            importance=0.7,
            metadata={"key": "value"},
            created_at=now,
            accessed_at=now,
            access_count=5,
        ))

        response = client_enabled.get(
            f"/api/v1/memories/{memory_id}?tenant_id={sample_tenant_id}"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["content"] == "Test content"
        assert data["importance"] == 0.7
        assert data["access_count"] == 5

    def test_get_memory_not_found(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Returns 404 when memory not found."""
        mock_memory_store.get_memory = AsyncMock(return_value=None)

        response = client_enabled.get(
            f"/api/v1/memories/{uuid4()}?tenant_id={sample_tenant_id}"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestUpdateMemory:
    """Test PUT /api/v1/memories/{memory_id} endpoint."""

    def test_update_memory_content(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can update memory content."""
        from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

        memory_id = uuid4()
        now = datetime.now(timezone.utc)
        mock_memory_store.update_memory = AsyncMock(return_value=ScopedMemory(
            id=memory_id,
            content="Updated content",
            scope=MemoryScope.GLOBAL,
            tenant_id=uuid4(),
            user_id=None,
            session_id=None,
            agent_id=None,
            importance=0.9,
            metadata={},
            created_at=now,
            accessed_at=now,
            access_count=0,
        ))

        response = client_enabled.put(
            f"/api/v1/memories/{memory_id}?tenant_id={sample_tenant_id}",
            json={"content": "Updated content", "importance": 0.9},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["content"] == "Updated content"
        assert data["importance"] == 0.9

    def test_update_memory_not_found(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Returns 404 when updating non-existent memory."""
        mock_memory_store.update_memory = AsyncMock(return_value=None)

        response = client_enabled.put(
            f"/api/v1/memories/{uuid4()}?tenant_id={sample_tenant_id}",
            json={"content": "Updated"},
        )

        assert response.status_code == 404


class TestDeleteMemory:
    """Test DELETE /api/v1/memories/{memory_id} endpoint."""

    def test_delete_memory_success(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can delete a memory."""
        mock_memory_store.delete_memory = AsyncMock(return_value=True)
        memory_id = uuid4()

        response = client_enabled.delete(
            f"/api/v1/memories/{memory_id}?tenant_id={sample_tenant_id}"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["deleted"] is True

    def test_delete_memory_not_found(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Returns 404 when deleting non-existent memory."""
        mock_memory_store.delete_memory = AsyncMock(return_value=False)

        response = client_enabled.delete(
            f"/api/v1/memories/{uuid4()}?tenant_id={sample_tenant_id}"
        )

        assert response.status_code == 404


class TestDeleteMemoriesByScope:
    """Test DELETE /api/v1/memories/scope/{scope} endpoint."""

    def test_delete_by_scope_session(
        self, client_enabled, mock_memory_store, sample_tenant_id, sample_session_id
    ):
        """Can delete all memories in SESSION scope."""
        mock_memory_store.delete_memories_by_scope = AsyncMock(return_value=5)

        response = client_enabled.delete(
            f"/api/v1/memories/scope/session?tenant_id={sample_tenant_id}&session_id={sample_session_id}"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["deleted_count"] == 5
        assert data["scope"] == "session"

    def test_delete_by_scope_global(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can delete all memories in GLOBAL scope."""
        mock_memory_store.delete_memories_by_scope = AsyncMock(return_value=10)

        response = client_enabled.delete(
            f"/api/v1/memories/scope/global?tenant_id={sample_tenant_id}"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["deleted_count"] == 10
        assert data["scope"] == "global"


class TestSearchMemories:
    """Test POST /api/v1/memories/search endpoint."""

    def test_search_memories(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Can search memories by query."""
        from agentic_rag_backend.memory.models import MemoryScope, ScopedMemory

        now = datetime.now(timezone.utc)
        memories = [
            ScopedMemory(
                id=uuid4(),
                content="User prefers dark mode",
                scope=MemoryScope.GLOBAL,
                tenant_id=uuid4(),
                user_id=None,
                session_id=None,
                agent_id=None,
                importance=0.8,
                metadata={},
                created_at=now,
                accessed_at=now,
                access_count=0,
            )
        ]
        mock_memory_store.search_memories = AsyncMock(
            return_value=(memories, [MemoryScope.GLOBAL])
        )

        response = client_enabled.post(
            "/api/v1/memories/search",
            json={
                "query": "dark mode",
                "scope": "global",
                "tenant_id": sample_tenant_id,
                "limit": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["memories"]) == 1
        assert data["query"] == "dark mode"
        assert "global" in data["scopes_searched"]

    def test_search_memories_with_parent_scopes(
        self, client_enabled, mock_memory_store, sample_tenant_id, sample_session_id
    ):
        """Can search with parent scope hierarchy."""
        from agentic_rag_backend.memory.models import MemoryScope

        mock_memory_store.search_memories = AsyncMock(
            return_value=([], [MemoryScope.SESSION, MemoryScope.USER, MemoryScope.GLOBAL])
        )

        response = client_enabled.post(
            "/api/v1/memories/search",
            json={
                "query": "test",
                "scope": "session",
                "tenant_id": sample_tenant_id,
                "session_id": sample_session_id,
                "include_parent_scopes": True,
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        scopes = data["scopes_searched"]
        assert "session" in scopes
        assert "user" in scopes
        assert "global" in scopes


class TestResponseFormat:
    """Test API response format compliance."""

    def test_success_response_format(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Success responses follow standard format."""
        mock_memory_store.list_memories = AsyncMock(return_value=([], 0))

        response = client_enabled.get(
            f"/api/v1/memories?tenant_id={sample_tenant_id}"
        )

        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert "requestId" in body["meta"]
        assert "timestamp" in body["meta"]

    def test_meta_timestamp_is_iso8601(
        self, client_enabled, mock_memory_store, sample_tenant_id
    ):
        """Meta timestamp is ISO 8601 format."""
        mock_memory_store.list_memories = AsyncMock(return_value=([], 0))

        response = client_enabled.get(
            f"/api/v1/memories?tenant_id={sample_tenant_id}"
        )

        timestamp = response.json()["meta"]["timestamp"]
        # Should end with Z for UTC
        assert timestamp.endswith("Z")
