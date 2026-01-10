"""Tests for Memory Consolidation (Story 20-A2).

This module tests:
- Importance decay calculation
- Cosine similarity calculation
- Memory consolidation workflow
- Duplicate detection and merging
- Low-importance memory cleanup
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agentic_rag_backend.memory.consolidation import (
    MemoryConsolidator,
    calculate_importance,
    cosine_similarity,
)
from agentic_rag_backend.memory.models import (
    ConsolidationResult,
    MemoryScope,
    ScopedMemory,
)


# Test fixtures


@pytest.fixture
def sample_embedding():
    """Sample normalized embedding vector."""
    import numpy as np
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def sample_memory(sample_embedding):
    """Create a sample memory for testing."""
    return ScopedMemory(
        id=uuid4(),
        tenant_id=uuid4(),
        content="Test memory content",
        scope=MemoryScope.USER,
        user_id=uuid4(),
        session_id=None,
        agent_id=None,
        importance=0.8,
        access_count=5,
        embedding=sample_embedding,
        metadata={"source": "test"},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        accessed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_store():
    """Create a mock memory store."""
    store = MagicMock()
    store.list_memories = AsyncMock(return_value=([], 0))
    store.get_memory = AsyncMock(return_value=None)
    store.get_memory_embeddings = AsyncMock(return_value={})
    store.update_memory = AsyncMock(return_value=None)
    store.delete_memory = AsyncMock(return_value=True)
    
    # Mock Redis for distributed locking
    mock_redis = MagicMock()
    mock_redis.client = MagicMock()
    # Support for manual SET pattern
    mock_redis.client.set = AsyncMock(return_value=True)
    mock_redis.client.eval = AsyncMock(return_value=1)
    # Support for redis-py lock() pattern
    mock_lock = AsyncMock()
    mock_lock.acquire = AsyncMock(return_value=True)
    mock_lock.release = AsyncMock(return_value=None)
    mock_redis.client.lock = MagicMock(return_value=mock_lock)
    
    store._redis = mock_redis
    
    store._postgres = MagicMock()
    store._postgres.pool = MagicMock()
    store._postgres.pool.acquire = MagicMock()
    return store


@pytest.fixture
def consolidator(mock_store):
    """Create a consolidator with mock store."""
    return MemoryConsolidator(
        store=mock_store,
        similarity_threshold=0.9,
        decay_half_life_days=30,
        min_importance=0.1,
        consolidation_batch_size=100,
    )


# Test calculate_importance function


class TestCalculateImportance:
    """Tests for importance decay calculation."""

    def test_no_decay_same_day(self):
        """Importance should not decay on the same day."""
        result = calculate_importance(
            current_importance=1.0,
            access_count=0,
            days_since_access=0.0,
            decay_half_life_days=30,
        )
        # With 0 accesses, access_boost = 0.5, so result = 1.0 * 1.0 * 0.5 = 0.5
        assert result == pytest.approx(0.5, rel=0.01)

    def test_with_access_boost_no_decay(self):
        """Access boost should increase effective importance."""
        result = calculate_importance(
            current_importance=1.0,
            access_count=10,
            days_since_access=0.0,
            decay_half_life_days=30,
        )
        # access_boost = min(1.0, 0.5 + 10 * 0.1) = 1.0
        # decay_factor = 1.0 (no decay)
        # result = 1.0 * 1.0 * 1.0 = 1.0
        assert result == pytest.approx(1.0, rel=0.01)

    def test_half_life_decay(self):
        """Importance should halve after half_life_days."""
        result = calculate_importance(
            current_importance=1.0,
            access_count=5,  # access_boost = 1.0
            days_since_access=30.0,  # One half-life
            decay_half_life_days=30,
        )
        # decay_factor = 0.5
        # access_boost = min(1.0, 0.5 + 5 * 0.1) = 1.0
        # result = 1.0 * 0.5 * 1.0 = 0.5
        assert result == pytest.approx(0.5, rel=0.01)

    def test_two_half_lives_decay(self):
        """Importance should quarter after two half-lives."""
        result = calculate_importance(
            current_importance=1.0,
            access_count=5,  # access_boost = 1.0
            days_since_access=60.0,  # Two half-lives
            decay_half_life_days=30,
        )
        # decay_factor = 0.25
        # result = 1.0 * 0.25 * 1.0 = 0.25
        assert result == pytest.approx(0.25, rel=0.01)

    def test_clamp_to_valid_range(self):
        """Result should always be clamped to 0.0-1.0."""
        # Very large values should clamp to 1.0
        result = calculate_importance(
            current_importance=2.0,  # Invalid but shouldn't crash
            access_count=100,
            days_since_access=0.0,
            decay_half_life_days=30,
        )
        assert 0.0 <= result <= 1.0

    def test_zero_importance_stays_zero(self):
        """Zero importance should remain zero."""
        result = calculate_importance(
            current_importance=0.0,
            access_count=10,
            days_since_access=0.0,
            decay_half_life_days=30,
        )
        assert result == 0.0


# Test cosine_similarity function


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=0.001)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(-1.0, rel=0.001)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.0, rel=0.001)

    def test_empty_vectors(self):
        """Empty vectors should return 0.0."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_zero_vectors(self):
        """Zero vectors should return 0.0 (avoid division by zero)."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0
        assert cosine_similarity(vec_a, vec_a) == 0.0

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity > 0.99


# Test MemoryConsolidator


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator class."""

    def test_consolidator_initialization(self, mock_store):
        """Test consolidator initializes with correct parameters."""
        consolidator = MemoryConsolidator(
            store=mock_store,
            similarity_threshold=0.85,
            decay_half_life_days=14,
            min_importance=0.05,
            consolidation_batch_size=50,
        )
        assert consolidator.similarity_threshold == 0.85
        assert consolidator.decay_half_life_days == 14
        assert consolidator.min_importance == 0.05
        assert consolidator.batch_size == 50

    @pytest.mark.asyncio
    async def test_consolidate_empty_memories(self, consolidator, mock_store):
        """Consolidation with no memories should return zero counts."""
        mock_store.list_memories.return_value = ([], 0)

        result = await consolidator.consolidate(
            tenant_id="test-tenant",
            scope=MemoryScope.USER,
        )

        assert result.memories_processed == 0
        assert result.duplicates_merged == 0
        assert result.memories_decayed == 0
        assert result.memories_removed == 0
        assert result.scope == MemoryScope.USER
        assert result.tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_consolidate_stores_last_result(self, consolidator, mock_store):
        """Consolidation should store last run time and result."""
        mock_store.list_memories.return_value = ([], 0)

        await consolidator.consolidate(tenant_id="test-tenant")

        assert consolidator.last_run_at is not None
        assert consolidator.last_result is not None
        assert isinstance(consolidator.last_result, ConsolidationResult)

    @pytest.mark.asyncio
    async def test_importance_decay_applied(self, consolidator, mock_store, sample_memory):
        """Test that importance decay is applied to old memories."""
        # Set memory to be accessed 30 days ago
        old_access_time = datetime.now(timezone.utc) - timedelta(days=30)
        sample_memory.accessed_at = old_access_time
        sample_memory.importance = 1.0
        sample_memory.access_count = 5  # Gives access_boost of 1.0

        mock_store.list_memories.return_value = ([sample_memory], 1)
        mock_store.get_memory.return_value = sample_memory

        # Mock _get_all_tenant_ids for full consolidation
        with patch.object(
            consolidator,
            "_get_all_tenant_ids",
            new_callable=AsyncMock,
            return_value=["test-tenant"],
        ):
            await consolidator.consolidate(tenant_id="test-tenant")

        # Should have updated the memory with decayed importance
        assert mock_store.update_memory.called
        call_args = mock_store.update_memory.call_args
        assert call_args.kwargs["importance"] < 1.0  # Importance should have decayed

    @pytest.mark.asyncio
    async def test_low_importance_removal(self, consolidator, mock_store, sample_memory):
        """Test that memories below min_importance are removed."""
        sample_memory.importance = 0.05  # Below threshold of 0.1

        mock_store.list_memories.return_value = ([sample_memory], 1)
        mock_store.get_memory.return_value = sample_memory

        result = await consolidator.consolidate(tenant_id="test-tenant")

        assert mock_store.delete_memory.called
        assert result.memories_removed >= 1

    def test_find_similar_memories(self, consolidator, sample_memory):
        """Test finding similar memories by embedding."""
        # Create a very similar embedding
        target_embedding = sample_memory.embedding.copy()
        target_embedding[0] += 0.001  # Tiny perturbation

        similar = consolidator.find_similar_memories(
            target_embedding=target_embedding,
            memories=[sample_memory],
            threshold=0.9,
        )

        assert len(similar) == 1
        assert similar[0][0] == sample_memory
        assert similar[0][1] > 0.99  # Very high similarity

    def test_find_similar_memories_below_threshold(self, consolidator, sample_memory):
        """Test that dissimilar memories are not returned."""
        import numpy as np

        # Create orthogonal embedding
        orthogonal = np.zeros(1536).tolist()
        orthogonal[0] = 1.0

        similar = consolidator.find_similar_memories(
            target_embedding=orthogonal,
            memories=[sample_memory],
            threshold=0.9,
        )

        assert len(similar) == 0

    def test_cleanup_low_importance_utility(self, consolidator, sample_memory):
        """Test the cleanup_low_importance utility method."""
        low_importance = ScopedMemory(
            **{**sample_memory.model_dump(), "id": uuid4(), "importance": 0.05}
        )
        high_importance = ScopedMemory(
            **{**sample_memory.model_dump(), "id": uuid4(), "importance": 0.9}
        )

        to_remove = consolidator.cleanup_low_importance([low_importance, high_importance])

        assert len(to_remove) == 1
        assert to_remove[0].importance == 0.05


class TestMemoryMerging:
    """Tests for memory merging functionality."""

    @pytest.fixture
    def two_similar_memories(self, sample_embedding):
        """Create two similar memories for merge testing."""
        tenant_id = uuid4()
        user_id = uuid4()

        memory1 = ScopedMemory(
            id=uuid4(),
            tenant_id=tenant_id,
            content="First memory about topic",
            scope=MemoryScope.USER,
            user_id=user_id,
            session_id=None,
            agent_id=None,
            importance=0.8,
            access_count=5,
            embedding=sample_embedding,
            metadata={"source": "first"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
        )

        memory2 = ScopedMemory(
            id=uuid4(),
            tenant_id=tenant_id,
            content="Second memory about topic",
            scope=MemoryScope.USER,
            user_id=user_id,
            session_id=None,
            agent_id=None,
            importance=0.6,
            access_count=3,
            embedding=sample_embedding,  # Same embedding = duplicate
            metadata={"source": "second"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
        )

        return memory1, memory2

    @pytest.mark.asyncio
    async def test_merge_memories_combines_metadata(
        self, consolidator, two_similar_memories
    ):
        """Test that merging combines metadata correctly."""
        memory1, memory2 = two_similar_memories

        merged = await consolidator._merge_memories(memory1, [memory2])

        # Primary metadata takes precedence
        assert merged["metadata"]["source"] == "first"
        # Merge tracking added
        assert "merged_count" in merged["metadata"]
        assert "merged_at" in merged["metadata"]
        assert "merged_from_ids" in merged["metadata"]
        assert str(memory2.id) in merged["metadata"]["merged_from_ids"]

    @pytest.mark.asyncio
    async def test_merge_memories_max_importance(
        self, consolidator, two_similar_memories
    ):
        """Test that merged importance is maximum of all memories."""
        memory1, memory2 = two_similar_memories

        merged = await consolidator._merge_memories(memory1, [memory2])

        assert merged["importance"] == max(memory1.importance, memory2.importance)

    @pytest.mark.asyncio
    async def test_merge_memories_sum_access_counts(
        self, consolidator, two_similar_memories
    ):
        """Test that access counts are summed during merge."""
        memory1, memory2 = two_similar_memories

        merged = await consolidator._merge_memories(memory1, [memory2])

        assert merged["access_count"] == memory1.access_count + memory2.access_count

    @pytest.mark.asyncio
    async def test_merge_respects_user_context(self, consolidator, sample_embedding):
        """Merges should not cross different user contexts."""
        tenant_id = uuid4()
        now = datetime.now(timezone.utc)

        memory1 = ScopedMemory(
            id=uuid4(),
            tenant_id=tenant_id,
            content="User A memory",
            scope=MemoryScope.USER,
            user_id=uuid4(),
            session_id=None,
            agent_id=None,
            importance=0.8,
            access_count=2,
            embedding=sample_embedding,
            metadata={},
            created_at=now,
            updated_at=now,
            accessed_at=now,
        )

        memory2 = ScopedMemory(
            id=uuid4(),
            tenant_id=tenant_id,
            content="User B memory",
            scope=MemoryScope.USER,
            user_id=uuid4(),
            session_id=None,
            agent_id=None,
            importance=0.7,
            access_count=1,
            embedding=sample_embedding,
            metadata={},
            created_at=now,
            updated_at=now,
            accessed_at=now,
        )

        consolidator.store.update_memory.return_value = memory1
        consolidator.store.delete_memory.return_value = True

        merged_count = await consolidator._merge_similar_memories(
            [memory1, memory2],
            tenant_id=str(tenant_id),
        )

        assert merged_count == 0
        consolidator.store.update_memory.assert_not_called()
        consolidator.store.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_skips_deletes_on_update_failure(
        self, consolidator, two_similar_memories
    ):
        """Secondary memories should not be deleted when update fails."""
        memory1, memory2 = two_similar_memories
        consolidator.store.update_memory.return_value = None

        merged_count = await consolidator._merge_similar_memories(
            [memory1, memory2],
            tenant_id=str(memory1.tenant_id),
        )

        assert merged_count == 0
        consolidator.store.delete_memory.assert_not_called()


class TestTenantIsolation:
    """Tests for multi-tenancy and tenant isolation."""

    @pytest.mark.asyncio
    async def test_consolidation_passes_tenant_id(self, consolidator, mock_store):
        """Ensure tenant_id is passed to all store operations."""
        mock_store.list_memories.return_value = ([], 0)

        await consolidator.consolidate(tenant_id="tenant-123")

        # Verify tenant_id was passed
        call_args = mock_store.list_memories.call_args
        assert call_args.kwargs["tenant_id"] == "tenant-123"

    @pytest.mark.asyncio
    async def test_consolidate_all_tenants_isolation(self, consolidator):
        """Test that each tenant is consolidated independently."""
        tenant_ids = ["tenant-a", "tenant-b", "tenant-c"]

        with patch.object(
            consolidator,
            "_get_all_tenant_ids",
            new_callable=AsyncMock,
            return_value=tenant_ids,
        ), patch.object(
            consolidator,
            "consolidate",
            new_callable=AsyncMock,
            return_value=ConsolidationResult(
                memories_processed=10,
                duplicates_merged=1,
                memories_decayed=5,
                memories_removed=0,
                processing_time_ms=100.0,
            ),
        ) as mock_consolidate:
            results = await consolidator.consolidate_all_tenants()

        assert len(results) == 3
        # Verify each tenant was consolidated separately
        calls = mock_consolidate.call_args_list
        called_tenants = [call.kwargs["tenant_id"] for call in calls]
        assert set(called_tenants) == set(tenant_ids)


class TestConsolidationResult:
    """Tests for ConsolidationResult model."""

    def test_consolidation_result_creation(self):
        """Test ConsolidationResult model creation."""
        result = ConsolidationResult(
            memories_processed=100,
            duplicates_merged=5,
            memories_decayed=30,
            memories_removed=2,
            processing_time_ms=1500.5,
            scope=MemoryScope.USER,
            tenant_id="test-tenant",
        )

        assert result.memories_processed == 100
        assert result.duplicates_merged == 5
        assert result.memories_decayed == 30
        assert result.memories_removed == 2
        assert result.processing_time_ms == 1500.5
        assert result.scope == MemoryScope.USER
        assert result.tenant_id == "test-tenant"

    def test_consolidation_result_json_serialization(self):
        """Test that ConsolidationResult serializes to JSON correctly."""
        result = ConsolidationResult(
            memories_processed=50,
            duplicates_merged=2,
            memories_decayed=10,
            memories_removed=1,
            processing_time_ms=500.0,
        )

        data = result.model_dump(mode="json")

        assert "memories_processed" in data
        assert "duplicates_merged" in data
        assert data["memories_processed"] == 50
