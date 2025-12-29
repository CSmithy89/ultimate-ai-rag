"""
Tests for HITLManager
Story 6-4: Human-in-the-Loop Source Validation
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from agentic_rag_backend.protocols.ag_ui_bridge import (
    HITLManager,
    HITLCheckpoint,
    HITLStatus,
    create_validate_sources_events,
)


# Test data
MOCK_SOURCES = [
    {
        "id": "source-1",
        "title": "Document One",
        "preview": "Preview of document one.",
        "similarity": 0.95,
    },
    {
        "id": "source-2",
        "title": "Document Two",
        "preview": "Preview of document two.",
        "similarity": 0.75,
    },
    {
        "id": "source-3",
        "title": "Document Three",
        "preview": "Preview of document three.",
        "similarity": 0.55,
    },
]


class TestHITLCheckpoint:
    """Tests for HITLCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test checkpoint creates with default values."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )

        assert checkpoint.checkpoint_id == "test-123"
        assert checkpoint.sources == MOCK_SOURCES
        assert checkpoint.query == "test query"
        assert checkpoint.status == HITLStatus.PENDING
        assert checkpoint.approved_source_ids == []
        assert checkpoint.rejected_source_ids == []

    def test_checkpoint_to_dict(self):
        """Test checkpoint serialization to dictionary."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )
        checkpoint.approved_source_ids = ["source-1"]
        checkpoint.rejected_source_ids = ["source-2", "source-3"]
        checkpoint.status = HITLStatus.APPROVED

        result = checkpoint.to_dict()

        assert result["checkpoint_id"] == "test-123"
        assert result["sources"] == MOCK_SOURCES
        assert result["query"] == "test query"
        assert result["status"] == "approved"
        assert result["approved_source_ids"] == ["source-1"]
        assert result["rejected_source_ids"] == ["source-2", "source-3"]


class TestHITLManager:
    """Tests for HITLManager class."""

    @pytest.fixture
    def manager(self):
        """Create a test HITL manager."""
        return HITLManager(timeout=1.0)

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, manager):
        """Test checkpoint creation."""
        checkpoint = await manager.create_checkpoint(
            sources=MOCK_SOURCES,
            query="test query",
        )

        assert checkpoint.sources == MOCK_SOURCES
        assert checkpoint.query == "test query"
        assert checkpoint.status == HITLStatus.PENDING
        assert checkpoint.checkpoint_id is not None

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_custom_id(self, manager):
        """Test checkpoint creation with custom ID."""
        checkpoint = await manager.create_checkpoint(
            sources=MOCK_SOURCES,
            query="test query",
            checkpoint_id="custom-id-123",
        )

        assert checkpoint.checkpoint_id == "custom-id-123"

    def test_get_checkpoint_events(self, manager):
        """Test generating AG-UI events for checkpoint."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )

        events = manager.get_checkpoint_events(checkpoint)

        assert len(events) == 2
        # First event should be TOOL_CALL_START
        assert events[0].data["tool_call_id"] == "test-123"
        assert events[0].data["tool_name"] == "validate_sources"
        # Second event should be TOOL_CALL_ARGS
        assert events[1].data["tool_call_id"] == "test-123"
        assert events[1].data["args"]["sources"] == MOCK_SOURCES
        assert events[1].data["args"]["query"] == "test query"

    def test_receive_validation_response_approved(self, manager):
        """Test receiving approval response."""
        # Create checkpoint first
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )
        manager._pending_checkpoints["test-123"] = checkpoint

        # Receive approval
        result = manager.receive_validation_response(
            checkpoint_id="test-123",
            approved_source_ids=["source-1", "source-2"],
        )

        assert result.status == HITLStatus.APPROVED
        assert result.approved_source_ids == ["source-1", "source-2"]
        assert "source-3" in result.rejected_source_ids
        assert result.response_event.is_set()

    def test_receive_validation_response_rejected(self, manager):
        """Test receiving rejection response (no approvals)."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )
        manager._pending_checkpoints["test-123"] = checkpoint

        result = manager.receive_validation_response(
            checkpoint_id="test-123",
            approved_source_ids=[],
        )

        assert result.status == HITLStatus.REJECTED
        assert result.approved_source_ids == []
        assert len(result.rejected_source_ids) == 3

    def test_receive_validation_response_not_found(self, manager):
        """Test error when checkpoint not found."""
        with pytest.raises(KeyError) as exc_info:
            manager.receive_validation_response(
                checkpoint_id="nonexistent",
                approved_source_ids=["source-1"],
            )

        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wait_for_validation_success(self, manager):
        """Test waiting for validation with successful response."""
        # Create checkpoint
        checkpoint = await manager.create_checkpoint(
            sources=MOCK_SOURCES,
            query="test query",
            checkpoint_id="test-123",
        )

        # Simulate response in background
        async def respond_later():
            await asyncio.sleep(0.1)
            manager.receive_validation_response(
                checkpoint_id="test-123",
                approved_source_ids=["source-1"],
            )

        # Start response task and wait
        asyncio.create_task(respond_later())
        result = await manager.wait_for_validation("test-123")

        assert result.status == HITLStatus.APPROVED
        assert result.approved_source_ids == ["source-1"]

    @pytest.mark.asyncio
    async def test_wait_for_validation_timeout(self, manager):
        """Test waiting for validation with timeout."""
        # Create checkpoint
        await manager.create_checkpoint(
            sources=MOCK_SOURCES,
            query="test query",
            checkpoint_id="test-123",
        )

        # Wait with short timeout (no response sent)
        result = await manager.wait_for_validation("test-123", timeout=0.1)

        assert result.status == HITLStatus.SKIPPED
        # All sources should be auto-approved on timeout
        assert len(result.approved_source_ids) == 3

    @pytest.mark.asyncio
    async def test_wait_for_validation_not_found(self, manager):
        """Test error when waiting for nonexistent checkpoint."""
        with pytest.raises(KeyError):
            await manager.wait_for_validation("nonexistent")

    def test_get_approved_sources(self, manager):
        """Test getting approved sources from checkpoint."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )
        checkpoint.approved_source_ids = ["source-1", "source-3"]
        manager._pending_checkpoints["test-123"] = checkpoint

        approved = manager.get_approved_sources("test-123")

        assert len(approved) == 2
        assert approved[0]["id"] == "source-1"
        assert approved[1]["id"] == "source-3"

    def test_get_approved_sources_not_found(self, manager):
        """Test getting approved sources for nonexistent checkpoint."""
        approved = manager.get_approved_sources("nonexistent")

        assert approved == []

    def test_cleanup_checkpoint(self, manager):
        """Test checkpoint cleanup."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )
        manager._pending_checkpoints["test-123"] = checkpoint

        manager.cleanup_checkpoint("test-123")

        assert "test-123" not in manager._pending_checkpoints

    def test_cleanup_nonexistent_checkpoint(self, manager):
        """Test cleanup of nonexistent checkpoint doesn't error."""
        # Should not raise
        manager.cleanup_checkpoint("nonexistent")

    def test_get_completion_events(self, manager):
        """Test generating completion events."""
        checkpoint = HITLCheckpoint(
            checkpoint_id="test-123",
            sources=MOCK_SOURCES,
            query="test query",
        )
        checkpoint.approved_source_ids = ["source-1"]
        checkpoint.status = HITLStatus.APPROVED

        events = manager.get_completion_events(checkpoint)

        assert len(events) == 2
        # First event should be TOOL_CALL_END
        assert events[0].data["tool_call_id"] == "test-123"
        # Second event should be STATE_SNAPSHOT
        assert "hitl_checkpoint" in events[1].data["state"]
        assert "approved_sources" in events[1].data["state"]


class TestCreateValidateSourcesEvents:
    """Tests for create_validate_sources_events helper."""

    def test_create_events_basic(self):
        """Test basic event creation."""
        events = create_validate_sources_events(
            sources=MOCK_SOURCES,
            query="test query",
        )

        assert len(events) == 2
        assert events[0].data["tool_name"] == "validate_sources"
        assert events[1].data["args"]["sources"] == MOCK_SOURCES
        assert events[1].data["args"]["query"] == "test query"

    def test_create_events_with_custom_id(self):
        """Test event creation with custom checkpoint ID."""
        events = create_validate_sources_events(
            sources=MOCK_SOURCES,
            query="test query",
            checkpoint_id="custom-123",
        )

        assert events[0].data["tool_call_id"] == "custom-123"
        assert events[1].data["tool_call_id"] == "custom-123"

    def test_create_events_generates_id(self):
        """Test that checkpoint ID is generated if not provided."""
        events = create_validate_sources_events(
            sources=MOCK_SOURCES,
            query="test query",
        )

        # Should have generated a valid UUID
        assert events[0].data["tool_call_id"] is not None
        assert len(events[0].data["tool_call_id"]) == 36  # UUID format
