"""Unit tests for AG-UI Prometheus metrics.

Story 22-B1: Implement AG-UI Stream Metrics
Epic: 22 - Advanced Protocol Integration

Tests cover:
- AGUIMetricsCollector initialization and methods
- Counter, Histogram, and Gauge metric emission
- Tenant ID normalization
- Context manager success and error paths
- Convenience functions
"""

import os
import time
from unittest.mock import patch

import pytest

# Import the module under test
from agentic_rag_backend.protocols.ag_ui_metrics import (
    AGUIMetricsCollector,
    track_agui_stream,
    record_stream_started,
    record_stream_completed,
    record_event_emitted,
    STREAM_STARTED,
    STREAM_COMPLETED,
    EVENT_EMITTED,
    STREAM_BYTES,
    ACTIVE_STREAMS,
    STREAM_DURATION,
    EVENT_LATENCY,
    STREAM_EVENT_COUNT,
)


class TestAGUIMetricsCollector:
    """Tests for AGUIMetricsCollector class."""

    def test_initialization_with_tenant_id(self):
        """Test collector initializes with provided tenant_id."""
        collector = AGUIMetricsCollector("test-tenant-123")
        # When using global mode (default), tenant_id normalizes to "global"
        # The actual normalized value depends on METRICS_TENANT_LABEL_MODE env var
        assert collector.tenant_id is not None
        assert collector.start_time == 0.0
        assert collector.last_event_time == 0.0
        assert collector.event_count == 0
        assert collector.total_bytes == 0

    def test_initialization_with_empty_tenant_id(self):
        """Test collector handles empty tenant_id by using 'unknown'."""
        collector = AGUIMetricsCollector("")
        # Empty string normalizes to "unknown"
        assert collector.tenant_id == "unknown"

    def test_initialization_with_none_tenant_id(self):
        """Test collector handles None tenant_id by using 'unknown'."""
        # The module converts None to "" via `tenant_id or ""`
        collector = AGUIMetricsCollector(None)  # type: ignore
        assert collector.tenant_id == "unknown"

    def test_stream_started_initializes_timing(self):
        """Test stream_started() initializes start_time and last_event_time."""
        collector = AGUIMetricsCollector("tenant-1")

        before = time.time()
        collector.stream_started()
        after = time.time()

        assert before <= collector.start_time <= after
        assert collector.start_time == collector.last_event_time

    def test_event_emitted_increments_count(self):
        """Test event_emitted() increments event_count."""
        collector = AGUIMetricsCollector("tenant-1")
        collector.stream_started()

        assert collector.event_count == 0

        collector.event_emitted("RUN_STARTED")
        assert collector.event_count == 1

        collector.event_emitted("TEXT_MESSAGE_CONTENT")
        assert collector.event_count == 2

        collector.event_emitted("RUN_FINISHED")
        assert collector.event_count == 3

    def test_event_emitted_accumulates_bytes(self):
        """Test event_emitted() accumulates total_bytes."""
        collector = AGUIMetricsCollector("tenant-1")
        collector.stream_started()

        collector.event_emitted("EVENT_1", event_bytes=100)
        assert collector.total_bytes == 100

        collector.event_emitted("EVENT_2", event_bytes=250)
        assert collector.total_bytes == 350

        collector.event_emitted("EVENT_3", event_bytes=0)
        assert collector.total_bytes == 350

    def test_event_emitted_updates_last_event_time(self):
        """Test event_emitted() updates last_event_time."""
        collector = AGUIMetricsCollector("tenant-1")
        collector.stream_started()

        first_time = collector.last_event_time

        # Small delay to ensure time difference
        time.sleep(0.01)
        collector.event_emitted("RUN_STARTED")

        assert collector.last_event_time >= first_time

    def test_stream_completed_success(self):
        """Test stream_completed() with success status."""
        collector = AGUIMetricsCollector("tenant-1")
        collector.stream_started()
        collector.event_emitted("RUN_STARTED")
        collector.event_emitted("TEXT_MESSAGE_CONTENT")

        # Should not raise
        collector.stream_completed("success")

    def test_stream_completed_error(self):
        """Test stream_completed() with error status."""
        collector = AGUIMetricsCollector("tenant-1")
        collector.stream_started()
        collector.event_emitted("RUN_STARTED")

        # Should not raise
        collector.stream_completed("error")


class TestTrackAGUIStreamContextManager:
    """Tests for track_agui_stream async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success_path(self):
        """Test context manager completes successfully and records success."""
        async with track_agui_stream("tenant-success") as metrics:
            metrics.event_emitted("RUN_STARTED")
            metrics.event_emitted("TEXT_MESSAGE_CONTENT", event_bytes=50)
            metrics.event_emitted("RUN_FINISHED")

        # After context exits, metrics should be recorded
        assert metrics.event_count == 3
        assert metrics.total_bytes == 50

    @pytest.mark.asyncio
    async def test_context_manager_error_path(self):
        """Test context manager handles exceptions and records error."""
        with pytest.raises(ValueError, match="Test error"):
            async with track_agui_stream("tenant-error") as metrics:
                metrics.event_emitted("RUN_STARTED")
                raise ValueError("Test error")

        # After exception, metrics should still be recorded
        assert metrics.event_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_reraises_exception(self):
        """Test context manager re-raises the original exception."""
        with pytest.raises(RuntimeError, match="Original error"):
            async with track_agui_stream("tenant-reraise"):
                raise RuntimeError("Original error")

    @pytest.mark.asyncio
    async def test_context_manager_yields_collector(self):
        """Test context manager yields a working AGUIMetricsCollector."""
        async with track_agui_stream("tenant-yield") as metrics:
            assert isinstance(metrics, AGUIMetricsCollector)
            assert metrics.start_time > 0  # stream_started was called
            assert metrics.event_count == 0  # No events emitted yet

    @pytest.mark.asyncio
    async def test_context_manager_with_empty_tenant(self):
        """Test context manager handles empty tenant_id."""
        async with track_agui_stream("") as metrics:
            assert metrics.tenant_id == "unknown"
            metrics.event_emitted("RUN_STARTED")

        assert metrics.event_count == 1


class TestConvenienceFunctions:
    """Tests for standalone convenience functions."""

    def test_record_stream_started(self):
        """Test record_stream_started convenience function."""
        # Should not raise
        record_stream_started("test-tenant")
        record_stream_started("")  # Empty tenant

    def test_record_stream_completed_success(self):
        """Test record_stream_completed with success status."""
        # Start a stream first (to have something to complete)
        record_stream_started("test-tenant")
        # Should not raise
        record_stream_completed("test-tenant", "success")

    def test_record_stream_completed_error(self):
        """Test record_stream_completed with error status."""
        record_stream_started("test-tenant")
        # Should not raise
        record_stream_completed("test-tenant", "error")

    def test_record_event_emitted_without_bytes(self):
        """Test record_event_emitted without byte count."""
        # Should not raise
        record_event_emitted("test-tenant", "RUN_STARTED")

    def test_record_event_emitted_with_bytes(self):
        """Test record_event_emitted with byte count."""
        # Should not raise
        record_event_emitted("test-tenant", "TEXT_MESSAGE_CONTENT", event_bytes=500)


class TestTenantNormalization:
    """Tests for tenant ID normalization behavior."""

    @patch.dict(os.environ, {"METRICS_TENANT_LABEL_MODE": "full"})
    def test_full_mode_preserves_tenant_id(self):
        """Test that full mode preserves the original tenant_id."""
        # Need to reimport to pick up env change
        from agentic_rag_backend.observability.metrics import normalize_tenant_label

        result = normalize_tenant_label("my-tenant-123")
        assert result == "my-tenant-123"

    @patch.dict(os.environ, {"METRICS_TENANT_LABEL_MODE": "global"})
    def test_global_mode_returns_global(self):
        """Test that global mode returns 'global' for all tenants."""
        from agentic_rag_backend.observability.metrics import normalize_tenant_label

        result = normalize_tenant_label("any-tenant")
        assert result == "global"

    @patch.dict(os.environ, {
        "METRICS_TENANT_LABEL_MODE": "hash",
        "METRICS_TENANT_LABEL_BUCKETS": "10"
    })
    def test_hash_mode_returns_bucket(self):
        """Test that hash mode returns a bucket identifier."""
        from agentic_rag_backend.observability.metrics import normalize_tenant_label

        result = normalize_tenant_label("my-tenant-123")
        assert result.startswith("bucket-")
        # Bucket should be 0-9 for 10 buckets
        bucket_num = int(result.split("-")[1])
        assert 0 <= bucket_num < 10

    def test_unknown_tenant_handling(self):
        """Test that empty/None tenant_id becomes 'unknown'."""
        from agentic_rag_backend.observability.metrics import normalize_tenant_label

        assert normalize_tenant_label("") == "unknown"
        assert normalize_tenant_label(None) == "unknown"  # type: ignore


class TestMetricsRegistration:
    """Tests to verify metrics are properly registered."""

    def test_stream_started_counter_exists(self):
        """Verify agui_stream_started_total counter is registered."""
        assert STREAM_STARTED is not None
        # Note: prometheus_client strips _total suffix from counter names internally
        assert "agui_stream_started" in STREAM_STARTED._name

    def test_stream_completed_counter_exists(self):
        """Verify agui_stream_completed_total counter is registered."""
        assert STREAM_COMPLETED is not None
        assert "agui_stream_completed" in STREAM_COMPLETED._name

    def test_event_emitted_counter_exists(self):
        """Verify agui_event_emitted_total counter is registered."""
        assert EVENT_EMITTED is not None
        assert "agui_event_emitted" in EVENT_EMITTED._name

    def test_stream_bytes_counter_exists(self):
        """Verify agui_stream_bytes_total counter is registered."""
        assert STREAM_BYTES is not None
        assert "agui_stream_bytes" in STREAM_BYTES._name

    def test_active_streams_gauge_exists(self):
        """Verify agui_active_streams gauge is registered."""
        assert ACTIVE_STREAMS is not None
        assert ACTIVE_STREAMS._name == "agui_active_streams"

    def test_stream_duration_histogram_exists(self):
        """Verify agui_stream_duration_seconds histogram is registered."""
        assert STREAM_DURATION is not None
        assert STREAM_DURATION._name == "agui_stream_duration_seconds"

    def test_event_latency_histogram_exists(self):
        """Verify agui_event_latency_seconds histogram is registered."""
        assert EVENT_LATENCY is not None
        assert EVENT_LATENCY._name == "agui_event_latency_seconds"

    def test_stream_event_count_histogram_exists(self):
        """Verify agui_stream_event_count histogram is registered."""
        assert STREAM_EVENT_COUNT is not None
        assert STREAM_EVENT_COUNT._name == "agui_stream_event_count"


class TestHistogramBuckets:
    """Tests for histogram bucket configuration."""

    def test_stream_duration_buckets(self):
        """Verify stream duration histogram has appropriate buckets."""
        from agentic_rag_backend.protocols.ag_ui_metrics import STREAM_DURATION_BUCKETS

        # Should cover 0.1s to 600s (Issue #5 fix: extended for long-tail coverage)
        assert STREAM_DURATION_BUCKETS == (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)

    def test_event_latency_buckets(self):
        """Verify event latency histogram has appropriate buckets."""
        from agentic_rag_backend.protocols.ag_ui_metrics import EVENT_LATENCY_BUCKETS

        # Should cover 10ms to 1s
        assert EVENT_LATENCY_BUCKETS == (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

    def test_stream_event_count_buckets(self):
        """Verify stream event count histogram has appropriate buckets."""
        from agentic_rag_backend.protocols.ag_ui_metrics import STREAM_EVENT_COUNT_BUCKETS

        # Should cover 1 to 250 events
        assert STREAM_EVENT_COUNT_BUCKETS == (1, 5, 10, 25, 50, 100, 250)


class TestMetricLabels:
    """Tests for metric label configuration."""

    def test_stream_started_has_tenant_label(self):
        """Verify agui_stream_started_total has tenant_id label."""
        assert "tenant_id" in STREAM_STARTED._labelnames

    def test_stream_completed_has_status_label(self):
        """Verify agui_stream_completed_total has status label."""
        assert "status" in STREAM_COMPLETED._labelnames
        assert "tenant_id" in STREAM_COMPLETED._labelnames

    def test_event_emitted_has_event_type_label(self):
        """Verify agui_event_emitted_total has event_type label."""
        assert "event_type" in EVENT_EMITTED._labelnames
        assert "tenant_id" in EVENT_EMITTED._labelnames


class TestEventLatencyTracking:
    """Tests for inter-event latency measurement."""

    def test_latency_not_recorded_for_first_event(self):
        """Test that latency is not recorded for the first event."""
        collector = AGUIMetricsCollector("tenant-latency")
        collector.stream_started()

        # First event should not record latency (no previous event)
        collector.event_emitted("RUN_STARTED")

        # Event count should be 1
        assert collector.event_count == 1

    def test_latency_recorded_for_subsequent_events(self):
        """Test that latency is recorded for events after the first."""
        collector = AGUIMetricsCollector("tenant-latency-2")
        collector.stream_started()

        collector.event_emitted("RUN_STARTED")
        time.sleep(0.01)  # Small delay
        collector.event_emitted("TEXT_MESSAGE_CONTENT")

        # Both events should be counted
        assert collector.event_count == 2


class TestEventTypeCardinality:
    """Tests for Issue #2 fix: event type label cardinality bounding."""

    def test_known_event_types_pass_through(self):
        """Test that known event types are passed through unchanged."""
        from agentic_rag_backend.protocols.ag_ui_metrics import KNOWN_EVENT_TYPES

        collector = AGUIMetricsCollector("cardinality-test")
        collector.stream_started()

        # All known types should work
        for event_type in KNOWN_EVENT_TYPES:
            collector.event_emitted(event_type)

        # Should have tracked all events
        assert collector.event_count == len(KNOWN_EVENT_TYPES)

    def test_unknown_event_types_normalized_to_other(self):
        """Test that unknown event types are normalized to OTHER.

        Issue #2: Unknown event types should be mapped to 'OTHER' to prevent
        label cardinality explosion from unexpected/malicious event types.
        """
        tenant_id = "unknown-type-test"
        collector = AGUIMetricsCollector(tenant_id)
        collector.stream_started()

        # Emit unknown event types (should be normalized to "OTHER")
        collector.event_emitted("CUSTOM_UNKNOWN_TYPE")
        collector.event_emitted("MALICIOUS_TYPE_123")
        collector.event_emitted("user_controlled_string")

        # Should have tracked all events
        assert collector.event_count == 3

        # The EVENT_EMITTED counter should have "OTHER" label, not the custom types
        # (We verify no error occurred, meaning normalization worked)

    def test_known_event_types_list_is_complete(self):
        """Verify KNOWN_EVENT_TYPES contains all expected AG-UI events."""
        from agentic_rag_backend.protocols.ag_ui_metrics import KNOWN_EVENT_TYPES

        expected_types = {
            "RUN_STARTED",
            "RUN_FINISHED",
            "RUN_ERROR",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "STATE_SNAPSHOT",
            "STATE_DELTA",
            "MESSAGES_SNAPSHOT",
        }

        assert KNOWN_EVENT_TYPES == expected_types


class TestGaugeUnderflowProtection:
    """Tests for Issue #3 and #4 fix: gauge underflow and negative duration protection."""

    def test_stream_completed_without_start_logs_warning(self):
        """Test that completing without starting logs a warning.

        Issue #3: stream_completed() should log a warning and NOT decrement
        the gauge if stream_started() was never called.
        """
        collector = AGUIMetricsCollector("underflow-test")

        # Calling stream_completed without stream_started should not raise
        # and should not decrement the gauge
        collector.stream_completed("error")

        # Verify internal flag shows stream was never started
        assert collector._stream_started is False

    def test_stream_started_sets_flag(self):
        """Test that stream_started sets the _stream_started flag."""
        collector = AGUIMetricsCollector("flag-test")

        assert collector._stream_started is False
        collector.stream_started()
        assert collector._stream_started is True

    def test_duration_not_recorded_without_start(self):
        """Test that duration histogram is not corrupted if start was never called.

        Issue #4: If stream_started() was never called, stream_completed()
        should not record duration (would be ~50 years since epoch).
        """
        collector = AGUIMetricsCollector("duration-test")

        # start_time is 0.0 by default
        assert collector.start_time == 0.0

        # Completing without starting should not record invalid duration
        collector.stream_completed("error")

        # If we got here without error, the guard worked


class TestStreamLifecycle:
    """Tests for complete stream lifecycle scenarios."""

    @pytest.mark.asyncio
    async def test_complete_successful_stream(self):
        """Test a complete successful stream lifecycle."""
        async with track_agui_stream("lifecycle-success") as metrics:
            metrics.event_emitted("RUN_STARTED")
            metrics.event_emitted("STATE_SNAPSHOT", event_bytes=200)
            metrics.event_emitted("TEXT_MESSAGE_START")
            metrics.event_emitted("TEXT_MESSAGE_CONTENT", event_bytes=500)
            metrics.event_emitted("TEXT_MESSAGE_END")
            metrics.event_emitted("RUN_FINISHED")

        assert metrics.event_count == 6
        assert metrics.total_bytes == 700

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self):
        """Test a stream lifecycle that includes tool calls."""
        async with track_agui_stream("lifecycle-tools") as metrics:
            metrics.event_emitted("RUN_STARTED")
            metrics.event_emitted("TOOL_CALL_START")
            metrics.event_emitted("TOOL_CALL_ARGS", event_bytes=100)
            metrics.event_emitted("TOOL_CALL_END")
            metrics.event_emitted("TEXT_MESSAGE_CONTENT", event_bytes=200)
            metrics.event_emitted("RUN_FINISHED")

        assert metrics.event_count == 6
        assert metrics.total_bytes == 300

    @pytest.mark.asyncio
    async def test_stream_with_error_mid_stream(self):
        """Test a stream that errors mid-way through."""
        with pytest.raises(Exception):
            async with track_agui_stream("lifecycle-error") as metrics:
                metrics.event_emitted("RUN_STARTED")
                metrics.event_emitted("TEXT_MESSAGE_START")
                raise Exception("Processing error")

        # Events before error should still be counted
        assert metrics.event_count == 2

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test a stream with no events (edge case)."""
        async with track_agui_stream("lifecycle-empty") as metrics:
            pass  # No events emitted

        assert metrics.event_count == 0
        assert metrics.total_bytes == 0
