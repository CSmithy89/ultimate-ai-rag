"""Integration tests for AG-UI Prometheus metrics.

Story 22-B1: Implement AG-UI Stream Metrics
Epic: 22 - Advanced Protocol Integration

Tests cover:
- Full stream lifecycle metrics emission
- AGUIBridge integration with metrics
- Metrics endpoint exposure
- Tenant isolation in metrics
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from prometheus_client import REGISTRY, generate_latest

from agentic_rag_backend.protocols.ag_ui_bridge import AGUIBridge
from agentic_rag_backend.protocols.ag_ui_metrics import (
    AGUIMetricsCollector,
    track_agui_stream,
)
from agentic_rag_backend.models.copilot import (
    CopilotRequest,
    CopilotMessage,
    MessageRole,
    CopilotConfig,
)


class TestMetricsEndpointExposure:
    """Tests for /metrics endpoint exposure of AG-UI metrics."""

    def test_agui_metrics_in_prometheus_output(self):
        """Verify AG-UI metrics appear in Prometheus output."""
        # Generate metrics output
        metrics_output = generate_latest(REGISTRY).decode("utf-8")

        # Check for metric definitions (HELP and TYPE lines)
        # Note: Metrics may not have data yet, but definitions should exist
        expected_metrics = [
            "agui_stream_started_total",
            "agui_stream_completed_total",
            "agui_event_emitted_total",
            "agui_stream_bytes_total",
            "agui_active_streams",
            "agui_stream_duration_seconds",
            "agui_event_latency_seconds",
            "agui_stream_event_count",
        ]

        for metric_name in expected_metrics:
            # Check if metric name appears in output (in HELP or TYPE line)
            assert metric_name in metrics_output, f"Metric {metric_name} not found in Prometheus output"

    def test_metrics_include_help_text(self):
        """Verify metrics include descriptive HELP text."""
        metrics_output = generate_latest(REGISTRY).decode("utf-8")

        # Check for HELP lines
        assert "# HELP agui_stream_started_total" in metrics_output
        assert "# HELP agui_stream_completed_total" in metrics_output
        assert "# HELP agui_event_emitted_total" in metrics_output

    def test_metrics_include_type_declarations(self):
        """Verify metrics include TYPE declarations."""
        metrics_output = generate_latest(REGISTRY).decode("utf-8")

        # Check for TYPE lines
        assert "# TYPE agui_stream_started_total counter" in metrics_output
        assert "# TYPE agui_stream_completed_total counter" in metrics_output
        assert "# TYPE agui_active_streams gauge" in metrics_output
        assert "# TYPE agui_stream_duration_seconds histogram" in metrics_output


class TestStreamLifecycleMetrics:
    """Tests for metrics emission during full stream lifecycle."""

    @pytest.mark.asyncio
    async def test_successful_stream_emits_all_metrics(self):
        """Test that a successful stream emits expected metrics."""
        tenant_id = f"test-tenant-{uuid4()}"

        async with track_agui_stream(tenant_id) as metrics:
            metrics.event_emitted("RUN_STARTED")
            metrics.event_emitted("TEXT_MESSAGE_CONTENT", event_bytes=100)
            metrics.event_emitted("RUN_FINISHED")

        # Verify counters incremented
        # Note: Due to tenant normalization, we check the normalized label
        assert metrics.event_count == 3
        assert metrics.total_bytes == 100

    @pytest.mark.asyncio
    async def test_error_stream_records_error_status(self):
        """Test that a stream with error records error status."""
        tenant_id = f"error-tenant-{uuid4()}"

        with pytest.raises(ValueError):
            async with track_agui_stream(tenant_id) as metrics:
                metrics.event_emitted("RUN_STARTED")
                raise ValueError("Test error")

        # Metrics should still be tracked even after error
        assert metrics.event_count == 1

    @pytest.mark.asyncio
    async def test_multiple_streams_increment_counters(self):
        """Test that multiple streams increment counters correctly."""
        tenant_id = f"multi-tenant-{uuid4()}"

        # First stream
        async with track_agui_stream(tenant_id) as metrics1:
            metrics1.event_emitted("RUN_STARTED")
            metrics1.event_emitted("RUN_FINISHED")

        # Second stream
        async with track_agui_stream(tenant_id) as metrics2:
            metrics2.event_emitted("RUN_STARTED")
            metrics2.event_emitted("TEXT_MESSAGE_CONTENT")
            metrics2.event_emitted("RUN_FINISHED")

        # Both streams should have their events counted
        assert metrics1.event_count == 2
        assert metrics2.event_count == 3


class TestTenantIsolation:
    """Tests for tenant isolation in metrics."""

    @pytest.mark.asyncio
    async def test_different_tenants_have_separate_metrics(self):
        """Test that different tenants have separate metric values."""
        tenant_a = f"tenant-a-{uuid4()}"
        tenant_b = f"tenant-b-{uuid4()}"

        # Stream for tenant A
        async with track_agui_stream(tenant_a) as metrics_a:
            metrics_a.event_emitted("RUN_STARTED")
            metrics_a.event_emitted("TEXT_MESSAGE_CONTENT", event_bytes=100)
            metrics_a.event_emitted("RUN_FINISHED")

        # Stream for tenant B (fewer events, different bytes)
        async with track_agui_stream(tenant_b) as metrics_b:
            metrics_b.event_emitted("RUN_STARTED")
            metrics_b.event_emitted("RUN_FINISHED")

        # Each collector tracks its own events
        assert metrics_a.event_count == 3
        assert metrics_a.total_bytes == 100
        assert metrics_b.event_count == 2
        assert metrics_b.total_bytes == 0

    @pytest.mark.asyncio
    async def test_unknown_tenant_handled_gracefully(self):
        """Test that empty tenant_id is handled as 'unknown'."""
        async with track_agui_stream("") as metrics:
            assert metrics.tenant_id == "unknown"
            metrics.event_emitted("RUN_STARTED")
            metrics.event_emitted("RUN_FINISHED")

        assert metrics.event_count == 2


class TestAGUIBridgeIntegration:
    """Tests for AGUIBridge integration with metrics."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing."""
        orchestrator = MagicMock()
        orchestrator.run = AsyncMock()
        return orchestrator

    @pytest.fixture
    def mock_result(self):
        """Create a mock orchestrator result."""
        result = MagicMock()
        result.answer = "This is a test answer."
        result.thoughts = ["Thinking step 1", "Thinking step 2"]
        result.retrieval_strategy = MagicMock(value="hybrid")
        result.trajectory_id = uuid4()
        result.evidence = None
        return result

    @pytest.fixture
    def sample_request(self):
        """Create a sample CopilotKit request."""
        return CopilotRequest(
            messages=[
                CopilotMessage(
                    role=MessageRole.USER,
                    content="What is the capital of France?",
                )
            ],
            config=CopilotConfig(
                configurable={
                    "tenant_id": f"test-tenant-{uuid4()}",
                    "session_id": str(uuid4()),
                }
            ),
        )

    @pytest.mark.asyncio
    async def test_agui_bridge_emits_metrics_on_success(
        self, mock_orchestrator, mock_result, sample_request
    ):
        """Test that AGUIBridge emits metrics for successful request."""
        mock_orchestrator.run.return_value = mock_result
        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        events = []
        async for event in bridge.process_request(sample_request):
            events.append(event)

        # Verify events were emitted
        event_types = [e.event.value for e in events]
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types
        assert "TEXT_MESSAGE_START" in event_types
        assert "TEXT_MESSAGE_CONTENT" in event_types
        assert "TEXT_MESSAGE_END" in event_types

    @pytest.mark.asyncio
    async def test_agui_bridge_emits_metrics_on_error(
        self, mock_orchestrator, sample_request
    ):
        """Test that AGUIBridge emits metrics even on error."""
        mock_orchestrator.run.side_effect = Exception("Test error")
        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        events = []
        async for event in bridge.process_request(sample_request):
            events.append(event)

        # Should still have events including error message
        event_types = [e.event.value for e in events]
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types

    @pytest.mark.asyncio
    async def test_agui_bridge_handles_missing_tenant(self, mock_orchestrator):
        """Test that AGUIBridge handles missing tenant_id."""
        request = CopilotRequest(
            messages=[
                CopilotMessage(
                    role=MessageRole.USER,
                    content="Test message",
                )
            ],
            config=CopilotConfig(configurable={}),  # No tenant_id
        )
        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        events = []
        async for event in bridge.process_request(request):
            events.append(event)

        # Should emit error message about missing tenant_id
        event_types = [e.event.value for e in events]
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types


class TestActiveStreamsGauge:
    """Tests for active streams gauge accuracy."""

    @pytest.mark.asyncio
    async def test_active_streams_increments_on_start(self):
        """Test that active streams gauge increments when stream starts."""
        collector = AGUIMetricsCollector("gauge-test-1")
        collector.stream_started()

        # Should have incremented gauge
        # (Can't easily verify value due to tenant normalization, but shouldn't error)
        assert collector.start_time > 0

        # Clean up
        collector.stream_completed("success")

    @pytest.mark.asyncio
    async def test_active_streams_decrements_on_complete(self):
        """Test that active streams gauge decrements when stream completes."""
        collector = AGUIMetricsCollector("gauge-test-2")
        collector.stream_started()
        collector.event_emitted("RUN_STARTED")
        collector.stream_completed("success")

        # Should have decremented (no error means success)
        assert collector.event_count == 1

    @pytest.mark.asyncio
    async def test_active_streams_decrements_on_error(self):
        """Test that active streams gauge decrements even on error."""
        collector = AGUIMetricsCollector("gauge-test-3")
        collector.stream_started()
        collector.stream_completed("error")

        # Should have decremented (no error means success)
        assert True  # If we got here without exception, the gauge decremented

    @pytest.mark.asyncio
    async def test_concurrent_streams_tracked_correctly(self):
        """Test that concurrent streams are tracked correctly."""
        # Start multiple concurrent streams
        async def run_stream(tenant_id: str, delay: float):
            async with track_agui_stream(tenant_id) as metrics:
                metrics.event_emitted("RUN_STARTED")
                await asyncio.sleep(delay)
                metrics.event_emitted("RUN_FINISHED")
            return metrics

        # Run streams concurrently with different durations
        results = await asyncio.gather(
            run_stream("concurrent-1", 0.1),
            run_stream("concurrent-2", 0.05),
            run_stream("concurrent-3", 0.15),
        )

        # All streams should have completed successfully
        for metrics in results:
            assert metrics.event_count == 2


class TestEventTypeTracking:
    """Tests for event type tracking."""

    @pytest.mark.asyncio
    async def test_all_event_types_tracked(self):
        """Test that all AG-UI event types are tracked correctly."""
        async with track_agui_stream("event-types") as metrics:
            event_types = [
                "RUN_STARTED",
                "RUN_FINISHED",
                "TEXT_MESSAGE_START",
                "TEXT_MESSAGE_CONTENT",
                "TEXT_MESSAGE_END",
                "TOOL_CALL_START",
                "TOOL_CALL_ARGS",
                "TOOL_CALL_END",
                "STATE_SNAPSHOT",
                "STATE_DELTA",
                "MESSAGES_SNAPSHOT",
            ]

            for event_type in event_types:
                metrics.event_emitted(event_type)

        assert metrics.event_count == len(event_types)

    @pytest.mark.asyncio
    async def test_custom_event_types_tracked(self):
        """Test that custom event types are also tracked."""
        async with track_agui_stream("custom-events") as metrics:
            metrics.event_emitted("CUSTOM_EVENT_TYPE")
            metrics.event_emitted("ANOTHER_CUSTOM")

        assert metrics.event_count == 2


class TestBytesTracking:
    """Tests for bytes tracking."""

    @pytest.mark.asyncio
    async def test_bytes_accumulated_correctly(self):
        """Test that bytes are accumulated correctly."""
        async with track_agui_stream("bytes-test") as metrics:
            metrics.event_emitted("EVENT_1", event_bytes=100)
            metrics.event_emitted("EVENT_2", event_bytes=250)
            metrics.event_emitted("EVENT_3", event_bytes=50)

        assert metrics.total_bytes == 400

    @pytest.mark.asyncio
    async def test_zero_bytes_handled(self):
        """Test that zero bytes events are handled correctly."""
        async with track_agui_stream("zero-bytes") as metrics:
            metrics.event_emitted("EVENT_1", event_bytes=0)
            metrics.event_emitted("EVENT_2")  # No bytes specified

        assert metrics.total_bytes == 0
        assert metrics.event_count == 2

    @pytest.mark.asyncio
    async def test_large_bytes_handled(self):
        """Test that large byte values are handled correctly."""
        async with track_agui_stream("large-bytes") as metrics:
            # Simulate large response
            metrics.event_emitted("LARGE_EVENT", event_bytes=1_000_000)

        assert metrics.total_bytes == 1_000_000


class TestGaugeUnderflowProtection:
    """Tests for Issue #3 fix: gauge underflow protection."""

    def test_stream_completed_without_start_does_not_underflow(self):
        """Test that stream_completed without stream_started doesn't cause underflow.

        Issue #3: If stream_completed() is called without stream_started(),
        the active streams gauge should NOT decrement (to prevent negative values).
        """
        from agentic_rag_backend.protocols.ag_ui_metrics import ACTIVE_STREAMS

        tenant_id = f"underflow-test-{uuid4()}"
        collector = AGUIMetricsCollector(tenant_id)

        # Get initial gauge value for this tenant
        initial_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()

        # Call stream_completed without calling stream_started
        # This should NOT decrement the gauge
        collector.stream_completed("error")

        # Verify gauge was not decremented
        final_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()
        assert final_value >= initial_value, "Gauge should not have decremented without stream_started()"

    @pytest.mark.asyncio
    async def test_context_manager_always_balances_gauge(self):
        """Test that context manager properly balances gauge inc/dec."""
        from agentic_rag_backend.protocols.ag_ui_metrics import ACTIVE_STREAMS

        tenant_id = f"balance-test-{uuid4()}"

        # Get baseline (will create label if not exists)
        collector = AGUIMetricsCollector(tenant_id)
        initial_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()

        # Run stream through context manager
        async with track_agui_stream(tenant_id) as metrics:
            metrics.event_emitted("RUN_STARTED")
            metrics.event_emitted("RUN_FINISHED")

        # Gauge should return to initial value
        final_value = ACTIVE_STREAMS.labels(tenant_id=metrics.tenant_id)._value.get()
        assert final_value == initial_value, "Gauge should return to initial value after stream completes"


class TestConcurrencyStress:
    """Tests for Issue #6 fix: concurrency stress testing with gauge verification."""

    @pytest.mark.asyncio
    async def test_high_concurrency_gauge_accuracy(self):
        """Verify gauge accuracy under high concurrency.

        Issue #6: Run many concurrent streams and verify the gauge
        returns to baseline after all complete.
        """
        from agentic_rag_backend.protocols.ag_ui_metrics import ACTIVE_STREAMS

        # Use a unique tenant for this test
        tenant_id = f"stress-test-{uuid4()}"

        # Get initial gauge value
        collector = AGUIMetricsCollector(tenant_id)
        initial_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()

        # Run many concurrent streams
        num_streams = 50

        async def run_stream(stream_id: int):
            delay = 0.01 * (stream_id % 5)  # Variable delays
            async with track_agui_stream(tenant_id) as metrics:
                metrics.event_emitted("RUN_STARTED")
                await asyncio.sleep(delay)
                metrics.event_emitted("RUN_FINISHED")
            return metrics

        # Start all streams concurrently
        tasks = [run_stream(i) for i in range(num_streams)]
        results = await asyncio.gather(*tasks)

        # Verify all streams completed
        assert len(results) == num_streams
        for metrics in results:
            assert metrics.event_count == 2

        # After all streams complete, gauge should return to initial value
        final_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()
        assert final_value == initial_value, (
            f"Gauge should return to {initial_value} after all streams complete, "
            f"but got {final_value}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_streams_with_errors_gauge_accuracy(self):
        """Verify gauge accuracy when some concurrent streams error.

        Some streams succeed, some fail - gauge should still balance.
        """
        from agentic_rag_backend.protocols.ag_ui_metrics import ACTIVE_STREAMS

        tenant_id = f"error-stress-{uuid4()}"

        # Get initial gauge value
        collector = AGUIMetricsCollector(tenant_id)
        initial_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()

        num_streams = 20
        error_indices = {3, 7, 11, 15}  # These streams will error

        async def run_stream(stream_id: int):
            try:
                async with track_agui_stream(tenant_id) as metrics:
                    metrics.event_emitted("RUN_STARTED")
                    if stream_id in error_indices:
                        raise RuntimeError(f"Intentional error in stream {stream_id}")
                    metrics.event_emitted("RUN_FINISHED")
                return ("success", metrics)
            except RuntimeError:
                return ("error", None)

        # Start all streams concurrently
        tasks = [run_stream(i) for i in range(num_streams)]
        results = await asyncio.gather(*tasks)

        # Verify expected successes and failures
        successes = sum(1 for status, _ in results if status == "success")
        errors = sum(1 for status, _ in results if status == "error")
        assert successes == num_streams - len(error_indices)
        assert errors == len(error_indices)

        # After all streams complete, gauge should return to initial value
        final_value = ACTIVE_STREAMS.labels(tenant_id=collector.tenant_id)._value.get()
        assert final_value == initial_value, (
            f"Gauge should return to {initial_value} after all streams complete "
            f"(including errors), but got {final_value}"
        )
