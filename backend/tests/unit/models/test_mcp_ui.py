"""Unit tests for MCP-UI Pydantic models.

Story 22-C1: Implement MCP-UI Renderer
"""

import pytest
from pydantic import ValidationError

from agentic_rag_backend.models.mcp_ui import (
    MCPUIConfig,
    MCPUIPayload,
    MCPUIResizeMessage,
    MCPUIResultMessage,
    MCPUIErrorMessage,
)


class TestMCPUIConfig:
    """Tests for MCPUIConfig model."""

    def test_default_values(self) -> None:
        """MCPUIConfig should have correct defaults."""
        config = MCPUIConfig()
        assert config.enabled is False
        assert config.allowed_origins == []

    def test_with_origins(self) -> None:
        """MCPUIConfig should accept allowed origins."""
        config = MCPUIConfig(
            enabled=True,
            allowed_origins=["https://example.com", "https://tools.example.com"],
        )
        assert config.enabled is True
        assert len(config.allowed_origins) == 2
        assert "https://example.com" in config.allowed_origins

    def test_empty_origins_list(self) -> None:
        """MCPUIConfig should allow empty origins list."""
        config = MCPUIConfig(enabled=True, allowed_origins=[])
        assert config.enabled is True
        assert config.allowed_origins == []


class TestMCPUIPayload:
    """Tests for MCPUIPayload model."""

    def test_required_fields(self) -> None:
        """MCPUIPayload should require tool_name and ui_url."""
        with pytest.raises(ValidationError) as exc_info:
            MCPUIPayload()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "tool_name" in field_names
        assert "ui_url" in field_names

    def test_valid_payload(self) -> None:
        """MCPUIPayload should accept valid data."""
        payload = MCPUIPayload(
            tool_name="calculator",
            ui_url="https://tools.example.com/calculator",  # type: ignore
        )
        assert payload.tool_name == "calculator"
        assert str(payload.ui_url) == "https://tools.example.com/calculator"
        assert payload.type == "mcp_ui"
        assert payload.ui_type == "iframe"

    def test_default_sandbox(self) -> None:
        """MCPUIPayload should have default sandbox with only allow-scripts.

        Note: allow-same-origin is NOT included by default as combining it
        with allow-scripts weakens security (iframe could escape sandbox).
        """
        payload = MCPUIPayload(
            tool_name="test",
            ui_url="https://example.com",  # type: ignore
        )
        assert "allow-scripts" in payload.sandbox
        assert "allow-same-origin" not in payload.sandbox

    def test_default_size(self) -> None:
        """MCPUIPayload should have default size."""
        payload = MCPUIPayload(
            tool_name="test",
            ui_url="https://example.com",  # type: ignore
        )
        assert payload.size["width"] == 600
        assert payload.size["height"] == 400

    def test_custom_size(self) -> None:
        """MCPUIPayload should accept custom size."""
        payload = MCPUIPayload(
            tool_name="test",
            ui_url="https://example.com",  # type: ignore
            size={"width": 800, "height": 600},
        )
        assert payload.size["width"] == 800
        assert payload.size["height"] == 600

    def test_custom_sandbox(self) -> None:
        """MCPUIPayload should accept custom sandbox permissions."""
        payload = MCPUIPayload(
            tool_name="test",
            ui_url="https://example.com",  # type: ignore
            sandbox=["allow-scripts"],
        )
        assert payload.sandbox == ["allow-scripts"]

    def test_data_field(self) -> None:
        """MCPUIPayload should accept arbitrary data."""
        payload = MCPUIPayload(
            tool_name="test",
            ui_url="https://example.com",  # type: ignore
            data={"initial_value": 42, "options": {"theme": "dark"}},
        )
        assert payload.data["initial_value"] == 42
        assert payload.data["options"]["theme"] == "dark"

    def test_invalid_url(self) -> None:
        """MCPUIPayload should reject invalid URLs."""
        with pytest.raises(ValidationError):
            MCPUIPayload(
                tool_name="test",
                ui_url="not-a-valid-url",  # type: ignore
            )

    def test_http_url(self) -> None:
        """MCPUIPayload should accept HTTP URLs (for dev environments)."""
        payload = MCPUIPayload(
            tool_name="test",
            ui_url="http://localhost:8080/tool",  # type: ignore
        )
        assert str(payload.ui_url) == "http://localhost:8080/tool"


class TestMCPUIResizeMessage:
    """Tests for MCPUIResizeMessage model."""

    def test_valid_resize(self) -> None:
        """MCPUIResizeMessage should accept valid dimensions."""
        msg = MCPUIResizeMessage(width=800, height=600)
        assert msg.type == "mcp_ui_resize"
        assert msg.width == 800
        assert msg.height == 600

    def test_min_dimensions(self) -> None:
        """MCPUIResizeMessage should enforce minimum dimensions."""
        with pytest.raises(ValidationError):
            MCPUIResizeMessage(width=50, height=600)  # width < 100
        with pytest.raises(ValidationError):
            MCPUIResizeMessage(width=800, height=10)  # height < 50

    def test_max_dimensions(self) -> None:
        """MCPUIResizeMessage should enforce maximum dimensions."""
        with pytest.raises(ValidationError):
            MCPUIResizeMessage(width=5000, height=600)  # width > 4000
        with pytest.raises(ValidationError):
            MCPUIResizeMessage(width=800, height=5000)  # height > 4000


class TestMCPUIResultMessage:
    """Tests for MCPUIResultMessage model."""

    def test_valid_result(self) -> None:
        """MCPUIResultMessage should accept any result type."""
        msg = MCPUIResultMessage(result={"value": 42})
        assert msg.type == "mcp_ui_result"
        assert msg.result == {"value": 42}

    def test_string_result(self) -> None:
        """MCPUIResultMessage should accept string result."""
        msg = MCPUIResultMessage(result="success")
        assert msg.result == "success"

    def test_list_result(self) -> None:
        """MCPUIResultMessage should accept list result."""
        msg = MCPUIResultMessage(result=[1, 2, 3])
        assert msg.result == [1, 2, 3]

    def test_none_result(self) -> None:
        """MCPUIResultMessage should accept None result."""
        msg = MCPUIResultMessage(result=None)
        assert msg.result is None


class TestMCPUIErrorMessage:
    """Tests for MCPUIErrorMessage model."""

    def test_valid_error(self) -> None:
        """MCPUIErrorMessage should accept error string."""
        msg = MCPUIErrorMessage(error="Something went wrong")
        assert msg.type == "mcp_ui_error"
        assert msg.error == "Something went wrong"

    def test_empty_error(self) -> None:
        """MCPUIErrorMessage should accept empty error string."""
        msg = MCPUIErrorMessage(error="")
        assert msg.error == ""

    def test_error_required(self) -> None:
        """MCPUIErrorMessage should require error field."""
        with pytest.raises(ValidationError):
            MCPUIErrorMessage()  # type: ignore
