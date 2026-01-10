"""Tests for MCP Client configuration.

Story 21-C2: Implement MCP Client Factory
"""

import pytest
from pydantic import ValidationError

from agentic_rag_backend.mcp_client.config import MCPClientSettings, MCPServerConfig


class TestMCPServerConfig:
    """Tests for MCPServerConfig model."""

    def test_minimal_config(self) -> None:
        """Test config with required fields only."""
        config = MCPServerConfig(
            name="test",
            url="https://mcp.example.com/sse",
        )
        assert config.name == "test"
        assert str(config.url) == "https://mcp.example.com/sse"
        assert config.api_key is None
        assert config.transport == "sse"
        assert config.timeout_ms == 30000

    def test_full_config(self) -> None:
        """Test config with all fields."""
        config = MCPServerConfig(
            name="github",
            url="https://mcp.github.com/sse",
            api_key="secret-key",
            transport="http",
            timeout_ms=60000,
        )
        assert config.name == "github"
        assert config.api_key == "secret-key"
        assert config.transport == "http"
        assert config.timeout_ms == 60000

    def test_alias_fields(self) -> None:
        """Test that alias fields work (apiKey, timeout)."""
        config = MCPServerConfig(
            name="test",
            url="https://mcp.example.com/sse",
            apiKey="aliased-key",  # Using alias
            timeout=15000,  # Using alias
        )
        assert config.api_key == "aliased-key"
        assert config.timeout_ms == 15000

    def test_invalid_url(self) -> None:
        """Test validation of invalid URL."""
        with pytest.raises(ValidationError):
            MCPServerConfig(
                name="test",
                url="not-a-valid-url",
            )

    def test_timeout_min_value(self) -> None:
        """Test timeout minimum value enforcement."""
        with pytest.raises(ValidationError):
            MCPServerConfig(
                name="test",
                url="https://mcp.example.com/sse",
                timeout_ms=500,  # Below 1000 minimum
            )


class TestMCPClientSettings:
    """Tests for MCPClientSettings model."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = MCPClientSettings()
        assert settings.enabled is False
        assert settings.servers == []
        assert settings.default_timeout_ms == 30000
        assert settings.retry_count == 3
        assert settings.retry_delay_ms == 1000

    def test_enabled_with_servers(self) -> None:
        """Test enabled settings with servers."""
        settings = MCPClientSettings(
            enabled=True,
            servers=[
                MCPServerConfig(name="server1", url="https://mcp1.example.com"),
                MCPServerConfig(name="server2", url="https://mcp2.example.com"),
            ],
        )
        assert settings.enabled is True
        assert len(settings.servers) == 2
        assert settings.servers[0].name == "server1"
        assert settings.servers[1].name == "server2"

    def test_retry_count_max(self) -> None:
        """Test retry count max value enforcement."""
        with pytest.raises(ValidationError):
            MCPClientSettings(retry_count=15)  # Above 10 max

    def test_retry_count_min(self) -> None:
        """Test retry count min value (0 is allowed)."""
        settings = MCPClientSettings(retry_count=0)
        assert settings.retry_count == 0

    def test_timeout_min_value(self) -> None:
        """Test timeout minimum value."""
        with pytest.raises(ValidationError):
            MCPClientSettings(default_timeout_ms=500)

    def test_retry_delay_min_value(self) -> None:
        """Test retry delay minimum value."""
        with pytest.raises(ValidationError):
            MCPClientSettings(retry_delay_ms=50)  # Below 100 minimum
