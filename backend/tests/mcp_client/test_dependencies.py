"""Tests for MCP Client dependencies.

Story 21-C2: Implement MCP Client Factory
"""

from unittest.mock import MagicMock

import pytest
from fastapi import Request

from agentic_rag_backend.mcp_client.client import MCPClientFactory
from agentic_rag_backend.mcp_client.dependencies import (
    create_mcp_client_factory,
    create_mcp_client_settings,
    get_mcp_factory,
)


class TestCreateMCPClientSettings:
    """Tests for create_mcp_client_settings function."""

    def test_creates_settings_from_app_settings(self) -> None:
        """Test creating MCP settings from app Settings."""
        # Mock application settings with MCP client fields
        app_settings = MagicMock()
        app_settings.mcp_clients_enabled = True
        app_settings.mcp_client_servers = [
            {"name": "github", "url": "https://mcp.github.com/sse", "apiKey": "key1"},
            {"name": "notion", "url": "https://mcp.notion.so/sse"},
        ]
        app_settings.mcp_client_timeout_ms = 45000
        app_settings.mcp_client_retry_count = 5
        app_settings.mcp_client_retry_delay_ms = 2000

        settings = create_mcp_client_settings(app_settings)

        assert settings.enabled is True
        assert len(settings.servers) == 2
        assert settings.servers[0].name == "github"
        assert settings.servers[0].api_key == "key1"
        assert settings.servers[1].name == "notion"
        assert settings.servers[1].api_key is None
        assert settings.default_timeout_ms == 45000
        assert settings.retry_count == 5
        assert settings.retry_delay_ms == 2000

    def test_skips_invalid_server_configs(self) -> None:
        """Test that invalid server configs are skipped."""
        app_settings = MagicMock()
        app_settings.mcp_clients_enabled = True
        app_settings.mcp_client_servers = [
            {"name": "valid", "url": "https://mcp.example.com"},
            {"invalid": "config"},  # Missing required fields
            {"name": "valid2", "url": "https://mcp2.example.com"},
        ]
        app_settings.mcp_client_timeout_ms = 30000
        app_settings.mcp_client_retry_count = 3
        app_settings.mcp_client_retry_delay_ms = 1000

        settings = create_mcp_client_settings(app_settings)

        # Should only have 2 valid servers
        assert len(settings.servers) == 2
        assert settings.servers[0].name == "valid"
        assert settings.servers[1].name == "valid2"

    def test_disabled_settings(self) -> None:
        """Test creating disabled settings."""
        app_settings = MagicMock()
        app_settings.mcp_clients_enabled = False
        app_settings.mcp_client_servers = []
        app_settings.mcp_client_timeout_ms = 30000
        app_settings.mcp_client_retry_count = 3
        app_settings.mcp_client_retry_delay_ms = 1000

        settings = create_mcp_client_settings(app_settings)

        assert settings.enabled is False
        assert settings.servers == []


class TestCreateMCPClientFactory:
    """Tests for create_mcp_client_factory function."""

    def test_creates_factory_from_app_settings(self) -> None:
        """Test creating factory from app Settings."""
        app_settings = MagicMock()
        app_settings.mcp_clients_enabled = True
        app_settings.mcp_client_servers = [
            {"name": "test", "url": "https://mcp.example.com"},
        ]
        app_settings.mcp_client_timeout_ms = 30000
        app_settings.mcp_client_retry_count = 3
        app_settings.mcp_client_retry_delay_ms = 1000

        factory = create_mcp_client_factory(app_settings)

        assert isinstance(factory, MCPClientFactory)
        assert factory.is_enabled is True
        assert factory.server_names == ["test"]


class TestGetMCPFactory:
    """Tests for get_mcp_factory dependency."""

    @pytest.mark.asyncio
    async def test_returns_factory_from_app_state(self) -> None:
        """Test getting factory from app state."""
        # Create mock request with app state
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_app = MagicMock()
        mock_app.state.mcp_client_factory = mock_factory
        mock_request = MagicMock(spec=Request)
        mock_request.app = mock_app

        factory = await get_mcp_factory(mock_request)

        assert factory is mock_factory

    @pytest.mark.asyncio
    async def test_returns_none_when_not_initialized(self) -> None:
        """Test returns None when factory not in state."""
        # Create a mock state without mcp_client_factory attribute
        class MockState:
            """State object without mcp_client_factory."""
            pass

        mock_app = MagicMock()
        mock_app.state = MockState()
        mock_request = MagicMock(spec=Request)
        mock_request.app = mock_app

        factory = await get_mcp_factory(mock_request)

        assert factory is None
