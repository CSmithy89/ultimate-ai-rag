"""Tests for MCP Client implementation.

Story 21-C2: Implement MCP Client Factory
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentic_rag_backend.mcp_client.client import MAX_BACKOFF_SECONDS, MCPClient, MCPClientFactory
from agentic_rag_backend.mcp_client.config import MCPClientSettings, MCPServerConfig
from agentic_rag_backend.mcp_client.errors import (
    MCPClientConnectionError,
    MCPClientError,
    MCPClientNotEnabledError,
    MCPClientTimeoutError,
    MCPProtocolError,
    MCPServerNotFoundError,
)


# --- Test Fixtures ---


@pytest.fixture
def server_config() -> MCPServerConfig:
    """Create test server configuration."""
    return MCPServerConfig(
        name="test-server",
        url="https://mcp.example.com/sse",
        api_key="test-api-key",
        transport="sse",
        timeout_ms=5000,
    )


@pytest.fixture
def client_settings(server_config: MCPServerConfig) -> MCPClientSettings:
    """Create test client settings."""
    return MCPClientSettings(
        enabled=True,
        servers=[server_config],
        default_timeout_ms=30000,
        retry_count=3,
        retry_delay_ms=100,
    )


@pytest.fixture
def disabled_settings() -> MCPClientSettings:
    """Create disabled client settings."""
    return MCPClientSettings(enabled=False)


# --- MCPClient Tests ---


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_init_builds_headers_with_api_key(self, server_config: MCPServerConfig) -> None:
        """Test headers include Authorization when API key provided."""
        client = MCPClient(server_config)
        headers = client._build_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_init_builds_headers_without_api_key(self) -> None:
        """Test headers without Authorization when no API key."""
        config = MCPServerConfig(
            name="no-auth",
            url="https://mcp.example.com/sse",
        )
        client = MCPClient(config)
        headers = client._build_headers()

        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_list_tools_success(self, server_config: MCPServerConfig) -> None:
        """Test successful tool listing."""
        client = MCPClient(server_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "tools": [
                    {"name": "search", "description": "Search tool"},
                    {"name": "fetch", "description": "Fetch tool"},
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            tools = await client.list_tools()

            assert len(tools) == 2
            assert tools[0]["name"] == "search"
            assert tools[1]["name"] == "fetch"

        await client.close()

    @pytest.mark.asyncio
    async def test_call_tool_success(self, server_config: MCPServerConfig) -> None:
        """Test successful tool call."""
        client = MCPClient(server_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"content": "test result"},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.call_tool("search", {"query": "test"})

            assert result == {"content": "test result"}

            # Verify request payload
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert payload["method"] == "tools/call"
            assert payload["params"]["name"] == "search"
            assert payload["params"]["arguments"] == {"query": "test"}

        await client.close()

    @pytest.mark.asyncio
    async def test_request_timeout_with_retry(self, server_config: MCPServerConfig) -> None:
        """Test request retries on timeout."""
        client = MCPClient(server_config, retry_count=2, retry_delay_ms=10)

        with patch.object(client._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")

            with pytest.raises(MCPClientTimeoutError) as exc_info:
                await client.list_tools()

            # Should have tried 3 times (initial + 2 retries)
            assert mock_post.call_count == 3
            assert exc_info.value.server_name == "test-server"
            assert exc_info.value.attempts == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_request_no_retry_on_4xx(self, server_config: MCPServerConfig) -> None:
        """Test no retry on 4xx client errors (raised as MCPProtocolError)."""
        client = MCPClient(server_config, retry_count=3, retry_delay_ms=10)

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(client._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )

            # 4xx errors are MCPProtocolError (semantic client errors, not connection errors)
            with pytest.raises(MCPProtocolError) as exc_info:
                await client.list_tools()

            # Should not retry 4xx errors
            assert mock_post.call_count == 1
            assert exc_info.value.error_code == 401
            assert "HTTP 401" in exc_info.value.error_message

        await client.close()

    @pytest.mark.asyncio
    async def test_request_retry_on_5xx(self, server_config: MCPServerConfig) -> None:
        """Test retry on 5xx server errors."""
        client = MCPClient(server_config, retry_count=2, retry_delay_ms=10)

        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(client._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Service Unavailable", request=MagicMock(), response=mock_response
            )

            with pytest.raises(MCPClientConnectionError):
                await client.list_tools()

            # Should retry on 5xx
            assert mock_post.call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_request_protocol_error(self, server_config: MCPServerConfig) -> None:
        """Test protocol error handling."""
        client = MCPClient(server_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {
                "code": -32600,
                "message": "Invalid Request",
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(MCPProtocolError) as exc_info:
                await client.list_tools()

            assert exc_info.value.error_code == -32600
            assert exc_info.value.error_message == "Invalid Request"

        await client.close()

    @pytest.mark.asyncio
    async def test_closed_client_raises_error(self, server_config: MCPServerConfig) -> None:
        """Test that closed client raises error on request."""
        client = MCPClient(server_config)
        await client.close()

        with pytest.raises(MCPClientError, match="closed"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_backoff_capped_at_max(self, server_config: MCPServerConfig) -> None:
        """Test that backoff delay is capped at MAX_BACKOFF_SECONDS."""
        client = MCPClient(server_config, retry_delay_ms=10000)

        # For attempt 10: base * 2^10 = 10 * 1024 = 10240 seconds
        # But should be capped at MAX_BACKOFF_SECONDS
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._backoff(10)
            called_delay = mock_sleep.call_args[0][0]
            assert called_delay <= MAX_BACKOFF_SECONDS

        await client.close()


# --- MCPClientFactory Tests ---


class TestMCPClientFactory:
    """Tests for MCPClientFactory class."""

    def test_is_enabled(self, client_settings: MCPClientSettings) -> None:
        """Test is_enabled property."""
        factory = MCPClientFactory(client_settings)
        assert factory.is_enabled is True

    def test_is_disabled(self, disabled_settings: MCPClientSettings) -> None:
        """Test is_enabled when disabled."""
        factory = MCPClientFactory(disabled_settings)
        assert factory.is_enabled is False

    def test_server_names(self, client_settings: MCPClientSettings) -> None:
        """Test server_names property."""
        factory = MCPClientFactory(client_settings)
        assert factory.server_names == ["test-server"]

    @pytest.mark.asyncio
    async def test_get_client_success(self, client_settings: MCPClientSettings) -> None:
        """Test getting client by name."""
        factory = MCPClientFactory(client_settings)

        client = await factory.get_client("test-server")

        assert isinstance(client, MCPClient)
        assert client.name == "test-server"

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_get_client_reuses_instance(self, client_settings: MCPClientSettings) -> None:
        """Test that get_client returns same instance."""
        factory = MCPClientFactory(client_settings)

        client1 = await factory.get_client("test-server")
        client2 = await factory.get_client("test-server")

        assert client1 is client2

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_get_client_when_disabled(self, disabled_settings: MCPClientSettings) -> None:
        """Test error when feature disabled."""
        factory = MCPClientFactory(disabled_settings)

        with pytest.raises(MCPClientNotEnabledError):
            await factory.get_client("any-server")

    @pytest.mark.asyncio
    async def test_get_client_unknown_server(self, client_settings: MCPClientSettings) -> None:
        """Test error for unknown server."""
        factory = MCPClientFactory(client_settings)

        with pytest.raises(MCPServerNotFoundError) as exc_info:
            await factory.get_client("unknown-server")

        assert exc_info.value.server_name == "unknown-server"
        assert "test-server" in exc_info.value.available_servers

    @pytest.mark.asyncio
    async def test_discover_all_tools_success(self, client_settings: MCPClientSettings) -> None:
        """Test tool discovery from all servers."""
        factory = MCPClientFactory(client_settings)

        with patch.object(MCPClient, "list_tools", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [{"name": "tool1"}]

            tools = await factory.discover_all_tools()

            assert "test-server" in tools
            assert tools["test-server"] == [{"name": "tool1"}]

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_discover_all_tools_partial_failure(self) -> None:
        """Test tool discovery continues on individual failures."""
        settings = MCPClientSettings(
            enabled=True,
            servers=[
                MCPServerConfig(name="server1", url="https://mcp1.example.com"),
                MCPServerConfig(name="server2", url="https://mcp2.example.com"),
            ],
            retry_count=0,
        )
        factory = MCPClientFactory(settings)

        with patch.object(MCPClient, "list_tools", new_callable=AsyncMock) as mock_list:
            # First server fails, second succeeds
            mock_list.side_effect = [
                Exception("Connection failed"),
                [{"name": "tool2"}],
            ]

            tools = await factory.discover_all_tools()

            # server1 failed but server2 should succeed
            assert "server1" not in tools
            assert tools.get("server2") == [{"name": "tool2"}]

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_discover_all_tools_when_disabled(
        self, disabled_settings: MCPClientSettings
    ) -> None:
        """Test empty result when disabled."""
        factory = MCPClientFactory(disabled_settings)

        tools = await factory.discover_all_tools()

        assert tools == {}

    @pytest.mark.asyncio
    async def test_call_tool_success(self, client_settings: MCPClientSettings) -> None:
        """Test tool call via factory."""
        factory = MCPClientFactory(client_settings)

        with patch.object(MCPClient, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"result": "success"}

            result = await factory.call_tool("test-server", "search", {"query": "test"})

            assert result == {"result": "success"}
            mock_call.assert_called_once_with("search", {"query": "test"})

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_close_all(self, client_settings: MCPClientSettings) -> None:
        """Test closing all clients."""
        factory = MCPClientFactory(client_settings)

        # Create a client
        client = await factory.get_client("test-server")

        # Close all
        await factory.close_all()

        # Client should be closed
        assert client.is_closed
        assert len(factory._clients) == 0

    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self, client_settings: MCPClientSettings) -> None:
        """Test lifespan context manager."""
        factory = MCPClientFactory(client_settings)

        async with factory.lifespan() as f:
            assert f is factory
            client = await f.get_client("test-server")
            assert not client.is_closed

        # After context exit, clients should be closed
        assert client.is_closed


# --- Error Tests ---


class TestMCPClientErrors:
    """Tests for MCP client error classes."""

    def test_timeout_error(self) -> None:
        """Test timeout error details."""
        error = MCPClientTimeoutError(
            server_name="test-server",
            attempts=3,
            timeout_ms=5000,
        )
        assert "test-server" in str(error)
        assert error.server_name == "test-server"
        assert error.attempts == 3
        assert error.timeout_ms == 5000

    def test_connection_error(self) -> None:
        """Test connection error details."""
        error = MCPClientConnectionError(
            server_name="test-server",
            url="https://example.com",
            reason="Connection refused",
        )
        assert "test-server" in str(error)
        assert error.url == "https://example.com"
        assert error.reason == "Connection refused"

    def test_server_not_found_error(self) -> None:
        """Test server not found error details."""
        error = MCPServerNotFoundError(
            server_name="unknown",
            available_servers=["server1", "server2"],
        )
        assert "unknown" in str(error)
        assert error.available_servers == ["server1", "server2"]

    def test_protocol_error(self) -> None:
        """Test protocol error details."""
        error = MCPProtocolError(
            server_name="test-server",
            error_code=-32600,
            error_message="Invalid Request",
            error_data={"detail": "missing field"},
        )
        assert error.error_code == -32600
        assert error.error_message == "Invalid Request"
        assert error.error_data == {"detail": "missing field"}

    def test_not_enabled_error(self) -> None:
        """Test not enabled error."""
        error = MCPClientNotEnabledError()
        assert "not enabled" in str(error)
        assert "MCP_CLIENTS_ENABLED" in str(error.details.get("hint", ""))
