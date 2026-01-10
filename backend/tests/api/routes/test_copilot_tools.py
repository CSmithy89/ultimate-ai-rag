"""Tests for CopilotKit MCP tools endpoints.

Story 21-C3: Wire MCP Client to CopilotRuntime
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag_backend.api.routes.copilot import router
from agentic_rag_backend.mcp_client import MCPClientFactory


class MockRateLimiter:
    """Mock rate limiter that always allows requests."""

    async def allow(self, key: str) -> bool:
        return True


@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    # Add mock rate limiter for rate-limited endpoints
    app.state.rate_limiter = MockRateLimiter()
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestListToolsEndpoint:
    """Tests for GET /copilot/tools endpoint."""

    def test_list_tools_no_factory(self, client: TestClient) -> None:
        """Test listing tools when factory is not initialized."""
        response = client.get("/api/v1/copilot/tools")

        assert response.status_code == 200
        data = response.json()
        assert data["tools"] == []
        assert data["mcpEnabled"] is False
        assert data["serverCount"] == 0

    def test_list_tools_with_disabled_factory(self, app: FastAPI, client: TestClient) -> None:
        """Test listing tools when factory is disabled."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = False
        mock_factory.server_names = []
        app.state.mcp_client_factory = mock_factory

        response = client.get("/api/v1/copilot/tools")

        assert response.status_code == 200
        data = response.json()
        assert data["mcpEnabled"] is False

    def test_list_tools_with_enabled_factory(self, app: FastAPI) -> None:
        """Test listing tools when factory is enabled with tools."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = True
        mock_factory.server_names = ["github", "notion"]
        mock_factory.discover_all_tools = AsyncMock(return_value={
            "github": [
                {
                    "name": "create_issue",
                    "description": "Create GitHub issue",
                    "inputSchema": {"type": "object"},
                },
            ],
        })
        app.state.mcp_client_factory = mock_factory

        # Need async client for async dependency
        with patch(
            "agentic_rag_backend.api.routes.copilot.get_mcp_factory",
            return_value=mock_factory,
        ):
            with TestClient(app) as test_client:
                response = test_client.get("/api/v1/copilot/tools")

        assert response.status_code == 200
        data = response.json()
        assert data["mcpEnabled"] is True
        assert data["serverCount"] == 2
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "github:create_issue"
        assert data["tools"][0]["source"] == "external"
        assert data["tools"][0]["serverName"] == "github"


class TestCallToolEndpoint:
    """Tests for POST /copilot/tools/call endpoint."""

    def test_call_tool_no_factory(self, client: TestClient) -> None:
        """Test calling tool when factory is not initialized."""
        response = client.post(
            "/api/v1/copilot/tools/call",
            json={"toolName": "github:create_issue", "arguments": {}},
            headers={"X-Tenant-ID": "tenant-123"},
        )

        assert response.status_code == 503
        # RFC 7807 format: detail is now a dict
        detail = response.json()["detail"]
        assert detail["status"] == 503
        assert "not enabled" in detail["detail"]

    def test_call_tool_missing_tenant_id(self, client: TestClient) -> None:
        """Test calling tool without tenant ID returns 400."""
        response = client.post(
            "/api/v1/copilot/tools/call",
            json={"toolName": "github:create_issue", "arguments": {}},
        )

        assert response.status_code == 400
        # RFC 7807 format: detail is now a dict
        detail = response.json()["detail"]
        assert "X-Tenant-ID" in detail["detail"]

    def test_call_tool_factory_disabled(self, app: FastAPI, client: TestClient) -> None:
        """Test calling tool when factory is disabled."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = False
        app.state.mcp_client_factory = mock_factory

        response = client.post(
            "/api/v1/copilot/tools/call",
            json={"toolName": "github:create_issue", "arguments": {}},
            headers={"X-Tenant-ID": "tenant-123"},
        )

        assert response.status_code == 503

    def test_call_internal_tool_rejected(self, app: FastAPI) -> None:
        """Test calling internal tool (no namespace) is rejected via validation."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = True
        app.state.mcp_client_factory = mock_factory

        with patch(
            "agentic_rag_backend.api.routes.copilot.get_mcp_factory",
            return_value=mock_factory,
        ):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/api/v1/copilot/tools/call",
                    json={"toolName": "search", "arguments": {}},
                    headers={"X-Tenant-ID": "tenant-123"},
                )

        # Validation rejects non-namespaced tool names (must be server_name:tool_name)
        assert response.status_code == 422
        assert "tool_name must be in format" in str(response.json())

    def test_call_external_tool_success(self, app: FastAPI) -> None:
        """Test successfully calling an external tool."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = True
        mock_factory.server_names = ["github"]  # Include server in allowed list
        mock_factory.call_tool = AsyncMock(return_value={
            "issue_url": "https://github.com/test/repo/issues/1",
            "issue_number": 1,
        })
        app.state.mcp_client_factory = mock_factory

        with patch(
            "agentic_rag_backend.api.routes.copilot.get_mcp_factory",
            return_value=mock_factory,
        ):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/api/v1/copilot/tools/call",
                    json={
                        "toolName": "github:create_issue",
                        "arguments": {"title": "Test", "body": "Test body"},
                    },
                    headers={"X-Tenant-ID": "tenant-123"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["serverName"] == "github"
        assert data["result"]["issue_number"] == 1

        mock_factory.call_tool.assert_called_once_with(
            server_name="github",
            tool_name="create_issue",
            arguments={"title": "Test", "body": "Test body"},
        )

    def test_call_tool_execution_error(self, app: FastAPI) -> None:
        """Test handling tool execution error (generic message returned)."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = True
        mock_factory.server_names = ["github"]  # Include server in allowed list
        mock_factory.call_tool = AsyncMock(
            side_effect=Exception("Tool execution failed")
        )
        app.state.mcp_client_factory = mock_factory

        with patch(
            "agentic_rag_backend.api.routes.copilot.get_mcp_factory",
            return_value=mock_factory,
        ):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/api/v1/copilot/tools/call",
                    json={"toolName": "github:create_issue", "arguments": {}},
                    headers={"X-Tenant-ID": "tenant-123"},
                )

        assert response.status_code == 500
        # Security: Generic error message returned (no internal details leaked)
        # RFC 7807 format: detail is now a dict
        detail = response.json()["detail"]
        assert detail["detail"] == "An error occurred during external tool execution"
        assert detail["status"] == 500

    def test_call_tool_with_tenant_header(self, app: FastAPI) -> None:
        """Test calling tool with tenant ID header."""
        mock_factory = MagicMock(spec=MCPClientFactory)
        mock_factory.is_enabled = True
        mock_factory.server_names = ["notion"]  # Include server in allowed list
        mock_factory.call_tool = AsyncMock(return_value={"success": True})
        app.state.mcp_client_factory = mock_factory

        with patch(
            "agentic_rag_backend.api.routes.copilot.get_mcp_factory",
            return_value=mock_factory,
        ):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/api/v1/copilot/tools/call",
                    json={"toolName": "notion:create_page", "arguments": {}},
                    headers={"X-Tenant-ID": "tenant-123"},
                )

        assert response.status_code == 200
