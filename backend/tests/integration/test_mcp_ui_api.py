"""Integration tests for MCP-UI API endpoints.

Story 22-C1: Implement MCP-UI Renderer
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Valid UUID format for tenant_id (matches TENANT_ID_PATTERN)
VALID_TENANT_ID = "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for MCP-UI tests."""
    settings = MagicMock()
    settings.mcp_ui_enabled = True
    settings.mcp_ui_allowed_origins = [
        "https://tools.example.com",
        "https://mcp-ui.example.com",
    ]
    settings.mcp_ui_signing_secret = "test-secret-key"
    return settings


@pytest.fixture
def client(mock_settings: MagicMock) -> TestClient:
    """Create test client with mocked settings."""
    with patch("agentic_rag_backend.api.routes.mcp.get_settings", return_value=mock_settings):
        # Import after patching to get the patched version
        from agentic_rag_backend.api.routes.mcp import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        yield TestClient(app)


class TestMCPUIConfigEndpoint:
    """Tests for GET /mcp/ui/config endpoint."""

    def test_get_config_success(self, client: TestClient, mock_settings: MagicMock) -> None:
        """Should return config when valid tenant ID provided."""
        response = client.get(
            "/api/v1/mcp/ui/config",
            headers={"X-Tenant-ID": VALID_TENANT_ID},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert len(data["allowed_origins"]) == 2
        assert "https://tools.example.com" in data["allowed_origins"]
        assert "https://mcp-ui.example.com" in data["allowed_origins"]

    def test_get_config_missing_tenant_header(self, client: TestClient) -> None:
        """Should return 422 when X-Tenant-ID header is missing."""
        response = client.get("/api/v1/mcp/ui/config")
        assert response.status_code == 422

    def test_get_config_empty_tenant_id(self, client: TestClient) -> None:
        """Should return 400 when X-Tenant-ID is empty."""
        response = client.get(
            "/api/v1/mcp/ui/config",
            headers={"X-Tenant-ID": ""},
        )
        assert response.status_code == 400
        assert "Invalid X-Tenant-ID" in response.json()["detail"]

    def test_get_config_whitespace_tenant_id(self, client: TestClient) -> None:
        """Should return 400 when X-Tenant-ID is whitespace only."""
        response = client.get(
            "/api/v1/mcp/ui/config",
            headers={"X-Tenant-ID": "   "},
        )
        assert response.status_code == 400

    def test_get_config_disabled(self, client: TestClient, mock_settings: MagicMock) -> None:
        """Should return enabled=false when MCP-UI is disabled."""
        mock_settings.mcp_ui_enabled = False

        response = client.get(
            "/api/v1/mcp/ui/config",
            headers={"X-Tenant-ID": VALID_TENANT_ID},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

    def test_get_config_empty_origins(self, client: TestClient, mock_settings: MagicMock) -> None:
        """Should return empty origins list when none configured."""
        mock_settings.mcp_ui_allowed_origins = []

        response = client.get(
            "/api/v1/mcp/ui/config",
            headers={"X-Tenant-ID": VALID_TENANT_ID},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["allowed_origins"] == []


class TestMCPUIConfigWithRealSettings:
    """Tests for MCP-UI config with actual Settings class."""

    def test_config_parsing_empty_origins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should parse empty origins correctly."""
        monkeypatch.setenv("MCP_UI_ENABLED", "true")
        monkeypatch.setenv("MCP_UI_ALLOWED_ORIGINS", "")

        # Clear the lru_cache to reload settings
        from agentic_rag_backend.config import get_settings, load_settings
        get_settings.cache_clear()

        try:
            settings = load_settings()
            assert settings.mcp_ui_enabled is True
            assert settings.mcp_ui_allowed_origins == []
        except ValueError:
            # May fail due to missing required env vars in test
            pytest.skip("Missing required environment variables for full settings test")

    def test_config_parsing_multiple_origins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should parse comma-separated origins."""
        monkeypatch.setenv("MCP_UI_ENABLED", "true")
        monkeypatch.setenv(
            "MCP_UI_ALLOWED_ORIGINS",
            "https://a.com, https://b.com , https://c.com",
        )

        from agentic_rag_backend.config import get_settings, load_settings
        get_settings.cache_clear()

        try:
            settings = load_settings()
            assert len(settings.mcp_ui_allowed_origins) == 3
            assert "https://a.com" in settings.mcp_ui_allowed_origins
            assert "https://b.com" in settings.mcp_ui_allowed_origins
            assert "https://c.com" in settings.mcp_ui_allowed_origins
        except ValueError:
            pytest.skip("Missing required environment variables for full settings test")

    def test_config_parsing_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MCP-UI should be disabled by default."""
        # Don't set MCP_UI_ENABLED
        monkeypatch.delenv("MCP_UI_ENABLED", raising=False)

        from agentic_rag_backend.config import get_settings, load_settings
        get_settings.cache_clear()

        try:
            settings = load_settings()
            assert settings.mcp_ui_enabled is False
        except ValueError:
            pytest.skip("Missing required environment variables for full settings test")
