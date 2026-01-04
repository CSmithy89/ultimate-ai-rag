"""Tests for MCP server tools.

Story 14-1: Expose RAG Engine via MCP Server
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag_backend.mcp_server.registry import MCPServerRegistry
from agentic_rag_backend.mcp_server.types import MCPError, MCPErrorCode
from agentic_rag_backend.mcp_server.tools.graphiti import (
    create_graphiti_search_tool,
    create_graphiti_add_episode_tool,
    register_graphiti_tools,
    _validate_tenant_id,
)
from agentic_rag_backend.mcp_server.tools.rag import (
    create_vector_search_tool,
    create_hybrid_retrieve_tool,
    create_ingest_text_tool,
    register_rag_tools,
)


# Test fixtures
@pytest.fixture
def mock_graphiti_client():
    """Create a mock Graphiti client."""
    client = MagicMock()
    client.is_connected = True
    client.client = MagicMock()

    # Mock search result
    mock_search_result = MagicMock()
    mock_search_result.nodes = []
    mock_search_result.edges = []
    client.client.search = AsyncMock(return_value=mock_search_result)

    # Mock add_episode result
    mock_episode = MagicMock()
    mock_episode.uuid = "episode-123"
    mock_episode.entity_references = []
    mock_episode.edge_references = []
    client.client.add_episode = AsyncMock(return_value=mock_episode)

    return client


@pytest.fixture
def mock_vector_service():
    """Create a mock vector search service."""
    service = MagicMock()
    service.search = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[])
    reranker.get_model = MagicMock(return_value="test-model")
    return reranker


class TestValidateTenantId:
    """Tests for tenant_id validation."""

    def test_valid_uuid(self):
        """Test valid UUID tenant_id."""
        # Should not raise
        _validate_tenant_id("00000000-0000-0000-0000-000000000001")

    def test_invalid_uuid(self):
        """Test invalid UUID tenant_id."""
        with pytest.raises(MCPError) as exc_info:
            _validate_tenant_id("not-a-uuid")
        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS

    def test_empty_tenant_id(self):
        """Test empty tenant_id."""
        with pytest.raises(MCPError) as exc_info:
            _validate_tenant_id("")
        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS


class TestGraphitiTools:
    """Tests for Graphiti tool wrappers."""

    @pytest.mark.asyncio
    async def test_graphiti_search_tool(self, mock_graphiti_client):
        """Test graphiti.search tool."""
        tool = create_graphiti_search_tool(mock_graphiti_client)
        assert tool.name == "graphiti.search"
        assert tool.category == "graphiti"

        result = await tool.handler({
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "query": "test query",
            "num_results": 5,
        })

        assert "nodes" in result
        assert "edges" in result
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_graphiti_search_missing_query(self, mock_graphiti_client):
        """Test graphiti.search with missing query."""
        tool = create_graphiti_search_tool(mock_graphiti_client)

        with pytest.raises(MCPError) as exc_info:
            await tool.handler({
                "tenant_id": "00000000-0000-0000-0000-000000000001",
            })
        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_graphiti_add_episode_tool(self, mock_graphiti_client):
        """Test graphiti.add_episode tool."""
        tool = create_graphiti_add_episode_tool(mock_graphiti_client)
        assert tool.name == "graphiti.add_episode"

        result = await tool.handler({
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "content": "This is test content for the episode.",
            "title": "Test Episode",
        })

        assert "episode_uuid" in result
        mock_graphiti_client.client.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_graphiti_add_episode_missing_content(self, mock_graphiti_client):
        """Test graphiti.add_episode with missing content."""
        tool = create_graphiti_add_episode_tool(mock_graphiti_client)

        with pytest.raises(MCPError) as exc_info:
            await tool.handler({
                "tenant_id": "00000000-0000-0000-0000-000000000001",
            })
        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS

    def test_register_graphiti_tools(self, mock_graphiti_client):
        """Test registering all Graphiti tools."""
        registry = MCPServerRegistry()
        registered = register_graphiti_tools(registry, mock_graphiti_client)

        assert len(registered) >= 3
        assert "graphiti.search" in registered
        assert "graphiti.add_episode" in registered


class TestRAGTools:
    """Tests for RAG extension tools."""

    @pytest.mark.asyncio
    async def test_vector_search_tool(self, mock_vector_service):
        """Test rag.vector_search tool."""
        tool = create_vector_search_tool(mock_vector_service)
        assert tool.name == "rag.vector_search"
        assert tool.category == "rag"

        result = await tool.handler({
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "query": "test query",
        })

        assert "hits" in result
        mock_vector_service.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_missing_query(self, mock_vector_service):
        """Test vector_search with missing query."""
        tool = create_vector_search_tool(mock_vector_service)

        with pytest.raises(MCPError) as exc_info:
            await tool.handler({
                "tenant_id": "00000000-0000-0000-0000-000000000001",
            })
        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_tool(
        self, mock_vector_service, mock_graphiti_client, mock_reranker
    ):
        """Test rag.hybrid_retrieve tool."""
        # Setup mock for graphiti search result
        mock_result = MagicMock()
        mock_result.nodes = []
        mock_result.edges = []
        mock_result.processing_time_ms = 100

        # Mock the graphiti_search function
        import agentic_rag_backend.mcp_server.tools.rag as rag_module
        original_search = rag_module.graphiti_search

        async def mock_search(*args, **kwargs):
            return mock_result

        rag_module.graphiti_search = mock_search

        try:
            tool = create_hybrid_retrieve_tool(
                mock_vector_service,
                mock_graphiti_client,
                mock_reranker,
            )
            assert tool.name == "rag.hybrid_retrieve"

            result = await tool.handler({
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "query": "test query",
                "use_reranking": False,
            })

            assert "vector_hits" in result
            assert "graph_nodes" in result
            assert result["retrieval_mode"] == "hybrid"
        finally:
            rag_module.graphiti_search = original_search

    @pytest.mark.asyncio
    async def test_ingest_text_tool(self, mock_graphiti_client):
        """Test rag.ingest_text tool."""
        tool = create_ingest_text_tool(mock_graphiti_client)
        assert tool.name == "rag.ingest_text"

        result = await tool.handler({
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "content": "This is a test document with sufficient content for ingestion.",
            "title": "Test Document",
        })

        assert "document_id" in result
        assert "episode_uuid" in result

    @pytest.mark.asyncio
    async def test_ingest_text_empty_content(self, mock_graphiti_client):
        """Test ingest_text with empty content."""
        tool = create_ingest_text_tool(mock_graphiti_client)

        with pytest.raises(MCPError) as exc_info:
            await tool.handler({
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "content": "",
            })
        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS

    def test_register_rag_tools_minimal(self, mock_graphiti_client):
        """Test registering RAG tools with minimal services."""
        registry = MCPServerRegistry()
        registered = register_rag_tools(
            registry,
            mock_graphiti_client,
            vector_service=None,
            reranker=None,
        )

        # Should have ingestion tools at minimum
        assert "rag.ingest_text" in registered
        assert "rag.ingest_url" in registered
        assert "rag.ingest_youtube" in registered

    def test_register_rag_tools_full(
        self, mock_graphiti_client, mock_vector_service, mock_reranker
    ):
        """Test registering RAG tools with all services."""
        registry = MCPServerRegistry()
        registered = register_rag_tools(
            registry,
            mock_graphiti_client,
            vector_service=mock_vector_service,
            reranker=mock_reranker,
        )

        # Should have all tools
        assert "rag.vector_search" in registered
        assert "rag.hybrid_retrieve" in registered
        assert "rag.query_with_reranking" in registered
        assert "rag.explain_answer" in registered


class TestToolInputSchemas:
    """Tests for tool input schemas."""

    def test_graphiti_search_schema(self, mock_graphiti_client):
        """Test graphiti.search input schema."""
        tool = create_graphiti_search_tool(mock_graphiti_client)
        schema = tool.input_schema

        assert schema["type"] == "object"
        assert "tenant_id" in schema["properties"]
        assert "query" in schema["properties"]
        assert "tenant_id" in schema["required"]
        assert "query" in schema["required"]

    def test_vector_search_schema(self, mock_vector_service):
        """Test rag.vector_search input schema."""
        tool = create_vector_search_tool(mock_vector_service)
        schema = tool.input_schema

        assert "tenant_id" in schema["properties"]
        assert "query" in schema["properties"]
        assert schema["additionalProperties"] is False


class TestCrossTenantIsolation:
    """Tests for cross-tenant isolation security."""

    @pytest.mark.asyncio
    async def test_registry_denies_cross_tenant_access(self):
        """Test that non-admin users cannot access other tenant's data via registry."""
        from agentic_rag_backend.mcp_server.auth import MCPAuthContext

        registry = MCPServerRegistry()

        # Create a simple test tool
        async def test_handler(args):
            return {"result": "ok", "tenant_id": args.get("tenant_id")}

        from agentic_rag_backend.mcp_server.types import MCPToolSpec, create_tool_input_schema

        tool = MCPToolSpec(
            name="test.tool",
            description="Test tool",
            input_schema=create_tool_input_schema(
                properties={
                    "tenant_id": {"type": "string"},
                },
                required=["tenant_id"],
            ),
            handler=test_handler,
            category="test",
        )
        registry.register(tool)

        # Create auth context for tenant A
        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"
        auth_context = MCPAuthContext(
            tenant_id=tenant_a,
            scopes=["tools:test.tool"],  # Regular user with limited scopes
        )

        # Tenant A should be able to access their own data
        result = await registry.call_tool(
            name="test.tool",
            arguments={"tenant_id": tenant_a},
            auth_context=auth_context,
        )
        assert result is not None

        # Tenant A should NOT be able to access Tenant B's data
        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool(
                name="test.tool",
                arguments={"tenant_id": tenant_b},
                auth_context=auth_context,
            )
        assert exc_info.value.code == MCPErrorCode.AUTHENTICATION_FAILED
        assert "Access denied to tenant" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_admin_can_access_any_tenant(self):
        """Test that admin users (scopes=None) can access any tenant."""
        from agentic_rag_backend.mcp_server.auth import MCPAuthContext

        registry = MCPServerRegistry()

        async def test_handler(args):
            return {"result": "ok", "tenant_id": args.get("tenant_id")}

        from agentic_rag_backend.mcp_server.types import MCPToolSpec, create_tool_input_schema

        tool = MCPToolSpec(
            name="test.tool",
            description="Test tool",
            input_schema=create_tool_input_schema(
                properties={
                    "tenant_id": {"type": "string"},
                },
                required=["tenant_id"],
            ),
            handler=test_handler,
            category="test",
        )
        registry.register(tool)

        # Create admin auth context (scopes=None means admin)
        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"
        admin_context = MCPAuthContext(
            tenant_id=tenant_a,
            scopes=None,  # Admin user
        )

        # Admin should be able to access any tenant
        result = await registry.call_tool(
            name="test.tool",
            arguments={"tenant_id": tenant_b},
            auth_context=admin_context,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_node_denies_cross_tenant_access(self, mock_graphiti_client):
        """Test that graphiti.get_node denies access to nodes from other tenants."""
        from agentic_rag_backend.mcp_server.tools.graphiti import create_graphiti_get_node_tool

        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"

        # Mock a node that belongs to tenant_b
        mock_node = MagicMock()
        mock_node.uuid = "node-uuid-123"
        mock_node.name = "Test Node"
        mock_node.summary = "Test summary"
        mock_node.labels = ["Entity"]
        mock_node.group_id = tenant_b  # Node belongs to tenant B

        mock_graphiti_client.client.get_node = AsyncMock(return_value=mock_node)

        tool = create_graphiti_get_node_tool(mock_graphiti_client)

        # Tenant A tries to access node that belongs to Tenant B
        with pytest.raises(MCPError) as exc_info:
            await tool.handler({
                "tenant_id": tenant_a,
                "node_uuid": "node-uuid-123",
            })

        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS
        assert "Resource not found or access denied" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_get_node_allows_same_tenant_access(self, mock_graphiti_client):
        """Test that graphiti.get_node allows access to nodes from same tenant."""
        from agentic_rag_backend.mcp_server.tools.graphiti import create_graphiti_get_node_tool

        tenant_a = "00000000-0000-0000-0000-000000000001"

        # Mock a node that belongs to tenant_a
        mock_node = MagicMock()
        mock_node.uuid = "node-uuid-123"
        mock_node.name = "Test Node"
        mock_node.summary = "Test summary"
        mock_node.labels = ["Entity"]
        mock_node.group_id = tenant_a  # Node belongs to tenant A

        mock_graphiti_client.client.get_node = AsyncMock(return_value=mock_node)

        tool = create_graphiti_get_node_tool(mock_graphiti_client)

        # Tenant A should be able to access their own node
        result = await tool.handler({
            "tenant_id": tenant_a,
            "node_uuid": "node-uuid-123",
        })

        assert result["uuid"] == "node-uuid-123"
        assert result["name"] == "Test Node"

    @pytest.mark.asyncio
    async def test_get_edges_denies_cross_tenant_access(self, mock_graphiti_client):
        """Test that graphiti.get_edges denies access to edges from other tenants."""
        from agentic_rag_backend.mcp_server.tools.graphiti import create_graphiti_get_edges_tool

        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"

        # Mock a node that belongs to tenant_b
        mock_node = MagicMock()
        mock_node.uuid = "node-uuid-123"
        mock_node.group_id = tenant_b  # Node belongs to tenant B

        mock_graphiti_client.client.get_node = AsyncMock(return_value=mock_node)

        tool = create_graphiti_get_edges_tool(mock_graphiti_client)

        # Tenant A tries to get edges for a node that belongs to Tenant B
        with pytest.raises(MCPError) as exc_info:
            await tool.handler({
                "tenant_id": tenant_a,
                "node_uuid": "node-uuid-123",
            })

        assert exc_info.value.code == MCPErrorCode.INVALID_PARAMS
        assert "Resource not found or access denied" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_get_edges_filters_cross_tenant_edges(self, mock_graphiti_client):
        """Test that graphiti.get_edges filters out edges from other tenants."""
        from agentic_rag_backend.mcp_server.tools.graphiti import create_graphiti_get_edges_tool

        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"

        # Mock a node that belongs to tenant_a
        mock_node = MagicMock()
        mock_node.uuid = "node-uuid-123"
        mock_node.group_id = tenant_a

        # Mock edges - some belong to tenant_a, some to tenant_b
        edge_a = MagicMock()
        edge_a.uuid = "edge-a"
        edge_a.source_node_uuid = "node-1"
        edge_a.target_node_uuid = "node-2"
        edge_a.name = "RELATES_TO"
        edge_a.fact = "Fact A"
        edge_a.group_id = tenant_a
        edge_a.valid_at = None
        edge_a.invalid_at = None

        edge_b = MagicMock()
        edge_b.uuid = "edge-b"
        edge_b.source_node_uuid = "node-1"
        edge_b.target_node_uuid = "node-3"
        edge_b.name = "RELATES_TO"
        edge_b.fact = "Fact B"
        edge_b.group_id = tenant_b  # Belongs to different tenant
        edge_b.valid_at = None
        edge_b.invalid_at = None

        mock_graphiti_client.client.get_node = AsyncMock(return_value=mock_node)
        mock_graphiti_client.client.get_edges_by_node = AsyncMock(return_value=[edge_a, edge_b])

        tool = create_graphiti_get_edges_tool(mock_graphiti_client)

        # Tenant A should only see their own edges
        result = await tool.handler({
            "tenant_id": tenant_a,
            "node_uuid": "node-uuid-123",
        })

        assert len(result["edges"]) == 1
        assert result["edges"][0]["uuid"] == "edge-a"
