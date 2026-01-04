"""Tests for MCP server types.

Story 14-1: Expose RAG Engine via MCP Server
"""

import pytest

from agentic_rag_backend.mcp_server.types import (
    MCPError,
    MCPErrorCode,
    MCPToolSpec,
    MCPToolResult,
    MCPRequest,
    MCPResponse,
    MCPCapabilities,
    MCPServerInfo,
    MCPInitializeResult,
    create_tool_input_schema,
)


class TestMCPError:
    """Tests for MCPError."""

    def test_create_error(self):
        """Test creating an MCP error."""
        error = MCPError(
            code=MCPErrorCode.INVALID_PARAMS,
            message="Test error",
            data={"field": "value"},
        )
        assert error.code == MCPErrorCode.INVALID_PARAMS
        assert error.message == "Test error"
        assert error.data == {"field": "value"}

    def test_error_to_dict(self):
        """Test converting error to dict."""
        error = MCPError(
            code=MCPErrorCode.TOOL_NOT_FOUND,
            message="Tool not found",
        )
        result = error.to_dict()
        assert result["code"] == "tool_not_found"
        assert result["message"] == "Tool not found"
        assert "data" not in result

    def test_error_to_dict_with_data(self):
        """Test converting error with data to dict."""
        error = MCPError(
            code=MCPErrorCode.RATE_LIMIT_EXCEEDED,
            message="Rate limited",
            data={"retry_after": 60},
        )
        result = error.to_dict()
        assert result["data"] == {"retry_after": 60}


class TestMCPToolResult:
    """Tests for MCPToolResult."""

    def test_text_result(self):
        """Test creating a text result."""
        result = MCPToolResult.text("Hello, world!")
        assert result.content == [{"type": "text", "text": "Hello, world!"}]
        assert not result.is_error

    def test_json_result(self):
        """Test creating a JSON result."""
        result = MCPToolResult.json({"key": "value"})
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert '"key": "value"' in result.content[0]["text"]

    def test_error_result(self):
        """Test creating an error result."""
        result = MCPToolResult.error("Something went wrong", MCPErrorCode.INTERNAL_ERROR)
        assert result.is_error
        assert result.content[0]["text"] == "Something went wrong"
        assert result.metadata["error_code"] == "internal_error"

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = MCPToolResult.text("test", metadata={"elapsed_ms": 100})
        d = result.to_dict()
        assert d["content"] == [{"type": "text", "text": "test"}]
        assert d["isError"] is False
        assert d["_meta"]["elapsed_ms"] == 100


class TestMCPRequest:
    """Tests for MCPRequest."""

    def test_create_request(self):
        """Test creating an MCP request."""
        request = MCPRequest(
            method="tools/list",
            params={"cursor": None},
        )
        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.params == {"cursor": None}

    def test_request_with_id(self):
        """Test creating a request with ID."""
        request = MCPRequest(
            id="request-123",
            method="tools/call",
            params={"name": "test"},
        )
        assert request.id == "request-123"


class TestMCPResponse:
    """Tests for MCPResponse."""

    def test_success_response(self):
        """Test creating a success response."""
        response = MCPResponse.success("req-1", {"tools": []})
        assert response.id == "req-1"
        assert response.result == {"tools": []}
        assert response.error is None

    def test_failure_response(self):
        """Test creating a failure response."""
        error = MCPError(MCPErrorCode.INTERNAL_ERROR, "Failed")
        response = MCPResponse.failure("req-1", error)
        assert response.id == "req-1"
        assert response.result is None
        assert response.error["code"] == "internal_error"


class TestMCPCapabilities:
    """Tests for MCPCapabilities."""

    def test_default_capabilities(self):
        """Test default capabilities."""
        caps = MCPCapabilities()
        assert caps.tools == {"listChanged": True}
        assert caps.resources is None
        assert caps.prompts is None


class TestCreateToolInputSchema:
    """Tests for create_tool_input_schema helper."""

    def test_basic_schema(self):
        """Test creating a basic schema."""
        schema = create_tool_input_schema(
            properties={
                "query": {"type": "string", "description": "Search query"},
            },
            required=["query"],
        )
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]
        assert schema["additionalProperties"] is False

    def test_schema_with_optional_fields(self):
        """Test schema with optional fields."""
        schema = create_tool_input_schema(
            properties={
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            required=["query"],
        )
        assert "limit" in schema["properties"]
        assert schema["required"] == ["query"]


class TestMCPToolSpec:
    """Tests for MCPToolSpec."""

    @pytest.mark.asyncio
    async def test_tool_spec_to_dict(self):
        """Test converting tool spec to dict."""

        async def dummy_handler(args):
            return {"result": "ok"}

        spec = MCPToolSpec(
            name="test.tool",
            description="A test tool",
            input_schema=create_tool_input_schema(
                properties={"input": {"type": "string"}},
                required=["input"],
            ),
            handler=dummy_handler,
            category="test",
        )

        d = spec.to_dict()
        assert d["name"] == "test.tool"
        assert d["description"] == "A test tool"
        assert "inputSchema" in d
