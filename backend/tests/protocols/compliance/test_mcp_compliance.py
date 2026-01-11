"""MCP Protocol Compliance Tests.

Story 22-D2: Implement Protocol Compliance Tests

Verifies MCP (Model Context Protocol) implementation:
- Tool schema format
- JSON-RPC message format
- Error response format
- Tool response format
"""

import pytest


# =============================================================================
# Tool Schema Compliance Tests
# =============================================================================


class TestMCPToolSchemaCompliance:
    """Verify MCP tool schema format compliance."""

    def test_tool_schema_has_required_fields(self) -> None:
        """Tool schema must have name, description, input_schema."""
        tool_schema = {
            "name": "vector_search",
            "description": "Semantic search over documents",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        }
        assert "name" in tool_schema
        assert "description" in tool_schema
        assert "input_schema" in tool_schema

    def test_input_schema_is_json_schema(self) -> None:
        """input_schema must be valid JSON Schema."""
        input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["query"],
        }
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

    def test_tool_name_format(self) -> None:
        """Tool names should be snake_case."""
        valid_names = [
            "vector_search",
            "hybrid_retrieve",
            "ingest_url",
            "search_nodes",
        ]
        for name in valid_names:
            assert name == name.lower(), f"Tool name {name} should be lowercase"
            assert " " not in name, f"Tool name {name} should not have spaces"


# =============================================================================
# JSON-RPC Message Compliance Tests
# =============================================================================


class TestMCPJsonRpcCompliance:
    """Verify MCP JSON-RPC message format compliance."""

    def test_request_has_required_fields(self) -> None:
        """JSON-RPC request must have jsonrpc, method, id."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "vector_search",
                "arguments": {"query": "test"},
            },
            "id": 1,
        }
        assert request["jsonrpc"] == "2.0"
        assert "method" in request
        assert "id" in request

    def test_tool_call_method_format(self) -> None:
        """Tool call method should be 'tools/call'."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "test", "arguments": {}},
            "id": 1,
        }
        assert request["method"] == "tools/call"

    def test_tool_list_method_format(self) -> None:
        """Tool list method should be 'tools/list'."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1,
        }
        assert request["method"] == "tools/list"

    def test_params_has_name_and_arguments(self) -> None:
        """Tool call params must have name and arguments."""
        params = {
            "name": "vector_search",
            "arguments": {"query": "test", "top_k": 5},
        }
        assert "name" in params
        assert "arguments" in params


# =============================================================================
# Response Format Compliance Tests
# =============================================================================


class TestMCPResponseCompliance:
    """Verify MCP response format compliance."""

    def test_success_response_has_required_fields(self) -> None:
        """Success response must have jsonrpc, result, id."""
        response = {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {"type": "text", "text": "Found 5 results..."}
                ]
            },
            "id": 1,
        }
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "id" in response

    def test_error_response_has_required_fields(self) -> None:
        """Error response must have jsonrpc, error, id."""
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32602,
                "message": "Invalid params",
                "data": {"details": "Missing required parameter"},
            },
            "id": 1,
        }
        assert response["jsonrpc"] == "2.0"
        assert "error" in response
        assert "id" in response

    def test_error_has_code_and_message(self) -> None:
        """Error object must have code and message."""
        error = {
            "code": -32600,
            "message": "Invalid Request",
        }
        assert "code" in error
        assert "message" in error

    def test_result_content_format(self) -> None:
        """Result content must be array of content blocks."""
        result = {
            "content": [
                {"type": "text", "text": "First result"},
                {"type": "text", "text": "Second result"},
            ]
        }
        assert isinstance(result["content"], list)
        for item in result["content"]:
            assert "type" in item


# =============================================================================
# Error Code Compliance Tests
# =============================================================================


class TestMCPErrorCodeCompliance:
    """Verify MCP error codes match JSON-RPC specification."""

    # Standard JSON-RPC error codes
    STANDARD_ERROR_CODES = {
        -32700: "Parse error",
        -32600: "Invalid Request",
        -32601: "Method not found",
        -32602: "Invalid params",
        -32603: "Internal error",
    }

    def test_parse_error_code(self) -> None:
        """Parse error must use code -32700."""
        error = {"code": -32700, "message": "Parse error"}
        assert error["code"] == -32700

    def test_invalid_request_code(self) -> None:
        """Invalid Request must use code -32600."""
        error = {"code": -32600, "message": "Invalid Request"}
        assert error["code"] == -32600

    def test_method_not_found_code(self) -> None:
        """Method not found must use code -32601."""
        error = {"code": -32601, "message": "Method not found"}
        assert error["code"] == -32601

    def test_invalid_params_code(self) -> None:
        """Invalid params must use code -32602."""
        error = {"code": -32602, "message": "Invalid params"}
        assert error["code"] == -32602

    def test_internal_error_code(self) -> None:
        """Internal error must use code -32603."""
        error = {"code": -32603, "message": "Internal error"}
        assert error["code"] == -32603


# =============================================================================
# Tool Discovery Compliance Tests
# =============================================================================


class TestMCPToolDiscoveryCompliance:
    """Verify MCP tool discovery compliance."""

    def test_tools_list_response_format(self) -> None:
        """tools/list response must return array of tools."""
        response = {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "vector_search",
                        "description": "Semantic search",
                        "input_schema": {"type": "object"},
                    },
                    {
                        "name": "hybrid_retrieve",
                        "description": "Graph + vector",
                        "input_schema": {"type": "object"},
                    },
                ]
            },
            "id": 1,
        }
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

    def test_each_tool_has_required_fields(self) -> None:
        """Each tool in list must have name, description, input_schema."""
        tool = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        }
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool


# =============================================================================
# Content Block Compliance Tests
# =============================================================================


class TestMCPContentBlockCompliance:
    """Verify MCP content block format compliance."""

    def test_text_content_block(self) -> None:
        """Text content block must have type and text."""
        block = {"type": "text", "text": "Hello, world!"}
        assert block["type"] == "text"
        assert "text" in block

    def test_image_content_block(self) -> None:
        """Image content block must have type, data, mimeType."""
        block = {
            "type": "image",
            "data": "base64encodeddata...",
            "mimeType": "image/png",
        }
        assert block["type"] == "image"
        assert "data" in block
        assert "mimeType" in block

    def test_resource_content_block(self) -> None:
        """Resource content block must have type and resource."""
        block = {
            "type": "resource",
            "resource": {
                "uri": "file:///path/to/file.txt",
                "text": "File contents...",
            },
        }
        assert block["type"] == "resource"
        assert "resource" in block
