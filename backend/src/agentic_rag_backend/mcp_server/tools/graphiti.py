"""Graphiti tool wrappers for MCP.

Provides MCP tools that wrap Graphiti operations with
tenant isolation enforcement.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

from typing import Any

import structlog

from ..types import MCPToolSpec, create_tool_input_schema, MCPError, MCPErrorCode
from ..registry import MCPServerRegistry
from ...db.graphiti import GraphitiClient
from ...retrieval.graphiti_retrieval import graphiti_search, GraphitiSearchResult
from ...indexing.graphiti_ingestion import ingest_document_as_episode
from ...models.documents import UnifiedDocument, DocumentMetadata, SourceType
from ...validation import is_valid_tenant_id

logger = structlog.get_logger(__name__)

# Maximum content size to prevent DoS attacks (1MB)
MAX_CONTENT_SIZE = 1_000_000


def _validate_content_size(content: str, max_size: int = MAX_CONTENT_SIZE) -> None:
    """Validate that content does not exceed maximum size.

    Args:
        content: Content string to validate
        max_size: Maximum allowed size in bytes

    Raises:
        MCPError: If content exceeds maximum size
    """
    if len(content.encode("utf-8")) > max_size:
        raise MCPError(
            code=MCPErrorCode.INVALID_PARAMS,
            message=f"Content exceeds maximum size of {max_size} bytes",
            data={"max_size": max_size},
        )


def _validate_tenant_id(tenant_id: str) -> None:
    """Validate tenant_id format.

    Raises:
        MCPError: If tenant_id is invalid
    """
    if not tenant_id or not is_valid_tenant_id(tenant_id):
        raise MCPError(
            code=MCPErrorCode.INVALID_PARAMS,
            message="Invalid tenant_id format. Must be a valid UUID.",
            data={"tenant_id": tenant_id},
        )


def _search_result_to_dict(result: GraphitiSearchResult) -> dict[str, Any]:
    """Convert search result to serializable dict."""
    return {
        "query": result.query,
        "tenant_id": result.tenant_id,
        "nodes": [
            {
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
                "labels": node.labels,
            }
            for node in result.nodes
        ],
        "edges": [
            {
                "uuid": edge.uuid,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "name": edge.name,
                "fact": edge.fact,
            }
            for edge in result.edges
        ],
        "processing_time_ms": result.processing_time_ms,
    }


def create_graphiti_search_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the graphiti.search tool.

    This tool performs hybrid search (semantic + graph) using Graphiti.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        query = arguments.get("query", "")
        if not query or not query.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Query is required",
            )

        num_results = arguments.get("num_results", 10)
        center_node_uuid = arguments.get("center_node_uuid")

        result = await graphiti_search(
            graphiti_client=graphiti_client,
            query=query,
            tenant_id=tenant_id,
            num_results=num_results,
            center_node_uuid=center_node_uuid,
        )

        return _search_result_to_dict(result)

    return MCPToolSpec(
        name="graphiti.search",
        description=(
            "Search the knowledge graph using hybrid retrieval (semantic + BM25 + graph). "
            "Returns relevant nodes and edges with their relationships."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
                "center_node_uuid": {
                    "type": "string",
                    "description": "Optional node UUID to center search around",
                },
            },
            required=["tenant_id", "query"],
        ),
        handler=handler,
        category="graphiti",
    )


def create_graphiti_add_episode_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the graphiti.add_episode tool.

    This tool adds a document as an episode to the knowledge graph.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        content = arguments.get("content", "")
        if not content or not content.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Content is required",
            )

        # Validate content size to prevent DoS attacks
        _validate_content_size(content)

        title = arguments.get("title")
        source_url = arguments.get("source_url")
        source_type_str = arguments.get("source_type", "text")

        # Map source type
        source_type_map = {
            "text": SourceType.TEXT,
            "url": SourceType.URL,
            "web": SourceType.URL,
            "pdf": SourceType.PDF,
        }
        source_type = source_type_map.get(source_type_str.lower(), SourceType.TEXT)

        # Create document for ingestion
        import hashlib
        from uuid import uuid4

        doc_id = uuid4()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        document = UnifiedDocument(
            id=doc_id,
            tenant_id=tenant_id,
            content=content,
            content_hash=content_hash,
            source_type=source_type,
            source_url=source_url,
            metadata=DocumentMetadata(title=title) if title else None,
        )

        result = await ingest_document_as_episode(
            graphiti_client=graphiti_client,
            document=document,
        )

        return {
            "document_id": result.document_id,
            "tenant_id": result.tenant_id,
            "episode_uuid": result.episode_uuid,
            "entities_extracted": result.entities_extracted,
            "edges_created": result.edges_created,
            "processing_time_ms": result.processing_time_ms,
            "source_description": result.source_description,
        }

    return MCPToolSpec(
        name="graphiti.add_episode",
        description=(
            "Add content as an episode to the knowledge graph. "
            f"Graphiti will extract entities and relationships automatically. "
            f"Maximum content size: {MAX_CONTENT_SIZE:,} bytes."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "content": {
                    "type": "string",
                    "description": f"Content to add to the knowledge graph (max {MAX_CONTENT_SIZE:,} bytes)",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the episode",
                },
                "source_url": {
                    "type": "string",
                    "description": "Optional source URL",
                },
                "source_type": {
                    "type": "string",
                    "description": "Source type: text, url, pdf (default: text)",
                    "enum": ["text", "url", "web", "pdf"],
                    "default": "text",
                },
            },
            required=["tenant_id", "content"],
        ),
        handler=handler,
        category="graphiti",
    )


def create_graphiti_get_node_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the graphiti.get_node tool.

    This tool retrieves a specific node by UUID with tenant isolation enforcement.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        node_uuid = arguments.get("node_uuid", "")
        if not node_uuid:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="node_uuid is required",
            )

        # Get node from Graphiti
        client = graphiti_client.client
        try:
            node = await client.get_node(node_uuid)
            if node is None:
                # Use generic message to prevent information leakage
                raise MCPError(
                    code=MCPErrorCode.INVALID_PARAMS,
                    message="Resource not found or access denied",
                )

            # CRITICAL: Enforce tenant isolation by verifying group_id matches tenant_id
            node_group_id = getattr(node, "group_id", None)
            if node_group_id != tenant_id:
                # Log the attempted cross-tenant access
                logger.warning(
                    "cross_tenant_access_attempt",
                    tenant_id=tenant_id,
                    node_group_id=node_group_id,
                    node_uuid=node_uuid,
                )
                # Return same error as not found to prevent information leakage
                raise MCPError(
                    code=MCPErrorCode.INVALID_PARAMS,
                    message="Resource not found or access denied",
                )

            return {
                "uuid": str(getattr(node, "uuid", "")),
                "name": getattr(node, "name", ""),
                "summary": getattr(node, "summary", ""),
                "labels": list(getattr(node, "labels", [])),
                "group_id": node_group_id,
            }
        except MCPError:
            raise
        except AttributeError:
            # Fallback if get_node not available
            raise MCPError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message="get_node operation not supported",
            )

    return MCPToolSpec(
        name="graphiti.get_node",
        description="Retrieve a specific node from the knowledge graph by its UUID.",
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "node_uuid": {
                    "type": "string",
                    "description": "Node UUID to retrieve",
                },
            },
            required=["tenant_id", "node_uuid"],
        ),
        handler=handler,
        category="graphiti",
    )


def create_graphiti_get_edges_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the graphiti.get_edges tool.

    This tool retrieves edges connected to a node with tenant isolation enforcement.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        node_uuid = arguments.get("node_uuid", "")
        if not node_uuid:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="node_uuid is required",
            )

        # Get edges from Graphiti
        client = graphiti_client.client
        try:
            # First verify the node belongs to the tenant
            node = await client.get_node(node_uuid)
            if node is None:
                raise MCPError(
                    code=MCPErrorCode.INVALID_PARAMS,
                    message="Resource not found or access denied",
                )

            # CRITICAL: Enforce tenant isolation by verifying group_id matches tenant_id
            node_group_id = getattr(node, "group_id", None)
            if node_group_id != tenant_id:
                logger.warning(
                    "cross_tenant_access_attempt",
                    tenant_id=tenant_id,
                    node_group_id=node_group_id,
                    node_uuid=node_uuid,
                )
                raise MCPError(
                    code=MCPErrorCode.INVALID_PARAMS,
                    message="Resource not found or access denied",
                )

            edges = await client.get_edges_by_node(node_uuid)

            # Filter edges to only include those belonging to the tenant
            filtered_edges = []
            for edge in edges:
                edge_group_id = getattr(edge, "group_id", None)
                # Include edge if it belongs to tenant or if group_id is not set
                # (for backwards compatibility with edges that might not have group_id)
                if edge_group_id is None or edge_group_id == tenant_id:
                    filtered_edges.append({
                        "uuid": str(getattr(edge, "uuid", "")),
                        "source_node_uuid": str(getattr(edge, "source_node_uuid", "")),
                        "target_node_uuid": str(getattr(edge, "target_node_uuid", "")),
                        "name": getattr(edge, "name", ""),
                        "fact": getattr(edge, "fact", ""),
                        "valid_at": str(getattr(edge, "valid_at", "")) if getattr(edge, "valid_at", None) else None,
                        "invalid_at": str(getattr(edge, "invalid_at", "")) if getattr(edge, "invalid_at", None) else None,
                    })

            return {
                "node_uuid": node_uuid,
                "edges": filtered_edges,
            }
        except MCPError:
            raise
        except AttributeError:
            # Fallback if operation not available
            raise MCPError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message="get_edges_by_node operation not supported",
            )

    return MCPToolSpec(
        name="graphiti.get_edges",
        description="Retrieve all edges connected to a specific node.",
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "node_uuid": {
                    "type": "string",
                    "description": "Node UUID to get edges for",
                },
            },
            required=["tenant_id", "node_uuid"],
        ),
        handler=handler,
        category="graphiti",
    )


def create_graphiti_delete_episode_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the graphiti.delete_episode tool.

    This tool deletes an episode from the knowledge graph.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        episode_uuid = arguments.get("episode_uuid", "")
        if not episode_uuid:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="episode_uuid is required",
            )

        # Delete episode from Graphiti
        client = graphiti_client.client
        try:
            await client.delete_episode(episode_uuid)
            return {
                "success": True,
                "episode_uuid": episode_uuid,
                "message": f"Episode {episode_uuid} deleted",
            }
        except Exception as e:
            raise MCPError(
                code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to delete episode: {e}",
            )

    return MCPToolSpec(
        name="graphiti.delete_episode",
        description="Delete an episode from the knowledge graph.",
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "episode_uuid": {
                    "type": "string",
                    "description": "Episode UUID to delete",
                },
            },
            required=["tenant_id", "episode_uuid"],
        ),
        handler=handler,
        category="graphiti",
    )


def register_graphiti_tools(
    registry: MCPServerRegistry,
    graphiti_client: GraphitiClient,
) -> list[str]:
    """Register all Graphiti tools with the registry.

    Args:
        registry: MCP server registry
        graphiti_client: Connected Graphiti client

    Returns:
        List of registered tool names
    """
    tools = [
        create_graphiti_search_tool(graphiti_client),
        create_graphiti_add_episode_tool(graphiti_client),
        create_graphiti_get_node_tool(graphiti_client),
        create_graphiti_get_edges_tool(graphiti_client),
        create_graphiti_delete_episode_tool(graphiti_client),
    ]

    registered = []
    for tool in tools:
        registry.register(tool)
        registered.append(tool.name)

    logger.info(
        "graphiti_tools_registered",
        tools=registered,
        count=len(registered),
    )

    return registered
