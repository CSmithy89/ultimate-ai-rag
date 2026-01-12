"""MCP stdio transport entry point.

Provides a standalone entry point for running the MCP server with stdio transport.
This enables native integration with Claude Desktop and other MCP clients that
communicate via stdin/stdout instead of HTTP.

Story 18-10: Implement MCP stdio Transport

Usage:
    # Direct execution
    python -m agentic_rag_backend.mcp_stdio

    # Via installed script
    rag-mcp-stdio

Environment Variables:
    All standard backend configuration variables are supported.
    For minimal operation, at minimum set:
    - LLM_PROVIDER and corresponding API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD for graph operations
    - DATABASE_URL for PostgreSQL/pgvector operations

    Optional:
    - MCP_TOOL_TIMEOUT_SECONDS: Default tool timeout (default: 30)
    - MCP_TOOL_MAX_TIMEOUT_SECONDS: Maximum tool timeout (default: 300)
    - MCP_TOOL_TIMEOUT_OVERRIDES: JSON object with per-tool timeouts

Claude Desktop Configuration:
    Add to your Claude Desktop MCP settings:
    {
        "mcpServers": {
            "agentic-rag": {
                "command": "rag-mcp-stdio",
                "args": []
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog

# Configure structlog for stderr (stdout is reserved for MCP protocol)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MCPStdioServer:
    """MCP Server runner for stdio transport.

    Initializes all required components (Graphiti, vector search, reranker)
    and runs the MCP server in stdio mode.
    """

    def __init__(self) -> None:
        """Initialize the stdio server runner."""
        self._server = None
        self._graphiti_client = None
        self._neo4j_client = None
        self._postgres_client = None
        self._redis_client = None

    async def _initialize_clients(self) -> None:
        """Initialize database and service clients."""
        from .config import load_settings
        from .llm import get_llm_adapter

        settings = load_settings()

        # Initialize LLM adapter
        llm_adapter = get_llm_adapter(settings)

        # Initialize Neo4j client (optional but recommended)
        if settings.neo4j_uri:
            try:
                from .db.neo4j import Neo4jClient

                self._neo4j_client = Neo4jClient(
                    settings.neo4j_uri,
                    settings.neo4j_user,
                    settings.neo4j_password,
                )
                await self._neo4j_client.connect()
                logger.info("neo4j_connected")
            except Exception as e:
                logger.warning("neo4j_connection_failed", error=str(e))
                self._neo4j_client = None

        # Initialize PostgreSQL client (optional for vector search)
        if settings.database_url:
            try:
                from .db.postgres import PostgresClient

                self._postgres_client = PostgresClient(settings.database_url)
                await self._postgres_client.connect()
                logger.info("postgres_connected")
            except Exception as e:
                logger.warning("postgres_connection_failed", error=str(e))
                self._postgres_client = None

        # Initialize Graphiti client (optional but recommended)
        try:
            from .db.graphiti import GraphitiClient, GRAPHITI_AVAILABLE

            if GRAPHITI_AVAILABLE and settings.neo4j_uri:
                self._graphiti_client = GraphitiClient(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                    llm_provider=llm_adapter.provider,
                    llm_api_key=llm_adapter.api_key,
                    llm_base_url=llm_adapter.base_url,
                    embedding_provider=settings.embedding_provider,
                    embedding_api_key=settings.embedding_api_key,
                    embedding_base_url=settings.embedding_base_url,
                    embedding_model=settings.graphiti_embedding_model,
                    llm_model=settings.graphiti_llm_model,
                )
                await self._graphiti_client.connect()
                await self._graphiti_client.build_indices()
                logger.info("graphiti_connected")
            else:
                logger.warning(
                    "graphiti_not_available",
                    reason="graphiti-core not installed or Neo4j not configured",
                )
        except Exception as e:
            logger.warning("graphiti_initialization_failed", error=str(e))
            self._graphiti_client = None

        return settings, llm_adapter

    async def _create_server(self, settings, llm_adapter) -> None:
        """Create and configure the MCP server with tools."""
        from .mcp_server.server import MCPServerFactory
        from .mcp_server.tools import register_graphiti_tools, register_rag_tools
        from .retrieval import (
            create_reranker_client,
            get_reranker_adapter,
        )
        from .retrieval.vector_search import VectorSearchService
        from .retrieval.pipeline import RetrievalPipeline

        # Create MCP server (no auth for stdio - trust local process)
        self._server = MCPServerFactory.create_server(
            name="agentic-rag-mcp-stdio",
            version="1.0.0",
            enable_auth=False,  # stdio transport trusts local process
            rate_limit_requests=0,  # No rate limiting for local
            rate_limit_window=60,
            default_timeout=settings.mcp_tool_timeout_seconds,
        )

        # Register Graphiti tools if available
        if self._graphiti_client and getattr(self._graphiti_client, "is_connected", False):
            register_graphiti_tools(self._server.registry, self._graphiti_client)
            logger.info("graphiti_tools_registered")

            # Create vector search service if PostgreSQL available
            vector_service = None
            if self._postgres_client:
                try:
                    vector_service = VectorSearchService(
                        postgres=self._postgres_client,
                        embedding_provider=settings.embedding_provider,
                        embedding_api_key=settings.embedding_api_key,
                        embedding_base_url=settings.embedding_base_url,
                        embedding_model=settings.embedding_model,
                    )
                    logger.info("vector_service_initialized")
                except Exception as e:
                    logger.warning("vector_service_init_failed", error=str(e))

            # Create reranker if enabled
            reranker = None
            if settings.reranker_enabled:
                try:
                    reranker_adapter = get_reranker_adapter(settings)
                    reranker = create_reranker_client(reranker_adapter)
                    logger.info(
                        "reranker_initialized",
                        provider=settings.reranker_provider,
                        model=settings.reranker_model,
                    )
                except Exception as e:
                    logger.warning("reranker_init_failed", error=str(e))

            # Create retrieval pipeline if both services available
            retrieval_pipeline = None
            if vector_service and self._graphiti_client:
                try:
                    retrieval_pipeline = RetrievalPipeline(
                        vector_service=vector_service,
                        graphiti_client=self._graphiti_client,
                        reranker=reranker,
                        reranker_top_k=settings.reranker_top_k,
                    )
                    logger.info("retrieval_pipeline_initialized")
                except Exception as e:
                    logger.warning("retrieval_pipeline_init_failed", error=str(e))

            # Register RAG tools
            if vector_service:
                register_rag_tools(
                    registry=self._server.registry,
                    graphiti_client=self._graphiti_client,
                    vector_service=vector_service,
                    reranker=reranker,
                    retrieval_pipeline=retrieval_pipeline,
                )
                logger.info("rag_tools_registered")
        else:
            logger.warning(
                "mcp_tools_limited",
                reason="Graphiti not available, only basic tools registered",
            )

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._server:
            self._server.stop()

        if self._graphiti_client:
            try:
                await self._graphiti_client.disconnect()
            except Exception:
                pass

        if self._neo4j_client:
            try:
                await self._neo4j_client.disconnect()
            except Exception:
                pass

        if self._postgres_client:
            try:
                await self._postgres_client.disconnect()
            except Exception:
                pass

        logger.info("mcp_stdio_cleanup_complete")

    async def run(self) -> None:
        """Run the MCP server in stdio mode."""
        logger.info("mcp_stdio_starting")

        try:
            # Initialize clients
            settings, llm_adapter = await self._initialize_clients()

            # Create server with tools
            await self._create_server(settings, llm_adapter)

            if not self._server:
                logger.error("mcp_server_creation_failed")
                return

            # Log available tools
            tools = self._server.registry.list_tools()
            logger.info(
                "mcp_stdio_ready",
                tools_count=len(tools),
                tools=[t["name"] for t in tools],
            )

            # Run stdio transport
            await self._server.run_stdio()

        except KeyboardInterrupt:
            logger.info("mcp_stdio_interrupted")
        except Exception as e:
            logger.exception("mcp_stdio_error", error=str(e))
            raise
        finally:
            await self._cleanup()


def main() -> None:
    """Entry point for the MCP stdio server."""
    # Set up signal handlers
    server = MCPStdioServer()

    def signal_handler(signum, frame):
        logger.info("mcp_stdio_signal_received", signal=signum)
        if server._server:
            server._server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the server
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
