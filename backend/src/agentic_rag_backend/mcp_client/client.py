"""MCP Client implementation for connecting to external MCP servers.

Story 21-C2: Implement MCP Client Factory

This module provides:
- MCPClient: HTTP/SSE client for MCP server communication
- MCPClientFactory: Factory for creating and managing client instances
"""

import asyncio
import random
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import httpx
import structlog

from agentic_rag_backend.mcp_client.config import MCPClientSettings, MCPServerConfig
from agentic_rag_backend.mcp_client.errors import (
    MCPClientConnectionError,
    MCPClientError,
    MCPClientNotEnabledError,
    MCPClientTimeoutError,
    MCPProtocolError,
    MCPServerNotFoundError,
)

logger = structlog.get_logger(__name__)

# Maximum backoff delay in seconds
MAX_BACKOFF_SECONDS = 30.0


class MCPClient:
    """Client for connecting to external MCP servers.

    Handles HTTP/SSE communication with MCP servers, including:
    - Tool discovery via tools/list
    - Tool execution via tools/call
    - Retry logic with exponential backoff
    - Timeout handling

    Attributes:
        config: Server configuration
        name: Server name for logging/identification
    """

    def __init__(
        self,
        config: MCPServerConfig,
        retry_count: int = 3,
        retry_delay_ms: int = 1000,
    ) -> None:
        """Initialize MCP client.

        Args:
            config: Server configuration
            retry_count: Number of retry attempts (from factory settings)
            retry_delay_ms: Base delay between retries in ms (from factory settings)
        """
        self.config = config
        self.name = config.name
        self._retry_count = retry_count
        self._retry_delay_ms = retry_delay_ms

        # Build HTTP client with configured timeout
        # Default to 30s if timeout_ms not set (defensive - factory should always set it)
        timeout_ms = config.timeout_ms if config.timeout_ms is not None else 30000
        timeout_seconds = timeout_ms / 1000.0
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers=self._build_headers(),
            follow_redirects=True,
        )
        # Use Event for thread-safe closed state (is_set() is atomic)
        self._closed_event = asyncio.Event()
        # Lock to protect close operation from concurrent calls
        self._close_lock = asyncio.Lock()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the MCP server.

        Returns:
            List of tool definitions from the server
        """
        logger.debug("mcp_list_tools", server=self.name)
        response = await self._request("tools/list", {})
        
        # Warn if tools key missing - may indicate server issue or protocol mismatch
        if "tools" not in response:
            logger.warning(
                "mcp_list_tools_missing_key",
                server=self.name,
                response_keys=list(response.keys()),
            )
            return []
        
        tools = response["tools"]
        logger.info(
            "mcp_tools_discovered",
            server=self.name,
            tool_count=len(tools),
            tools=[t.get("name") for t in tools],
        )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool on the MCP server.

        Args:
            name: Tool name to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        logger.debug("mcp_call_tool", server=self.name, tool=name)
        result = await self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        logger.info("mcp_tool_called", server=self.name, tool=name)
        return result

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send JSON-RPC request to MCP server with retry logic.

        Args:
            method: MCP method to call (e.g., "tools/list", "tools/call")
            params: Method parameters

        Returns:
            Response result dictionary

        Raises:
            MCPClientTimeoutError: If request times out after all retries
            MCPClientConnectionError: If unable to connect to server
            MCPProtocolError: If server returns an error response
        """
        if self._closed_event.is_set():
            raise MCPClientError(f"MCP client '{self.name}' is closed")

        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        last_error: Optional[Exception] = None
        url = str(self.config.url)

        for attempt in range(self._retry_count + 1):
            try:
                logger.debug(
                    "mcp_request",
                    server=self.name,
                    method=method,
                    attempt=attempt + 1,
                    request_id=request_id,
                )

                response = await self._http_client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                # Check for JSON-RPC error response
                if "error" in data:
                    error = data["error"]
                    raise MCPProtocolError(
                        server_name=self.name,
                        error_code=error.get("code"),
                        error_message=error.get("message"),
                        error_data=error.get("data"),
                    )

                return data.get("result", {})

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "mcp_request_timeout",
                    server=self.name,
                    method=method,
                    attempt=attempt + 1,
                    max_attempts=self._retry_count + 1,
                )
                if attempt == self._retry_count:
                    raise MCPClientTimeoutError(
                        server_name=self.name,
                        attempts=attempt + 1,
                        timeout_ms=self.config.timeout_ms,
                    ) from e
                await self._backoff(attempt)

            except httpx.HTTPStatusError as e:
                # Don't retry 4xx client errors - use MCPProtocolError for semantic clarity
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        "mcp_client_error",
                        server=self.name,
                        method=method,
                        status_code=e.response.status_code,
                    )
                    raise MCPProtocolError(
                        server_name=self.name,
                        error_code=e.response.status_code,
                        error_message=f"HTTP {e.response.status_code}",
                    ) from e

                last_error = e
                logger.warning(
                    "mcp_request_http_error",
                    server=self.name,
                    method=method,
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                )
                if attempt == self._retry_count:
                    raise MCPClientConnectionError(
                        server_name=self.name,
                        url=url,
                        reason=f"HTTP {e.response.status_code} after {attempt + 1} attempts",
                    ) from e
                await self._backoff(attempt)

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    "mcp_connection_error",
                    server=self.name,
                    method=method,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == self._retry_count:
                    raise MCPClientConnectionError(
                        server_name=self.name,
                        url=url,
                        reason=f"Connection failed after {attempt + 1} attempts",
                    ) from e
                await self._backoff(attempt)

            except MCPProtocolError:
                # Don't retry protocol errors - they're application-level
                raise

        # Should not reach here, but just in case
        raise MCPClientError(
            f"Unexpected error after {self._retry_count + 1} attempts: {last_error}"
        )

    async def _backoff(self, attempt: int) -> None:
        """Calculate and wait for exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-indexed)
        """
        # Exponential backoff: base_delay * 2^attempt
        base_delay = self._retry_delay_ms / 1000.0
        delay = base_delay * (2**attempt)
        # Add jitter (10% of delay)
        jitter = random.uniform(0, delay * 0.1)
        delay = min(delay + jitter, MAX_BACKOFF_SECONDS)
        logger.debug("mcp_backoff", server=self.name, delay_seconds=delay)
        await asyncio.sleep(delay)

    async def close(self) -> None:
        """Close the HTTP client and release resources.

        Thread-safe: Uses lock to prevent double-close race conditions.
        """
        async with self._close_lock:
            if not self._closed_event.is_set():
                self._closed_event.set()
                await self._http_client.aclose()
                logger.debug("mcp_client_closed", server=self.name)

    @property
    def is_closed(self) -> bool:
        """Check if client is closed.

        Thread-safe: Uses asyncio.Event.is_set() which is atomic.
        """
        return self._closed_event.is_set()


class MCPClientFactory:
    """Factory for creating and managing MCP client instances.

    Manages client lifecycle, including:
    - Lazy client creation on demand
    - Connection pooling (reuse clients)
    - Tool discovery across all servers
    - Graceful shutdown

    Attributes:
        settings: MCP client configuration settings
    """

    def __init__(self, settings: MCPClientSettings) -> None:
        """Initialize factory with settings.

        Args:
            settings: MCP client configuration
        """
        self.settings = settings
        self._clients: dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()

    @property
    def is_enabled(self) -> bool:
        """Check if MCP client feature is enabled."""
        return self.settings.enabled

    @property
    def server_names(self) -> list[str]:
        """Get list of configured server names."""
        return [server.name for server in self.settings.servers]

    async def get_client(self, name: str) -> MCPClient:
        """Get or create an MCP client by server name.

        Args:
            name: Server name from configuration

        Returns:
            MCPClient instance

        Raises:
            MCPClientNotEnabledError: If MCP client feature is disabled
            MCPServerNotFoundError: If server name is not in configuration
        """
        if not self.settings.enabled:
            raise MCPClientNotEnabledError()

        async with self._lock:
            if name in self._clients and not self._clients[name].is_closed:
                return self._clients[name]

            config = self._find_server_config(name)
            if not config:
                raise MCPServerNotFoundError(
                    server_name=name,
                    available_servers=self.server_names,
                )

            client = MCPClient(
                config=config,
                retry_count=self.settings.retry_count,
                retry_delay_ms=self.settings.retry_delay_ms,
            )
            self._clients[name] = client
            logger.info("mcp_client_created", server=name, url=str(config.url))
            return client

    def _find_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """Find server configuration by name.

        Args:
            name: Server name to find

        Returns:
            Server configuration with global timeout applied if not explicitly set
        """
        for server in self.settings.servers:
            if server.name == name:
                # Apply global default_timeout_ms if server timeout not explicitly set
                # None means "use factory default", any explicit value is preserved
                if server.timeout_ms is None:
                    return MCPServerConfig(
                        name=server.name,
                        url=server.url,
                        api_key=server.api_key,
                        transport=server.transport,
                        timeout_ms=self.settings.default_timeout_ms,
                    )
                return server
        return None

    async def discover_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        """Discover tools from all configured MCP servers.

        Returns:
            Dictionary mapping server names to their tool definitions

        Note:
            Failures for individual servers are logged but don't prevent
            discovery from other servers.
        """
        if not self.settings.enabled:
            logger.debug("mcp_discovery_skipped", reason="feature_disabled")
            return {}

        tools: dict[str, list[dict[str, Any]]] = {}
        for server in self.settings.servers:
            try:
                client = await self.get_client(server.name)
                server_tools = await client.list_tools()
                tools[server.name] = server_tools
            except Exception as e:
                logger.warning(
                    "mcp_tool_discovery_failed",
                    server=server.name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool on a specific MCP server.

        Args:
            server_name: Server name from configuration
            tool_name: Tool name to execute
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPClientNotEnabledError: If MCP client feature is disabled
            MCPServerNotFoundError: If server name is not in configuration
            MCPClientError: If tool execution fails
        """
        client = await self.get_client(server_name)
        return await client.call_tool(tool_name, arguments)

    async def close_all(self) -> None:
        """Close all MCP clients and release resources.

        Should be called during application shutdown.
        """
        async with self._lock:
            # Create a copy of items before iterating to avoid dict modification issues
            clients_copy = list(self._clients.items())
            count = len(clients_copy)
            for name, client in clients_copy:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(
                        "mcp_client_close_error",
                        server=name,
                        error=str(e),
                    )
            self._clients.clear()
            logger.info("mcp_clients_closed", count=count)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator["MCPClientFactory"]:
        """Context manager for factory lifecycle.

        Example:
            async with factory.lifespan() as f:
                tools = await f.discover_all_tools()
        """
        try:
            yield self
        finally:
            await self.close_all()
