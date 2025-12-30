"""Graphiti client wrapper for temporal knowledge graph operations.

Provides a managed client for interacting with Graphiti's temporal
knowledge graph, including connection management, index building,
and custom entity type configuration.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional

import structlog

try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIClient
    from graphiti_core.embedder import OpenAIEmbedder

    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    Graphiti = None  # type: ignore
    OpenAIClient = None  # type: ignore
    OpenAIEmbedder = None  # type: ignore

from ..models.entity_types import ENTITY_TYPES, EDGE_TYPE_MAPPINGS

logger = logging.getLogger(__name__)
struct_logger = structlog.get_logger(__name__)

# Connection configuration constants
# 5 seconds allows graceful shutdown while preventing hung connections
DEFAULT_DISCONNECT_TIMEOUT = 5.0


class ConnectionState(str, Enum):
    """State machine for GraphitiClient connection lifecycle."""
    NEW = "new"              # Never connected
    CONNECTING = "connecting"  # Connection in progress
    CONNECTED = "connected"    # Successfully connected
    DISCONNECTED = "disconnected"  # Disconnected, cannot reconnect


class GraphitiClient:
    """Managed Graphiti client for temporal knowledge graph operations.

    Handles connection lifecycle, index management, and provides
    access to Graphiti's core functionality with custom entity types.
    
    Connection State Machine:
        NEW -> CONNECTING -> CONNECTED -> DISCONNECTED
        
    Once disconnected, the client cannot be reconnected (credentials cleared).
    Create a new instance instead.

    Attributes:
        uri: Neo4j connection URI
        user: Neo4j username
        client: The underlying Graphiti instance
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize Graphiti client configuration.

        Args:
            uri: Neo4j connection URI (bolt:// or neo4j://)
            user: Neo4j username
            password: Neo4j password
            openai_api_key: OpenAI API key for embeddings and LLM
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model for entity extraction
        """
        if not GRAPHITI_AVAILABLE:
            raise ImportError(
                "graphiti-core is not installed. "
                "Install with: uv add graphiti-core"
            )

        self.uri = uri
        self.user = user
        self._password: Optional[str] = password
        self._openai_api_key: Optional[str] = openai_api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._client: Optional[Graphiti] = None
        self._state = ConnectionState.NEW
        self._connect_lock = asyncio.Lock()

    @property
    def client(self) -> Graphiti:
        """Get the underlying Graphiti client.

        Returns:
            The Graphiti instance

        Raises:
            RuntimeError: If client is not connected
        """
        if self._client is None or self._state != ConnectionState.CONNECTED:
            raise RuntimeError(
                f"Graphiti client is not connected (state: {self._state.value}). "
                "Call connect() first."
            )
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._state == ConnectionState.CONNECTED and self._client is not None

    async def connect(self) -> None:
        """Establish connection to Neo4j and initialize Graphiti.

        Creates the Graphiti instance with configured LLM client,
        embedder, and custom entity types. Clears sensitive credentials
        from memory after successful connection for security.
        
        Thread-safe via asyncio lock to prevent race conditions.
        
        Raises:
            RuntimeError: If client is in invalid state for connection
        """
        async with self._connect_lock:
            # State machine validation
            if self._state == ConnectionState.CONNECTED:
                struct_logger.warning("graphiti_already_connected")
                return
            
            if self._state == ConnectionState.CONNECTING:
                raise RuntimeError(
                    "Connection already in progress. Wait for completion."
                )
            
            if self._state == ConnectionState.DISCONNECTED:
                raise RuntimeError(
                    "Cannot reconnect after disconnect. "
                    "Create a new GraphitiClient instance."
                )

            # Transition to connecting state
            self._state = ConnectionState.CONNECTING

            try:
                # Create OpenAI clients for LLM and embeddings
                llm_client = OpenAIClient(  # type: ignore[call-arg]
                    api_key=self._openai_api_key,
                    model=self.llm_model,
                )
                embedder = OpenAIEmbedder(  # type: ignore[call-arg]
                    api_key=self._openai_api_key,
                    model=self.embedding_model,
                )

                # Clear the API key from memory after passing to clients
                self._openai_api_key = None

                # Initialize Graphiti with custom configuration
                self._client = Graphiti(
                    uri=self.uri,
                    user=self.user,
                    password=self._password,
                    llm_client=llm_client,
                    embedder=embedder,
                )

                # Clear password from memory after successful connection
                self._password = None

                # Transition to connected state
                self._state = ConnectionState.CONNECTED
                struct_logger.info(
                    "graphiti_connected",
                    uri=self.uri,
                    embedding_model=self.embedding_model,
                    llm_model=self.llm_model,
                )

            except Exception as e:
                # Transition back to NEW on failure (credentials still available)
                self._state = ConnectionState.NEW
                self._client = None
                struct_logger.error("graphiti_connection_failed", error=str(e))
                raise

    async def disconnect(self, timeout: float = DEFAULT_DISCONNECT_TIMEOUT) -> None:
        """Close the Graphiti connection and cleanup resources.
        
        This is a fire-and-forget operation - timeout and errors are logged
        but not re-raised to allow graceful shutdown even with connection issues.
        
        After disconnect, the client cannot be reconnected (credentials cleared).
        
        Args:
            timeout: Maximum seconds to wait for graceful disconnect
        """
        if self._state != ConnectionState.CONNECTED or self._client is None:
            return

        try:
            # Graphiti uses Neo4j driver internally, close it with timeout
            if hasattr(self._client, "driver") and self._client.driver:
                await asyncio.wait_for(
                    self._client.driver.close(),
                    timeout=timeout,
                )
            struct_logger.info("graphiti_disconnected")

        except asyncio.TimeoutError:
            struct_logger.error(
                "graphiti_disconnect_timeout",
                timeout_seconds=timeout,
            )
        except Exception as e:
            struct_logger.warning("graphiti_disconnect_error", error=str(e))
        finally:
            self._client = None
            self._state = ConnectionState.DISCONNECTED

    async def build_indices(self) -> None:
        """Build required Neo4j indices for Graphiti operations.

        Creates indices for efficient entity and relationship queries,
        including vector indices for semantic search.
        """
        if self._state != ConnectionState.CONNECTED:
            raise RuntimeError(
                f"Client not connected (state: {self._state.value}). "
                "Call connect() first."
            )

        try:
            await self.client.build_indices_and_constraints()
            struct_logger.info("graphiti_indices_built")
        except Exception as e:
            struct_logger.error("graphiti_index_build_failed", error=str(e))
            raise

    def get_entity_types(self) -> list[type]:
        """Get configured custom entity types.

        Returns:
            List of entity type classes
        """
        return ENTITY_TYPES

    def get_edge_type_mappings(self) -> dict[tuple[str, str], list[str]]:
        """Get edge type mappings for entity pairs.

        Returns:
            Dictionary mapping entity type pairs to valid edge types
        """
        return EDGE_TYPE_MAPPINGS

    def __repr__(self) -> str:
        """Safe repr that hides sensitive credentials."""
        return (
            f"GraphitiClient(uri={self.uri!r}, user={self.user!r}, "
            f"state={self._state.value!r})"
        )


async def create_graphiti_client(
    uri: str,
    user: str,
    password: str,
    openai_api_key: str,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
) -> GraphitiClient:
    """Factory function to create and connect a Graphiti client.

    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        openai_api_key: OpenAI API key
        embedding_model: OpenAI embedding model name
        llm_model: OpenAI LLM model name

    Returns:
        Connected GraphitiClient instance
    """
    client = GraphitiClient(
        uri=uri,
        user=user,
        password=password,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    await client.connect()
    return client
