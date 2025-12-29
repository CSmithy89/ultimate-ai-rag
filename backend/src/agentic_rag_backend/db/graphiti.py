"""Graphiti client wrapper for temporal knowledge graph operations.

Provides a managed client for interacting with Graphiti's temporal
knowledge graph, including connection management, index building,
and custom entity type configuration.
"""

import logging
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


class GraphitiClient:
    """Managed Graphiti client for temporal knowledge graph operations.

    Handles connection lifecycle, index management, and provides
    access to Graphiti's core functionality with custom entity types.

    Attributes:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        openai_api_key: OpenAI API key for LLM operations
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
        self.password = password
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._client: Optional[Graphiti] = None
        self._connected = False

    @property
    def client(self) -> Graphiti:
        """Get the underlying Graphiti client.

        Returns:
            The Graphiti instance

        Raises:
            RuntimeError: If client is not connected
        """
        if self._client is None or not self._connected:
            raise RuntimeError(
                "Graphiti client is not connected. Call connect() first."
            )
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None

    async def connect(self) -> None:
        """Establish connection to Neo4j and initialize Graphiti.

        Creates the Graphiti instance with configured LLM client,
        embedder, and custom entity types.
        """
        if self._connected:
            struct_logger.warning("graphiti_already_connected")
            return

        try:
            # Create OpenAI clients for LLM and embeddings
            llm_client = OpenAIClient(
                api_key=self.openai_api_key,
                model=self.llm_model,
            )
            embedder = OpenAIEmbedder(
                api_key=self.openai_api_key,
                model=self.embedding_model,
            )

            # Initialize Graphiti with custom configuration
            self._client = Graphiti(
                uri=self.uri,
                user=self.user,
                password=self.password,
                llm_client=llm_client,
                embedder=embedder,
            )

            self._connected = True
            struct_logger.info(
                "graphiti_connected",
                uri=self.uri,
                embedding_model=self.embedding_model,
                llm_model=self.llm_model,
            )

        except Exception as e:
            self._connected = False
            self._client = None
            struct_logger.error("graphiti_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close the Graphiti connection and cleanup resources."""
        if not self._connected or self._client is None:
            return

        try:
            # Graphiti uses Neo4j driver internally, close it
            if hasattr(self._client, "driver") and self._client.driver:
                await self._client.driver.close()

            struct_logger.info("graphiti_disconnected")
        except Exception as e:
            struct_logger.warning("graphiti_disconnect_error", error=str(e))
        finally:
            self._client = None
            self._connected = False

    async def build_indices(self) -> None:
        """Build required Neo4j indices for Graphiti operations.

        Creates indices for efficient entity and relationship queries,
        including vector indices for semantic search.
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Call connect() first.")

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
