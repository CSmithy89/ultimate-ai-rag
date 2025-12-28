"""Neo4j async client for knowledge graph operations."""

from typing import Any, Optional

import structlog
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError as Neo4jDriverError

from agentic_rag_backend.core.errors import Neo4jError

logger = structlog.get_logger(__name__)


class Neo4jClient:
    """
    Async Neo4j client for knowledge graph operations.

    Implements multi-tenancy through tenant_id filtering on all queries.
    Uses MERGE operations for idempotent node and relationship creation.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info("neo4j_connected", uri=self.uri)

    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    @property
    def driver(self):
        """Get the Neo4j driver, raising error if not connected."""
        if self._driver is None:
            raise Neo4jError("connection", "Neo4j driver not connected")
        return self._driver

    async def create_indexes(self) -> None:
        """
        Create required indexes for performance.

        Creates indexes on entity_id, tenant_id, type, and name for
        efficient queries and deduplication lookups.
        """
        try:
            async with self.driver.session() as session:
                # Entity indexes
                await session.run(
                    "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)"
                )
                await session.run(
                    "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)"
                )
                await session.run(
                    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)"
                )
                await session.run(
                    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                )
                # Document indexes
                await session.run(
                    "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)"
                )
                await session.run(
                    "CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id)"
                )
                # Chunk indexes
                await session.run(
                    "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
                )
                await session.run(
                    "CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id)"
                )
                logger.info("neo4j_indexes_created")
        except Neo4jDriverError as e:
            raise Neo4jError("create_indexes", str(e)) from e

    async def create_entity(
        self,
        entity_id: str,
        tenant_id: str,
        name: str,
        entity_type: str,
        description: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        source_chunk_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create or merge an entity node.

        Uses MERGE for idempotent creation. If entity exists, updates
        properties and appends chunk reference to source_chunks array.

        Args:
            entity_id: Unique entity identifier (UUID)
            tenant_id: Tenant identifier for multi-tenancy
            name: Entity name
            entity_type: Entity type (Person, Organization, Technology, Concept, Location)
            description: Optional entity description
            properties: Optional additional properties
            source_chunk_id: Optional chunk ID where entity was found

        Returns:
            Dictionary with created/updated entity properties
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.tenant_id = $tenant_id,
                        e.name = $name,
                        e.type = $type,
                        e.description = COALESCE($description, e.description),
                        e.properties = COALESCE($properties, e.properties),
                        e.source_chunks = CASE
                            WHEN $chunk_id IS NOT NULL AND NOT $chunk_id IN COALESCE(e.source_chunks, [])
                            THEN COALESCE(e.source_chunks, []) + $chunk_id
                            ELSE COALESCE(e.source_chunks, [])
                        END,
                        e.updated_at = datetime()
                    ON CREATE SET e.created_at = datetime()
                    RETURN e
                    """,
                    id=entity_id,
                    tenant_id=tenant_id,
                    name=name,
                    type=entity_type,
                    description=description,
                    properties=properties,
                    chunk_id=source_chunk_id,
                )
                record = await result.single()
                if record:
                    entity_data = dict(record["e"])
                    logger.info(
                        "entity_created",
                        entity_id=entity_id,
                        name=name,
                        type=entity_type,
                    )
                    return entity_data
                return {}
        except Neo4jDriverError as e:
            raise Neo4jError("create_entity", str(e)) from e

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        tenant_id: str,
        confidence: float,
        chunk_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Create a relationship between two entities.

        Uses MERGE to avoid duplicate relationships. The relationship type
        must be one of: MENTIONS, AUTHORED_BY, PART_OF, USES, RELATED_TO.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            tenant_id: Tenant identifier for filtering
            confidence: Confidence score (0.0-1.0)
            chunk_id: Optional source chunk ID
            description: Optional relationship description

        Returns:
            True if relationship was created, False otherwise
        """
        # Validate relationship type to prevent injection
        valid_types = {"MENTIONS", "AUTHORED_BY", "PART_OF", "USES", "RELATED_TO"}
        if relationship_type not in valid_types:
            logger.warning(
                "invalid_relationship_type",
                type=relationship_type,
                valid_types=list(valid_types),
            )
            return False

        try:
            async with self.driver.session() as session:
                # Dynamic relationship type requires string formatting
                # We've validated the type above to prevent injection
                query = f"""
                MATCH (source:Entity {{id: $source_id, tenant_id: $tenant_id}})
                MATCH (target:Entity {{id: $target_id, tenant_id: $tenant_id}})
                MERGE (source)-[r:{relationship_type}]->(target)
                SET r.confidence = $confidence,
                    r.source_chunk = $chunk_id,
                    r.description = $description,
                    r.created_at = COALESCE(r.created_at, datetime())
                RETURN r
                """
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    tenant_id=tenant_id,
                    confidence=confidence,
                    chunk_id=chunk_id,
                    description=description,
                )
                record = await result.single()
                if record:
                    logger.info(
                        "relationship_created",
                        source=source_id,
                        target=target_id,
                        type=relationship_type,
                        confidence=confidence,
                    )
                    return True
                return False
        except Neo4jDriverError as e:
            raise Neo4jError("create_relationship", str(e)) from e

    async def find_similar_entity(
        self,
        tenant_id: str,
        name: str,
        entity_type: str,
    ) -> Optional[dict[str, Any]]:
        """
        Find an existing entity by normalized name and type.

        Used for deduplication before creating new entities. Performs
        case-insensitive matching with trimmed whitespace.

        Args:
            tenant_id: Tenant identifier for filtering
            name: Entity name to search for
            entity_type: Entity type to match

        Returns:
            Entity dictionary if found, None otherwise
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id, type: $type})
                    WHERE toLower(trim(e.name)) = toLower(trim($name))
                    RETURN e
                    LIMIT 1
                    """,
                    tenant_id=tenant_id,
                    name=name,
                    type=entity_type,
                )
                record = await result.single()
                if record:
                    return dict(record["e"])
                return None
        except Neo4jDriverError as e:
            raise Neo4jError("find_similar_entity", str(e)) from e

    async def get_entity(
        self,
        entity_id: str,
        tenant_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get an entity by ID with tenant filtering.

        Args:
            entity_id: Entity UUID
            tenant_id: Tenant identifier for access control

        Returns:
            Entity dictionary if found, None otherwise
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {id: $id, tenant_id: $tenant_id})
                    RETURN e
                    """,
                    id=entity_id,
                    tenant_id=tenant_id,
                )
                record = await result.single()
                if record:
                    return dict(record["e"])
                return None
        except Neo4jDriverError as e:
            raise Neo4jError("get_entity", str(e)) from e

    async def get_entities_by_type(
        self,
        tenant_id: str,
        entity_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get all entities of a specific type for a tenant.

        Args:
            tenant_id: Tenant identifier
            entity_type: Entity type to filter by
            limit: Maximum number of entities to return

        Returns:
            List of entity dictionaries
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id, type: $type})
                    RETURN e
                    ORDER BY e.name
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    type=entity_type,
                    limit=limit,
                )
                records = await result.data()
                return [dict(r["e"]) for r in records]
        except Neo4jDriverError as e:
            raise Neo4jError("get_entities_by_type", str(e)) from e

    async def create_document_node(
        self,
        document_id: str,
        tenant_id: str,
        title: Optional[str] = None,
        source_url: Optional[str] = None,
        source_type: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create or merge a document node in the graph.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier
            title: Document title
            source_url: Source URL for web documents
            source_type: Type of source (url, pdf, text)
            content_hash: Content hash for deduplication

        Returns:
            Dictionary with document node properties
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MERGE (d:Document {id: $id})
                    SET d.tenant_id = $tenant_id,
                        d.title = COALESCE($title, d.title),
                        d.source_url = COALESCE($source_url, d.source_url),
                        d.source_type = COALESCE($source_type, d.source_type),
                        d.content_hash = COALESCE($content_hash, d.content_hash),
                        d.updated_at = datetime()
                    ON CREATE SET d.created_at = datetime()
                    RETURN d
                    """,
                    id=document_id,
                    tenant_id=tenant_id,
                    title=title,
                    source_url=source_url,
                    source_type=source_type,
                    content_hash=content_hash,
                )
                record = await result.single()
                if record:
                    logger.info("document_node_created", document_id=document_id)
                    return dict(record["d"])
                return {}
        except Neo4jDriverError as e:
            raise Neo4jError("create_document_node", str(e)) from e

    async def create_chunk_node(
        self,
        chunk_id: str,
        tenant_id: str,
        document_id: str,
        chunk_index: int,
        preview: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a chunk node and link it to its document.

        Args:
            chunk_id: Chunk UUID
            tenant_id: Tenant identifier
            document_id: Parent document ID
            chunk_index: Position in document
            preview: First 200 characters of chunk content

        Returns:
            Dictionary with chunk node properties
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MERGE (c:Chunk {id: $id})
                    SET c.tenant_id = $tenant_id,
                        c.document_id = $document_id,
                        c.chunk_index = $chunk_index,
                        c.preview = $preview,
                        c.created_at = COALESCE(c.created_at, datetime())
                    WITH c
                    MATCH (d:Document {id: $document_id, tenant_id: $tenant_id})
                    MERGE (d)-[:CONTAINS]->(c)
                    RETURN c
                    """,
                    id=chunk_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    preview=preview[:200] if preview else None,
                )
                record = await result.single()
                if record:
                    return dict(record["c"])
                return {}
        except Neo4jDriverError as e:
            raise Neo4jError("create_chunk_node", str(e)) from e

    async def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_id: str,
        tenant_id: str,
        confidence: float = 1.0,
    ) -> bool:
        """
        Create a MENTIONS relationship from chunk to entity.

        Args:
            chunk_id: Chunk UUID
            entity_id: Entity UUID
            tenant_id: Tenant identifier
            confidence: Confidence score

        Returns:
            True if relationship created, False otherwise
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id, tenant_id: $tenant_id})
                    MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.confidence = $confidence,
                        r.created_at = COALESCE(r.created_at, datetime())
                    RETURN r
                    """,
                    chunk_id=chunk_id,
                    entity_id=entity_id,
                    tenant_id=tenant_id,
                    confidence=confidence,
                )
                record = await result.single()
                return record is not None
        except Neo4jDriverError as e:
            raise Neo4jError("link_chunk_to_entity", str(e)) from e

    async def get_graph_stats(self, tenant_id: str) -> dict[str, int]:
        """
        Get statistics for the knowledge graph.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary with node and relationship counts
        """
        try:
            async with self.driver.session() as session:
                # Count entities
                entity_result = await session.run(
                    "MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e) as count",
                    tenant_id=tenant_id,
                )
                entity_record = await entity_result.single()
                entity_count = entity_record["count"] if entity_record else 0

                # Count documents
                doc_result = await session.run(
                    "MATCH (d:Document {tenant_id: $tenant_id}) RETURN count(d) as count",
                    tenant_id=tenant_id,
                )
                doc_record = await doc_result.single()
                doc_count = doc_record["count"] if doc_record else 0

                # Count chunks
                chunk_result = await session.run(
                    "MATCH (c:Chunk {tenant_id: $tenant_id}) RETURN count(c) as count",
                    tenant_id=tenant_id,
                )
                chunk_record = await chunk_result.single()
                chunk_count = chunk_record["count"] if chunk_record else 0

                # Count relationships
                rel_result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})-[r]-()
                    RETURN count(r) as count
                    """,
                    tenant_id=tenant_id,
                )
                rel_record = await rel_result.single()
                rel_count = rel_record["count"] if rel_record else 0

                return {
                    "entity_count": entity_count,
                    "document_count": doc_count,
                    "chunk_count": chunk_count,
                    "relationship_count": rel_count,
                }
        except Neo4jDriverError as e:
            raise Neo4jError("get_graph_stats", str(e)) from e


# Global Neo4j client instance
_neo4j_client: Optional[Neo4jClient] = None


async def get_neo4j_client(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Neo4jClient:
    """
    Get or create the global Neo4j client instance.

    Args:
        uri: Neo4j connection URI. Required on first call.
        user: Neo4j username. Required on first call.
        password: Neo4j password. Required on first call.

    Returns:
        Neo4jClient instance
    """
    global _neo4j_client
    if _neo4j_client is None:
        if uri is None or user is None or password is None:
            raise Neo4jError(
                "init",
                "Neo4j URI, user, and password required for first initialization",
            )
        _neo4j_client = Neo4jClient(uri, user, password)
        await _neo4j_client.connect()
        await _neo4j_client.create_indexes()
    return _neo4j_client


async def close_neo4j_client() -> None:
    """Close the global Neo4j client connection."""
    global _neo4j_client
    if _neo4j_client is not None:
        await _neo4j_client.disconnect()
        _neo4j_client = None
