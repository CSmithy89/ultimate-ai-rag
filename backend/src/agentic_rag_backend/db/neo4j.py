"""Neo4j async client for knowledge graph operations."""

import asyncio
import re
from typing import Any, Optional

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError as Neo4jDriverError

from agentic_rag_backend.core.errors import Neo4jError

logger = structlog.get_logger(__name__)

DEFAULT_ALLOWED_RELATIONSHIPS = (
    "MENTIONS",
    "AUTHORED_BY",
    "PART_OF",
    "USES",
    "RELATED_TO",
    "CALLS",
    "IMPORTS",
    "EXTENDS",
    "IMPLEMENTS",
    "DEFINED_IN",
    "USES_TYPE",
)
TERM_SAFE_PATTERN = re.compile(r"^[a-z0-9_-]{2,}$")


class Neo4jClient:
    """
    Async Neo4j client for knowledge graph operations.

    Implements multi-tenancy through tenant_id filtering on all queries.
    Uses MERGE operations for idempotent node and relationship creation.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        pool_min_size: int = 1,
        pool_max_size: int = 50,
        pool_acquire_timeout: float = 30.0,
        connection_timeout: float = 30.0,
        max_connection_lifetime: int = 3600,
    ) -> None:
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
        self._driver: Optional[AsyncDriver] = None
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.pool_acquire_timeout = pool_acquire_timeout
        self.connection_timeout = connection_timeout
        self.max_connection_lifetime = max_connection_lifetime

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self.pool_max_size,
                connection_acquisition_timeout=self.pool_acquire_timeout,
                connection_timeout=self.connection_timeout,
                max_connection_lifetime=self.max_connection_lifetime,
            )
            logger.info(
                "neo4j_pool_configured",
                uri=self.uri,
                pool_min_size=self.pool_min_size,
                pool_max_size=self.pool_max_size,
                pool_acquire_timeout=self.pool_acquire_timeout,
                connection_timeout=self.connection_timeout,
                max_connection_lifetime=self.max_connection_lifetime,
            )
            # Neo4j driver does not support a min pool size; we warm manually instead.
            if self.pool_min_size > 0:
                await self._warm_pool(self.pool_min_size)
            logger.info("neo4j_connected", uri=self.uri)

    async def _warm_pool(self, target_size: int) -> None:
        """Warm the Neo4j pool by opening concurrent sessions."""
        async def _ping() -> None:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1")
                await result.consume()

        await asyncio.gather(*[_ping() for _ in range(target_size)])
        logger.info("neo4j_pool_warmed", pool_min_size=target_size)

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

        Uses MERGE with tenant_id for tenant isolation. If entity exists,
        updates properties and appends chunk reference to source_chunks array.

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
                    MERGE (e:Entity {id: $id, tenant_id: $tenant_id})
                    SET e.name = $name,
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
        valid_types = {
            "MENTIONS",
            "AUTHORED_BY",
            "PART_OF",
            "USES",
            "RELATED_TO",
            "CALLS",
            "IMPORTS",
            "EXTENDS",
            "IMPLEMENTS",
            "DEFINED_IN",
            "USES_TYPE",
        }
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

    async def get_entity_relationships(
        self,
        entity_id: str,
        tenant_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetch outgoing relationships for an entity.

        Args:
            entity_id: Entity UUID
            tenant_id: Tenant identifier
            limit: Maximum relationships to return

        Returns:
            List of relationship dictionaries with target data
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Entity {id: $id, tenant_id: $tenant_id})-[r]->(target:Entity)
                    WHERE target.tenant_id = $tenant_id
                    RETURN type(r) as type,
                           target.id as target_id,
                           target.name as target_name,
                           target.type as target_type,
                           r.confidence as confidence,
                           r.description as description
                    LIMIT $limit
                    """,
                    id=entity_id,
                    tenant_id=tenant_id,
                    limit=limit,
                )
                records = await result.data()
                return [
                    {
                        "type": record.get("type"),
                        "target_id": record.get("target_id"),
                        "target_name": record.get("target_name"),
                        "target_type": record.get("target_type"),
                        "confidence": record.get("confidence"),
                        "description": record.get("description"),
                    }
                    for record in records
                ]
        except Neo4jDriverError as e:
            raise Neo4jError("get_entity_relationships", str(e)) from e

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

        Uses MERGE with tenant_id for tenant isolation.

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
                    MERGE (d:Document {id: $id, tenant_id: $tenant_id})
                    SET d.title = COALESCE($title, d.title),
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

        Uses MERGE with tenant_id for tenant isolation.

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
                    MERGE (c:Chunk {id: $id, tenant_id: $tenant_id})
                    SET c.document_id = $document_id,
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

    async def add_chunk_reference_to_entity(
        self,
        entity_id: str,
        tenant_id: str,
        chunk_id: str,
    ) -> bool:
        """
        Add a chunk reference to an existing entity without modifying other properties.

        This method is used during deduplication to add new source chunk references
        to an existing entity without overwriting its description or other properties.

        Args:
            entity_id: Entity UUID
            tenant_id: Tenant identifier
            chunk_id: Chunk ID to add to source_chunks array

        Returns:
            True if entity was updated, False otherwise
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {id: $id, tenant_id: $tenant_id})
                    SET e.source_chunks = CASE
                        WHEN NOT $chunk_id IN COALESCE(e.source_chunks, [])
                        THEN COALESCE(e.source_chunks, []) + $chunk_id
                        ELSE e.source_chunks
                    END,
                    e.updated_at = datetime()
                    RETURN e
                    """,
                    id=entity_id,
                    tenant_id=tenant_id,
                    chunk_id=chunk_id,
                )
                record = await result.single()
                if record:
                    logger.debug(
                        "chunk_reference_added",
                        entity_id=entity_id,
                        chunk_id=chunk_id,
                    )
                    return True
                return False
        except Neo4jDriverError as e:
            raise Neo4jError("add_chunk_reference_to_entity", str(e)) from e

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

    async def search_entities_by_terms(
        self,
        tenant_id: str,
        terms: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find entities whose names contain any of the provided terms.

        Args:
            tenant_id: Tenant identifier
            terms: Lowercased search terms (sanitized to prevent Cypher keyword injection)
            limit: Maximum number of entities to return

        Returns:
            List of entity property dictionaries
        """
        if not terms:
            return []
        safe_terms: list[str] = []
        for term in terms:
            if not term:
                continue
            normalized = term.lower()
            if TERM_SAFE_PATTERN.fullmatch(normalized):
                safe_terms.append(normalized)
        if not safe_terms:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})
                    WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS term)
                    RETURN e
                    ORDER BY e.name
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    terms=safe_terms,
                    limit=limit,
                )
                records = await result.data()
                return [dict(record["e"]) for record in records if record.get("e")]
        except Neo4jDriverError as e:
            raise Neo4jError("search_entities_by_terms", str(e)) from e

    async def traverse_paths(
        self,
        tenant_id: str,
        start_entity_ids: list[str],
        max_hops: int = 2,
        limit: int = 10,
        allowed_relationships: Optional[list[str]] = None,
    ) -> list[Any]:
        """
        Traverse bounded paths from starting entities.

        Args:
            tenant_id: Tenant identifier
            start_entity_ids: List of starting entity IDs
            max_hops: Maximum relationship hops to traverse
            limit: Maximum number of paths to return
            allowed_relationships: Allowed relationship types

        Returns:
            List of Neo4j Path objects
        """
        if not start_entity_ids:
            return []
        if not (1 <= max_hops <= 5):
            raise ValueError("max_hops must be between 1 and 5")
        rel_types = allowed_relationships or list(DEFAULT_ALLOWED_RELATIONSHIPS)
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH p=(start:Entity)-[r*1..]-(target:Entity)
                    WHERE start.id IN $start_ids
                      AND start.tenant_id = $tenant_id
                      AND all(n IN nodes(p) WHERE n.tenant_id = $tenant_id)
                      AND all(rel IN relationships(p) WHERE type(rel) IN $rel_types)
                      AND length(p) <= $max_hops
                    RETURN p
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    start_ids=start_entity_ids,
                    max_hops=max_hops,
                    rel_types=rel_types,
                    limit=limit,
                )
                records = await result.data()
                return [record["p"] for record in records if record.get("p")]
        except Neo4jDriverError as e:
            raise Neo4jError("traverse_paths", str(e)) from e

    # Story 4.4 - Knowledge Graph Visualization Methods

    async def get_graph_data(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fetch graph data for visualization.

        Args:
            tenant_id: Tenant identifier for filtering
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip
            entity_type: Optional filter by entity type
            relationship_type: Optional filter by relationship type

        Returns:
            Dictionary with nodes and edges lists
        """
        try:
            async with self.driver.session() as session:
                # Build the query based on filters
                if entity_type:
                    node_query = """
                    MATCH (n:Entity {tenant_id: $tenant_id, type: $entity_type})
                    RETURN n
                    ORDER BY n.name
                    SKIP $offset
                    LIMIT $limit
                    """
                    node_params = {
                        "tenant_id": tenant_id,
                        "entity_type": entity_type,
                        "offset": offset,
                        "limit": limit,
                    }
                else:
                    node_query = """
                    MATCH (n:Entity {tenant_id: $tenant_id})
                    RETURN n
                    ORDER BY n.name
                    SKIP $offset
                    LIMIT $limit
                    """
                    node_params = {
                        "tenant_id": tenant_id,
                        "offset": offset,
                        "limit": limit,
                    }

                # Fetch nodes
                node_result = await session.run(node_query, **node_params)
                node_records = await node_result.data()

                # Collect node IDs for edge query
                node_ids = [r["n"]["id"] for r in node_records if r.get("n")]

                # Check which nodes are orphans
                orphan_ids = set()
                if node_ids:
                    orphan_query = """
                    MATCH (n:Entity)
                    WHERE n.id IN $node_ids AND n.tenant_id = $tenant_id
                    AND NOT (n)-[]-()
                    RETURN n.id as id
                    """
                    orphan_result = await session.run(
                        orphan_query,
                        node_ids=node_ids,
                        tenant_id=tenant_id,
                    )
                    orphan_records = await orphan_result.data()
                    orphan_ids = {r["id"] for r in orphan_records}

                # Build nodes list
                nodes = []
                for record in node_records:
                    if record.get("n"):
                        node_data = dict(record["n"])
                        nodes.append({
                            "id": node_data.get("id", ""),
                            "label": node_data.get("name", ""),
                            "type": node_data.get("type", ""),
                            "properties": {
                                "description": node_data.get("description"),
                                "source_chunks": node_data.get("source_chunks", []),
                            },
                            "is_orphan": node_data.get("id", "") in orphan_ids,
                        })

                # Fetch edges between these nodes
                edges = []
                edge_records = []
                if len(node_ids) > 1:
                    if relationship_type:
                        # Validate relationship type
                        valid_types = {"MENTIONS", "AUTHORED_BY", "PART_OF", "USES", "RELATED_TO"}
                        if relationship_type in valid_types:
                            edge_query = f"""
                            MATCH (source:Entity)-[r:{relationship_type}]->(target:Entity)
                            WHERE source.id IN $node_ids AND target.id IN $node_ids
                            AND source.tenant_id = $tenant_id
                            RETURN source.id as source_id, target.id as target_id, 
                                   type(r) as rel_type, r.confidence as confidence,
                                   r.description as description
                            """
                            edge_result = await session.run(
                                edge_query,
                                node_ids=node_ids,
                                tenant_id=tenant_id,
                            )
                            edge_records = await edge_result.data()
                    else:
                        edge_query = """
                        MATCH (source:Entity)-[r]->(target:Entity)
                        WHERE source.id IN $node_ids AND target.id IN $node_ids
                        AND source.tenant_id = $tenant_id
                        AND type(r) IN ['MENTIONS', 'AUTHORED_BY', 'PART_OF', 'USES', 'RELATED_TO']
                        RETURN source.id as source_id, target.id as target_id, 
                               type(r) as rel_type, r.confidence as confidence,
                               r.description as description
                        """
                        edge_result = await session.run(
                            edge_query,
                            node_ids=node_ids,
                            tenant_id=tenant_id,
                        )
                        edge_records = await edge_result.data()

                    for i, record in enumerate(edge_records):
                        edges.append({
                            "id": f"edge-{record['source_id'][:8]}-{record['target_id'][:8]}-{i}",
                            "source": record["source_id"],
                            "target": record["target_id"],
                            "type": record["rel_type"],
                            "label": record["rel_type"],
                            "properties": {
                                "confidence": record.get("confidence"),
                                "description": record.get("description"),
                            },
                        })

                return {"nodes": nodes, "edges": edges}

        except Neo4jDriverError as e:
            raise Neo4jError("get_graph_data", str(e)) from e

    async def get_visualization_stats(self, tenant_id: str) -> dict[str, Any]:
        """
        Get detailed statistics for knowledge graph visualization.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary with node count, edge count, orphan count, and type breakdowns
        """
        try:
            async with self.driver.session() as session:
                # Count total entities
                entity_result = await session.run(
                    "MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e) as count",
                    tenant_id=tenant_id,
                )
                entity_record = await entity_result.single()
                node_count = entity_record["count"] if entity_record else 0

                # Count entity-to-entity relationships
                rel_result = await session.run(
                    """
                    MATCH (e1:Entity {tenant_id: $tenant_id})-[r]->(e2:Entity {tenant_id: $tenant_id})
                    WHERE type(r) IN ['MENTIONS', 'AUTHORED_BY', 'PART_OF', 'USES', 'RELATED_TO']
                    RETURN count(r) as count
                    """,
                    tenant_id=tenant_id,
                )
                rel_record = await rel_result.single()
                edge_count = rel_record["count"] if rel_record else 0

                # Count orphan nodes
                orphan_result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})
                    WHERE NOT (e)-[]-()
                    RETURN count(e) as count
                    """,
                    tenant_id=tenant_id,
                )
                orphan_record = await orphan_result.single()
                orphan_count = orphan_record["count"] if orphan_record else 0

                # Count by entity type
                type_result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})
                    RETURN e.type as type, count(e) as count
                    """,
                    tenant_id=tenant_id,
                )
                type_records = await type_result.data()
                entity_type_counts = {r["type"]: r["count"] for r in type_records if r.get("type")}

                # Count by relationship type
                rel_type_result = await session.run(
                    """
                    MATCH (e1:Entity {tenant_id: $tenant_id})-[r]->(e2:Entity {tenant_id: $tenant_id})
                    WHERE type(r) IN ['MENTIONS', 'AUTHORED_BY', 'PART_OF', 'USES', 'RELATED_TO']
                    RETURN type(r) as type, count(r) as count
                    """,
                    tenant_id=tenant_id,
                )
                rel_type_records = await rel_type_result.data()
                relationship_type_counts = {r["type"]: r["count"] for r in rel_type_records if r.get("type")}

                return {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "orphan_count": orphan_count,
                    "entity_type_counts": entity_type_counts,
                    "relationship_type_counts": relationship_type_counts,
                }

        except Neo4jDriverError as e:
            raise Neo4jError("get_visualization_stats", str(e)) from e

    async def get_orphan_nodes(
        self,
        tenant_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get nodes with no relationships (orphans).

        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of orphan nodes to return

        Returns:
            List of orphan node dictionaries
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})
                    WHERE NOT (e)-[]-()
                    RETURN e
                    ORDER BY e.name
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    limit=limit,
                )
                records = await result.data()
                
                orphans = []
                for record in records:
                    if record.get("e"):
                        node_data = dict(record["e"])
                        orphans.append({
                            "id": node_data.get("id", ""),
                            "label": node_data.get("name", ""),
                            "type": node_data.get("type", ""),
                            "properties": {
                                "description": node_data.get("description"),
                                "source_chunks": node_data.get("source_chunks", []),
                            },
                            "is_orphan": True,
                        })
                
                return orphans

        except Neo4jDriverError as e:
            raise Neo4jError("get_orphan_nodes", str(e)) from e

# Global Neo4j client instance
_neo4j_client: Optional[Neo4jClient] = None


async def get_neo4j_client(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    pool_min_size: Optional[int] = None,
    pool_max_size: Optional[int] = None,
    pool_acquire_timeout: Optional[float] = None,
    connection_timeout: Optional[float] = None,
    max_connection_lifetime: Optional[int] = None,
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
        _neo4j_client = Neo4jClient(
            uri,
            user,
            password,
            pool_min_size=pool_min_size if pool_min_size is not None else 1,
            pool_max_size=pool_max_size if pool_max_size is not None else 50,
            pool_acquire_timeout=(
                pool_acquire_timeout if pool_acquire_timeout is not None else 30.0
            ),
            connection_timeout=(
                connection_timeout if connection_timeout is not None else 30.0
            ),
            max_connection_lifetime=(
                max_connection_lifetime if max_connection_lifetime is not None else 3600
            ),
        )
        await _neo4j_client.connect()
        await _neo4j_client.create_indexes()
    return _neo4j_client


async def close_neo4j_client() -> None:
    """Close the global Neo4j client connection."""
    global _neo4j_client
    if _neo4j_client is not None:
        await _neo4j_client.disconnect()
        _neo4j_client = None
