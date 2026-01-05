"""Community Detection for Graph Intelligence (Story 20-B1).

This module implements community detection algorithms for the knowledge graph,
competing with Microsoft GraphRAG's approach to handling "global" queries.

The CommunityDetector class:
- Exports the Neo4j graph to NetworkX format
- Runs Louvain or Leiden community detection algorithms
- Generates hierarchical community structures
- Creates LLM summaries for each community
- Stores communities back to Neo4j with relationships

Key Features:
- Multi-tenancy: All operations filter by tenant_id
- Hierarchical: Supports multiple levels of community abstraction
- Feature-flagged: Controlled by COMMUNITY_DETECTION_ENABLED
- Configurable: Algorithm, min size, max levels via env vars
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import structlog

from .errors import CommunityDetectionError, CommunityNotFoundError, GraphTooSmallError
from .models import Community, CommunityAlgorithm

logger = structlog.get_logger(__name__)

# Check for optional dependencies
try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    louvain_communities = None

try:
    import leidenalg
    import igraph as ig

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    leidenalg = None
    ig = None


class CommunityDetector:
    """Detect and manage entity communities in the knowledge graph.

    This class implements community detection using Louvain or Leiden algorithms,
    which identify clusters of related entities that are more densely connected
    to each other than to the rest of the graph.

    Attributes:
        neo4j_client: Neo4j client for graph operations
        llm_client: Optional LLM client for summary generation
        algorithm: Community detection algorithm to use
        min_community_size: Minimum entities per community
        max_hierarchy_levels: Maximum levels in community hierarchy
        summary_model: LLM model for generating summaries
    """

    def __init__(
        self,
        neo4j_client: Any,
        llm_client: Any = None,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LOUVAIN,
        min_community_size: int = 3,
        max_hierarchy_levels: int = 3,
        summary_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize CommunityDetector.

        Args:
            neo4j_client: Neo4j client for graph operations
            llm_client: Optional LLM client for summary generation
            algorithm: Algorithm to use (louvain or leiden)
            min_community_size: Minimum entities per community
            max_hierarchy_levels: Maximum hierarchy depth
            summary_model: LLM model for summaries

        Raises:
            ImportError: If networkx is not installed
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for community detection. "
                "Install with: pip install networkx>=3.0"
            )

        self._neo4j = neo4j_client
        self._llm_client = llm_client
        self.algorithm = algorithm
        self.min_community_size = min_community_size
        self.max_hierarchy_levels = max_hierarchy_levels
        self.summary_model = summary_model

    async def detect_communities(
        self,
        tenant_id: str,
        generate_summaries: bool = True,
        algorithm: Optional[CommunityAlgorithm] = None,
        min_size: Optional[int] = None,
        max_levels: Optional[int] = None,
    ) -> list[Community]:
        """Detect communities in the knowledge graph.

        This is the main entry point for community detection. It:
        1. Exports the graph to NetworkX format
        2. Runs the community detection algorithm
        3. Builds the hierarchical community structure
        4. Generates LLM summaries (if enabled)
        5. Stores communities to Neo4j

        Args:
            tenant_id: Tenant identifier for filtering
            generate_summaries: Whether to generate LLM summaries
            algorithm: Override default algorithm
            min_size: Override minimum community size
            max_levels: Override maximum hierarchy levels

        Returns:
            List of detected Community objects

        Raises:
            GraphTooSmallError: If graph has fewer nodes than min_size
            CommunityDetectionError: If detection fails
        """
        start_time = time.perf_counter()
        algo = algorithm or self.algorithm
        min_community_size = min_size or self.min_community_size
        max_hierarchy_levels = max_levels or self.max_hierarchy_levels

        logger.info(
            "community_detection_started",
            tenant_id=tenant_id,
            algorithm=algo.value,
            min_size=min_community_size,
            max_levels=max_hierarchy_levels,
        )

        try:
            # Step 1: Export graph to NetworkX
            G = await self._build_networkx_graph(tenant_id)

            if len(G.nodes) < min_community_size:
                raise GraphTooSmallError(
                    node_count=len(G.nodes),
                    min_required=min_community_size,
                    tenant_id=tenant_id,
                )

            # Step 2: Run community detection algorithm
            if algo == CommunityAlgorithm.LEIDEN:
                partition = self._run_leiden(G)
            else:
                partition = self._run_louvain(G)

            # Step 3: Build community objects from partition
            communities = self._build_communities(
                partition=partition,
                graph=G,
                tenant_id=tenant_id,
                min_size=min_community_size,
            )

            if not communities:
                logger.info(
                    "no_communities_detected",
                    tenant_id=tenant_id,
                    reason="All communities below minimum size",
                )
                return []

            # Step 4: Build hierarchy (multiple levels)
            hierarchical_communities = self._build_hierarchy(
                communities=communities,
                graph=G,
                tenant_id=tenant_id,
                max_levels=max_hierarchy_levels,
            )

            # Step 5: Generate summaries using LLM
            if generate_summaries and self._llm_client:
                await self._generate_summaries(hierarchical_communities, tenant_id)

            # Step 6: Store communities to Neo4j
            await self._store_communities(hierarchical_communities, tenant_id)

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info(
                "community_detection_completed",
                tenant_id=tenant_id,
                communities_created=len(hierarchical_communities),
                processing_time_ms=processing_time_ms,
                algorithm=algo.value,
            )

            return hierarchical_communities

        except GraphTooSmallError:
            raise
        except Exception as e:
            logger.error(
                "community_detection_failed",
                tenant_id=tenant_id,
                error=str(e),
                algorithm=algo.value,
            )
            raise CommunityDetectionError(str(e), tenant_id, algo.value) from e

    async def _build_networkx_graph(self, tenant_id: str) -> "nx.Graph":
        """Export Neo4j graph to NetworkX format.

        Fetches all entities and relationships for the tenant and builds
        an undirected NetworkX graph for community detection.

        Args:
            tenant_id: Tenant identifier

        Returns:
            NetworkX Graph with entities as nodes and relationships as edges
        """
        async with self._neo4j.driver.session() as session:
            # Fetch all entities for the tenant
            node_result = await session.run(
                """
                MATCH (e:Entity {tenant_id: $tenant_id})
                RETURN e.id as id, e.name as name, e.type as type,
                       e.description as description
                """,
                tenant_id=tenant_id,
            )
            node_records = await node_result.data()

            # Fetch all relationships between entities
            edge_result = await session.run(
                """
                MATCH (source:Entity {tenant_id: $tenant_id})-[r]-(target:Entity {tenant_id: $tenant_id})
                WHERE source.id < target.id
                RETURN source.id as source_id, target.id as target_id,
                       type(r) as rel_type, r.confidence as confidence
                """,
                tenant_id=tenant_id,
            )
            edge_records = await edge_result.data()

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes with attributes
        for node in node_records:
            G.add_node(
                node["id"],
                name=node.get("name", ""),
                type=node.get("type", ""),
                description=node.get("description", ""),
            )

        # Add edges with weights
        for edge in edge_records:
            weight = edge.get("confidence", 1.0) or 1.0
            G.add_edge(
                edge["source_id"],
                edge["target_id"],
                weight=weight,
                rel_type=edge.get("rel_type", "RELATED_TO"),
            )

        logger.debug(
            "networkx_graph_built",
            tenant_id=tenant_id,
            nodes=len(G.nodes),
            edges=len(G.edges),
        )

        return G

    def _run_louvain(self, G: "nx.Graph") -> dict[str, int]:
        """Run Louvain community detection algorithm.

        Uses NetworkX's built-in Louvain implementation for community detection.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary mapping node IDs to community IDs
        """
        communities = louvain_communities(G, weight="weight", resolution=1.0)

        # Convert to partition dict: {node_id: community_id}
        partition = {}
        for community_idx, community_nodes in enumerate(communities):
            for node_id in community_nodes:
                partition[node_id] = community_idx

        logger.debug(
            "louvain_completed",
            communities_found=len(communities),
            nodes_assigned=len(partition),
        )

        return partition

    def _run_leiden(self, G: "nx.Graph") -> dict[str, int]:
        """Run Leiden community detection algorithm.

        Leiden provides higher quality communities but requires the optional
        leidenalg package. Falls back to Louvain if not available.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary mapping node IDs to community IDs
        """
        if not LEIDEN_AVAILABLE:
            logger.warning(
                "leiden_not_available_falling_back_to_louvain",
                hint="Install leidenalg and igraph for Leiden algorithm",
            )
            return self._run_louvain(G)

        try:
            # Convert NetworkX to igraph
            ig_graph = ig.Graph.from_networkx(G)

            # Run Leiden algorithm
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
            )

            # Map back to node IDs
            node_list = list(G.nodes())
            result = {}
            for node_idx, membership in enumerate(partition.membership):
                result[node_list[node_idx]] = membership

            logger.debug(
                "leiden_completed",
                communities_found=len(set(partition.membership)),
                nodes_assigned=len(result),
            )

            return result

        except Exception as e:
            logger.warning(
                "leiden_failed_falling_back_to_louvain",
                error=str(e),
            )
            return self._run_louvain(G)

    def _build_communities(
        self,
        partition: dict[str, int],
        graph: "nx.Graph",
        tenant_id: str,
        min_size: int,
    ) -> list[Community]:
        """Convert partition dictionary to Community objects.

        Filters out communities smaller than min_size and creates
        Community objects with entity information.

        Args:
            partition: Node to community mapping
            graph: NetworkX graph with node attributes
            tenant_id: Tenant identifier
            min_size: Minimum community size

        Returns:
            List of Community objects
        """
        # Group nodes by community
        community_nodes: dict[int, list[str]] = {}
        for node_id, community_id in partition.items():
            if community_id not in community_nodes:
                community_nodes[community_id] = []
            community_nodes[community_id].append(node_id)

        communities = []
        for community_id, node_ids in community_nodes.items():
            # Filter by minimum size
            if len(node_ids) < min_size:
                continue

            # Get entity names for naming
            entity_names = [
                graph.nodes[nid].get("name", nid)
                for nid in node_ids[:5]  # Use first 5 for naming
            ]

            # Generate a simple name from entity names
            if len(entity_names) == 1:
                name = f"{entity_names[0]} Community"
            else:
                name = f"{entity_names[0]}, {entity_names[1]} and {len(node_ids) - 2} others"

            community = Community(
                id=str(uuid4()),
                name=name,
                level=0,
                tenant_id=tenant_id,
                entity_ids=node_ids,
                entity_count=len(node_ids),
                keywords=[],
                created_at=datetime.now(timezone.utc),
            )
            communities.append(community)

        logger.debug(
            "communities_built",
            tenant_id=tenant_id,
            total=len(communities),
            filtered_out=len(community_nodes) - len(communities),
        )

        return communities

    def _build_hierarchy(
        self,
        communities: list[Community],
        graph: "nx.Graph",
        tenant_id: str,
        max_levels: int,
    ) -> list[Community]:
        """Build hierarchical community structure.

        Creates multiple levels of abstraction by recursively grouping
        communities. Level 0 is the most granular (original partition).

        Args:
            communities: Base-level communities
            graph: NetworkX graph
            tenant_id: Tenant identifier
            max_levels: Maximum hierarchy depth

        Returns:
            All communities across all levels
        """
        all_communities = list(communities)

        if max_levels <= 1 or len(communities) < 2:
            return all_communities

        current_level_communities = communities
        current_level = 0

        while current_level < max_levels - 1 and len(current_level_communities) > 1:
            # Build a meta-graph where each community is a node
            meta_graph = nx.Graph()
            community_map = {c.id: c for c in current_level_communities}

            for community in current_level_communities:
                meta_graph.add_node(
                    community.id,
                    entity_count=community.entity_count,
                )

            # Add edges between communities based on cross-community relationships
            for c1 in current_level_communities:
                for c2 in current_level_communities:
                    if c1.id >= c2.id:
                        continue
                    # Count edges between communities
                    edge_count = sum(
                        1 for e1 in c1.entity_ids for e2 in c2.entity_ids
                        if graph.has_edge(e1, e2)
                    )
                    if edge_count > 0:
                        meta_graph.add_edge(c1.id, c2.id, weight=edge_count)

            # Skip if no edges between communities
            if len(meta_graph.edges) == 0:
                break

            # Run Louvain on meta-graph
            meta_partition = self._run_louvain(meta_graph)

            # Group communities into higher-level communities
            higher_level: dict[int, list[Community]] = {}
            for comm_id, meta_comm_id in meta_partition.items():
                if meta_comm_id not in higher_level:
                    higher_level[meta_comm_id] = []
                higher_level[meta_comm_id].append(community_map[comm_id])

            # Create higher-level community objects
            new_level = current_level + 1
            new_communities = []

            for meta_id, child_communities in higher_level.items():
                if len(child_communities) < 2:
                    continue  # Skip single-community groups

                # Aggregate entity IDs
                all_entity_ids = []
                for child in child_communities:
                    all_entity_ids.extend(child.entity_ids)

                # Generate name from child communities
                child_names = [c.name.split(",")[0] for c in child_communities[:3]]
                name = f"{', '.join(child_names)} Group"

                parent = Community(
                    id=str(uuid4()),
                    name=name,
                    level=new_level,
                    tenant_id=tenant_id,
                    entity_ids=all_entity_ids,
                    entity_count=len(all_entity_ids),
                    child_ids=[c.id for c in child_communities],
                    keywords=[],
                    created_at=datetime.now(timezone.utc),
                )

                # Update children with parent reference
                for child in child_communities:
                    child.parent_id = parent.id

                new_communities.append(parent)

            if not new_communities:
                break

            all_communities.extend(new_communities)
            current_level_communities = new_communities
            current_level = new_level

        logger.debug(
            "hierarchy_built",
            tenant_id=tenant_id,
            levels=current_level + 1,
            total_communities=len(all_communities),
        )

        return all_communities

    async def _generate_summaries(
        self,
        communities: list[Community],
        tenant_id: str,
    ) -> None:
        """Generate LLM summaries for communities.

        Uses the configured LLM to generate a summary and keywords
        for each community based on its entity names and descriptions.

        Args:
            communities: List of communities to summarize
            tenant_id: Tenant identifier
        """
        if not self._llm_client:
            logger.warning(
                "llm_client_not_available",
                tenant_id=tenant_id,
                hint="Community summaries will not be generated",
            )
            return

        for community in communities:
            try:
                # Get entity details for context
                entity_context = await self._get_entity_context(
                    community.entity_ids[:20],  # Limit for context length
                    tenant_id,
                )

                # Generate summary using LLM
                prompt = self._build_summary_prompt(community, entity_context)
                response = await self._llm_client.generate(
                    prompt=prompt,
                    model=self.summary_model,
                )

                # Parse response
                summary, keywords = self._parse_summary_response(response)
                community.summary = summary
                community.keywords = keywords

            except Exception as e:
                logger.warning(
                    "community_summary_generation_failed",
                    community_id=community.id,
                    tenant_id=tenant_id,
                    error=str(e),
                )
                # Set a default summary
                community.summary = f"A community of {community.entity_count} related entities"
                community.keywords = []

    async def _get_entity_context(
        self,
        entity_ids: list[str],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Get entity details for summary generation.

        Args:
            entity_ids: List of entity IDs
            tenant_id: Tenant identifier

        Returns:
            List of entity dictionaries with name, type, description
        """
        async with self._neo4j.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {tenant_id: $tenant_id})
                WHERE e.id IN $entity_ids
                RETURN e.id as id, e.name as name, e.type as type,
                       e.description as description
                """,
                tenant_id=tenant_id,
                entity_ids=entity_ids,
            )
            records = await result.data()
            return records

    def _build_summary_prompt(
        self,
        community: Community,
        entities: list[dict[str, Any]],
    ) -> str:
        """Build prompt for LLM summary generation.

        Args:
            community: Community to summarize
            entities: Entity details

        Returns:
            Prompt string for LLM
        """
        entity_descriptions = []
        for e in entities:
            desc = f"- {e.get('name', 'Unknown')} ({e.get('type', 'Entity')})"
            if e.get("description"):
                desc += f": {e['description'][:100]}"
            entity_descriptions.append(desc)

        entity_text = "\n".join(entity_descriptions)

        return f"""Analyze this community of related entities and provide:
1. A concise summary (2-3 sentences) describing the theme and relationships
2. 3-5 keywords that capture the main topics

Entities in this community:
{entity_text}

Respond in this exact format:
SUMMARY: <your summary here>
KEYWORDS: keyword1, keyword2, keyword3
"""

    def _parse_summary_response(
        self,
        response: str,
    ) -> tuple[str, list[str]]:
        """Parse LLM response into summary and keywords.

        Args:
            response: LLM response text

        Returns:
            Tuple of (summary, keywords list)
        """
        summary = ""
        keywords = []

        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("KEYWORDS:"):
                keyword_text = line.replace("KEYWORDS:", "").strip()
                keywords = [k.strip() for k in keyword_text.split(",") if k.strip()]

        return summary or "A community of related entities", keywords

    async def _store_communities(
        self,
        communities: list[Community],
        tenant_id: str,
    ) -> None:
        """Store communities to Neo4j.

        Creates Community nodes and BELONGS_TO relationships from entities,
        as well as PARENT_OF/CHILD_OF relationships for hierarchy.

        Args:
            communities: Communities to store
            tenant_id: Tenant identifier
        """
        async with self._neo4j.driver.session() as session:
            for community in communities:
                # Create community node
                await session.run(
                    """
                    MERGE (c:Community {id: $id, tenant_id: $tenant_id})
                    SET c.name = $name,
                        c.level = $level,
                        c.summary = $summary,
                        c.keywords = $keywords,
                        c.entity_count = $entity_count,
                        c.parent_id = $parent_id,
                        c.child_ids = $child_ids,
                        c.created_at = datetime($created_at),
                        c.updated_at = datetime()
                    """,
                    id=community.id,
                    tenant_id=tenant_id,
                    name=community.name,
                    level=community.level,
                    summary=community.summary or "",
                    keywords=community.keywords,
                    entity_count=community.entity_count,
                    parent_id=community.parent_id,
                    child_ids=community.child_ids,
                    created_at=community.created_at.isoformat() if community.created_at else datetime.now(timezone.utc).isoformat(),
                )

                # Create BELONGS_TO relationships from entities
                for entity_id in community.entity_ids:
                    await session.run(
                        """
                        MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
                        MATCH (c:Community {id: $community_id, tenant_id: $tenant_id})
                        MERGE (e)-[:BELONGS_TO]->(c)
                        """,
                        entity_id=entity_id,
                        community_id=community.id,
                        tenant_id=tenant_id,
                    )

                # Create hierarchy relationships
                if community.parent_id:
                    await session.run(
                        """
                        MATCH (child:Community {id: $child_id, tenant_id: $tenant_id})
                        MATCH (parent:Community {id: $parent_id, tenant_id: $tenant_id})
                        MERGE (parent)-[:PARENT_OF]->(child)
                        MERGE (child)-[:CHILD_OF]->(parent)
                        """,
                        child_id=community.id,
                        parent_id=community.parent_id,
                        tenant_id=tenant_id,
                    )

        logger.info(
            "communities_stored",
            tenant_id=tenant_id,
            count=len(communities),
        )

    async def get_community(
        self,
        community_id: str,
        tenant_id: str,
        include_entities: bool = False,
    ) -> Community:
        """Get a community by ID.

        Args:
            community_id: Community UUID
            tenant_id: Tenant identifier
            include_entities: Whether to include entity details

        Returns:
            Community object

        Raises:
            CommunityNotFoundError: If community not found
        """
        async with self._neo4j.driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Community {id: $id, tenant_id: $tenant_id})
                OPTIONAL MATCH (e:Entity {tenant_id: $tenant_id})-[:BELONGS_TO]->(c)
                RETURN c, collect(e.id) as entity_ids
                """,
                id=community_id,
                tenant_id=tenant_id,
            )
            record = await result.single()

            if not record:
                raise CommunityNotFoundError(community_id, tenant_id)

            community_data = dict(record["c"])
            entity_ids = record["entity_ids"]

            community = Community(
                id=community_data.get("id", community_id),
                name=community_data.get("name", ""),
                level=community_data.get("level", 0),
                tenant_id=tenant_id,
                entity_ids=entity_ids or [],
                entity_count=community_data.get("entity_count", 0),
                summary=community_data.get("summary"),
                keywords=community_data.get("keywords", []),
                parent_id=community_data.get("parent_id"),
                child_ids=community_data.get("child_ids", []),
                created_at=community_data.get("created_at"),
                updated_at=community_data.get("updated_at"),
            )

            return community

    async def list_communities(
        self,
        tenant_id: str,
        level: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Community], int]:
        """List communities for a tenant.

        Args:
            tenant_id: Tenant identifier
            level: Filter by hierarchy level (optional)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (communities list, total count)
        """
        async with self._neo4j.driver.session() as session:
            # Build query based on level filter
            if level is not None:
                count_query = """
                    MATCH (c:Community {tenant_id: $tenant_id, level: $level})
                    RETURN count(c) as total
                """
                list_query = """
                    MATCH (c:Community {tenant_id: $tenant_id, level: $level})
                    RETURN c
                    ORDER BY c.entity_count DESC
                    SKIP $offset
                    LIMIT $limit
                """
                params = {"tenant_id": tenant_id, "level": level, "offset": offset, "limit": limit}
            else:
                count_query = """
                    MATCH (c:Community {tenant_id: $tenant_id})
                    RETURN count(c) as total
                """
                list_query = """
                    MATCH (c:Community {tenant_id: $tenant_id})
                    RETURN c
                    ORDER BY c.level, c.entity_count DESC
                    SKIP $offset
                    LIMIT $limit
                """
                params = {"tenant_id": tenant_id, "offset": offset, "limit": limit}

            # Get total count
            count_result = await session.run(count_query, **params)
            count_record = await count_result.single()
            total = count_record["total"] if count_record else 0

            # Get communities
            list_result = await session.run(list_query, **params)
            records = await list_result.data()

            communities = []
            for record in records:
                c = record["c"]
                communities.append(
                    Community(
                        id=c.get("id", ""),
                        name=c.get("name", ""),
                        level=c.get("level", 0),
                        tenant_id=tenant_id,
                        entity_ids=[],  # Don't load all IDs for listing
                        entity_count=c.get("entity_count", 0),
                        summary=c.get("summary"),
                        keywords=c.get("keywords", []),
                        parent_id=c.get("parent_id"),
                        child_ids=c.get("child_ids", []),
                        created_at=c.get("created_at"),
                        updated_at=c.get("updated_at"),
                    )
                )

            return communities, total

    async def delete_community(
        self,
        community_id: str,
        tenant_id: str,
    ) -> bool:
        """Delete a community and its relationships.

        Args:
            community_id: Community UUID
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._neo4j.driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Community {id: $id, tenant_id: $tenant_id})
                DETACH DELETE c
                RETURN count(c) as deleted
                """,
                id=community_id,
                tenant_id=tenant_id,
            )
            record = await result.single()
            deleted = record["deleted"] > 0 if record else False

            if deleted:
                logger.info(
                    "community_deleted",
                    community_id=community_id,
                    tenant_id=tenant_id,
                )

            return deleted

    async def delete_all_communities(
        self,
        tenant_id: str,
    ) -> int:
        """Delete all communities for a tenant.

        Useful before re-running community detection.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of communities deleted
        """
        async with self._neo4j.driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Community {tenant_id: $tenant_id})
                WITH c, count(c) as total
                DETACH DELETE c
                RETURN total
                """,
                tenant_id=tenant_id,
            )
            record = await result.single()
            count = record["total"] if record else 0

            logger.info(
                "all_communities_deleted",
                tenant_id=tenant_id,
                count=count,
            )

            return count

    async def search_communities(
        self,
        query: str,
        tenant_id: str,
        level: Optional[int] = None,
        limit: int = 10,
    ) -> list[Community]:
        """Search communities by keyword or summary content.

        Performs a text search on community names, summaries, and keywords.

        Args:
            query: Search query string
            tenant_id: Tenant identifier
            level: Filter by hierarchy level (optional)
            limit: Maximum results

        Returns:
            List of matching communities
        """
        async with self._neo4j.driver.session() as session:
            # Build query with text matching
            if level is not None:
                cypher = """
                    MATCH (c:Community {tenant_id: $tenant_id, level: $level})
                    WHERE toLower(c.name) CONTAINS toLower($query)
                       OR toLower(c.summary) CONTAINS toLower($query)
                       OR any(kw IN c.keywords WHERE toLower(kw) CONTAINS toLower($query))
                    RETURN c
                    ORDER BY c.entity_count DESC
                    LIMIT $limit
                """
                params = {"tenant_id": tenant_id, "level": level, "query": query, "limit": limit}
            else:
                cypher = """
                    MATCH (c:Community {tenant_id: $tenant_id})
                    WHERE toLower(c.name) CONTAINS toLower($query)
                       OR toLower(c.summary) CONTAINS toLower($query)
                       OR any(kw IN c.keywords WHERE toLower(kw) CONTAINS toLower($query))
                    RETURN c
                    ORDER BY c.level, c.entity_count DESC
                    LIMIT $limit
                """
                params = {"tenant_id": tenant_id, "query": query, "limit": limit}

            result = await session.run(cypher, **params)
            records = await result.data()

            communities = []
            for record in records:
                c = record["c"]
                communities.append(
                    Community(
                        id=c.get("id", ""),
                        name=c.get("name", ""),
                        level=c.get("level", 0),
                        tenant_id=tenant_id,
                        entity_ids=[],
                        entity_count=c.get("entity_count", 0),
                        summary=c.get("summary"),
                        keywords=c.get("keywords", []),
                        parent_id=c.get("parent_id"),
                        child_ids=c.get("child_ids", []),
                        created_at=c.get("created_at"),
                        updated_at=c.get("updated_at"),
                    )
                )

            return communities
