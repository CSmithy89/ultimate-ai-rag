"""LazyRAG Query-Time Summarization (Story 20-B2).

This module implements the LazyRAG pattern for query-time graph summarization,
achieving up to 99% reduction in indexing costs compared to MS GraphRAG's
eager summarization approach.

LazyRAG Algorithm:
1. Find Seed Entities - Use Graphiti hybrid search to find relevant entities
2. Expand Subgraph - N-hop traversal from seed entities via Neo4j
3. Get Community Context - Optional community summaries from Story 20-B1
4. Generate Summary - LLM-based summary generation at query time
5. Estimate Confidence - Based on entity coverage and relationship density

Key Features:
- Multi-tenancy via group_ids/tenant_id filtering
- Feature flag: LAZY_RAG_ENABLED
- Integration with CommunityDetector from Story 20-B1
- Performance target: <3 seconds for typical queries (<50 entities)

Configuration:
- LAZY_RAG_ENABLED: Enable/disable feature (default: false)
- LAZY_RAG_MAX_ENTITIES: Maximum entities in context (default: 50)
- LAZY_RAG_MAX_HOPS: Relationship expansion depth (default: 2)
- LAZY_RAG_USE_COMMUNITIES: Include community context (default: true)
- LAZY_RAG_SUMMARY_MODEL: LLM model for summaries (default: gpt-4o-mini)
"""

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog
from openai import AsyncOpenAI

from ..config import Settings
from ..db.graphiti import GraphitiClient, GRAPHITI_AVAILABLE
from ..llm.providers import get_llm_adapter, OPENAI_COMPATIBLE_LLM_PROVIDERS
from ..observability.metrics import record_llm_call
from ..rate_limit import RateLimiter
from .lazy_rag_models import (
    LazyRAGCommunity,
    LazyRAGEntity,
    LazyRAGRelationship,
    LazyRAGResult,
    SubgraphExpansionResult,
    SummaryResult,
)

logger = structlog.get_logger(__name__)

# Safe hop patterns for Cypher queries (prevents injection via max_hops)
# Keys are validated max_hops values (1-5), values are Cypher path patterns
_SAFE_HOP_PATTERNS: dict[int, str] = {
    1: "[*1..1]",
    2: "[*1..2]",
    3: "[*1..3]",
    4: "[*1..4]",
    5: "[*1..5]",
}

# Default timeout for LLM API calls in seconds
LLM_CALL_TIMEOUT_SECONDS = 30.0

# Context formatting limits (for LLM prompt context window management)
MAX_ENTITIES_IN_CONTEXT = 50
MAX_RELATIONSHIPS_IN_CONTEXT = 50
MAX_ENTITY_DESCRIPTION_LENGTH = 200
MAX_RELATIONSHIP_FACT_LENGTH = 100
LOG_QUERY_TRUNCATE_LENGTH = 50


# Summary prompt template
LAZY_RAG_SUMMARY_PROMPT = """Based on the following knowledge graph subset, answer the query.

Query: {query}

Entities ({entity_count} total):
{entity_context}

Relationships ({relationship_count} total):
{relationship_context}

{community_context_section}

Instructions:
1. Provide a comprehensive answer based ONLY on the information above.
2. Reference specific entities and relationships when relevant.
3. If the information is insufficient to fully answer the query, indicate what's missing.
4. Be concise but thorough.

Answer:"""


@dataclass
class CircuitBreakerState:
    """State for per-tenant circuit breaker."""
    failure_count: int = 0
    open_until: Optional[datetime] = None


class LazyRAGRetriever:
    """LazyRAG query-time summarization retriever.

    This class implements the LazyRAG pattern that defers graph summarization
    to query time, eliminating expensive pre-computed summaries.

    Attributes:
        graphiti_client: Graphiti client for seed entity search
        neo4j_client: Neo4j client for subgraph expansion
        settings: Application settings with LazyRAG configuration
        community_detector: Optional CommunityDetector from Story 20-B1
    """

    def __init__(
        self,
        graphiti_client: Optional[GraphitiClient],
        neo4j_client: Any,
        settings: Settings,
        community_detector: Optional[Any] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """Initialize LazyRAGRetriever.

        Args:
            graphiti_client: Graphiti client for hybrid search (seed entities)
            neo4j_client: Neo4j client for graph traversal
            settings: Application settings
            community_detector: Optional CommunityDetector instance from 20-B1
            rate_limiter: Optional rate limiter for LLM calls
        """
        self._graphiti = graphiti_client
        self._neo4j = neo4j_client
        self._settings = settings
        self._community_detector = community_detector
        self._rate_limiter = rate_limiter

        # Extract settings with defaults
        self.max_entities = getattr(settings, "lazy_rag_max_entities", 50)
        self.max_hops = getattr(settings, "lazy_rag_max_hops", 2)
        self.use_communities = getattr(settings, "lazy_rag_use_communities", True)
        self.summary_model = getattr(settings, "lazy_rag_summary_model", "gpt-4o-mini")

        # Circuit breaker state for Graphiti failures (per tenant)
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        self._graphiti_circuit_threshold: int = 3  # failures before opening circuit
        self._graphiti_circuit_timeout: int = 30  # seconds to keep circuit open

    def _get_circuit_breaker(self, tenant_id: str) -> CircuitBreakerState:
        """Get or create circuit breaker state for a tenant."""
        if tenant_id not in self._circuit_breakers:
            self._circuit_breakers[tenant_id] = CircuitBreakerState()
        return self._circuit_breakers[tenant_id]

    def _maybe_open_circuit_breaker(self, tenant_id: str) -> None:
        """Open circuit breaker if failure threshold reached."""
        cb = self._get_circuit_breaker(tenant_id)
        if cb.failure_count >= self._graphiti_circuit_threshold:
            cb.open_until = datetime.now() + timedelta(
                seconds=self._graphiti_circuit_timeout
            )
            logger.warning(
                "graphiti_circuit_breaker_opened",
                tenant_id=tenant_id,
                failure_count=cb.failure_count,
                open_until=cb.open_until.isoformat(),
            )

    async def query(
        self,
        query: str,
        tenant_id: str,
        max_entities: Optional[int] = None,
        max_hops: Optional[int] = None,
        use_communities: Optional[bool] = None,
        include_summary: bool = True,
    ) -> LazyRAGResult:
        """Execute a LazyRAG query with query-time summarization.

        This is the main entry point for LazyRAG queries. It:
        1. Finds seed entities via Graphiti hybrid search
        2. Expands the subgraph via N-hop Neo4j traversal
        3. Optionally includes community context from 20-B1
        4. Generates an LLM summary at query time

        Args:
            query: Natural language query
            tenant_id: Tenant identifier for multi-tenancy
            max_entities: Override max entities (default: from settings)
            max_hops: Override max hops (default: from settings)
            use_communities: Override community usage (default: from settings)
            include_summary: Generate LLM summary (default: True)

        Returns:
            LazyRAGResult with entities, relationships, summary, and confidence

        Raises:
            RuntimeError: If required clients are not available
        """
        start_time = time.perf_counter()

        # Apply overrides
        effective_max_entities = max_entities or self.max_entities
        effective_max_hops = max_hops or self.max_hops
        effective_use_communities = (
            use_communities if use_communities is not None else self.use_communities
        )

        logger.info(
            "lazy_rag_query_started",
            query=query[:100],
            tenant_id=tenant_id,
            max_entities=effective_max_entities,
            max_hops=effective_max_hops,
            use_communities=effective_use_communities,
        )

        try:
            # Step 1: Find seed entities via Graphiti
            seed_entities = await self._find_seed_entities(
                query=query,
                tenant_id=tenant_id,
                num_results=min(10, effective_max_entities),
            )

            # Step 2: Expand subgraph from seed entities
            expansion_result = await self._expand_subgraph(
                seed_entity_ids=[e.id for e in seed_entities],
                tenant_id=tenant_id,
                max_hops=effective_max_hops,
                max_entities=effective_max_entities,
            )

            # Combine seed entities with expanded entities (dedup)
            all_entities = self._merge_entities(seed_entities, expansion_result.entities)
            all_entities = all_entities[:effective_max_entities]

            # Step 3: Get community context (if enabled and available)
            communities: list[LazyRAGCommunity] = []
            if effective_use_communities and self._community_detector:
                communities = await self._get_community_context(
                    entity_ids=[e.id for e in all_entities],
                    tenant_id=tenant_id,
                )

            # Step 4: Generate summary (if requested)
            summary_result: Optional[SummaryResult] = None
            if include_summary and all_entities:
                summary_result = await self._generate_summary(
                    query=query,
                    entities=all_entities,
                    relationships=expansion_result.relationships,
                    communities=communities,
                    tenant_id=tenant_id,
                )
            elif not all_entities:
                # No entities found - create empty result with explanation
                summary_result = SummaryResult(
                    text="",
                    confidence=0.0,
                    missing_info="No relevant entities found in the knowledge graph for this query.",
                )

            # Calculate final confidence
            confidence = self._estimate_confidence(
                query=query,
                entities=all_entities,
                relationships=expansion_result.relationships,
            )

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            result = LazyRAGResult(
                query=query,
                tenant_id=tenant_id,
                entities=all_entities,
                relationships=expansion_result.relationships,
                communities=communities,
                summary=summary_result.text if summary_result else None,
                confidence=summary_result.confidence if summary_result else confidence,
                seed_entity_count=len(seed_entities),
                expanded_entity_count=len(all_entities),
                processing_time_ms=processing_time_ms,
                missing_info=summary_result.missing_info if summary_result else None,
            )

            logger.info(
                "lazy_rag_query_completed",
                query=query[:100],
                tenant_id=tenant_id,
                seed_entities=len(seed_entities),
                total_entities=len(all_entities),
                relationships=len(expansion_result.relationships),
                communities=len(communities),
                confidence=result.confidence,
                processing_time_ms=processing_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "lazy_rag_query_failed",
                query=query[:100],
                tenant_id=tenant_id,
                error=str(e),
            )
            raise

    async def expand_only(
        self,
        query: str,
        tenant_id: str,
        max_entities: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> SubgraphExpansionResult:
        """Expand subgraph without generating summary (debug endpoint).

        Args:
            query: Natural language query for seed entity search
            tenant_id: Tenant identifier
            max_entities: Override max entities
            max_hops: Override max hops

        Returns:
            SubgraphExpansionResult with entities and relationships
        """
        effective_max_entities = max_entities or self.max_entities
        effective_max_hops = max_hops or self.max_hops

        # Find seed entities
        seed_entities = await self._find_seed_entities(
            query=query,
            tenant_id=tenant_id,
            num_results=min(10, effective_max_entities),
        )

        # Expand subgraph
        expansion_result = await self._expand_subgraph(
            seed_entity_ids=[e.id for e in seed_entities],
            tenant_id=tenant_id,
            max_hops=effective_max_hops,
            max_entities=effective_max_entities,
        )

        # Merge and return
        all_entities = self._merge_entities(seed_entities, expansion_result.entities)
        all_entities = all_entities[:effective_max_entities]

        return SubgraphExpansionResult(
            entities=all_entities,
            relationships=expansion_result.relationships,
            seed_count=len(seed_entities),
            expanded_count=len(all_entities),
        )

    async def _find_seed_entities(
        self,
        query: str,
        tenant_id: str,
        num_results: int = 10,
    ) -> list[LazyRAGEntity]:
        """Find seed entities using Graphiti hybrid search.

        Uses Graphiti's integrated semantic + BM25 + graph search to find
        the most relevant entities for the query.

        Implements circuit breaker pattern to avoid repeated failures
        when Graphiti is unhealthy.

        Args:
            query: Natural language query
            tenant_id: Tenant identifier (used as group_id)
            num_results: Maximum number of seed entities

        Returns:
            List of seed LazyRAGEntity objects
        """
        if not self._graphiti or not GRAPHITI_AVAILABLE:
            logger.warning(
                "graphiti_not_available_for_seed_search",
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

        if not self._graphiti.is_connected:
            logger.warning(
                "graphiti_not_connected",
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

        # Check circuit breaker - skip Graphiti if circuit is open
        cb = self._get_circuit_breaker(tenant_id)
        if cb.open_until and datetime.now() < cb.open_until:
            logger.debug(
                "graphiti_circuit_breaker_open",
                tenant_id=tenant_id,
                open_until=cb.open_until.isoformat(),
                hint="Using fallback due to recent failures",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

        try:
            search_result = await self._graphiti.client.search(
                query=query,
                group_ids=[tenant_id],
                num_results=num_results,
            )

            # Success - reset circuit breaker
            cb.failure_count = 0
            cb.open_until = None

            entities = []
            for node in getattr(search_result, "nodes", []):
                entity = LazyRAGEntity(
                    id=str(getattr(node, "uuid", "")),
                    name=getattr(node, "name", ""),
                    type=getattr(node, "type", "Entity"),
                    description=getattr(node, "summary", None),
                    summary=getattr(node, "summary", None),
                    labels=list(getattr(node, "labels", [])),
                )
                entities.append(entity)

            logger.debug(
                "seed_entities_found",
                count=len(entities),
                query=query[:LOG_QUERY_TRUNCATE_LENGTH],
            )

            return entities

        except asyncio.TimeoutError:
            logger.warning(
                "graphiti_search_timeout",
                query=query[:LOG_QUERY_TRUNCATE_LENGTH],
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

        except (ConnectionError, OSError) as e:
            # Network-related errors: ConnectionRefusedError, ConnectionResetError, etc.
            cb.failure_count += 1
            self._maybe_open_circuit_breaker(tenant_id)
            logger.warning(
                "graphiti_connection_error",
                error=str(e),
                error_type=type(e).__name__,
                failure_count=cb.failure_count,
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

        except ValueError as e:
            # Invalid parameters or data format issues
            logger.warning(
                "graphiti_search_value_error",
                error=str(e),
                query=query[:LOG_QUERY_TRUNCATE_LENGTH],
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

        except Exception as e:
            # Unexpected errors - track for circuit breaker
            cb.failure_count += 1
            self._maybe_open_circuit_breaker(tenant_id)
            logger.warning(
                "graphiti_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                failure_count=cb.failure_count,
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_seed_search(query, tenant_id, num_results)

    async def _fallback_seed_search(
        self,
        query: str,
        tenant_id: str,
        num_results: int,
    ) -> list[LazyRAGEntity]:
        """Fallback seed search using direct Neo4j text matching.

        Used when Graphiti is not available. Performs simple text matching
        on entity names and descriptions.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            num_results: Maximum results

        Returns:
            List of matching entities
        """
        if not self._neo4j:
            logger.warning("neo4j_not_available_for_fallback_search")
            return []

        try:
            # Extract key terms from query for matching
            query_lower = query.lower()

            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})
                    WHERE toLower(e.name) CONTAINS $query_lower
                       OR toLower(e.description) CONTAINS $query_lower
                       OR toLower(e.summary) CONTAINS $query_lower
                    RETURN e.id AS id, e.name AS name, e.type AS type,
                           e.description AS description, e.summary AS summary
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    query_lower=query_lower,
                    limit=num_results,
                )
                records = await result.data()

            return [
                LazyRAGEntity(
                    id=str(r.get("id", "")),
                    name=r.get("name", ""),
                    type=r.get("type", "Entity"),
                    description=r.get("description"),
                    summary=r.get("summary"),
                )
                for r in records
            ]

        except Exception as e:
            logger.error("fallback_seed_search_failed", error=str(e))
            return []

    async def _expand_subgraph(
        self,
        seed_entity_ids: list[str],
        tenant_id: str,
        max_hops: int,
        max_entities: int,
    ) -> SubgraphExpansionResult:
        """Expand subgraph from seed entities via N-hop traversal.

        Traverses relationships from seed entities up to max_hops depth,
        collecting all connected entities and relationships.

        Args:
            seed_entity_ids: List of seed entity IDs
            tenant_id: Tenant identifier
            max_hops: Maximum hops to traverse
            max_entities: Maximum entities to collect

        Returns:
            SubgraphExpansionResult with expanded entities and relationships
        """
        if not self._neo4j or not seed_entity_ids:
            return SubgraphExpansionResult(
                entities=[],
                relationships=[],
                seed_count=len(seed_entity_ids),
                expanded_count=0,
            )

        try:
            async with self._neo4j.driver.session() as session:
                # Expand subgraph from seeds
                expansion_result = await session.run(
                    """
                    MATCH (seed:Entity {tenant_id: $tenant_id})
                    WHERE seed.id IN $seed_ids
                    CALL apoc.path.subgraphAll(seed, {
                        maxLevel: $max_hops,
                        relationshipFilter: null,
                        labelFilter: '+Entity'
                    })
                    YIELD nodes, relationships
                    UNWIND nodes AS n
                    WITH DISTINCT n
                    WHERE n.tenant_id = $tenant_id
                    RETURN n.id AS id, n.name AS name, n.type AS type,
                           n.description AS description, n.summary AS summary
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    seed_ids=seed_entity_ids,
                    max_hops=max_hops,
                    limit=max_entities,
                )
                entity_records = await expansion_result.data()

        except Exception as apoc_error:
            # APOC not available, use standard Cypher variable-length path
            logger.debug(
                "apoc_not_available",
                error=str(apoc_error),
                hint="Using standard Cypher for expansion",
            )
            try:
                # SECURITY: Cypher Injection Prevention
                # hop_pattern uses a whitelist lookup (_SAFE_HOP_PATTERNS) that only
                # allows values [*1..1] through [*1..5]. This is NOT user-controlled
                # input - max_hops is validated to 1-5 range by Pydantic models.
                # The f-string is safe because hop_pattern can only be one of the
                # pre-defined safe patterns.
                hop_pattern = _SAFE_HOP_PATTERNS.get(max_hops, "[*1..2]")
                async with self._neo4j.driver.session() as session:
                    expansion_result = await session.run(
                        f"""
                        MATCH (seed:Entity {{tenant_id: $tenant_id}})
                        WHERE seed.id IN $seed_ids
                        MATCH path = (seed)-{hop_pattern}-(related:Entity {{tenant_id: $tenant_id}})
                        WITH DISTINCT related
                        RETURN related.id AS id, related.name AS name, related.type AS type,
                               related.description AS description, related.summary AS summary
                        LIMIT $limit
                        """,
                        tenant_id=tenant_id,
                        seed_ids=seed_entity_ids,
                        limit=max_entities,
                    )
                    entity_records = await expansion_result.data()
            except Exception as e:
                logger.error("subgraph_expansion_failed", error=str(e))
                return SubgraphExpansionResult(
                    entities=[],
                    relationships=[],
                    seed_count=len(seed_entity_ids),
                    expanded_count=0,
                )

        # Build entities list
        entities = [
            LazyRAGEntity(
                id=str(r.get("id", "")),
                name=r.get("name", ""),
                type=r.get("type", "Entity"),
                description=r.get("description"),
                summary=r.get("summary"),
            )
            for r in entity_records
        ]

        # Get all entity IDs including seeds
        all_entity_ids = list(set(seed_entity_ids + [e.id for e in entities]))

        # Fetch relationships between entities
        relationships = await self._get_relationships_between_entities(
            entity_ids=all_entity_ids,
            tenant_id=tenant_id,
        )

        return SubgraphExpansionResult(
            entities=entities,
            relationships=relationships,
            seed_count=len(seed_entity_ids),
            expanded_count=len(entities),
        )

    async def _get_relationships_between_entities(
        self,
        entity_ids: list[str],
        tenant_id: str,
    ) -> list[LazyRAGRelationship]:
        """Get all relationships between a set of entities.

        Args:
            entity_ids: List of entity IDs
            tenant_id: Tenant identifier

        Returns:
            List of relationships between the entities
        """
        if not self._neo4j or not entity_ids:
            return []

        try:
            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (a:Entity {tenant_id: $tenant_id})-[r]->(b:Entity {tenant_id: $tenant_id})
                    WHERE a.id IN $entity_ids AND b.id IN $entity_ids
                    RETURN a.id AS source_id, type(r) AS rel_type, r.fact AS fact, b.id AS target_id
                    """,
                    tenant_id=tenant_id,
                    entity_ids=entity_ids,
                )
                records = await result.data()

            return [
                LazyRAGRelationship(
                    source_id=str(r.get("source_id", "")),
                    target_id=str(r.get("target_id", "")),
                    type=r.get("rel_type", "RELATED_TO"),
                    fact=r.get("fact"),
                )
                for r in records
            ]

        except Exception as e:
            logger.error("get_relationships_failed", error=str(e))
            return []

    async def _get_community_context(
        self,
        entity_ids: list[str],
        tenant_id: str,
    ) -> list[LazyRAGCommunity]:
        """Get community context for entities from Story 20-B1.

        Looks up which communities contain the given entities and returns
        their summaries for high-level context.

        Args:
            entity_ids: List of entity IDs
            tenant_id: Tenant identifier

        Returns:
            List of relevant communities with summaries
        """
        if not self._community_detector or not entity_ids:
            return []

        try:
            # Query communities containing these entities
            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})-[:BELONGS_TO]->(c:Community {tenant_id: $tenant_id})
                    WHERE e.id IN $entity_ids
                    RETURN DISTINCT c.id AS id, c.name AS name, c.summary AS summary,
                           c.keywords AS keywords, c.level AS level
                    ORDER BY c.level DESC
                    LIMIT 5
                    """,
                    tenant_id=tenant_id,
                    entity_ids=entity_ids,
                )
                records = await result.data()

            return [
                LazyRAGCommunity(
                    id=str(r.get("id", "")),
                    name=r.get("name", ""),
                    summary=r.get("summary"),
                    keywords=tuple(r.get("keywords", [])) if r.get("keywords") else (),
                    level=r.get("level", 0),
                )
                for r in records
            ]

        except Exception as e:
            logger.warning("get_community_context_failed", error=str(e))
            return []

    async def _generate_summary(
        self,
        query: str,
        entities: list[LazyRAGEntity],
        relationships: list[LazyRAGRelationship],
        communities: list[LazyRAGCommunity],
        tenant_id: str,
    ) -> SummaryResult:
        """Generate LLM summary at query time.

        This is the core of LazyRAG - generating focused summaries only when
        needed, eliminating expensive pre-computed summaries.

        Args:
            query: Original query
            entities: Entities in the subgraph
            relationships: Relationships in the subgraph
            communities: Community context
            tenant_id: Tenant identifier for rate limiting

        Returns:
            SummaryResult with text and confidence
        """
        # Check rate limit
        if self._rate_limiter and not await self._rate_limiter.allow(f"llm:{tenant_id}"):
            logger.warning("llm_rate_limit_exceeded", tenant_id=tenant_id)
            return SummaryResult(
                text=f"Rate limit exceeded. Found {len(entities)} relevant entities with {len(relationships)} relationships.",
                confidence=self._estimate_confidence(query, entities, relationships),
                missing_info="LLM summary unavailable due to rate limits",
            )

        if not entities:
            return SummaryResult(
                text="",
                confidence=0.0,
                missing_info="No relevant entities found in the knowledge graph.",
            )

        # Format entity context
        entity_context = self._format_entities(entities)

        # Format relationship context
        relationship_context = self._format_relationships(relationships)

        # Format community context
        community_context_section = ""
        if communities:
            community_lines = []
            for c in communities:
                if c.summary:
                    community_lines.append(f"- {c.name} (Level {c.level}): {c.summary}")
            if community_lines:
                community_context_section = (
                    "Community Context (high-level themes):\n" + "\n".join(community_lines)
                )

        # Build prompt
        prompt = LAZY_RAG_SUMMARY_PROMPT.format(
            query=query,
            entity_count=len(entities),
            entity_context=entity_context,
            relationship_count=len(relationships),
            relationship_context=relationship_context,
            community_context_section=community_context_section,
        )

        try:
            # Get LLM adapter for OpenAI-compatible providers
            llm_adapter = get_llm_adapter(self._settings)

            if llm_adapter.provider in OPENAI_COMPATIBLE_LLM_PROVIDERS:
                client = AsyncOpenAI(**llm_adapter.openai_kwargs())

                # Wrap LLM call with timeout to prevent indefinite blocking
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=self.summary_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a knowledge graph analyst providing accurate, concise answers based on graph data.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.3,
                            max_tokens=1000,
                        ),
                        timeout=LLM_CALL_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "llm_call_timeout",
                        timeout_seconds=LLM_CALL_TIMEOUT_SECONDS,
                        model=self.summary_model,
                    )
                    return SummaryResult(
                        text=f"Summary generation timed out after {LLM_CALL_TIMEOUT_SECONDS}s. Based on {len(entities)} entities found.",
                        confidence=0.3,
                        missing_info="LLM summary timed out",
                    )

                # Record metric
                record_llm_call(
                    model=self.summary_model,
                    operation="summary",
                    tenant_id=tenant_id,
                )

                summary_text = response.choices[0].message.content or ""

                # Check if LLM indicated missing information
                missing_info = None
                if "insufficient" in summary_text.lower() or "missing" in summary_text.lower():
                    # Try to extract what's missing
                    missing_match = re.search(
                        r"(?:missing|insufficient)[^.]*\.",
                        summary_text,
                        re.IGNORECASE,
                    )
                    if missing_match:
                        missing_info = missing_match.group(0).strip()

                confidence = self._estimate_confidence(query, entities, relationships)

                return SummaryResult(
                    text=summary_text,
                    confidence=confidence,
                    missing_info=missing_info,
                )
            else:
                # Non-OpenAI provider - return context-based response
                logger.warning(
                    "llm_provider_not_openai_compatible",
                    provider=llm_adapter.provider,
                    hint="Returning entity context without LLM summary",
                )
                return SummaryResult(
                    text=f"Found {len(entities)} relevant entities with {len(relationships)} relationships. "
                    f"Key entities: {', '.join(e.name for e in entities[:5])}.",
                    confidence=self._estimate_confidence(query, entities, relationships),
                    missing_info=None,
                )

        except Exception as e:
            logger.error("summary_generation_failed", error=str(e))
            # Return a graceful fallback
            return SummaryResult(
                text=f"Found {len(entities)} relevant entities with {len(relationships)} relationships. "
                f"Summary generation failed: {str(e)[:100]}",
                confidence=self._estimate_confidence(query, entities, relationships) * 0.5,
                missing_info="LLM summary generation failed",
            )

    def _format_entities(self, entities: list[LazyRAGEntity]) -> str:
        """Format entities for prompt context.

        Args:
            entities: List of entities

        Returns:
            Formatted entity string
        """
        lines = []
        for e in entities[:MAX_ENTITIES_IN_CONTEXT]:
            desc = e.description or e.summary or "No description"
            lines.append(f"- {e.name} ({e.type}): {desc[:MAX_ENTITY_DESCRIPTION_LENGTH]}")
        return "\n".join(lines) if lines else "No entities found."

    def _format_relationships(self, relationships: list[LazyRAGRelationship]) -> str:
        """Format relationships for prompt context.

        Args:
            relationships: List of relationships

        Returns:
            Formatted relationship string
        """
        lines = []
        for r in relationships[:MAX_RELATIONSHIPS_IN_CONTEXT]:
            fact_text = f" - {r.fact[:MAX_RELATIONSHIP_FACT_LENGTH]}" if r.fact else ""
            lines.append(f"- [{r.source_id}] --{r.type}--> [{r.target_id}]{fact_text}")
        return "\n".join(lines) if lines else "No relationships found."

    def _estimate_confidence(
        self,
        query: str,
        entities: list[LazyRAGEntity],
        relationships: list[LazyRAGRelationship],
    ) -> float:
        """Estimate confidence based on entity coverage.

        Confidence factors:
        - Entity count relative to max_entities (higher = more context)
        - Query term coverage in entity names/descriptions
        - Relationship density (more relationships = better connected)

        Args:
            query: Original query
            entities: Retrieved entities
            relationships: Retrieved relationships

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not entities:
            return 0.0

        # Factor 1: Entity coverage (0.0-0.4)
        entity_ratio = min(1.0, len(entities) / self.max_entities)
        entity_score = entity_ratio * 0.4

        # Factor 2: Query term coverage in entities (0.0-0.4)
        # Extract simple terms from query
        query_terms = [
            t.lower() for t in re.findall(r"\b\w{3,}\b", query) if len(t) >= 3
        ]
        if query_terms:
            entity_text = " ".join(
                [
                    f"{e.name} {e.description or ''}"
                    for e in entities
                ]
            ).lower()
            terms_found = sum(1 for term in query_terms if term in entity_text)
            term_coverage = terms_found / len(query_terms)
        else:
            term_coverage = 0.5  # Neutral if no query terms
        term_score = term_coverage * 0.4

        # Factor 3: Relationship density (0.0-0.2)
        if len(entities) > 1:
            max_possible_rels = len(entities) * (len(entities) - 1)
            rel_density = min(1.0, len(relationships) / max(1, max_possible_rels))
        else:
            rel_density = 0.0
        rel_score = rel_density * 0.2

        return round(entity_score + term_score + rel_score, 2)

    def _merge_entities(
        self,
        seed_entities: list[LazyRAGEntity],
        expanded_entities: list[LazyRAGEntity],
    ) -> list[LazyRAGEntity]:
        """Merge and deduplicate entities.

        Args:
            seed_entities: Seed entities (higher priority)
            expanded_entities: Expanded entities

        Returns:
            Deduplicated list with seed entities first
        """
        seen_ids: set[str] = set()
        result: list[LazyRAGEntity] = []

        # Add seed entities first
        for e in seed_entities:
            if e.id and e.id not in seen_ids:
                seen_ids.add(e.id)
                result.append(e)

        # Add expanded entities
        for e in expanded_entities:
            if e.id and e.id not in seen_ids:
                seen_ids.add(e.id)
                result.append(e)

        return result
