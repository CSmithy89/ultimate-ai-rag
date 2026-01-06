"""Dual-Level Retrieval (Story 20-C2).

This module implements dual-level retrieval combining:
- Low-level: Entity/chunk granular retrieval via Graphiti
- High-level: Community/theme retrieval via CommunityDetector (20-B1)

The dual-level approach is inspired by LightRAG's architecture that
combines specific entity knowledge with broader thematic context for
more comprehensive answers.

Key Features:
- Parallel execution of low-level and high-level retrieval
- Multi-tenancy via tenant_id/group_ids filtering
- Graceful fallback when one level has no results
- LLM-based synthesis of both levels
- Configurable weights for result ranking
- Feature flag: DUAL_LEVEL_RETRIEVAL_ENABLED

Configuration:
- DUAL_LEVEL_RETRIEVAL_ENABLED: Enable/disable feature (default: false)
- DUAL_LEVEL_LOW_WEIGHT: Weight for low-level results (default: 0.6)
- DUAL_LEVEL_HIGH_WEIGHT: Weight for high-level results (default: 0.4)
- DUAL_LEVEL_LOW_LIMIT: Max low-level results (default: 10)
- DUAL_LEVEL_HIGH_LIMIT: Max high-level results (default: 5)
- DUAL_LEVEL_SYNTHESIS_MODEL: LLM model for synthesis (default: gpt-4o-mini)

Performance target: <300ms additional latency
"""

import asyncio
import time
from typing import Any, Optional

import structlog
from openai import AsyncOpenAI

from ..config import Settings
from ..db.graphiti import GraphitiClient, GRAPHITI_AVAILABLE
from ..llm.providers import get_llm_adapter, OPENAI_COMPATIBLE_LLM_PROVIDERS
from .dual_level_models import (
    DualLevelResult,
    HighLevelResult,
    LowLevelResult,
    SynthesisResult,
)

logger = structlog.get_logger(__name__)


# Synthesis prompt template for combining both levels
DUAL_LEVEL_SYNTHESIS_PROMPT = """You are a knowledge synthesis expert. Combine the following two perspectives to answer the query comprehensively.

Query: {query}

=== LOW-LEVEL CONTEXT (Specific Facts & Entities) ===
{low_level_context}

=== HIGH-LEVEL CONTEXT (Themes & Patterns) ===
{high_level_context}

Instructions:
1. Synthesize BOTH perspectives into a coherent answer.
2. Use specific facts from low-level context for precision.
3. Frame the answer within the broader themes from high-level context.
4. If contexts conflict, prefer low-level facts but acknowledge the broader pattern.
5. Indicate confidence: HIGH (both levels agree), MEDIUM (partial overlap), LOW (one level only).
6. Be concise but comprehensive.

Synthesis:"""


class DualLevelRetriever:
    """Dual-level retrieval combining entity and community perspectives.

    This class implements the LightRAG-inspired dual-level pattern:
    - Low-level: Granular entity/chunk retrieval for specific facts
    - High-level: Community/theme retrieval for broader context
    - Synthesis: LLM combines both for comprehensive answers

    Attributes:
        graphiti_client: Graphiti client for low-level entity search
        neo4j_client: Neo4j client for graph queries
        settings: Application settings with dual-level configuration
        community_detector: CommunityDetector from Story 20-B1
    """

    def __init__(
        self,
        graphiti_client: Optional[GraphitiClient],
        neo4j_client: Any,
        settings: Settings,
        community_detector: Optional[Any] = None,
    ) -> None:
        """Initialize DualLevelRetriever.

        Args:
            graphiti_client: Graphiti client for entity search
            neo4j_client: Neo4j client for graph queries
            settings: Application settings
            community_detector: CommunityDetector instance from 20-B1
        """
        self._graphiti = graphiti_client
        self._neo4j = neo4j_client
        self._settings = settings
        self._community_detector = community_detector

        # Extract settings with defaults
        self.low_weight = getattr(settings, "dual_level_low_weight", 0.6)
        self.high_weight = getattr(settings, "dual_level_high_weight", 0.4)
        self.low_limit = getattr(settings, "dual_level_low_limit", 10)
        self.high_limit = getattr(settings, "dual_level_high_limit", 5)
        self.synthesis_model = getattr(settings, "dual_level_synthesis_model", "gpt-4o-mini")
        self.synthesis_temperature = getattr(settings, "dual_level_synthesis_temperature", 0.3)

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        low_level_limit: Optional[int] = None,
        high_level_limit: Optional[int] = None,
        include_synthesis: bool = True,
        low_weight: Optional[float] = None,
        high_weight: Optional[float] = None,
    ) -> DualLevelResult:
        """Execute dual-level retrieval with optional synthesis.

        This is the main entry point that:
        1. Runs low-level and high-level retrieval in parallel
        2. Calculates combined confidence based on coverage
        3. Optionally synthesizes both perspectives via LLM

        Args:
            query: Natural language query
            tenant_id: Tenant identifier for multi-tenancy
            low_level_limit: Override max low-level results
            high_level_limit: Override max high-level results
            include_synthesis: Generate LLM synthesis (default: True)
            low_weight: Override low-level weight
            high_weight: Override high-level weight

        Returns:
            DualLevelResult with both levels and optional synthesis

        Raises:
            RuntimeError: If both retrieval levels fail
        """
        start_time = time.perf_counter()

        # Apply overrides
        effective_low_limit = low_level_limit or self.low_limit
        effective_high_limit = high_level_limit or self.high_limit
        effective_low_weight = low_weight if low_weight is not None else self.low_weight
        effective_high_weight = high_weight if high_weight is not None else self.high_weight

        logger.info(
            "dual_level_retrieval_started",
            query=query[:100],
            tenant_id=tenant_id,
            low_limit=effective_low_limit,
            high_limit=effective_high_limit,
            include_synthesis=include_synthesis,
        )

        try:
            # Execute both retrievals in parallel for performance
            low_level_task = self._retrieve_low_level(
                query=query,
                tenant_id=tenant_id,
                limit=effective_low_limit,
            )
            high_level_task = self._retrieve_high_level(
                query=query,
                tenant_id=tenant_id,
                limit=effective_high_limit,
            )

            low_level_results, high_level_results = await asyncio.gather(
                low_level_task, high_level_task, return_exceptions=True
            )

            # Handle exceptions from parallel execution
            if isinstance(low_level_results, Exception):
                logger.warning(
                    "low_level_retrieval_failed",
                    error=str(low_level_results),
                )
                low_level_results = []

            if isinstance(high_level_results, Exception):
                logger.warning(
                    "high_level_retrieval_failed",
                    error=str(high_level_results),
                )
                high_level_results = []

            # Determine if fallback was used (one level empty)
            fallback_used = (
                (len(low_level_results) == 0 and len(high_level_results) > 0) or
                (len(high_level_results) == 0 and len(low_level_results) > 0)
            )

            # Calculate confidence based on coverage
            confidence = self._calculate_confidence(
                low_level_results=low_level_results,
                high_level_results=high_level_results,
                low_weight=effective_low_weight,
                high_weight=effective_high_weight,
            )

            # Generate synthesis if requested and we have results
            synthesis: Optional[str] = None
            if include_synthesis and (low_level_results or high_level_results):
                synthesis_result = await self._synthesize(
                    query=query,
                    low_level_results=low_level_results,
                    high_level_results=high_level_results,
                )
                synthesis = synthesis_result.text if synthesis_result else None
                # Update confidence with synthesis confidence if available
                if synthesis_result and synthesis_result.confidence > 0:
                    confidence = (confidence + synthesis_result.confidence) / 2

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            weighted_low_results = [
                LowLevelResult(
                    id=result.id,
                    name=result.name,
                    type=result.type,
                    content=result.content,
                    score=round(result.score * effective_low_weight, 3),
                    source=result.source,
                    labels=list(result.labels),
                )
                for result in low_level_results
            ]
            weighted_high_results = [
                HighLevelResult(
                    id=result.id,
                    name=result.name,
                    summary=result.summary,
                    keywords=result.keywords,
                    level=result.level,
                    entity_count=result.entity_count,
                    score=round(result.score * effective_high_weight, 3),
                    entity_ids=result.entity_ids,
                )
                for result in high_level_results
            ]

            result = DualLevelResult(
                query=query,
                tenant_id=tenant_id,
                low_level_results=weighted_low_results,
                high_level_results=weighted_high_results,
                synthesis=synthesis,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                fallback_used=fallback_used,
            )

            logger.info(
                "dual_level_retrieval_completed",
                query=query[:100],
                tenant_id=tenant_id,
                low_level_count=len(low_level_results),
                high_level_count=len(high_level_results),
                confidence=confidence,
                fallback_used=fallback_used,
                processing_time_ms=processing_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "dual_level_retrieval_failed",
                query=query[:100],
                tenant_id=tenant_id,
                error=str(e),
            )
            raise

    async def _retrieve_low_level(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[LowLevelResult]:
        """Retrieve low-level (entity/chunk) results via Graphiti.

        Uses Graphiti's hybrid search (semantic + BM25 + graph) to find
        specific entities and facts relevant to the query.

        Args:
            query: Natural language query
            tenant_id: Tenant identifier (used as group_id)
            limit: Maximum results to return

        Returns:
            List of LowLevelResult objects
        """
        if not self._graphiti or not GRAPHITI_AVAILABLE:
            logger.warning(
                "graphiti_not_available_for_low_level",
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_low_level_search(query, tenant_id, limit)

        if not self._graphiti.is_connected:
            logger.warning(
                "graphiti_not_connected",
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_low_level_search(query, tenant_id, limit)

        try:
            search_result = await self._graphiti.client.search(
                query=query,
                group_ids=[tenant_id],
                num_results=limit,
            )

            results = []
            for idx, node in enumerate(getattr(search_result, "nodes", [])):
                # Calculate score based on position (first = highest)
                score = 1.0 - (idx / max(1, limit))

                result = LowLevelResult(
                    id=str(getattr(node, "uuid", "")),
                    name=getattr(node, "name", ""),
                    type=getattr(node, "type", "Entity"),
                    content=getattr(node, "summary", None) or getattr(node, "description", None),
                    score=round(score, 3),
                    source=getattr(node, "source_id", None),
                    labels=list(getattr(node, "labels", [])),
                )
                results.append(result)

            logger.debug(
                "low_level_retrieved",
                count=len(results),
                query=query[:50],
            )

            return results

        except Exception as e:
            logger.warning(
                "graphiti_search_failed",
                error=str(e),
                hint="Falling back to direct Neo4j search",
            )
            return await self._fallback_low_level_search(query, tenant_id, limit)

    async def _fallback_low_level_search(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[LowLevelResult]:
        """Fallback low-level search using direct Neo4j text matching.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            limit: Maximum results

        Returns:
            List of matching LowLevelResult objects
        """
        if not self._neo4j:
            logger.warning("neo4j_not_available_for_fallback")
            return []

        try:
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
                    limit=limit,
                )
                records = await result.data()

            return [
                LowLevelResult(
                    id=str(r.get("id", "")),
                    name=r.get("name", ""),
                    type=r.get("type", "Entity"),
                    content=r.get("description") or r.get("summary"),
                    score=0.5,  # Lower score for fallback results
                )
                for r in records
            ]

        except Exception as e:
            logger.error("fallback_low_level_search_failed", error=str(e))
            return []

    async def _retrieve_high_level(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[HighLevelResult]:
        """Retrieve high-level (community/theme) results.

        Uses CommunityDetector from Story 20-B1 to find relevant
        communities and their summaries for broader context.

        Args:
            query: Natural language query
            tenant_id: Tenant identifier
            limit: Maximum results to return

        Returns:
            List of HighLevelResult objects
        """
        if not self._neo4j:
            logger.warning("neo4j_not_available_for_high_level")
            return []

        try:
            if self._community_detector:
                results = await self._semantic_high_level_search(
                    query=query,
                    tenant_id=tenant_id,
                    limit=limit,
                )
                if results:
                    return results

            # Query communities with text matching on name, summary, keywords
            query_lower = query.lower()

            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Community {tenant_id: $tenant_id})
                    WHERE toLower(c.name) CONTAINS $query_lower
                       OR toLower(c.summary) CONTAINS $query_lower
                       OR ANY(kw IN c.keywords WHERE toLower(kw) CONTAINS $query_lower)
                    OPTIONAL MATCH (e:Entity {tenant_id: $tenant_id})-[:BELONGS_TO]->(c)
                    WITH c, COUNT(e) AS entity_count
                    RETURN c.id AS id, c.name AS name, c.summary AS summary,
                           c.keywords AS keywords, c.level AS level,
                           entity_count
                    ORDER BY entity_count DESC, c.level DESC
                    LIMIT $limit
                    """,
                    tenant_id=tenant_id,
                    query_lower=query_lower,
                    limit=limit,
                )
                records = await result.data()

            results = []
            for idx, r in enumerate(records):
                # Calculate score based on position
                score = 1.0 - (idx / max(1, limit))

                result = HighLevelResult(
                    id=str(r.get("id", "")),
                    name=r.get("name", ""),
                    summary=r.get("summary"),
                    keywords=tuple(r.get("keywords", []) or []),
                    level=r.get("level", 0) or 0,
                    entity_count=r.get("entity_count", 0) or 0,
                    score=round(score, 3),
                )
                results.append(result)

            # If no results from text matching, try semantic approach via community embedding
            if not results and self._community_detector:
                results = await self._semantic_high_level_search(
                    query=query,
                    tenant_id=tenant_id,
                    limit=limit,
                )

            logger.debug(
                "high_level_retrieved",
                count=len(results),
                query=query[:50],
            )

            return results

        except Exception as e:
            logger.error("high_level_retrieval_failed", error=str(e))
            return []

    async def _semantic_high_level_search(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[HighLevelResult]:
        """Semantic search for communities using embeddings.

        Fallback when text matching doesn't find communities.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            limit: Maximum results

        Returns:
            List of HighLevelResult objects
        """
        if not self._community_detector:
            return []

        try:
            # Use community detector's search if available
            if hasattr(self._community_detector, "search_communities"):
                communities = await self._community_detector.search_communities(
                    query=query,
                    tenant_id=tenant_id,
                    limit=limit,
                )
                return [
                    HighLevelResult(
                        id=str(c.get("id", "")),
                        name=c.get("name", ""),
                        summary=c.get("summary"),
                        keywords=tuple(c.get("keywords", [])),
                        level=c.get("level", 0),
                        entity_count=c.get("entity_count", 0),
                        score=c.get("score", 0.5),
                    )
                    for c in communities
                ]
        except Exception as e:
            logger.warning("semantic_high_level_search_failed", error=str(e))

        return []

    async def _synthesize(
        self,
        query: str,
        low_level_results: list[LowLevelResult],
        high_level_results: list[HighLevelResult],
    ) -> Optional[SynthesisResult]:
        """Synthesize low-level and high-level results via LLM.

        Creates a coherent answer that combines specific facts with
        broader thematic context.

        Args:
            query: Original query
            low_level_results: Low-level entity/chunk results
            high_level_results: High-level community/theme results

        Returns:
            SynthesisResult with combined text and confidence
        """
        if not low_level_results and not high_level_results:
            return SynthesisResult(
                text="",
                confidence=0.0,
                reasoning="No results from either level to synthesize.",
            )

        # Format low-level context
        if low_level_results:
            low_lines = []
            for r in low_level_results[:10]:  # Limit for context window
                content = r.content or "No description"
                low_lines.append(f"- {r.name} ({r.type}): {content[:200]}")
            low_level_context = "\n".join(low_lines)
        else:
            low_level_context = "No specific entities found."

        # Format high-level context
        if high_level_results:
            high_lines = []
            for r in high_level_results[:5]:  # Limit for context window
                keywords = ", ".join(r.keywords[:5]) if r.keywords else "N/A"
                summary = r.summary or "No summary"
                high_lines.append(
                    f"- {r.name} (Level {r.level}, {r.entity_count} entities): {summary[:200]}\n  Keywords: {keywords}"
                )
            high_level_context = "\n".join(high_lines)
        else:
            high_level_context = "No thematic communities found."

        # Build prompt
        prompt = DUAL_LEVEL_SYNTHESIS_PROMPT.format(
            query=query,
            low_level_context=low_level_context,
            high_level_context=high_level_context,
        )

        try:
            # Get LLM adapter for OpenAI-compatible providers
            llm_adapter = get_llm_adapter(self._settings)

            if llm_adapter.provider in OPENAI_COMPATIBLE_LLM_PROVIDERS:
                client = AsyncOpenAI(**llm_adapter.openai_kwargs())

                response = await client.chat.completions.create(
                    model=self.synthesis_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a knowledge synthesis expert that combines multiple perspectives into coherent answers.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.synthesis_temperature,
                    max_tokens=1000,
                )

                synthesis_text = response.choices[0].message.content or ""

                # Estimate confidence based on synthesis content
                confidence = self._estimate_synthesis_confidence(
                    synthesis_text=synthesis_text,
                    low_count=len(low_level_results),
                    high_count=len(high_level_results),
                )

                return SynthesisResult(
                    text=synthesis_text,
                    confidence=confidence,
                )
            else:
                # Non-OpenAI provider - return context-based response
                logger.warning(
                    "llm_provider_not_openai_compatible",
                    provider=llm_adapter.provider,
                    hint="Returning combined context without LLM synthesis",
                )
                return SynthesisResult(
                    text=f"Found {len(low_level_results)} specific entities and {len(high_level_results)} thematic communities. "
                    f"Key entities: {', '.join(r.name for r in low_level_results[:3])}.",
                    confidence=0.5,
                )

        except Exception as e:
            logger.error("synthesis_generation_failed", error=str(e))
            # Return graceful fallback
            return SynthesisResult(
                text=f"Found {len(low_level_results)} entities and {len(high_level_results)} communities. "
                f"Synthesis generation failed: {str(e)[:100]}",
                confidence=0.3,
                reasoning=f"LLM synthesis failed: {str(e)[:100]}",
            )

    def _calculate_confidence(
        self,
        low_level_results: list[LowLevelResult],
        high_level_results: list[HighLevelResult],
        low_weight: float,
        high_weight: float,
    ) -> float:
        """Calculate confidence based on coverage from both levels.

        Confidence factors:
        - Low-level coverage: Do we have specific facts?
        - High-level coverage: Do we have thematic context?
        - Balance: Both levels contributing improves confidence

        Args:
            low_level_results: Low-level results
            high_level_results: High-level results
            low_weight: Weight for low-level
            high_weight: Weight for high-level

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not low_level_results and not high_level_results:
            return 0.0

        # Low-level contribution (0.0-0.5 based on weight)
        low_score = 0.0
        if low_level_results:
            # Average the individual scores
            avg_low_score = sum(r.score for r in low_level_results) / len(low_level_results)
            low_score = avg_low_score * low_weight

        # High-level contribution (0.0-0.5 based on weight)
        high_score = 0.0
        if high_level_results:
            avg_high_score = sum(r.score for r in high_level_results) / len(high_level_results)
            high_score = avg_high_score * high_weight

        # Normalize weights if they don't sum to 1
        total_weight = low_weight + high_weight
        if total_weight > 0:
            confidence = (low_score + high_score) / total_weight
        else:
            confidence = 0.5

        # Bonus for having both levels (balanced retrieval)
        if low_level_results and high_level_results:
            confidence = min(1.0, confidence * 1.1)  # 10% bonus

        return round(min(1.0, max(0.0, confidence)), 2)

    def _estimate_synthesis_confidence(
        self,
        synthesis_text: str,
        low_count: int,
        high_count: int,
    ) -> float:
        """Estimate confidence based on synthesis content.

        Args:
            synthesis_text: Generated synthesis text
            low_count: Number of low-level results
            high_count: Number of high-level results

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # Check for explicit confidence markers in synthesis
        text_lower = synthesis_text.lower()
        if "high confidence" in text_lower or "confident" in text_lower:
            confidence = 0.8
        elif "medium confidence" in text_lower or "moderate" in text_lower:
            confidence = 0.6
        elif "low confidence" in text_lower or "uncertain" in text_lower:
            confidence = 0.4
        elif "insufficient" in text_lower or "not enough" in text_lower:
            confidence = 0.3

        # Boost for having both levels
        if low_count > 0 and high_count > 0:
            confidence = min(1.0, confidence * 1.15)

        # Slight boost for more results
        if low_count >= 5:
            confidence = min(1.0, confidence * 1.05)
        if high_count >= 3:
            confidence = min(1.0, confidence * 1.05)

        return round(confidence, 2)
