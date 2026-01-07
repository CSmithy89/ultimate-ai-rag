"""Query Router for Global/Local Query Classification (Story 20-B3).

This module implements the QueryRouter class that routes queries to either:
- GLOBAL: Community-level retrieval (uses 20-B1 CommunityDetector)
- LOCAL: Entity-level retrieval (uses 20-B2 LazyRAGRetriever)
- HYBRID: Weighted combination of both approaches

The routing decision is made using:
1. Rule-based classification via regex pattern matching (fast, <10ms)
2. Optional LLM classification for ambiguous queries (when confidence < threshold)

Based on Microsoft GraphRAG's global vs local query distinction, this enables
the system to answer abstract questions using community summaries while
providing precise entity-level answers for specific questions.
"""

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Pattern

import structlog
from openai import AsyncOpenAI

from ..config import Settings
from ..llm.providers import get_llm_adapter, OPENAI_COMPATIBLE_LLM_PROVIDERS
from .query_router_models import QueryType, RoutingDecision

logger = structlog.get_logger(__name__)

# Maximum query length for routing to prevent DoS via expensive regex operations
MAX_QUERY_LENGTH = 10000

# LRU Cache settings for routing decisions
ROUTING_CACHE_MAX_SIZE = 1000  # Maximum number of cached entries
ROUTING_CACHE_TTL_SECONDS = 300  # 5 minutes TTL for cached decisions


@dataclass
class _CacheEntry:
    """Cache entry for routing decisions with TTL."""

    decision: RoutingDecision
    timestamp: float


# Compiled regex patterns for rule-based classification
# Patterns are compiled once at module load for performance (<10ms per query)

# SECURITY: ReDoS Prevention
# These patterns are designed to be safe against Regular Expression Denial of Service (ReDoS).
# - Non-greedy quantifiers (e.g., .{0,N}?) are used to limit backtracking.
# - Bounded repetitions (e.g., {0,20}) prevent catastrophic backtracking on long inputs.
# - MAX_QUERY_LENGTH check ensures inputs are within safe limits.
# - Anchors (\b) are used to match whole words where appropriate.

# Global patterns indicate need for corpus-wide understanding
# Note: Use non-greedy .{0,N}? to prevent ReDoS while matching across words
GLOBAL_PATTERNS: list[Pattern[str]] = [
    # Theme/topic patterns
    re.compile(r"\b(what|which)\s+(are|is)\s+the\s+(main|primary|key|overall|central)\b", re.IGNORECASE),
    re.compile(r"\b(main|primary|key|central|major)\s+(themes?|topics?|concepts?|ideas?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+themes?\b", re.IGNORECASE),
    # Summary/overview patterns
    re.compile(r"\b(summarize|summary|overview|synopsis)\b", re.IGNORECASE),
    re.compile(r"\b(give|provide)\s+(me\s+)?(a\s+)?(general\s+)?(overview|summary)\b", re.IGNORECASE),
    re.compile(r"\bgeneral\s+(understanding|overview|summary)\b", re.IGNORECASE),
    # Aggregation patterns (use non-greedy ? to prevent ReDoS)
    re.compile(r"\b(all|every|each)\s+.{0,20}?(types?|kinds?|categories?)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(many|much)\s+.{0,30}?(total|overall|in\s+general)\b", re.IGNORECASE),
    # Trend/analysis patterns
    re.compile(r"\b(trends?|patterns?|tendencies?)\s+(in|across)\b", re.IGNORECASE),
    re.compile(r"\b(overall|in\s+general|as\s+a\s+whole)\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+.{0,30}?(in\s+general|overall|as\s+a\s+whole)\b", re.IGNORECASE),
    # Comparison across corpus
    re.compile(r"\bcompare\s+.{0,30}?(across|between|among)\b", re.IGNORECASE),
]

# Local patterns indicate need for specific entity details
# Note: Use non-greedy .{0,N}? to prevent ReDoS while matching across words
LOCAL_PATTERNS: list[Pattern[str]] = [
    # What is X patterns
    re.compile(r"\bwhat\s+is\s+(\w+)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(does|did)\s+(\w+)\b", re.IGNORECASE),
    # Who is X patterns
    re.compile(r"\bwho\s+(is|was|are|were)\s+", re.IGNORECASE),
    # Where/when patterns
    re.compile(r"\bwhere\s+(is|was|does|did|are|were)\b", re.IGNORECASE),
    re.compile(r"\bwhen\s+(did|was|is|does|are|were)\b", re.IGNORECASE),
    # How does X patterns
    re.compile(r"\bhow\s+(do|does|did)\s+(\w+)\b", re.IGNORECASE),
    # Specific entity patterns
    re.compile(r"\b(specific|particular|exact|precise)\b", re.IGNORECASE),
    re.compile(r"\b(this|that|the)\s+(\w+)\s+(function|class|method|file|module)\b", re.IGNORECASE),
    # Find/locate patterns (use non-greedy ? to prevent ReDoS)
    re.compile(r"\b(find|locate|get)\s+.{0,20}?(named|called|about|for)\b", re.IGNORECASE),
    # Definition patterns
    re.compile(r"\bdefine\s+(\w+)\b", re.IGNORECASE),
    re.compile(r"\bdefinition\s+of\s+", re.IGNORECASE),
    # Function/code patterns (common in codebase queries)
    re.compile(r"\bfunction\s+(named|called)\b", re.IGNORECASE),
    re.compile(r"\b(what|how)\s+.{0,10}?function\b", re.IGNORECASE),
]

# LLM Classification prompt template
LLM_CLASSIFICATION_PROMPT = """You are a query classifier. Determine if the following query requires:
- GLOBAL: High-level, abstract understanding across the entire knowledge base (themes, summaries, trends)
- LOCAL: Specific information about particular entities, facts, or details
- HYBRID: Both high-level context and specific details

Query: {query}

Respond with exactly one of: GLOBAL, LOCAL, or HYBRID
Then provide a confidence score from 0.0 to 1.0
Then provide a brief reasoning.

Format:
TYPE: [GLOBAL|LOCAL|HYBRID]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""


class QueryRouter:
    """Routes queries to appropriate retrieval strategies.

    The QueryRouter analyzes incoming queries and determines whether they
    should be processed using:
    - GLOBAL retrieval: Community-level search via CommunityDetector (20-B1)
    - LOCAL retrieval: Entity-level search via LazyRAGRetriever (20-B2)
    - HYBRID retrieval: Weighted combination of both approaches

    Routing decisions are made using:
    1. Fast rule-based classification via regex patterns (<10ms)
    2. Optional LLM classification for ambiguous queries (when enabled)

    Attributes:
        settings: Application settings
        use_llm: Whether to use LLM for uncertain queries
        llm_model: Model to use for LLM classification
        confidence_threshold: Threshold below which LLM/hybrid fallback is used
    """

    def __init__(
        self,
        settings: Settings,
        use_llm: Optional[bool] = None,
        llm_model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ):
        """Initialize the QueryRouter.

        Args:
            settings: Application settings
            use_llm: Override for LLM classification (None = use settings)
            llm_model: Override for LLM model (None = use settings)
            confidence_threshold: Override for confidence threshold (None = use settings)
        """
        self._settings = settings
        self.use_llm = use_llm if use_llm is not None else settings.query_routing_use_llm
        self.llm_model = llm_model or settings.query_routing_llm_model
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.query_routing_confidence_threshold
        )
        self._openai_client: Optional[AsyncOpenAI] = None

        # LRU cache for routing decisions (OrderedDict for LRU ordering)
        self._routing_cache: OrderedDict[str, _CacheEntry] = OrderedDict()

    async def route(
        self,
        query: str,
        tenant_id: str,
        use_llm: Optional[bool] = None,
    ) -> RoutingDecision:
        """Route a query to the appropriate retrieval strategy.

        This is the main entry point for query routing. It:
        1. Runs rule-based classification (fast pattern matching)
        2. If confidence < threshold and LLM is enabled, runs LLM classification
        3. Combines results to make final routing decision

        Args:
            query: The query text to classify
            tenant_id: Tenant identifier for logging
            use_llm: Override for LLM classification (None = use instance setting)

        Returns:
            RoutingDecision with query type, confidence, and reasoning
        """
        start_time = time.perf_counter()

        # Validate input
        if not query or not query.strip():
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                confidence=0.0,
                reasoning="Empty query defaults to hybrid",
                global_weight=0.5,
                local_weight=0.5,
                classification_method="rule_based",
                processing_time_ms=0,
            )

        # Validate query length to prevent DoS via expensive regex operations
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
            )

        # Check cache for existing routing decision
        should_use_llm = use_llm if use_llm is not None else self.use_llm
        cache_key = self._get_cache_key(query, should_use_llm)
        cached = self._get_cached_decision(cache_key)
        if cached is not None:
            # Update processing time for cache hit
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            cached.processing_time_ms = processing_time_ms
            logger.debug(
                "routing_cache_hit",
                query=query[:100],
                tenant_id=tenant_id,
                query_type=cached.query_type.value,
            )
            return cached

        # Step 1: Rule-based classification (always runs, fast)
        rule_decision = self._rule_based_classify(query)

        logger.debug(
            "rule_based_classification",
            query=query[:100],
            tenant_id=tenant_id,
            query_type=rule_decision.query_type.value,
            confidence=rule_decision.confidence,
            global_matches=rule_decision.global_matches,
            local_matches=rule_decision.local_matches,
        )

        # Step 2: Check if LLM classification is needed (should_use_llm already set above)
        if (
            should_use_llm
            and rule_decision.confidence < self.confidence_threshold
        ):
            # LLM classification for uncertain queries
            try:
                llm_decision = await self._llm_classify(query)

                logger.debug(
                    "llm_classification",
                    query=query[:100],
                    tenant_id=tenant_id,
                    query_type=llm_decision.query_type.value,
                    confidence=llm_decision.confidence,
                )

                # Combine decisions
                final_decision = self._combine_decisions(rule_decision, llm_decision)
                final_decision.classification_method = "combined"

            except Exception as e:
                logger.warning(
                    "llm_classification_failed",
                    error=str(e),
                    tenant_id=tenant_id,
                    fallback="rule_based",
                )
                # Fall back to rule-based decision
                final_decision = rule_decision
        else:
            final_decision = rule_decision

        # Calculate processing time
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        final_decision.processing_time_ms = processing_time_ms

        logger.info(
            "query_routed",
            query=query[:100],
            tenant_id=tenant_id,
            query_type=final_decision.query_type.value,
            confidence=final_decision.confidence,
            method=final_decision.classification_method,
            processing_time_ms=processing_time_ms,
        )

        # Cache the decision for future queries
        self._cache_decision(cache_key, final_decision)

        return final_decision

    def _rule_based_classify(self, query: str) -> RoutingDecision:
        """Classify query using regex pattern matching.

        This is the fast path for classification, completing in <10ms.
        It matches the query against predefined global and local patterns.

        Args:
            query: The query text to classify

        Returns:
            RoutingDecision based on pattern matching
        """
        # Count pattern matches
        global_matches = sum(
            1 for pattern in GLOBAL_PATTERNS
            if pattern.search(query)
        )
        local_matches = sum(
            1 for pattern in LOCAL_PATTERNS
            if pattern.search(query)
        )

        total = global_matches + local_matches

        # No patterns matched - uncertain, default to hybrid with low confidence
        if total == 0:
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                confidence=0.3,
                reasoning="No pattern matches - defaulting to hybrid",
                global_weight=0.5,
                local_weight=0.5,
                classification_method="rule_based",
                global_matches=0,
                local_matches=0,
            )

        # Calculate ratio
        global_ratio = global_matches / total

        # Determine query type based on ratio
        if global_ratio >= 0.7:
            # Strongly global
            confidence = min(0.9, 0.6 + (global_ratio * 0.3))
            return RoutingDecision(
                query_type=QueryType.GLOBAL,
                confidence=confidence,
                reasoning=f"Global patterns matched: {global_matches} (ratio: {global_ratio:.2f})",
                global_weight=1.0,
                local_weight=0.0,
                classification_method="rule_based",
                global_matches=global_matches,
                local_matches=local_matches,
            )
        elif global_ratio <= 0.3:
            # Strongly local
            local_ratio = 1 - global_ratio
            confidence = min(0.9, 0.6 + (local_ratio * 0.3))
            return RoutingDecision(
                query_type=QueryType.LOCAL,
                confidence=confidence,
                reasoning=f"Local patterns matched: {local_matches} (ratio: {local_ratio:.2f})",
                global_weight=0.0,
                local_weight=1.0,
                classification_method="rule_based",
                global_matches=global_matches,
                local_matches=local_matches,
            )
        else:
            # Mixed patterns - hybrid
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                confidence=0.6,
                reasoning=f"Mixed patterns: global={global_matches}, local={local_matches}",
                global_weight=global_ratio,
                local_weight=1 - global_ratio,
                classification_method="rule_based",
                global_matches=global_matches,
                local_matches=local_matches,
            )

    async def _llm_classify(self, query: str) -> RoutingDecision:
        """Classify query using LLM for ambiguous cases.

        Uses the configured LLM model to classify queries that the
        rule-based classifier is uncertain about.

        Args:
            query: The query text to classify

        Returns:
            RoutingDecision from LLM classification
        """
        # Get or create OpenAI client
        if self._openai_client is None:
            llm_adapter = get_llm_adapter(self._settings)
            if llm_adapter.provider not in OPENAI_COMPATIBLE_LLM_PROVIDERS:
                # Fall back to hybrid for non-OpenAI providers
                return RoutingDecision(
                    query_type=QueryType.HYBRID,
                    confidence=0.5,
                    reasoning="LLM provider not compatible with classification",
                    global_weight=0.5,
                    local_weight=0.5,
                    classification_method="llm",
                )
            self._openai_client = AsyncOpenAI(**llm_adapter.openai_kwargs())

        # Call LLM for classification
        prompt = LLM_CLASSIFICATION_PROMPT.format(query=query)

        response = await self._openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for deterministic classification
            max_tokens=100,   # Short response expected
        )

        response_text = response.choices[0].message.content or ""
        return self._parse_llm_response(response_text)

    def _parse_llm_response(self, response: str) -> RoutingDecision:
        """Parse LLM classification response.

        Extracts the query type, confidence, and reasoning from
        the LLM's structured response.

        Args:
            response: Raw LLM response text

        Returns:
            RoutingDecision parsed from the response
        """
        lines = response.strip().split("\n")

        query_type = QueryType.HYBRID  # Default
        confidence = 0.5
        reasoning = "Unable to parse LLM response"

        for line in lines:
            line = line.strip()
            if line.upper().startswith("TYPE:"):
                type_str = line[5:].strip().upper()
                if type_str == "GLOBAL":
                    query_type = QueryType.GLOBAL
                elif type_str == "LOCAL":
                    query_type = QueryType.LOCAL
                else:
                    query_type = QueryType.HYBRID
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif line.upper().startswith("REASONING:"):
                reasoning = line[10:].strip()

        # Calculate weights based on type
        if query_type == QueryType.GLOBAL:
            global_weight, local_weight = 1.0, 0.0
        elif query_type == QueryType.LOCAL:
            global_weight, local_weight = 0.0, 1.0
        else:
            global_weight, local_weight = 0.5, 0.5

        return RoutingDecision(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            global_weight=global_weight,
            local_weight=local_weight,
            classification_method="llm",
        )

    def _combine_decisions(
        self,
        rule_decision: RoutingDecision,
        llm_decision: RoutingDecision,
    ) -> RoutingDecision:
        """Combine rule-based and LLM classification decisions.

        When both classifiers have run, this method combines their
        results using a weighted approach based on confidence scores.

        Args:
            rule_decision: Decision from rule-based classification
            llm_decision: Decision from LLM classification

        Returns:
            Combined RoutingDecision
        """
        # If both agree, boost confidence
        if rule_decision.query_type == llm_decision.query_type:
            combined_confidence = min(
                0.95,
                (rule_decision.confidence + llm_decision.confidence) / 2 + 0.1
            )
            return RoutingDecision(
                query_type=rule_decision.query_type,
                confidence=combined_confidence,
                reasoning=f"Both classifiers agree: {llm_decision.reasoning}",
                global_weight=llm_decision.global_weight,
                local_weight=llm_decision.local_weight,
                classification_method="combined",
                global_matches=rule_decision.global_matches,
                local_matches=rule_decision.local_matches,
            )

        # If they disagree, prefer the one with higher confidence
        if llm_decision.confidence > rule_decision.confidence:
            # Prefer LLM decision but slightly reduce confidence due to disagreement
            return RoutingDecision(
                query_type=llm_decision.query_type,
                confidence=llm_decision.confidence * 0.9,
                reasoning=f"LLM preferred over rules: {llm_decision.reasoning}",
                global_weight=llm_decision.global_weight,
                local_weight=llm_decision.local_weight,
                classification_method="combined",
                global_matches=rule_decision.global_matches,
                local_matches=rule_decision.local_matches,
            )
        else:
            # Prefer rule-based decision
            return RoutingDecision(
                query_type=rule_decision.query_type,
                confidence=rule_decision.confidence * 0.9,
                reasoning=f"Rules preferred over LLM: {rule_decision.reasoning}",
                global_weight=rule_decision.global_weight,
                local_weight=rule_decision.local_weight,
                classification_method="combined",
                global_matches=rule_decision.global_matches,
                local_matches=rule_decision.local_matches,
            )

    def _get_cache_key(self, query: str, use_llm: bool) -> str:
        """Generate a cache key for a routing decision.

        Args:
            query: The query text
            use_llm: Whether LLM classification is enabled

        Returns:
            A hash-based cache key
        """
        # Include use_llm in key since it affects the routing decision
        key_data = f"{query}|{use_llm}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _get_cached_decision(self, cache_key: str) -> Optional[RoutingDecision]:
        """Get a cached routing decision if valid.

        Args:
            cache_key: The cache key to look up

        Returns:
            The cached RoutingDecision if valid, None otherwise
        """
        if cache_key not in self._routing_cache:
            return None

        entry = self._routing_cache[cache_key]
        now = time.time()

        # Check if entry has expired
        if now - entry.timestamp > ROUTING_CACHE_TTL_SECONDS:
            del self._routing_cache[cache_key]
            return None

        # Move to end for LRU ordering
        self._routing_cache.move_to_end(cache_key)

        # Return a copy to avoid mutation issues
        decision = entry.decision
        return RoutingDecision(
            query_type=decision.query_type,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            global_weight=decision.global_weight,
            local_weight=decision.local_weight,
            classification_method=f"{decision.classification_method}_cached",
            processing_time_ms=decision.processing_time_ms,
            global_matches=decision.global_matches,
            local_matches=decision.local_matches,
        )

    def _cache_decision(self, cache_key: str, decision: RoutingDecision) -> None:
        """Cache a routing decision.

        Args:
            cache_key: The cache key
            decision: The routing decision to cache
        """
        # Evict oldest entries if cache is full
        while len(self._routing_cache) >= ROUTING_CACHE_MAX_SIZE:
            self._routing_cache.popitem(last=False)

        self._routing_cache[cache_key] = _CacheEntry(
            decision=decision,
            timestamp=time.time(),
        )

    def clear_cache(self) -> int:
        """Clear the routing decision cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._routing_cache)
        self._routing_cache.clear()
        logger.debug("routing_cache_cleared", entries_cleared=count)
        return count

    @classmethod
    def get_global_patterns(cls) -> list[str]:
        """Get list of global pattern strings for debugging.

        Returns:
            List of regex pattern strings
        """
        return [pattern.pattern for pattern in GLOBAL_PATTERNS]

    @classmethod
    def get_local_patterns(cls) -> list[str]:
        """Get list of local pattern strings for debugging.

        Returns:
            List of regex pattern strings
        """
        return [pattern.pattern for pattern in LOCAL_PATTERNS]

    async def close(self) -> None:
        """Clean up resources, including the OpenAI client.

        Should be called when the QueryRouter instance is no longer needed
        to properly close connection pools.
        """
        if self._openai_client is not None:
            try:
                await self._openai_client.close()
            except Exception as e:
                logger.warning("openai_client_close_error", error=str(e))
            finally:
                self._openai_client = None
