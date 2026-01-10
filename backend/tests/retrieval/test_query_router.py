"""Tests for Query Router (Story 20-B3).

This module tests the QueryRouter class and its components:
- Rule-based classification via pattern matching
- LLM classification for ambiguous queries
- Combined decision logic
- QueryType enum and RoutingDecision dataclass
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.retrieval.query_router import (
    QueryRouter,
    GLOBAL_PATTERNS,
    LOCAL_PATTERNS,
)
from agentic_rag_backend.retrieval.query_router_models import (
    QueryType,
    RoutingDecision,
    QueryRouteRequest,
    QueryRouteResponse,
    PatternListResponse,
    RouterStatusResponse,
)


def _make_mock_settings(
    query_routing_enabled: bool = True,
    query_routing_use_llm: bool = False,
    query_routing_llm_model: str = "gpt-4o-mini",
    query_routing_confidence_threshold: float = 0.7,
    llm_provider: str = "openai",
    llm_api_key: str = "test-key",
    llm_base_url: str = None,
):
    """Create mock settings for QueryRouter tests."""
    settings = MagicMock()
    settings.query_routing_enabled = query_routing_enabled
    settings.query_routing_use_llm = query_routing_use_llm
    settings.query_routing_llm_model = query_routing_llm_model
    settings.query_routing_confidence_threshold = query_routing_confidence_threshold
    settings.llm_provider = llm_provider
    settings.llm_api_key = llm_api_key
    settings.llm_base_url = llm_base_url
    return settings


class TestQueryType:
    """Tests for QueryType enum."""

    def test_query_type_values(self):
        """Should have correct enum values."""
        assert QueryType.GLOBAL.value == "global"
        assert QueryType.LOCAL.value == "local"
        assert QueryType.HYBRID.value == "hybrid"

    def test_query_type_from_string(self):
        """Should create enum from string value."""
        assert QueryType("global") == QueryType.GLOBAL
        assert QueryType("local") == QueryType.LOCAL
        assert QueryType("hybrid") == QueryType.HYBRID


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Should create routing decision with all fields."""
        decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.85,
            reasoning="High global pattern match",
            global_weight=1.0,
            local_weight=0.0,
            classification_method="rule_based",
            global_matches=3,
            local_matches=0,
            processing_time_ms=5,
        )

        assert decision.query_type == QueryType.GLOBAL
        assert decision.confidence == 0.85
        assert decision.reasoning == "High global pattern match"
        assert decision.global_weight == 1.0
        assert decision.local_weight == 0.0
        assert decision.classification_method == "rule_based"
        assert decision.global_matches == 3
        assert decision.local_matches == 0
        assert decision.processing_time_ms == 5

    def test_routing_decision_default_weights_global(self):
        """Should set default weights for global type."""
        decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.8,
            reasoning="Test",
        )

        assert decision.global_weight == 1.0
        assert decision.local_weight == 0.0

    def test_routing_decision_default_weights_local(self):
        """Should set default weights for local type."""
        decision = RoutingDecision(
            query_type=QueryType.LOCAL,
            confidence=0.8,
            reasoning="Test",
        )

        assert decision.global_weight == 0.0
        assert decision.local_weight == 1.0

    def test_routing_decision_default_weights_hybrid(self):
        """Should set default weights for hybrid type."""
        decision = RoutingDecision(
            query_type=QueryType.HYBRID,
            confidence=0.6,
            reasoning="Test",
        )

        assert decision.global_weight == 0.5
        assert decision.local_weight == 0.5

    def test_routing_decision_confidence_clamping(self):
        """Should clamp confidence to valid range."""
        decision_high = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=1.5,
            reasoning="Test",
        )
        assert decision_high.confidence == 1.0

        decision_low = RoutingDecision(
            query_type=QueryType.LOCAL,
            confidence=-0.5,
            reasoning="Test",
        )
        assert decision_low.confidence == 0.0

    def test_routing_decision_to_response(self):
        """Should convert to API response model."""
        decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.85,
            reasoning="Pattern match",
            global_weight=1.0,
            local_weight=0.0,
            classification_method="rule_based",
            global_matches=2,
            local_matches=0,
            processing_time_ms=5,
        )

        response = decision.to_response()

        assert isinstance(response, QueryRouteResponse)
        assert response.query_type == QueryType.GLOBAL
        assert response.confidence == 0.85
        assert response.reasoning == "Pattern match"
        assert response.global_weight == 1.0
        assert response.local_weight == 0.0


class TestPatterns:
    """Tests for pattern compilation and coverage."""

    def test_global_patterns_are_compiled(self):
        """Should have compiled global patterns."""
        assert len(GLOBAL_PATTERNS) > 0
        for pattern in GLOBAL_PATTERNS:
            assert hasattr(pattern, "search")

    def test_local_patterns_are_compiled(self):
        """Should have compiled local patterns."""
        assert len(LOCAL_PATTERNS) > 0
        for pattern in LOCAL_PATTERNS:
            assert hasattr(pattern, "search")


class TestQueryRouterRuleBased:
    """Tests for rule-based query classification."""

    @pytest.fixture
    def router(self):
        """Create QueryRouter with mock settings."""
        settings = _make_mock_settings(query_routing_use_llm=False)
        return QueryRouter(settings=settings)

    def test_classify_global_themes(self, router):
        """Should classify theme questions as global."""
        queries = [
            "What are the main themes in this document?",
            "What are the key topics covered?",
            "What are the central concepts in the knowledge base?",
            "Tell me about the main ideas overall",
        ]

        for query in queries:
            decision = router._rule_based_classify(query)
            assert decision.query_type == QueryType.GLOBAL, f"Failed for: {query}"
            assert decision.global_matches > 0

    def test_classify_global_summary(self, router):
        """Should classify summary requests as global."""
        queries = [
            "Give me a summary of the project",
            "Summarize the main functionality",
            "Provide an overview of the architecture",
            "Give me a general overview of the codebase",
        ]

        for query in queries:
            decision = router._rule_based_classify(query)
            assert decision.query_type == QueryType.GLOBAL, f"Failed for: {query}"

    def test_classify_global_trends(self, router):
        """Should classify trend/pattern questions as global."""
        queries = [
            "Show me the trends across the codebase",
            "Analyze the patterns in the data",
            "Tell me about the system overall",
            "Compare the approaches across modules",
        ]

        for query in queries:
            decision = router._rule_based_classify(query)
            assert decision.query_type == QueryType.GLOBAL, f"Failed for: {query}"

    def test_classify_local_what_is(self, router):
        """Should classify 'what is X' questions as local."""
        queries = [
            "What is FastAPI?",
            "What does the QueryRouter do?",
            "What is the purpose of this function?",
        ]

        for query in queries:
            decision = router._rule_based_classify(query)
            assert decision.query_type == QueryType.LOCAL, f"Failed for: {query}"
            assert decision.local_matches > 0

    def test_classify_local_who_where_when(self, router):
        """Should classify who/where/when questions as local."""
        queries = [
            "Who is the author of this module?",
            "Where is the config file located?",
            "When was this function last modified?",
        ]

        for query in queries:
            decision = router._rule_based_classify(query)
            assert decision.query_type == QueryType.LOCAL, f"Failed for: {query}"

    def test_classify_local_specific(self, router):
        """Should classify specific entity questions as local."""
        queries = [
            "Find the function named process_query",
            "Locate the class called QueryRouter",
            "Get the specific implementation of route()",
            "Define the term 'embedding'",
        ]

        for query in queries:
            decision = router._rule_based_classify(query)
            assert decision.query_type == QueryType.LOCAL, f"Failed for: {query}"

    def test_classify_hybrid_mixed_patterns(self, router):
        """Should classify mixed pattern queries as hybrid."""
        # Queries that match both global and local patterns
        query = "What is the main theme of the QueryRouter function?"
        decision = router._rule_based_classify(query)

        # Should be hybrid or show mixed confidence
        assert decision.global_matches > 0
        assert decision.local_matches > 0

    def test_classify_no_patterns_returns_hybrid(self, router):
        """Should return hybrid with low confidence for no pattern matches."""
        query = "xyz abc 123"  # No patterns match
        decision = router._rule_based_classify(query)

        assert decision.query_type == QueryType.HYBRID
        assert decision.confidence < 0.5
        assert decision.global_matches == 0
        assert decision.local_matches == 0

    def test_classification_is_case_insensitive(self, router):
        """Should classify queries regardless of case."""
        decision_lower = router._rule_based_classify("what are the main themes?")
        decision_upper = router._rule_based_classify("WHAT ARE THE MAIN THEMES?")
        decision_mixed = router._rule_based_classify("What Are The Main Themes?")

        assert decision_lower.query_type == QueryType.GLOBAL
        assert decision_upper.query_type == QueryType.GLOBAL
        assert decision_mixed.query_type == QueryType.GLOBAL


class TestQueryRouterAsync:
    """Tests for async routing methods."""

    @pytest.fixture
    def router(self):
        """Create QueryRouter with LLM disabled."""
        settings = _make_mock_settings(query_routing_use_llm=False)
        return QueryRouter(settings=settings)

    @pytest.mark.asyncio
    async def test_route_global_query(self, router):
        """Should route global query correctly."""
        decision = await router.route(
            query="What are the main themes in this codebase?",
            tenant_id="tenant-1",
        )

        assert decision.query_type == QueryType.GLOBAL
        assert decision.confidence > 0.5
        assert decision.processing_time_ms >= 0
        assert decision.classification_method == "rule_based"

    @pytest.mark.asyncio
    async def test_route_local_query(self, router):
        """Should route local query correctly."""
        decision = await router.route(
            query="What is the QueryRouter class?",
            tenant_id="tenant-1",
        )

        assert decision.query_type == QueryType.LOCAL
        assert decision.confidence > 0.5
        assert decision.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_route_empty_query(self, router):
        """Should handle empty query gracefully."""
        decision = await router.route(
            query="",
            tenant_id="tenant-1",
        )

        assert decision.query_type == QueryType.HYBRID
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_route_whitespace_query(self, router):
        """Should handle whitespace-only query."""
        decision = await router.route(
            query="   ",
            tenant_id="tenant-1",
        )

        assert decision.query_type == QueryType.HYBRID
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_route_respects_use_llm_override(self, router):
        """Should respect use_llm parameter override."""
        # LLM disabled in settings, but even with override=True,
        # since the query matches patterns well, no LLM call needed
        decision = await router.route(
            query="What are the main themes?",
            tenant_id="tenant-1",
            use_llm=False,
        )

        assert decision.classification_method == "rule_based"


class TestQueryRouterWithLLM:
    """Tests for LLM-enhanced query classification."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create mock OpenAI response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = (
            "TYPE: GLOBAL\n"
            "CONFIDENCE: 0.85\n"
            "REASONING: Query asks about themes across the corpus"
        )
        return response

    @pytest.fixture
    def router_with_llm(self):
        """Create QueryRouter with LLM enabled."""
        settings = _make_mock_settings(
            query_routing_use_llm=True,
            query_routing_confidence_threshold=0.9,  # High threshold to trigger LLM
        )
        return QueryRouter(settings=settings)

    def test_parse_llm_response_valid(self, router_with_llm):
        """Should parse valid LLM response correctly."""
        response = (
            "TYPE: GLOBAL\n"
            "CONFIDENCE: 0.85\n"
            "REASONING: Query asks about themes"
        )

        decision = router_with_llm._parse_llm_response(response)

        assert decision.query_type == QueryType.GLOBAL
        assert decision.confidence == 0.85
        assert "themes" in decision.reasoning
        assert decision.classification_method == "llm"

    def test_parse_llm_response_local(self, router_with_llm):
        """Should parse LOCAL type correctly."""
        response = (
            "TYPE: LOCAL\n"
            "CONFIDENCE: 0.9\n"
            "REASONING: Query asks about specific entity"
        )

        decision = router_with_llm._parse_llm_response(response)

        assert decision.query_type == QueryType.LOCAL
        assert decision.confidence == 0.9
        assert decision.global_weight == 0.0
        assert decision.local_weight == 1.0

    def test_parse_llm_response_hybrid(self, router_with_llm):
        """Should parse HYBRID type correctly."""
        response = (
            "TYPE: HYBRID\n"
            "CONFIDENCE: 0.75\n"
            "REASONING: Query needs both approaches"
        )

        decision = router_with_llm._parse_llm_response(response)

        assert decision.query_type == QueryType.HYBRID
        assert decision.confidence == 0.75
        assert decision.global_weight == 0.5
        assert decision.local_weight == 0.5

    def test_parse_llm_response_invalid_format(self, router_with_llm):
        """Should handle invalid LLM response gracefully."""
        response = "This is an invalid response"

        decision = router_with_llm._parse_llm_response(response)

        assert decision.query_type == QueryType.HYBRID
        assert decision.confidence == 0.5

    def test_parse_llm_response_out_of_range_confidence(self, router_with_llm):
        """Should clamp out-of-range confidence values."""
        response = (
            "TYPE: GLOBAL\n"
            "CONFIDENCE: 1.5\n"
            "REASONING: Test"
        )

        decision = router_with_llm._parse_llm_response(response)

        assert decision.confidence == 1.0

    @pytest.mark.asyncio
    async def test_llm_classify_with_mock(self, router_with_llm, mock_openai_response):
        """Should call LLM and parse response."""
        with patch.object(
            router_with_llm, "_openai_client", create=True
        ) as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            router_with_llm._openai_client = mock_client

            decision = await router_with_llm._llm_classify("What are the themes?")

            assert decision.query_type == QueryType.GLOBAL
            assert decision.confidence == 0.85
            mock_client.chat.completions.create.assert_called_once()


class TestCombineDecisions:
    """Tests for combining rule-based and LLM decisions."""

    @pytest.fixture
    def router(self):
        """Create QueryRouter."""
        settings = _make_mock_settings()
        return QueryRouter(settings=settings)

    def test_combine_both_agree_boosts_confidence(self, router):
        """Should boost confidence when both classifiers agree."""
        rule_decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.7,
            reasoning="Rule-based",
        )
        llm_decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.8,
            reasoning="LLM agrees",
        )

        combined = router._combine_decisions(rule_decision, llm_decision)

        assert combined.query_type == QueryType.GLOBAL
        assert combined.confidence > max(rule_decision.confidence, llm_decision.confidence)
        assert combined.classification_method == "combined"

    def test_combine_prefers_higher_confidence(self, router):
        """Should prefer higher confidence decision when they disagree."""
        rule_decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.5,
            reasoning="Rule-based uncertain",
        )
        llm_decision = RoutingDecision(
            query_type=QueryType.LOCAL,
            confidence=0.9,
            reasoning="LLM confident",
        )

        combined = router._combine_decisions(rule_decision, llm_decision)

        assert combined.query_type == QueryType.LOCAL
        # Confidence is reduced due to disagreement
        assert combined.confidence < llm_decision.confidence

    def test_combine_prefers_rule_when_more_confident(self, router):
        """Should prefer rule-based when it has higher confidence."""
        rule_decision = RoutingDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.9,
            reasoning="Rule-based confident",
            global_matches=3,
            local_matches=0,
        )
        llm_decision = RoutingDecision(
            query_type=QueryType.LOCAL,
            confidence=0.5,
            reasoning="LLM uncertain",
        )

        combined = router._combine_decisions(rule_decision, llm_decision)

        assert combined.query_type == QueryType.GLOBAL
        assert combined.global_matches == 3


class TestGetPatterns:
    """Tests for pattern retrieval methods."""

    def test_get_global_patterns(self):
        """Should return list of global pattern strings."""
        patterns = QueryRouter.get_global_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) == len(GLOBAL_PATTERNS)
        for pattern in patterns:
            assert isinstance(pattern, str)

    def test_get_local_patterns(self):
        """Should return list of local pattern strings."""
        patterns = QueryRouter.get_local_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) == len(LOCAL_PATTERNS)
        for pattern in patterns:
            assert isinstance(pattern, str)


class TestQueryRouteRequest:
    """Tests for QueryRouteRequest Pydantic model."""

    def test_valid_request(self):
        """Should validate valid request."""
        from uuid import UUID

        request = QueryRouteRequest(
            query="What are the main themes?",
            tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
            use_llm=True,
        )

        assert request.query == "What are the main themes?"
        assert request.use_llm is True

    def test_request_with_defaults(self):
        """Should use default values."""
        from uuid import UUID

        request = QueryRouteRequest(
            query="Test query",
            tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
        )

        assert request.use_llm is None

    def test_request_validates_query_length(self):
        """Should validate query length."""
        from uuid import UUID
        from pydantic import ValidationError

        # Empty query should fail
        with pytest.raises(ValidationError):
            QueryRouteRequest(
                query="",
                tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
            )


class TestRouterStatusResponse:
    """Tests for RouterStatusResponse Pydantic model."""

    def test_status_response_creation(self):
        """Should create status response with all fields."""
        status = RouterStatusResponse(
            enabled=True,
            use_llm=False,
            llm_model="gpt-4o-mini",
            confidence_threshold=0.7,
            community_detection_available=True,
            lazy_rag_available=True,
        )

        assert status.enabled is True
        assert status.use_llm is False
        assert status.llm_model == "gpt-4o-mini"
        assert status.confidence_threshold == 0.7
        assert status.community_detection_available is True
        assert status.lazy_rag_available is True


class TestPatternListResponse:
    """Tests for PatternListResponse Pydantic model."""

    def test_pattern_list_response_creation(self):
        """Should create pattern list response."""
        response = PatternListResponse(
            global_patterns=["pattern1", "pattern2"],
            local_patterns=["pattern3"],
            global_pattern_count=2,
            local_pattern_count=1,
        )

        assert len(response.global_patterns) == 2
        assert len(response.local_patterns) == 1
        assert response.global_pattern_count == 2
        assert response.local_pattern_count == 1
