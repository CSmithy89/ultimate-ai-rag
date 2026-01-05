"""Integration tests for the full retrieval pipeline.

Story 19-F1: Add Full Retrieval Pipeline Integration Test

Tests the complete pipeline: embed -> search -> rerank -> grade -> fallback
Covers edge cases: empty results, low scores, timeouts, fallback trigger.

These tests require:
- INTEGRATION_TESTS=1 environment variable
- Running PostgreSQL, Neo4j, and Redis services
- Valid embedding API key (or mocked embeddings)

CI Configuration:
- Tests should run with 60-second timeout per test
- Configure in CI via pytest --timeout=60 (requires pytest-timeout)
- Or set via pytest.ini: timeout = 60
"""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.protocols.a2a import A2ASessionManager
from agentic_rag_backend.protocols.mcp import MCPToolRegistry
from agentic_rag_backend.retrieval.grader import (
    FallbackStrategy,
    GraderResult,
    HeuristicGrader,
    RetrievalGrader,
    RetrievalHit,
    WebSearchFallback,
)
from agentic_rag_backend.retrieval.graph_traversal import GraphTraversalService
from agentic_rag_backend.retrieval.hybrid_synthesis import build_hybrid_prompt
from agentic_rag_backend.retrieval.reranking import (
    FlashRankRerankerClient,
    RerankedHit,
    RerankerProviderAdapter,
    RerankerProviderType,
    create_reranker_client,
)
from agentic_rag_backend.retrieval.types import GraphTraversalResult, VectorHit
from agentic_rag_backend.retrieval.vector_search import VectorSearchService
from agentic_rag_backend.schemas import PlanStep

# Note: Tests that use mocks don't require database cleanup fixtures.
# Database-hitting tests (marked with requires_databases) use integration_cleanup fixture.
# All tests should complete within 60 seconds to prevent CI hangs.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(60),  # Prevent hanging tests in CI
]

# Test constants
EMBEDDING_DIM = 1536
VALID_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def _content_hash(content: str) -> str:
    """Generate a content hash for document deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _embedding(seed: float = 1.0) -> list[float]:
    """Generate a deterministic embedding vector for testing."""
    vector = [0.0] * EMBEDDING_DIM
    vector[0] = seed
    vector[1] = seed * 0.5
    vector[2] = seed * 0.25
    return vector


def _create_sample_hits(count: int = 3, score_base: float = 0.8) -> list[VectorHit]:
    """Create sample VectorHit objects for testing."""
    hits = []
    for i in range(count):
        hits.append(
            VectorHit(
                chunk_id=f"chunk-{i}",
                document_id=f"doc-{i}",
                content=f"Sample content for chunk {i}. This is test data.",
                similarity=score_base - (i * 0.1),
                metadata={"source": "test", "index": i},
            )
        )
    return hits


def _create_retrieval_hits(
    count: int = 3, score_base: float = 0.8
) -> list[RetrievalHit]:
    """Create sample RetrievalHit objects for grading."""
    hits = []
    for i in range(count):
        hits.append(
            RetrievalHit(
                content=f"Sample content for retrieval hit {i}. This is test data.",
                score=score_base - (i * 0.1),
                metadata={"source": "test", "index": i},
            )
        )
    return hits


class DummyOrchestrator:
    """Dummy orchestrator for MCP testing."""

    def __init__(self, postgres_client: PostgresClient | None = None):
        self._postgres = postgres_client
        self.vector_search_service = None

    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> SimpleNamespace:
        """Simulate orchestrator query execution."""
        return SimpleNamespace(
            answer=f"answer:{query}",
            plan=[PlanStep(step="retrieve", status="completed")],
            thoughts=["Retrieved relevant documents"],
            retrieval_strategy=SimpleNamespace(value="hybrid"),
            trajectory_id=uuid4(),
            evidence=None,
        )


class DummyNeo4j:
    """Dummy Neo4j client for MCP testing."""

    async def get_visualization_stats(self, tenant_id: str) -> dict[str, Any]:
        """Return mock graph stats."""
        return {"nodes": 5, "edges": 10, "documents": 3}


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def generate_embedding(
        self,
        text: str,
        tenant_id: str | None = None,
    ) -> list[float]:
        """Generate a deterministic embedding based on text hash."""
        # Create a deterministic embedding based on text content
        hash_val = hashlib.md5(text.encode()).hexdigest()
        seed = int(hash_val[:8], 16) / (16**8)
        return _embedding(seed)


class TestRetrievalPipeline:
    """Integration tests for the full retrieval pipeline.

    Tests cover: embed -> search -> rerank -> grade -> fallback
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_vector_only(
        self,
        postgres_client: PostgresClient,
        integration_cleanup: str,
    ) -> None:
        """Vector search -> rerank -> grade -> response.

        This test verifies the complete vector-only retrieval pipeline:
        1. Insert test documents with embeddings
        2. Perform vector search
        3. Rerank results
        4. Grade results for quality
        5. Verify response quality
        """
        tenant_id = integration_cleanup
        tenant_uuid = UUID(tenant_id)

        # Step 1: Insert test documents
        test_content = "Python is a versatile programming language used for data science and web development."
        doc_id = await postgres_client.create_document(
            tenant_id=tenant_uuid,
            source_type="text",
            content_hash=_content_hash(test_content),
        )

        query_embedding = _embedding(0.9)
        chunk_id = await postgres_client.create_chunk(
            tenant_id=tenant_uuid,
            document_id=doc_id,
            content=test_content,
            chunk_index=0,
            token_count=15,
            embedding=query_embedding,
        )

        # Step 2: Vector search
        mock_embedding_gen = MockEmbeddingGenerator()
        vector_service = VectorSearchService(
            postgres=postgres_client,
            embedding_generator=mock_embedding_gen,
            limit=10,
            similarity_threshold=0.1,
            timeout_seconds=10.0,
            cache_ttl_seconds=0,  # Disable cache for testing
        )

        # Use same embedding as stored to ensure match
        with patch.object(
            mock_embedding_gen,
            "generate_embedding",
            return_value=query_embedding,
        ):
            vector_service.embedding_generator = mock_embedding_gen
            vector_hits = await vector_service.search(
                query="Python programming language",
                tenant_id=tenant_id,
            )

        assert len(vector_hits) >= 1
        assert any(hit.chunk_id == str(chunk_id) for hit in vector_hits)

        # Step 3: Rerank results (using mocked FlashRank)
        mock_rerank_results = [
            {"id": 0, "text": vector_hits[0].content, "score": 0.95}
        ]

        with patch("flashrank.Ranker") as mock_ranker_cls:
            mock_ranker = MagicMock()
            mock_ranker.rerank = MagicMock(return_value=mock_rerank_results)
            mock_ranker_cls.return_value = mock_ranker

            adapter = RerankerProviderAdapter(
                provider=RerankerProviderType.FLASHRANK,
                api_key=None,
                model="ms-marco-MiniLM-L-12-v2",
            )
            reranker = create_reranker_client(adapter)
            reranked = await reranker.rerank(
                query="Python programming language",
                hits=vector_hits,
                top_k=5,
            )

        assert len(reranked) >= 1
        assert reranked[0].rerank_score > 0

        # Step 4: Grade results
        retrieval_hits = [
            RetrievalHit(
                content=hit.hit.content,
                score=hit.rerank_score,
                metadata={"chunk_id": hit.hit.chunk_id},
            )
            for hit in reranked
        ]

        grader = HeuristicGrader(top_k=5)
        result = await grader.grade(
            query="Python programming language",
            hits=retrieval_hits,
            threshold=0.5,
        )

        # Step 5: Verify response quality
        assert result.score > 0
        assert result.grading_time_ms >= 0
        # With high rerank scores, should pass threshold
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_full_pipeline_hybrid(
        self,
        postgres_client: PostgresClient,
        neo4j_client: Neo4jClient,
        integration_cleanup: str,
    ) -> None:
        """Vector + Graph -> merge -> rerank -> grade -> response.

        This test verifies the hybrid retrieval pipeline:
        1. Insert test documents in both vector store and graph
        2. Perform vector search
        3. Perform graph traversal
        4. Merge results using hybrid synthesis
        5. Grade final results
        """
        tenant_id = integration_cleanup
        tenant_uuid = UUID(tenant_id)

        # Step 1a: Insert vector data
        vector_content = "FastAPI is a modern web framework that uses Starlette and Pydantic."
        doc_id = await postgres_client.create_document(
            tenant_id=tenant_uuid,
            source_type="text",
            content_hash=_content_hash(vector_content),
        )

        vector_embedding = _embedding(0.85)
        chunk_id = await postgres_client.create_chunk(
            tenant_id=tenant_uuid,
            document_id=doc_id,
            content=vector_content,
            chunk_index=0,
            token_count=12,
            embedding=vector_embedding,
        )

        # Step 1b: Insert graph data
        entity_fastapi = str(uuid4())
        entity_starlette = str(uuid4())

        await neo4j_client.create_entity(
            entity_id=entity_fastapi,
            tenant_id=tenant_id,
            name="FastAPI",
            entity_type="Framework",
        )
        await neo4j_client.create_entity(
            entity_id=entity_starlette,
            tenant_id=tenant_id,
            name="Starlette",
            entity_type="Library",
        )
        await neo4j_client.create_relationship(
            source_id=entity_fastapi,
            target_id=entity_starlette,
            relationship_type="USES",
            tenant_id=tenant_id,
            confidence=0.95,
        )

        # Step 2: Vector search
        rows = await postgres_client.search_similar_chunks(
            tenant_id=tenant_uuid,
            embedding=vector_embedding,
            similarity_threshold=0.1,
            limit=5,
        )

        vector_hits = [
            VectorHit(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                content=row["content"],
                similarity=float(row["similarity"]),
                metadata=row.get("metadata"),
            )
            for row in rows
        ]

        assert len(vector_hits) >= 1

        # Step 3: Graph traversal
        traversal = GraphTraversalService(
            neo4j=neo4j_client,
            max_hops=2,
            path_limit=5,
            entity_limit=10,
            cache_ttl_seconds=0,  # Disable cache for testing
        )
        graph_result = await traversal.traverse(
            query="FastAPI uses Starlette",
            tenant_id=tenant_id,
        )

        assert graph_result.nodes
        assert graph_result.paths

        # Step 4: Merge using hybrid synthesis
        hybrid_prompt = build_hybrid_prompt(
            query="FastAPI web framework",
            vector_hits=vector_hits,
            graph_result=graph_result,
        )

        assert f"[vector:{chunk_id}]" in hybrid_prompt
        assert "[graph:" in hybrid_prompt
        assert "FastAPI" in hybrid_prompt

        # Step 5: Grade merged results
        all_hits = [
            RetrievalHit(
                content=hit.content,
                score=hit.similarity,
                metadata={"source": "vector"},
            )
            for hit in vector_hits
        ]
        # Add graph-derived hits
        for node in graph_result.nodes:
            all_hits.append(
                RetrievalHit(
                    content=f"{node.name} ({node.type})",
                    score=0.8,  # Default score for graph nodes
                    metadata={"source": "graph", "node_id": node.id},
                )
            )

        grader = HeuristicGrader(top_k=5)
        result = await grader.grade(
            query="FastAPI web framework",
            hits=all_hits,
            threshold=0.4,
        )

        assert result.score > 0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fallback_on_low_score(self) -> None:
        """Low grader score triggers Tavily fallback.

        Tests that when the grader score is below threshold,
        the fallback handler is triggered appropriately.
        """
        # Create hits with low scores
        low_score_hits = _create_retrieval_hits(count=3, score_base=0.2)

        # Create mock fallback handler
        mock_fallback = AsyncMock()
        mock_fallback.execute.return_value = [
            RetrievalHit(
                content="Fallback result from web search with relevant info.",
                score=0.85,
                metadata={"source": "tavily_web_search", "url": "https://example.com"},
            )
        ]

        # Create grader with fallback enabled
        base_grader = HeuristicGrader(top_k=3)
        grader = RetrievalGrader(
            grader=base_grader,
            threshold=0.5,  # Set threshold higher than hit scores
            fallback_enabled=True,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=mock_fallback,
        )

        # Grade and trigger fallback
        result, fallback_hits = await grader.grade_and_fallback(
            query="test query for fallback",
            hits=low_score_hits,
            tenant_id=VALID_TENANT_ID,
        )

        # Verify fallback was triggered
        assert result.passed is False
        assert result.fallback_triggered is True
        assert result.fallback_strategy == FallbackStrategy.WEB_SEARCH
        assert len(fallback_hits) == 1
        assert fallback_hits[0].content == "Fallback result from web search with relevant info."
        assert fallback_hits[0].metadata["source"] == "tavily_web_search"

        # Verify fallback handler was called with correct args
        mock_fallback.execute.assert_called_once_with(
            "test query for fallback",
            VALID_TENANT_ID,
        )

    @pytest.mark.asyncio
    async def test_fallback_on_empty_results(self) -> None:
        """No results triggers fallback.

        Tests that when vector search returns no results,
        the grader fails and triggers fallback.
        """
        # Create mock fallback handler
        mock_fallback = AsyncMock()
        mock_fallback.execute.return_value = [
            RetrievalHit(
                content="Fallback result for empty query.",
                score=0.75,
                metadata={"source": "tavily_web_search"},
            )
        ]

        # Create grader with fallback enabled
        base_grader = HeuristicGrader(top_k=3)
        grader = RetrievalGrader(
            grader=base_grader,
            threshold=0.3,
            fallback_enabled=True,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=mock_fallback,
        )

        # Grade with empty hits
        result, fallback_hits = await grader.grade_and_fallback(
            query="obscure query with no matches",
            hits=[],  # Empty results
            tenant_id=VALID_TENANT_ID,
        )

        # Verify empty results trigger fallback
        assert result.score == 0.0
        assert result.passed is False
        assert result.fallback_triggered is True
        assert len(fallback_hits) == 1
        mock_fallback.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_a2a_query_endpoint(self) -> None:
        """A2A API endpoint returns valid response.

        Tests the A2A session creation and message flow.
        """
        manager = A2ASessionManager(max_sessions_total=10)

        # Create session
        session = await manager.create_session(VALID_TENANT_ID)
        assert "session_id" in session
        assert session["tenant_id"] == VALID_TENANT_ID

        session_id = session["session_id"]

        # Add message to session
        updated = await manager.add_message(
            session_id=session_id,
            tenant_id=VALID_TENANT_ID,
            sender="agent",
            content="Query: What is Python?",
            metadata={"capability": "knowledge.query"},
        )

        assert len(updated["messages"]) == 1
        assert updated["messages"][0]["content"] == "Query: What is Python?"
        assert updated["messages"][0]["sender"] == "agent"

        # Fetch session
        fetched = await manager.get_session(session_id)
        assert fetched is not None
        assert fetched["session_id"] == session_id
        assert fetched["tenant_id"] == VALID_TENANT_ID

        # Add response message
        response = await manager.add_message(
            session_id=session_id,
            tenant_id=VALID_TENANT_ID,
            sender="rag_agent",
            content="Python is a programming language.",
            metadata={"strategy": "hybrid"},
        )

        assert len(response["messages"]) == 2

    @pytest.mark.asyncio
    async def test_mcp_tool_invocation(self) -> None:
        """MCP knowledge.query tool works end-to-end.

        Tests that the MCP tool registry correctly invokes
        the knowledge.query tool and returns valid results.
        """
        # Create MCP registry with dummy orchestrator
        orchestrator = DummyOrchestrator()
        neo4j = DummyNeo4j()
        registry = MCPToolRegistry(
            orchestrator=orchestrator,
            neo4j=neo4j,
            timeout_seconds=30.0,
        )

        # Verify tool is registered
        tools = registry.list_tools()
        tool_names = [tool["name"] for tool in tools]
        assert "knowledge.query" in tool_names
        assert "knowledge.graph_stats" in tool_names

        # Test knowledge.query tool
        query_result = await registry.call_tool(
            "knowledge.query",
            {
                "query": "What is Python?",
                "tenant_id": VALID_TENANT_ID,
            },
        )

        assert "answer" in query_result
        assert query_result["answer"] == "answer:What is Python?"
        assert "retrieval_strategy" in query_result
        assert query_result["retrieval_strategy"] == "hybrid"
        assert "plan" in query_result
        assert len(query_result["plan"]) >= 1

        # Test knowledge.graph_stats tool
        stats_result = await registry.call_tool(
            "knowledge.graph_stats",
            {"tenant_id": VALID_TENANT_ID},
        )

        assert "nodes" in stats_result
        assert stats_result["nodes"] == 5
        assert "edges" in stats_result
        assert stats_result["edges"] == 10

    @pytest.mark.asyncio
    async def test_pipeline_with_timeout(
        self,
        postgres_client: PostgresClient,
        integration_cleanup: str,
    ) -> None:
        """Test pipeline handles timeout gracefully.

        Tests that the pipeline properly handles timeout scenarios
        and raises appropriate exceptions.
        """
        tenant_id = integration_cleanup

        # Create vector search service with very short timeout
        mock_embedding_gen = MockEmbeddingGenerator()

        # Mock a slow embedding generation
        async def slow_embedding(*args, **kwargs):
            await asyncio.sleep(2.0)  # Simulate slow response
            return _embedding(0.5)

        with patch.object(
            mock_embedding_gen,
            "generate_embedding",
            side_effect=slow_embedding,
        ):
            vector_service = VectorSearchService(
                postgres=postgres_client,
                embedding_generator=mock_embedding_gen,
                limit=10,
                similarity_threshold=0.1,
                timeout_seconds=0.1,  # Very short timeout
                cache_ttl_seconds=0,
            )

            with pytest.raises(asyncio.TimeoutError):
                await vector_service.search(
                    query="test query",
                    tenant_id=tenant_id,
                )

    @pytest.mark.asyncio
    async def test_reranking_improves_results(self) -> None:
        """Test that reranking actually reorders results based on relevance.

        Verifies that the reranker produces different ordering
        than the original similarity scores when appropriate.
        """
        # Create hits where similarity order differs from semantic relevance
        hits = [
            VectorHit(
                chunk_id="chunk-0",
                document_id="doc-0",
                content="JavaScript is a scripting language.",  # Less relevant
                similarity=0.95,  # Higher similarity
                metadata={},
            ),
            VectorHit(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Python is excellent for machine learning and data science.",
                similarity=0.85,  # Lower similarity
                metadata={},
            ),
            VectorHit(
                chunk_id="chunk-2",
                document_id="doc-2",
                content="Python syntax is clean and readable.",
                similarity=0.80,
                metadata={},
            ),
        ]

        # Mock FlashRank to prefer Python-related content for a Python query
        mock_rerank_results = [
            {"id": 1, "text": hits[1].content, "score": 0.95},  # Python ML -> top
            {"id": 2, "text": hits[2].content, "score": 0.90},  # Python syntax
            {"id": 0, "text": hits[0].content, "score": 0.60},  # JavaScript -> bottom
        ]

        with patch("flashrank.Ranker") as mock_ranker_cls:
            mock_ranker = MagicMock()
            mock_ranker.rerank = MagicMock(return_value=mock_rerank_results)
            mock_ranker_cls.return_value = mock_ranker

            reranker = FlashRankRerankerClient(model="ms-marco-MiniLM-L-12-v2")
            reranked = await reranker.rerank(
                query="Python machine learning",
                hits=hits,
                top_k=3,
            )

        # Verify reranking changed the order
        assert len(reranked) == 3
        assert reranked[0].hit.chunk_id == "chunk-1"  # Python ML is now first
        assert reranked[0].rerank_score == 0.95
        assert reranked[2].hit.chunk_id == "chunk-0"  # JavaScript is now last

    @pytest.mark.asyncio
    async def test_grader_with_cross_encoder_fallback(self) -> None:
        """Test grader configuration with different fallback strategies.

        Tests multiple fallback scenarios:
        - Web search fallback
        - Expanded query fallback
        - Disabled fallback
        """
        base_grader = HeuristicGrader(top_k=3)
        low_score_hits = _create_retrieval_hits(count=3, score_base=0.2)

        # Test 1: Web search fallback
        web_fallback = AsyncMock()
        web_fallback.execute.return_value = [
            RetrievalHit(content="Web result", score=0.8)
        ]

        grader_web = RetrievalGrader(
            grader=base_grader,
            threshold=0.5,
            fallback_enabled=True,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=web_fallback,
        )

        result_web, hits_web = await grader_web.grade_and_fallback(
            "query", low_score_hits
        )
        assert result_web.fallback_strategy == FallbackStrategy.WEB_SEARCH
        assert len(hits_web) == 1

        # Test 2: Expanded query fallback
        expand_fallback = AsyncMock()
        expand_fallback.execute.return_value = []

        grader_expand = RetrievalGrader(
            grader=base_grader,
            threshold=0.5,
            fallback_enabled=True,
            fallback_strategy=FallbackStrategy.EXPANDED_QUERY,
            fallback_handler=expand_fallback,
        )

        result_expand, hits_expand = await grader_expand.grade_and_fallback(
            "query", low_score_hits
        )
        assert result_expand.fallback_strategy == FallbackStrategy.EXPANDED_QUERY

        # Test 3: Disabled fallback
        grader_disabled = RetrievalGrader(
            grader=base_grader,
            threshold=0.5,
            fallback_enabled=False,
            fallback_strategy=FallbackStrategy.WEB_SEARCH,
            fallback_handler=web_fallback,
        )

        web_fallback.reset_mock()
        result_disabled, hits_disabled = await grader_disabled.grade_and_fallback(
            "query", low_score_hits
        )
        assert result_disabled.passed is False
        assert len(hits_disabled) == 0
        web_fallback.execute.assert_not_called()


class TestRetrievalPipelineEdgeCases:
    """Edge case tests for the retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self) -> None:
        """Test handling of empty or whitespace-only queries."""
        grader = HeuristicGrader(top_k=3)

        result = await grader.grade(
            query="",
            hits=[],
            threshold=0.5,
        )

        assert result.score == 0.0
        assert result.passed is False
        assert result.fallback_triggered is True

    @pytest.mark.asyncio
    async def test_very_long_content_handling(self) -> None:
        """Test handling of very long content in hits."""
        long_content = "A" * 10000  # Very long content
        hits = [
            RetrievalHit(content=long_content, score=0.8),
        ]

        grader = HeuristicGrader(top_k=3)
        result = await grader.grade(
            query="test query",
            hits=hits,
            threshold=0.5,
        )

        # Should still work with long content
        assert result.score > 0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self) -> None:
        """Test handling of special characters in queries."""
        grader = HeuristicGrader(top_k=3)
        hits = _create_retrieval_hits(count=3, score_base=0.8)

        # Query with special characters
        special_query = "What is Python's __init__() method? <script>alert('xss')</script>"

        result = await grader.grade(
            query=special_query,
            hits=hits,
            threshold=0.5,
        )

        # Should handle gracefully
        assert result.score > 0
        assert result.grading_time_ms >= 0

    @pytest.mark.asyncio
    async def test_concurrent_grading(self) -> None:
        """Test concurrent grading operations don't interfere."""
        grader = HeuristicGrader(top_k=3)

        async def grade_query(query_id: int, score_base: float):
            hits = _create_retrieval_hits(count=3, score_base=score_base)
            return await grader.grade(
                query=f"query {query_id}",
                hits=hits,
                threshold=0.5,
            )

        # Run multiple grading operations concurrently
        results = await asyncio.gather(
            grade_query(1, 0.9),  # Should pass
            grade_query(2, 0.3),  # Should fail
            grade_query(3, 0.7),  # Should pass
            grade_query(4, 0.2),  # Should fail
        )

        assert results[0].passed is True
        assert results[1].passed is False
        assert results[2].passed is True
        assert results[3].passed is False

    @pytest.mark.asyncio
    async def test_mcp_tool_not_found(self) -> None:
        """Test MCP registry handles unknown tool gracefully."""
        from agentic_rag_backend.protocols.mcp import MCPToolNotFoundError

        orchestrator = DummyOrchestrator()
        neo4j = DummyNeo4j()
        registry = MCPToolRegistry(orchestrator=orchestrator, neo4j=neo4j)

        with pytest.raises(MCPToolNotFoundError):
            await registry.call_tool(
                "nonexistent.tool",
                {"tenant_id": VALID_TENANT_ID},
            )

    @pytest.mark.asyncio
    async def test_a2a_session_tenant_isolation(self) -> None:
        """Test A2A sessions are isolated per tenant."""
        manager = A2ASessionManager(max_sessions_total=10)

        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"

        # Create sessions for different tenants
        session_a = await manager.create_session(tenant_a)
        session_b = await manager.create_session(tenant_b)

        assert session_a["tenant_id"] == tenant_a
        assert session_b["tenant_id"] == tenant_b
        assert session_a["session_id"] != session_b["session_id"]

        # Add message to tenant A's session
        await manager.add_message(
            session_id=session_a["session_id"],
            tenant_id=tenant_a,
            sender="agent",
            content="Tenant A message",
        )

        # Verify tenant B can't access tenant A's session messages
        session_a_fetched = await manager.get_session(session_a["session_id"])
        session_b_fetched = await manager.get_session(session_b["session_id"])

        assert len(session_a_fetched["messages"]) == 1
        assert len(session_b_fetched["messages"]) == 0

    @pytest.mark.asyncio
    async def test_hybrid_prompt_with_empty_graph(self) -> None:
        """Test hybrid prompt generation with empty graph results."""
        vector_hits = _create_sample_hits(count=2)
        empty_graph = GraphTraversalResult(nodes=[], edges=[], paths=[])

        prompt = build_hybrid_prompt(
            query="test query",
            vector_hits=vector_hits,
            graph_result=empty_graph,
        )

        # Should include vector evidence but no graph evidence
        assert "Vector Evidence:" in prompt
        assert "[vector:chunk-0]" in prompt
        assert "Graph Nodes:" not in prompt
        assert "Graph Paths:" not in prompt

    @pytest.mark.asyncio
    async def test_hybrid_prompt_with_empty_vector(self) -> None:
        """Test hybrid prompt generation with empty vector results."""
        from agentic_rag_backend.retrieval.types import GraphNode, GraphPath

        graph_result = GraphTraversalResult(
            nodes=[
                GraphNode(id="node-1", name="Python", type="Language"),
                GraphNode(id="node-2", name="Django", type="Framework"),
            ],
            edges=[],
            paths=[
                GraphPath(node_ids=["node-1", "node-2"], edge_types=["USES"]),
            ],
        )

        prompt = build_hybrid_prompt(
            query="test query",
            vector_hits=[],
            graph_result=graph_result,
        )

        # Should include graph evidence but no vector evidence
        assert "Vector Evidence:" not in prompt
        assert "Graph Nodes:" in prompt
        assert "[graph:node-1]" in prompt
