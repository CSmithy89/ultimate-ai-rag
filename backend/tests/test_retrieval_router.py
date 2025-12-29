from agentic_rag_backend.retrieval_router import RetrievalStrategy, select_retrieval_strategy


def test_select_retrieval_strategy_semantic() -> None:
    query = "Provide a semantic similarity overview"
    assert select_retrieval_strategy(query) == RetrievalStrategy.VECTOR


def test_select_retrieval_strategy_relational() -> None:
    query = "Show the relationship between entities in the graph"
    assert select_retrieval_strategy(query) == RetrievalStrategy.GRAPH


def test_select_retrieval_strategy_hybrid() -> None:
    query = "Run a multi-hop query across the graph and summarize"
    assert select_retrieval_strategy(query) == RetrievalStrategy.HYBRID


def test_select_retrieval_strategy_not_hybrid_on_and() -> None:
    query = "Apples and oranges"
    assert select_retrieval_strategy(query) != RetrievalStrategy.HYBRID
