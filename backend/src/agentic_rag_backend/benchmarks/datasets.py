"""Dataset loading and management for retrieval quality benchmarks.

This module handles loading and validating evaluation datasets in a
custom JSON format compatible with BEIR-style datasets.

Dataset Format:
{
    "name": "dataset-name",
    "description": "Dataset description",
    "version": "1.0.0",
    "queries": [
        {
            "query_id": "q1",
            "text": "What is machine learning?",
            "relevant_docs": {
                "doc1": 1.0,   // Fully relevant
                "doc2": 0.5    // Partially relevant
            }
        }
    ],
    "documents": [
        {
            "doc_id": "doc1",
            "text": "Machine learning is...",
            "metadata": {}
        }
    ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Query:
    """A benchmark query with relevance judgments.

    Attributes:
        query_id: Unique identifier for the query
        text: The query text
        relevant_docs: Mapping of doc_id -> relevance grade (0.0-1.0)
    """

    query_id: str
    text: str
    relevant_docs: dict[str, float] = field(default_factory=dict)


@dataclass
class Document:
    """A document in the evaluation corpus.

    Attributes:
        doc_id: Unique identifier for the document
        text: The document text/content
        metadata: Optional metadata about the document
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset with queries and documents.

    Attributes:
        name: Dataset name
        description: Dataset description
        version: Dataset version string
        queries: List of benchmark queries
        documents: List of corpus documents
    """

    name: str
    description: str
    version: str
    queries: list[Query]
    documents: list[Document]

    @property
    def num_queries(self) -> int:
        """Number of queries in the dataset."""
        return len(self.queries)

    @property
    def num_documents(self) -> int:
        """Number of documents in the dataset."""
        return len(self.documents)

    @property
    def avg_relevant_per_query(self) -> float:
        """Average number of relevant documents per query."""
        if not self.queries:
            return 0.0
        total_relevant = sum(len(q.relevant_docs) for q in self.queries)
        return total_relevant / len(self.queries)

    def get_document_by_id(self, doc_id: str) -> Document | None:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_document_text_map(self) -> dict[str, str]:
        """Get mapping of doc_id -> text for all documents."""
        return {doc.doc_id: doc.text for doc in self.documents}

    def summary(self) -> dict[str, Any]:
        """Get dataset summary for logging/reporting."""
        return {
            "name": self.name,
            "version": self.version,
            "num_queries": self.num_queries,
            "num_documents": self.num_documents,
            "avg_relevant_per_query": round(self.avg_relevant_per_query, 2),
        }


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


def validate_dataset_schema(data: dict[str, Any]) -> None:
    """Validate dataset JSON schema.

    Args:
        data: Parsed JSON data

    Raises:
        DatasetValidationError: If validation fails
    """
    required_fields = ["name", "queries", "documents"]
    for field_name in required_fields:
        if field_name not in data:
            raise DatasetValidationError(f"Missing required field: {field_name}")

    if not isinstance(data["queries"], list):
        raise DatasetValidationError("'queries' must be a list")
    if not isinstance(data["documents"], list):
        raise DatasetValidationError("'documents' must be a list")

    # Validate queries
    for i, query in enumerate(data["queries"]):
        if "query_id" not in query:
            raise DatasetValidationError(f"Query at index {i} missing 'query_id'")
        if "text" not in query:
            raise DatasetValidationError(f"Query at index {i} missing 'text'")
        if "relevant_docs" in query and not isinstance(query["relevant_docs"], dict):
            raise DatasetValidationError(
                f"Query {query['query_id']}: 'relevant_docs' must be a dict"
            )

    # Validate documents
    for i, doc in enumerate(data["documents"]):
        if "doc_id" not in doc:
            raise DatasetValidationError(f"Document at index {i} missing 'doc_id'")
        if "text" not in doc:
            raise DatasetValidationError(f"Document at index {i} missing 'text'")

    # Validate relevance grade values
    for query in data["queries"]:
        for doc_id, grade in query.get("relevant_docs", {}).items():
            if not isinstance(grade, (int, float)):
                raise DatasetValidationError(
                    f"Query {query['query_id']}: relevance grade for {doc_id} "
                    f"must be numeric, got {type(grade).__name__}"
                )
            if grade < 0 or grade > 1:
                raise DatasetValidationError(
                    f"Query {query['query_id']}: relevance grade for {doc_id} "
                    f"must be between 0 and 1, got {grade}"
                )


def load_dataset(path: Path | str) -> EvaluationDataset:
    """Load an evaluation dataset from a JSON file.

    Args:
        path: Path to the dataset JSON file

    Returns:
        Parsed EvaluationDataset

    Raises:
        FileNotFoundError: If the file doesn't exist
        DatasetValidationError: If validation fails
        json.JSONDecodeError: If JSON parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    validate_dataset_schema(data)

    queries = [
        Query(
            query_id=q["query_id"],
            text=q["text"],
            relevant_docs=q.get("relevant_docs", {}),
        )
        for q in data["queries"]
    ]

    documents = [
        Document(
            doc_id=d["doc_id"],
            text=d["text"],
            metadata=d.get("metadata", {}),
        )
        for d in data["documents"]
    ]

    return EvaluationDataset(
        name=data["name"],
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        queries=queries,
        documents=documents,
    )


def create_sample_dataset() -> EvaluationDataset:
    """Create a sample evaluation dataset for testing.

    Returns:
        A small sample dataset with 5 queries and 10 documents
    """
    queries = [
        Query(
            query_id="q1",
            text="What is machine learning?",
            relevant_docs={"doc1": 1.0, "doc2": 0.7},
        ),
        Query(
            query_id="q2",
            text="How does neural network training work?",
            relevant_docs={"doc3": 1.0, "doc4": 0.5},
        ),
        Query(
            query_id="q3",
            text="What are transformers in NLP?",
            relevant_docs={"doc5": 1.0, "doc6": 0.8},
        ),
        Query(
            query_id="q4",
            text="Explain gradient descent optimization",
            relevant_docs={"doc7": 1.0, "doc3": 0.3},
        ),
        Query(
            query_id="q5",
            text="What is retrieval augmented generation?",
            relevant_docs={"doc8": 1.0, "doc9": 0.9, "doc10": 0.4},
        ),
    ]

    documents = [
        Document(
            doc_id="doc1",
            text="Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without explicit programming.",
            metadata={"source": "ai_textbook"},
        ),
        Document(
            doc_id="doc2",
            text="AI and machine learning are transforming industries by enabling "
            "automated decision-making and pattern recognition.",
            metadata={"source": "tech_blog"},
        ),
        Document(
            doc_id="doc3",
            text="Neural network training involves forward propagation, loss calculation, "
            "and backpropagation to update weights using gradient descent.",
            metadata={"source": "ml_course"},
        ),
        Document(
            doc_id="doc4",
            text="Deep learning models use multiple layers of neurons to learn "
            "hierarchical representations of data.",
            metadata={"source": "research_paper"},
        ),
        Document(
            doc_id="doc5",
            text="Transformers are attention-based models that revolutionized NLP by "
            "enabling parallel processing and capturing long-range dependencies.",
            metadata={"source": "nlp_textbook"},
        ),
        Document(
            doc_id="doc6",
            text="The attention mechanism in transformers computes weighted sums of "
            "values based on query-key compatibility.",
            metadata={"source": "arxiv"},
        ),
        Document(
            doc_id="doc7",
            text="Gradient descent is an optimization algorithm that iteratively adjusts "
            "parameters in the direction that minimizes the loss function.",
            metadata={"source": "math_textbook"},
        ),
        Document(
            doc_id="doc8",
            text="Retrieval Augmented Generation (RAG) combines retrieval systems with "
            "generative models to provide grounded, factual responses.",
            metadata={"source": "rag_paper"},
        ),
        Document(
            doc_id="doc9",
            text="RAG systems first retrieve relevant documents from a knowledge base, "
            "then use them as context for language model generation.",
            metadata={"source": "tutorial"},
        ),
        Document(
            doc_id="doc10",
            text="Knowledge grounding through retrieval helps reduce hallucinations in "
            "large language models.",
            metadata={"source": "blog"},
        ),
    ]

    return EvaluationDataset(
        name="sample-benchmark",
        description="Sample evaluation dataset for testing retrieval benchmarks",
        version="1.0.0",
        queries=queries,
        documents=documents,
    )


def save_dataset(dataset: EvaluationDataset, path: Path | str) -> None:
    """Save an evaluation dataset to a JSON file.

    Args:
        dataset: The dataset to save
        path: Path to save the JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": dataset.name,
        "description": dataset.description,
        "version": dataset.version,
        "queries": [
            {
                "query_id": q.query_id,
                "text": q.text,
                "relevant_docs": q.relevant_docs,
            }
            for q in dataset.queries
        ],
        "documents": [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "metadata": d.metadata,
            }
            for d in dataset.documents
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
