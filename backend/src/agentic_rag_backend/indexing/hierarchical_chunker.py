"""Hierarchical Chunking for Parent-Child Chunk Hierarchy (Story 20-C3).

This module implements hierarchical document chunking with multiple levels:
- Level 0: 256 tokens (smallest - used for embedding and matching)
- Level 1: 512 tokens
- Level 2: 1024 tokens (default return level)
- Level 3: 2048 tokens (largest)

The small-to-big retrieval pattern uses small chunks for precise matching
but returns parent chunks at a configurable level for complete context.

Key Features:
- Multi-level hierarchical chunking with configurable sizes
- Parent-child relationships between chunk levels
- Only Level 0 chunks get embeddings (storage optimization)
- Configurable overlap ratio between chunks
- Feature flag: HIERARCHICAL_CHUNKS_ENABLED

Configuration:
- HIERARCHICAL_CHUNKS_ENABLED: Enable/disable feature (default: false)
- HIERARCHICAL_CHUNK_LEVELS: Token sizes per level (default: 256,512,1024,2048)
- HIERARCHICAL_OVERLAP_RATIO: Overlap between chunks (default: 0.1)
- HIERARCHICAL_EMBEDDING_LEVEL: Which level gets embeddings (default: 0)

Performance target: <500ms additional latency for typical documents
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
import tiktoken

from ..core.errors import ChunkingError

logger = structlog.get_logger(__name__)

# Use cl100k_base encoding for GPT-4/GPT-3.5 compatibility
ENCODING = tiktoken.get_encoding("cl100k_base")

# Default hierarchical chunking configuration
DEFAULT_LEVEL_SIZES = [256, 512, 1024, 2048]
DEFAULT_OVERLAP_RATIO = 0.1
DEFAULT_EMBEDDING_LEVEL = 0
MAX_OVERLAP_PROBE_CHARS = 2000


@dataclass
class HierarchicalChunk:
    """A chunk with parent-child hierarchy relationships.

    Attributes:
        id: Unique identifier for this chunk
        content: Text content of the chunk
        level: Hierarchy level (0 = smallest, higher = larger)
        parent_id: Reference to parent chunk (None for top level)
        child_ids: List of child chunk IDs
        document_id: Source document identifier
        tenant_id: Tenant identifier for multi-tenancy
        token_count: Number of tokens in content
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        metadata: Additional metadata (source, position, etc.)
        embedding: Vector embedding (only for embedding_level chunks)
    """

    id: str
    content: str
    level: int
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)
    document_id: str = ""
    tenant_id: str = ""
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to dictionary.

        Returns:
            Dictionary representation of the chunk
        """
        return {
            "id": self.id,
            "content": self.content,
            "level": self.level,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalChunk":
        """Deserialize chunk from dictionary.

        Args:
            data: Dictionary representation of chunk

        Returns:
            HierarchicalChunk instance
        """
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            level=data.get("level", 0),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
            document_id=data.get("document_id", ""),
            tenant_id=data.get("tenant_id", ""),
            token_count=data.get("token_count", 0),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class HierarchicalChunkResult:
    """Result of hierarchical chunking operation.

    Attributes:
        chunks_by_level: Dictionary mapping level to list of chunks
        total_chunks: Total number of chunks created
        processing_time_ms: Time taken to chunk the document
        document_id: Source document identifier
        tenant_id: Tenant identifier
    """

    chunks_by_level: dict[int, list[HierarchicalChunk]]
    total_chunks: int
    processing_time_ms: int
    document_id: str
    tenant_id: str

    @property
    def all_chunks(self) -> list[HierarchicalChunk]:
        """Get all chunks across all levels, sorted by level then position."""
        result = []
        for level in sorted(self.chunks_by_level.keys()):
            result.extend(self.chunks_by_level[level])
        return result


class HierarchicalChunker:
    """Creates hierarchical chunk trees from documents.

    This class implements the multi-level chunking strategy:
    1. Create Level 0 chunks at smallest size (for precise matching)
    2. Combine Level 0 chunks into Level 1 parents
    3. Continue combining up to the highest configured level
    4. Maintain parent-child relationships between levels

    Attributes:
        level_sizes: Token sizes for each level (must be increasing)
        overlap_ratio: Overlap between chunks (0.0-0.5)
        embedding_level: Which level gets embeddings
    """

    def __init__(
        self,
        level_sizes: Optional[list[int]] = None,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
        embedding_level: int = DEFAULT_EMBEDDING_LEVEL,
    ) -> None:
        """Initialize HierarchicalChunker.

        Args:
            level_sizes: Token sizes for each level (default: [256, 512, 1024, 2048])
            overlap_ratio: Overlap between chunks (default: 0.1)
            embedding_level: Which level gets embeddings (default: 0)

        Raises:
            ValueError: If level_sizes is not strictly increasing or overlap_ratio is invalid
        """
        self.level_sizes = level_sizes or DEFAULT_LEVEL_SIZES.copy()
        self.overlap_ratio = overlap_ratio
        self.embedding_level = embedding_level

        # Validate level sizes are strictly increasing
        for i in range(1, len(self.level_sizes)):
            if self.level_sizes[i] <= self.level_sizes[i - 1]:
                raise ValueError(
                    f"Level sizes must be strictly increasing. "
                    f"Got {self.level_sizes[i]} <= {self.level_sizes[i - 1]} at index {i}."
                )

        # Validate overlap ratio
        if not 0.0 <= self.overlap_ratio <= 0.5:
            raise ValueError(
                f"Overlap ratio must be between 0.0 and 0.5. Got {self.overlap_ratio}."
            )

        # Validate embedding level
        if not 0 <= self.embedding_level < len(self.level_sizes):
            raise ValueError(
                f"Embedding level must be between 0 and {len(self.level_sizes) - 1}. "
                f"Got {self.embedding_level}."
            )

        logger.debug(
            "hierarchical_chunker_initialized",
            level_sizes=self.level_sizes,
            overlap_ratio=self.overlap_ratio,
            embedding_level=self.embedding_level,
        )

    def chunk_document(
        self,
        content: str,
        document_id: str,
        tenant_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> HierarchicalChunkResult:
        """Create hierarchical chunks from document content.

        This is the main entry point that:
        1. Creates Level 0 chunks at smallest size
        2. Iteratively combines into higher-level parent chunks
        3. Links parent-child relationships

        Args:
            content: Document text content
            document_id: Unique document identifier
            tenant_id: Tenant identifier for multi-tenancy
            metadata: Additional metadata to include in chunks

        Returns:
            HierarchicalChunkResult with all levels of chunks

        Raises:
            ChunkingError: If chunking fails
        """
        start_time = time.perf_counter()

        if not content or not content.strip():
            return HierarchicalChunkResult(
                chunks_by_level={},
                total_chunks=0,
                processing_time_ms=0,
                document_id=document_id,
                tenant_id=tenant_id,
            )

        try:
            chunks_by_level: dict[int, list[HierarchicalChunk]] = {}

            # Step 1: Create Level 0 (smallest) chunks
            level_0_chunks = self._create_level_chunks(
                content=content,
                level=0,
                chunk_size=self.level_sizes[0],
                document_id=document_id,
                tenant_id=tenant_id,
                metadata=metadata or {},
            )
            chunks_by_level[0] = level_0_chunks

            # Step 2: Create higher-level chunks by combining lower levels
            for level in range(1, len(self.level_sizes)):
                parent_chunks = self._combine_to_level(
                    child_chunks=chunks_by_level[level - 1],
                    level=level,
                    target_size=self.level_sizes[level],
                    document_id=document_id,
                    tenant_id=tenant_id,
                    metadata=metadata or {},
                )
                chunks_by_level[level] = parent_chunks

                # Link children to parents
                self._link_parent_children(
                    parent_chunks=parent_chunks,
                    child_chunks=chunks_by_level[level - 1],
                )

            total_chunks = sum(len(chunks) for chunks in chunks_by_level.values())
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info(
                "hierarchical_chunking_completed",
                document_id=document_id,
                tenant_id=tenant_id,
                total_chunks=total_chunks,
                chunks_per_level={level: len(chunks) for level, chunks in chunks_by_level.items()},
                processing_time_ms=processing_time_ms,
            )

            return HierarchicalChunkResult(
                chunks_by_level=chunks_by_level,
                total_chunks=total_chunks,
                processing_time_ms=processing_time_ms,
                document_id=document_id,
                tenant_id=tenant_id,
            )

        except Exception as e:
            logger.error(
                "hierarchical_chunking_failed",
                document_id=document_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            raise ChunkingError(
                document_id, f"Hierarchical chunking failed: {str(e)}"
            ) from e

    def _create_level_chunks(
        self,
        content: str,
        level: int,
        chunk_size: int,
        document_id: str,
        tenant_id: str,
        metadata: dict[str, Any],
    ) -> list[HierarchicalChunk]:
        """Create chunks at a specific level from raw content.

        Args:
            content: Text content to chunk
            level: Target hierarchy level
            chunk_size: Target chunk size in tokens
            document_id: Document identifier
            tenant_id: Tenant identifier
            metadata: Additional metadata

        Returns:
            List of chunks at the specified level
        """
        tokens = ENCODING.encode(content)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        chunks = []
        overlap_tokens = int(chunk_size * self.overlap_ratio)
        step_size = max(1, chunk_size - overlap_tokens)
        token_start = 0
        chunk_index = 0

        while token_start < total_tokens:
            token_end = min(token_start + chunk_size, total_tokens)
            chunk_tokens = tokens[token_start:token_end]
            chunk_text = ENCODING.decode(chunk_tokens).strip()

            if chunk_text:
                # Calculate character positions
                start_char = len(ENCODING.decode(tokens[:token_start]))
                end_char = start_char + len(ENCODING.decode(chunk_tokens))

                chunk_id = self._generate_chunk_id(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    level=level,
                    chunk_index=chunk_index,
                )

                chunk = HierarchicalChunk(
                    id=chunk_id,
                    content=chunk_text,
                    level=level,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    token_count=len(chunk_tokens),
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "level_size": chunk_size,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            if token_end >= total_tokens:
                break
            token_start += step_size

        return chunks

    def _combine_to_level(
        self,
        child_chunks: list[HierarchicalChunk],
        level: int,
        target_size: int,
        document_id: str,
        tenant_id: str,
        metadata: dict[str, Any],
    ) -> list[HierarchicalChunk]:
        """Combine lower-level chunks into higher-level parents.

        Args:
            child_chunks: Chunks from the previous (lower) level
            level: Target hierarchy level
            target_size: Target size in tokens for parent chunks
            document_id: Document identifier
            tenant_id: Tenant identifier
            metadata: Additional metadata

        Returns:
            List of parent chunks at the target level
        """
        if not child_chunks:
            return []

        parent_chunks = []
        current_children: list[HierarchicalChunk] = []
        current_tokens = 0
        parent_index = 0

        for child in child_chunks:
            projected_children = current_children + [child]
            projected_tokens = self._estimate_combined_tokens(projected_children)

            # Check if adding this child would exceed target size
            if projected_tokens > target_size and current_children:
                # Create parent from accumulated children
                parent = self._create_parent_chunk(
                    children=current_children,
                    level=level,
                    parent_index=parent_index,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    metadata=metadata,
                )
                parent_chunks.append(parent)
                parent_index += 1

                # Start new accumulation
                current_children = [child]
                current_tokens = self._estimate_combined_tokens(current_children)
            else:
                current_children = projected_children
                current_tokens = projected_tokens

        # Handle remaining children
        if current_children:
            parent = self._create_parent_chunk(
                children=current_children,
                level=level,
                parent_index=parent_index,
                document_id=document_id,
                tenant_id=tenant_id,
                metadata=metadata,
            )
            parent_chunks.append(parent)

        return parent_chunks

    def _create_parent_chunk(
        self,
        children: list[HierarchicalChunk],
        level: int,
        parent_index: int,
        document_id: str,
        tenant_id: str,
        metadata: dict[str, Any],
    ) -> HierarchicalChunk:
        """Create a parent chunk from multiple child chunks.

        Args:
            children: Child chunks to combine
            level: Target hierarchy level
            parent_index: Index of this parent in its level
            document_id: Document identifier
            tenant_id: Tenant identifier
            metadata: Additional metadata

        Returns:
            Parent chunk containing combined content
        """
        # Combine content from children while trimming overlaps
        combined_content = self._merge_child_contents(children)

        # Calculate positions from children
        start_char = children[0].start_char if children else 0
        end_char = children[-1].end_char if children else 0

        # Calculate token count based on merged content
        token_count = len(ENCODING.encode(combined_content))

        chunk_id = self._generate_chunk_id(
            document_id=document_id,
            tenant_id=tenant_id,
            level=level,
            chunk_index=parent_index,
        )

        return HierarchicalChunk(
            id=chunk_id,
            content=combined_content,
            level=level,
            child_ids=[child.id for child in children],
            document_id=document_id,
            tenant_id=tenant_id,
            token_count=token_count,
            start_char=start_char,
            end_char=end_char,
            metadata={
                **metadata,
                "chunk_index": parent_index,
                "level_size": self.level_sizes[level],
                "child_count": len(children),
            },
        )

    def _merge_child_contents(self, children: list[HierarchicalChunk]) -> str:
        """Merge child chunk contents while removing overlaps."""
        if not children:
            return ""
        merged = children[0].content
        for child in children[1:]:
            overlap = self._find_overlap(merged, child.content)
            if overlap > 0:
                merged = f"{merged}{child.content[overlap:]}"
            else:
                merged = f"{merged} {child.content}"
        return merged.strip()

    def _find_overlap(self, left: str, right: str) -> int:
        """Find the length of the largest suffix/prefix overlap."""
        max_probe = min(len(left), len(right), MAX_OVERLAP_PROBE_CHARS)
        for size in range(max_probe, 0, -1):
            if left.endswith(right[:size]):
                return size
        return 0

    def _estimate_combined_tokens(self, children: list[HierarchicalChunk]) -> int:
        """Estimate combined token count for a group of children."""
        combined_content = self._merge_child_contents(children)
        return len(ENCODING.encode(combined_content)) if combined_content else 0

    def _link_parent_children(
        self,
        parent_chunks: list[HierarchicalChunk],
        child_chunks: list[HierarchicalChunk],
    ) -> None:
        """Link parent and child chunks bidirectionally.

        Args:
            parent_chunks: Parent chunks at higher level
            child_chunks: Child chunks at lower level
        """
        # Build parent lookup from child_ids
        child_to_parent: dict[str, str] = {}
        for parent in parent_chunks:
            for child_id in parent.child_ids:
                child_to_parent[child_id] = parent.id

        # Set parent_id on children
        for child in child_chunks:
            if child.id in child_to_parent:
                child.parent_id = child_to_parent[child.id]

    def _generate_chunk_id(
        self,
        document_id: str,
        tenant_id: str,
        level: int,
        chunk_index: int,
    ) -> str:
        """Generate a deterministic, unique chunk ID.

        Uses hash of tenant_id + document_id + level + index for idempotent
        re-ingestion while preventing cross-tenant ID collisions.

        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier for multi-tenancy
            level: Hierarchy level
            chunk_index: Index within level

        Returns:
            Unique chunk identifier
        """
        # Create deterministic ID including tenant to prevent cross-tenant collisions
        id_string = f"{tenant_id}:{document_id}:L{level}:C{chunk_index}"
        hash_digest = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        return f"chunk_{level}_{hash_digest}"


def create_hierarchical_chunker(
    level_sizes: Optional[list[int]] = None,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    embedding_level: int = DEFAULT_EMBEDDING_LEVEL,
) -> HierarchicalChunker:
    """Factory function to create a configured HierarchicalChunker.

    Args:
        level_sizes: Token sizes for each level
        overlap_ratio: Overlap between chunks
        embedding_level: Which level gets embeddings

    Returns:
        Configured HierarchicalChunker instance
    """
    return HierarchicalChunker(
        level_sizes=level_sizes,
        overlap_ratio=overlap_ratio,
        embedding_level=embedding_level,
    )
