"""Semantic chunking with tiktoken token counting."""

from dataclasses import dataclass
from typing import Optional

import structlog
import tiktoken

from agentic_rag_backend.core.errors import ChunkingError

logger = structlog.get_logger(__name__)

# Use cl100k_base encoding for GPT-4/GPT-3.5 compatibility
ENCODING = tiktoken.get_encoding("cl100k_base")

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


@dataclass
class ChunkData:
    """A chunk of document content."""

    content: str
    chunk_index: int
    token_count: int
    start_char: int
    end_char: int


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    return len(ENCODING.encode(text))


def _find_sentence_boundary(text: str, position: int, direction: int = 1) -> int:
    """
    Find the nearest sentence boundary from a position.

    Args:
        text: Text to search in
        position: Starting position
        direction: 1 for forward, -1 for backward

    Returns:
        Position of sentence boundary, or original position if not found
    """
    sentence_endings = ".!?"
    search_range = 100  # Look up to 100 characters for a boundary

    if direction > 0:
        # Search forward
        for i in range(position, min(position + search_range, len(text))):
            if text[i] in sentence_endings and (i + 1 >= len(text) or text[i + 1].isspace()):
                return i + 1
    else:
        # Search backward
        for i in range(position, max(position - search_range, 0), -1):
            if text[i - 1] in sentence_endings and text[i].isspace():
                return i

    return position


def _find_paragraph_boundary(text: str, position: int, direction: int = 1) -> Optional[int]:
    """
    Find the nearest paragraph boundary (double newline) from a position.

    Args:
        text: Text to search in
        position: Starting position
        direction: 1 for forward, -1 for backward

    Returns:
        Position of paragraph boundary, or None if not found within range
    """
    search_range = 200  # Look up to 200 characters for a paragraph break

    if direction > 0:
        # Search forward
        idx = text.find("\n\n", position, position + search_range)
        return idx + 2 if idx != -1 else None
    else:
        # Search backward
        idx = text.rfind("\n\n", max(0, position - search_range), position)
        return idx + 2 if idx != -1 else None


def chunk_document(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    preserve_boundaries: bool = True,
) -> list[ChunkData]:
    """
    Split document content into overlapping semantic chunks.

    Uses tiktoken for accurate token counting and attempts to preserve
    natural boundaries (sentences, paragraphs) when possible.

    Args:
        content: Document text content
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        preserve_boundaries: If True, try to break at sentence/paragraph boundaries

    Returns:
        List of ChunkData objects

    Raises:
        ChunkingError: If chunking fails
    """
    if not content or not content.strip():
        return []

    try:
        tokens = ENCODING.encode(content)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        chunks = []
        chunk_index = 0
        token_start = 0
        
        # Track cumulative character position to avoid O(n^2) decoding
        current_char_pos = 0
        last_token_idx = 0

        while token_start < total_tokens:
            # Update current character position incrementally
            if token_start > last_token_idx:
                current_char_pos += len(ENCODING.decode(tokens[last_token_idx:token_start]))
                last_token_idx = token_start

            # Calculate end position
            token_end = min(token_start + chunk_size, total_tokens)

            # Extract tokens for this chunk
            chunk_tokens = tokens[token_start:token_end]

            # Decode back to text
            chunk_text = ENCODING.decode(chunk_tokens)

            # Find character positions
            start_char = current_char_pos
            end_char = start_char + len(chunk_text)

            # Try to adjust to natural boundaries if not at the end
            if preserve_boundaries and token_end < total_tokens:
                # Try paragraph boundary first
                para_boundary = _find_paragraph_boundary(content, end_char, direction=-1)
                if para_boundary and para_boundary > start_char + len(chunk_text) // 2:
                    # Use paragraph boundary if it's in the latter half
                    end_char = para_boundary
                    chunk_text = content[start_char:end_char]
                    chunk_tokens = ENCODING.encode(chunk_text)
                else:
                    # Try sentence boundary
                    sent_boundary = _find_sentence_boundary(content, end_char, direction=-1)
                    if sent_boundary > start_char + len(chunk_text) // 2:
                        end_char = sent_boundary
                        chunk_text = content[start_char:end_char]
                        chunk_tokens = ENCODING.encode(chunk_text)

            # Clean up the chunk text
            chunk_text = chunk_text.strip()
            if not chunk_text:
                # Skip empty chunks
                token_start = token_end
                continue

            # Recalculate token count for the final chunk
            final_token_count = len(ENCODING.encode(chunk_text))

            chunks.append(ChunkData(
                content=chunk_text,
                chunk_index=chunk_index,
                token_count=final_token_count,
                start_char=start_char,
                end_char=end_char,
            ))

            # Calculate next start position with overlap
            # Use character-based calculation for more accurate overlap
            if token_end >= total_tokens:
                break

            # Move forward by (chunk_size - overlap) tokens
            advancement = max(1, len(chunk_tokens) - chunk_overlap)
            token_start = token_start + advancement
            chunk_index += 1

            # Safety check to prevent infinite loops
            if chunk_index > total_tokens:
                logger.warning(
                    "chunking_safety_break",
                    chunk_index=chunk_index,
                    total_tokens=total_tokens,
                )
                break

        logger.info(
            "document_chunked",
            total_tokens=total_tokens,
            chunks_created=len(chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return chunks

    except Exception as e:
        raise ChunkingError("unknown", f"Failed to chunk document: {str(e)}") from e


def chunk_sections(
    sections: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkData]:
    """
    Chunk document by sections, then by token size.

    Preserves section headers and tries to keep sections together
    when possible, only splitting when a section exceeds chunk_size.

    Args:
        sections: List of section dictionaries with 'heading' and 'content' keys
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of ChunkData objects
    """
    all_chunks = []
    global_index = 0
    current_char = 0

    for section in sections:
        heading = section.get("heading", "")
        content = section.get("content", "")

        # Combine heading and content
        if heading:
            section_text = f"## {heading}\n\n{content}"
        else:
            section_text = content

        section_tokens = count_tokens(section_text)

        if section_tokens <= chunk_size:
            # Section fits in one chunk
            all_chunks.append(ChunkData(
                content=section_text.strip(),
                chunk_index=global_index,
                token_count=section_tokens,
                start_char=current_char,
                end_char=current_char + len(section_text),
            ))
            global_index += 1
            current_char += len(section_text)
        else:
            # Section needs to be split
            section_chunks = chunk_document(
                section_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for chunk in section_chunks:
                all_chunks.append(ChunkData(
                    content=chunk.content,
                    chunk_index=global_index,
                    token_count=chunk.token_count,
                    start_char=current_char + chunk.start_char,
                    end_char=current_char + chunk.end_char,
                ))
                global_index += 1
            current_char += len(section_text)

    return all_chunks


def estimate_chunks(content: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """
    Estimate the number of chunks that will be created.

    Useful for progress tracking and resource planning.

    Args:
        content: Document text content
        chunk_size: Target chunk size in tokens

    Returns:
        Estimated number of chunks
    """
    if not content:
        return 0

    total_tokens = count_tokens(content)
    if total_tokens <= chunk_size:
        return 1

    # Estimate based on overlap (roughly 80% of chunk_size effective per chunk)
    effective_chunk_size = int(chunk_size * 0.8)
    return max(1, (total_tokens + effective_chunk_size - 1) // effective_chunk_size)
