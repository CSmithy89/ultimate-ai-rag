"""Chunking utilities for codebase indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from ..types import CodeSymbol, SymbolType

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CodeChunk:
    """A chunk of code content tied to a symbol."""

    content: str
    symbol_name: str
    symbol_type: str
    file_path: str
    line_start: int
    line_end: int
    chunk_index: int


class CodeChunker:
    """Build code chunks for embeddings and retrieval."""

    def __init__(self, max_chunk_size: int = 1000, include_class_context: bool = True) -> None:
        self.max_chunk_size = max_chunk_size
        self.include_class_context = include_class_context

    def chunk_file(
        self,
        file_path: str,
        symbols: list[CodeSymbol],
        display_path: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Create chunks for a file based on extracted symbols."""
        if not symbols:
            return []

        display_path = display_path or file_path

        try:
            lines = Path(file_path).read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as exc:
            logger.warning("code_chunk_read_failed", file_path=file_path, error=str(exc))
            return []

        class_headers = self._build_class_headers(symbols, lines)
        chunks: list[CodeChunk] = []

        for symbol in sorted(symbols, key=lambda s: s.line_start):
            if symbol.type not in {SymbolType.FUNCTION, SymbolType.METHOD, SymbolType.CLASS}:
                continue

            start_idx = max(symbol.line_start - 1, 0)
            end_idx = min(symbol.line_end, len(lines))
            snippet_lines = lines[start_idx:end_idx]
            snippet = "".join(snippet_lines).strip()
            if not snippet:
                continue

            if self.include_class_context and symbol.type == SymbolType.METHOD and symbol.parent:
                header = class_headers.get(symbol.parent)
                if header:
                    snippet = f"{header}\n{snippet}"

            chunks.extend(
                self._split_chunk(
                    snippet,
                    symbol,
                    display_path,
                    symbol.line_start,
                    symbol.line_end,
                )
            )

        return chunks

    def _build_class_headers(
        self,
        symbols: list[CodeSymbol],
        lines: list[str],
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        for symbol in symbols:
            if symbol.type != SymbolType.CLASS:
                continue
            if symbol.signature:
                headers[symbol.name] = symbol.signature
                continue
            start_idx = max(symbol.line_start - 1, 0)
            end_idx = min(symbol.line_start, len(lines))
            header_line = "".join(lines[start_idx:end_idx]).strip()
            if header_line:
                headers[symbol.name] = header_line
        return headers

    def _split_chunk(
        self,
        text: str,
        symbol: CodeSymbol,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> list[CodeChunk]:
        if len(text) <= self.max_chunk_size:
            return [
                CodeChunk(
                    content=text,
                    symbol_name=symbol.name,
                    symbol_type=symbol.type.value,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    chunk_index=0,
                )
            ]

        chunks: list[CodeChunk] = []
        current_lines: list[str] = []
        current_len = 0
        chunk_start_line = line_start
        line_cursor = line_start

        for line in text.splitlines(keepends=True):
            if current_len + len(line) > self.max_chunk_size and current_lines:
                chunk_text = "".join(current_lines).strip()
                if chunk_text:
                    chunks.append(
                        CodeChunk(
                            content=chunk_text,
                            symbol_name=symbol.name,
                            symbol_type=symbol.type.value,
                            file_path=file_path,
                            line_start=chunk_start_line,
                            line_end=line_cursor - 1,
                            chunk_index=len(chunks),
                        )
                    )
                current_lines = []
                current_len = 0
                chunk_start_line = line_cursor

            current_lines.append(line)
            current_len += len(line)
            line_cursor += 1

        if current_lines:
            chunk_text = "".join(current_lines).strip()
            if chunk_text:
                chunks.append(
                    CodeChunk(
                        content=chunk_text,
                        symbol_name=symbol.name,
                        symbol_type=symbol.type.value,
                        file_path=file_path,
                        line_start=chunk_start_line,
                        line_end=line_end,
                        chunk_index=len(chunks),
                    )
                )

        return chunks
