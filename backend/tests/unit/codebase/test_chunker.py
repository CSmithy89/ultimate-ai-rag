"""Tests for codebase chunker."""

from pathlib import Path

from agentic_rag_backend.codebase.indexing.chunker import CodeChunker
from agentic_rag_backend.codebase.types import CodeSymbol, SymbolScope, SymbolType


def test_chunker_includes_class_context(tmp_path: Path):
    source = (
        "class User:\n"
        "    def greet(self):\n"
        "        return \"hi\"\n"
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    class_symbol = CodeSymbol(
        name="User",
        type=SymbolType.CLASS,
        scope=SymbolScope.GLOBAL,
        file_path="sample.py",
        line_start=1,
        line_end=3,
        signature="class User:",
    )
    method_symbol = CodeSymbol(
        name="greet",
        type=SymbolType.METHOD,
        scope=SymbolScope.CLASS,
        file_path="sample.py",
        line_start=2,
        line_end=3,
        parent="User",
    )

    chunker = CodeChunker(max_chunk_size=200, include_class_context=True)
    chunks = chunker.chunk_file(
        file_path=str(file_path),
        symbols=[class_symbol, method_symbol],
        display_path="sample.py",
    )

    assert chunks
    assert "class User" in chunks[0].content
    assert "def greet" in chunks[0].content
