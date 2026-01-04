"""Tests for code graph builder."""

from agentic_rag_backend.codebase.indexing.graph_builder import CodeGraphBuilder
from agentic_rag_backend.codebase.symbol_table import SymbolTable
from agentic_rag_backend.codebase.types import CodeSymbol, SymbolScope, SymbolType


def test_graph_builder_extracts_calls():
    content = "def a():\n    b()\n\ndef b():\n    pass\n"
    symbols = [
        CodeSymbol(
            name="a",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="test.py",
            line_start=1,
            line_end=2,
        ),
        CodeSymbol(
            name="b",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="test.py",
            line_start=4,
            line_end=5,
        ),
    ]
    table = SymbolTable(tenant_id="tenant", repo_path="/repo")
    for sym in symbols:
        table.add(sym)

    builder = CodeGraphBuilder()
    relationships = builder.build_relationships(table, "test.py", content, symbols)

    assert any(
        rel.relationship_type == "CALLS" and rel.source_name == "a" and rel.target_name == "b"
        for rel in relationships
    )


def test_graph_builder_extracts_imports():
    content = "import os\nfrom typing import Optional\n"
    table = SymbolTable(tenant_id="tenant", repo_path="/repo")
    builder = CodeGraphBuilder()

    relationships = builder.build_relationships(table, "test.py", content, [])
    assert any(rel.relationship_type == "IMPORTS" for rel in relationships)
