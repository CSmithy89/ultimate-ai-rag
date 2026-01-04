"""Relationship extraction for codebase indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import structlog

from ..symbol_table import SymbolTable
from ..types import CodeSymbol, SymbolType

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CodeRelationship:
    """Represents a relationship between code entities."""

    source_name: str
    source_type: str
    target_name: str
    target_type: str
    relationship_type: str
    source_file: str
    target_file: Optional[str] = None


class CodeGraphBuilder:
    """Build lightweight code relationships from source text."""

    _CALL_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)
    _IMPORT_PATTERN = re.compile(
        r"(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+)(?!\s+from))"
    )
    _IMPORT_JS_PATTERN = re.compile(
        r'import\s+(?:[^;]*?\s+from\s+)?["\']([^"\']+)["\']'
    )
    _REQUIRE_JS_PATTERN = re.compile(r'require\(\s*["\']([^"\']+)["\']\s*\)')

    _EXCLUDED_WORDS = {
        "the", "and", "for", "with", "this", "that", "from", "import",
        "class", "def", "return", "if", "else", "elif", "try", "except",
        "while", "for", "in", "is", "not", "or", "and", "none", "true",
        "false", "async", "await", "yield", "raise", "pass", "break",
        "continue", "lambda", "global", "nonlocal", "assert", "del",
    }

    def build_relationships(
        self,
        symbol_table: SymbolTable,
        file_path: str,
        file_content: str,
        symbols: list[CodeSymbol],
    ) -> list[CodeRelationship]:
        relationships: list[CodeRelationship] = []

        relationships.extend(self._build_import_relationships(file_path, file_content))

        for symbol in symbols:
            if symbol.type not in {SymbolType.FUNCTION, SymbolType.METHOD}:
                continue
            snippet = self._extract_symbol_content(file_content, symbol)
            for match in self._CALL_PATTERN.finditer(snippet):
                called = match.group(1)
                called_name = called.lower()
                if called_name in self._EXCLUDED_WORDS:
                    continue
                if self._is_definition_reference(snippet, match.start()):
                    continue
                matches = symbol_table.lookup(called)
                if not matches:
                    continue
                target = matches[0]
                relationships.append(
                    CodeRelationship(
                        source_name=symbol.name,
                        source_type=symbol.type.value,
                        target_name=target.name,
                        target_type=target.type.value,
                        relationship_type="CALLS",
                        source_file=file_path,
                        target_file=target.file_path,
                    )
                )

        return relationships

    def _build_import_relationships(
        self,
        file_path: str,
        file_content: str,
    ) -> list[CodeRelationship]:
        relationships: list[CodeRelationship] = []
        seen: set[str] = set()

        for match in self._IMPORT_JS_PATTERN.finditer(file_content):
            module = match.group(1)
            if module and module not in seen:
                seen.add(module)
                relationships.append(
                    CodeRelationship(
                        source_name=file_path,
                        source_type="file",
                        target_name=module,
                        target_type="module",
                        relationship_type="IMPORTS",
                        source_file=file_path,
                    )
                )

        for match in self._REQUIRE_JS_PATTERN.finditer(file_content):
            module = match.group(1)
            if module and module not in seen:
                seen.add(module)
                relationships.append(
                    CodeRelationship(
                        source_name=file_path,
                        source_type="file",
                        target_name=module,
                        target_type="module",
                        relationship_type="IMPORTS",
                        source_file=file_path,
                    )
                )

        for match in self._IMPORT_PATTERN.finditer(file_content):
            module = match.group(1) or match.group(2)
            if module and module not in seen:
                seen.add(module)
                relationships.append(
                    CodeRelationship(
                        source_name=file_path,
                        source_type="file",
                        target_name=module,
                        target_type="module",
                        relationship_type="IMPORTS",
                        source_file=file_path,
                    )
                )

        return relationships

    def _extract_symbol_content(self, file_content: str, symbol: CodeSymbol) -> str:
        lines = file_content.splitlines(keepends=True)
        start_idx = max(symbol.line_start - 1, 0)
        end_idx = min(symbol.line_end, len(lines))
        return "".join(lines[start_idx:end_idx])

    def _is_definition_reference(self, snippet: str, position: int) -> bool:
        line_start = snippet.rfind("\n", 0, position) + 1
        line_end = snippet.find("\n", position)
        if line_end == -1:
            line_end = len(snippet)
        line = snippet[line_start:line_end].lstrip()
        return line.startswith("def ") or line.startswith("class ") or line.startswith("function ")
