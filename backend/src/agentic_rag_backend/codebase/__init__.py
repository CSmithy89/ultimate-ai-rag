"""Codebase intelligence module for hallucination detection.

This module provides AST-based validation of LLM responses to catch
references to non-existent code symbols, files, or API endpoints.

Key components:
- types: Core data types (SymbolType, CodeSymbol, ValidationResult, etc.)
- symbol_table: In-memory symbol table with multiple indexes
- parser: Tree-sitter AST parsing wrapper
- extractor: Symbol extraction from AST
- detector: Main HallucinationDetector class
- validators: Validation logic for symbols, paths, APIs, imports
"""

from .types import (
    CodeSymbol,
    HallucinationReport,
    SymbolScope,
    SymbolType,
    ValidationResult,
)
from .symbol_table import SymbolTable
from .parser import ASTParser
from .extractor import SymbolExtractor
from .detector import DetectorMode, HallucinationDetector

__all__ = [
    # Types
    "CodeSymbol",
    "HallucinationReport",
    "SymbolScope",
    "SymbolType",
    "ValidationResult",
    # Core classes
    "SymbolTable",
    "ASTParser",
    "SymbolExtractor",
    "DetectorMode",
    "HallucinationDetector",
]
