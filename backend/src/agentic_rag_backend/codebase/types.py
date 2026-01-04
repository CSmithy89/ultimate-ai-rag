"""Core type definitions for codebase hallucination detection.

This module defines the fundamental data types used throughout the
codebase intelligence system for symbol extraction, validation, and
hallucination detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SymbolType(str, Enum):
    """Types of code symbols that can be extracted and validated."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    INTERFACE = "interface"  # TypeScript
    ENUM = "enum"
    PROPERTY = "property"
    PARAMETER = "parameter"


class SymbolScope(str, Enum):
    """Scope context where a symbol is defined."""

    GLOBAL = "global"
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    BLOCK = "block"


@dataclass(frozen=True)
class CodeSymbol:
    """A code symbol extracted from AST.

    Represents a named entity in the codebase such as a function,
    class, method, variable, or import statement.

    Attributes:
        name: The symbol name (e.g., 'create_user')
        type: The symbol type (function, class, etc.)
        scope: The scope where the symbol is defined
        file_path: Relative path to the file containing the symbol
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (1-indexed)
        signature: Optional function/method signature
        parent: Optional parent symbol name (e.g., class name for methods)
        docstring: Optional docstring/documentation
        qualified_name: Optional fully qualified name (module.class.method)
    """

    name: str
    type: SymbolType
    scope: SymbolScope
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str] = None
    parent: Optional[str] = None
    docstring: Optional[str] = None
    qualified_name: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a symbol reference in an LLM response.

    Represents the outcome of checking whether a referenced symbol,
    file path, or API endpoint exists in the codebase.

    Attributes:
        symbol_name: The symbol/reference that was validated
        is_valid: Whether the reference is valid (exists in codebase)
        confidence: Confidence score (0.0-1.0) in the validation result
        reason: Human-readable explanation of the validation result
        suggestions: List of similar symbols if the reference is invalid
        location_in_response: Start and end positions in the response text
        symbol_type: The detected type of the symbol (if applicable)
    """

    symbol_name: str
    is_valid: bool
    confidence: float
    reason: str
    suggestions: list[str] = field(default_factory=list)
    location_in_response: Optional[tuple[int, int]] = None
    symbol_type: Optional[SymbolType] = None


@dataclass
class HallucinationReport:
    """Complete hallucination detection report for an LLM response.

    Aggregates all validation results and provides summary metrics
    for the hallucination detection analysis.

    Attributes:
        total_symbols_checked: Total number of symbols validated
        valid_symbols: Count of valid symbol references
        invalid_symbols: Count of invalid (hallucinated) references
        uncertain_symbols: Count of uncertain validations (low confidence)
        validation_results: Detailed results for each validated symbol
        files_checked: List of file paths that were validated
        processing_time_ms: Time taken to process the validation in milliseconds
        confidence_score: Overall confidence in the response (0.0-1.0)
        response_text: The original response text that was validated
        tenant_id: Tenant identifier for multi-tenancy tracking
    """

    total_symbols_checked: int
    valid_symbols: int
    invalid_symbols: int
    uncertain_symbols: int
    validation_results: list[ValidationResult]
    files_checked: list[str]
    processing_time_ms: int
    confidence_score: float
    response_text: Optional[str] = None
    tenant_id: Optional[str] = None


class Language(str, Enum):
    """Supported programming languages for AST parsing."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    TSX = "tsx"
    JSX = "jsx"


# Language file extension mapping
LANGUAGE_EXTENSIONS: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JSX,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
}


def get_language_for_file(file_path: str) -> Optional[Language]:
    """Determine the programming language from a file path.

    Args:
        file_path: Path to the source file

    Returns:
        Language enum value if recognized, None otherwise
    """
    import os

    _, ext = os.path.splitext(file_path)
    return LANGUAGE_EXTENSIONS.get(ext.lower())
