"""Symbol validator for codebase hallucination detection.

Validates that symbol references (functions, classes, methods) exist
in the codebase symbol table.
"""

from typing import Optional

import structlog

from ..symbol_table import SymbolTable
from ..types import SymbolType, ValidationResult
from .base import BaseValidator

logger = structlog.get_logger(__name__)


class SymbolValidator(BaseValidator):
    """Validator for code symbol references.

    Checks that referenced functions, classes, methods, and other
    symbols exist in the codebase symbol table. Provides suggestions
    for similar symbols when a reference is not found.

    Attributes:
        symbol_table: The SymbolTable to validate against
    """

    def __init__(self, symbol_table: SymbolTable) -> None:
        """Initialize the symbol validator.

        Args:
            symbol_table: SymbolTable containing indexed codebase symbols
        """
        self._symbol_table = symbol_table

    @property
    def validator_type(self) -> str:
        """Return the validator type identifier."""
        return "symbol"

    def validate(
        self,
        reference: str,
        location: Optional[tuple[int, int]] = None,
        context: Optional[dict] = None,
    ) -> ValidationResult:
        """Validate a symbol reference against the symbol table.

        Args:
            reference: The symbol name to validate
            location: Optional position in the response text
            context: Optional context with 'expected_type' key

        Returns:
            ValidationResult indicating if the symbol exists
        """
        expected_type = None
        if context and "expected_type" in context:
            expected_type = context["expected_type"]
            if isinstance(expected_type, str):
                try:
                    expected_type = SymbolType(expected_type)
                except ValueError:
                    expected_type = None

        return self.validate_symbol(reference, expected_type, location)

    def validate_symbol(
        self,
        name: str,
        expected_type: Optional[SymbolType] = None,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate a symbol name against the symbol table.

        Checks if the symbol exists and optionally validates its type.

        Args:
            name: The symbol name to validate
            expected_type: Optional expected symbol type
            location: Optional position in the response text

        Returns:
            ValidationResult with validation outcome and suggestions
        """
        # Skip common built-in names that shouldn't be flagged
        builtins = {
            "print", "len", "range", "str", "int", "float", "bool", "list",
            "dict", "set", "tuple", "open", "type", "isinstance", "hasattr",
            "getattr", "setattr", "None", "True", "False", "self", "cls",
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
            "AttributeError", "ImportError", "RuntimeError", "StopIteration",
        }
        if name in builtins:
            return ValidationResult(
                symbol_name=name,
                is_valid=True,
                confidence=1.0,
                reason=f"'{name}' is a Python built-in",
                suggestions=[],
                location_in_response=location,
                symbol_type=expected_type,
            )

        # Look up the symbol
        symbols = self._symbol_table.lookup(name)

        if symbols:
            # Symbol exists
            if expected_type is not None:
                # Check if any symbol matches the expected type
                matching = [s for s in symbols if s.type == expected_type]
                if matching:
                    symbol = matching[0]
                    return ValidationResult(
                        symbol_name=name,
                        is_valid=True,
                        confidence=1.0,
                        reason=f"Symbol '{name}' found as {expected_type.value} in {symbol.file_path}",
                        suggestions=[],
                        location_in_response=location,
                        symbol_type=expected_type,
                    )
                else:
                    # Symbol exists but wrong type
                    actual_types = sorted(set(s.type.value for s in symbols))
                    return ValidationResult(
                        symbol_name=name,
                        is_valid=False,
                        confidence=0.9,
                        reason=f"Symbol '{name}' exists but as {', '.join(actual_types)}, not {expected_type.value}",
                        suggestions=[f"{name} is a {t}" for t in actual_types],
                        location_in_response=location,
                        symbol_type=expected_type,
                    )
            else:
                # No expected type, symbol exists
                symbol = symbols[0]
                return ValidationResult(
                    symbol_name=name,
                    is_valid=True,
                    confidence=1.0,
                    reason=f"Symbol '{name}' found as {symbol.type.value} in {symbol.file_path}",
                    suggestions=[],
                    location_in_response=location,
                    symbol_type=symbol.type,
                )

        # Symbol not found - look for similar names
        similar = self._symbol_table.find_similar(name, limit=5, cutoff=0.6)

        if similar:
            return ValidationResult(
                symbol_name=name,
                is_valid=False,
                confidence=0.85,
                reason=f"Symbol '{name}' not found in codebase. Did you mean: {', '.join(similar)}?",
                suggestions=similar,
                location_in_response=location,
                symbol_type=expected_type,
            )
        else:
            return ValidationResult(
                symbol_name=name,
                is_valid=False,
                confidence=0.8,
                reason=f"Symbol '{name}' not found in codebase",
                suggestions=[],
                location_in_response=location,
                symbol_type=expected_type,
            )

    def validate_qualified(
        self,
        qualified_name: str,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate a fully qualified symbol name.

        Args:
            qualified_name: Qualified name (e.g., 'MyClass.my_method')
            location: Optional position in the response text

        Returns:
            ValidationResult with validation outcome
        """
        symbol = self._symbol_table.lookup_qualified(qualified_name)

        if symbol:
            return ValidationResult(
                symbol_name=qualified_name,
                is_valid=True,
                confidence=1.0,
                reason=f"Qualified symbol '{qualified_name}' found in {symbol.file_path}",
                suggestions=[],
                location_in_response=location,
                symbol_type=symbol.type,
            )

        # Try to find partial matches
        parts = qualified_name.split(".")
        if len(parts) > 1:
            # Try the last part as a simple name
            last_part = parts[-1]
            similar = self._symbol_table.find_similar(last_part, limit=5)
            if similar:
                suggestions = [
                    f"{'.'.join(parts[:-1])}.{s}" for s in similar
                ]
            else:
                suggestions = []
        else:
            suggestions = self._symbol_table.find_similar(qualified_name, limit=5)

        return ValidationResult(
            symbol_name=qualified_name,
            is_valid=False,
            confidence=0.85,
            reason=f"Qualified symbol '{qualified_name}' not found in codebase",
            suggestions=suggestions,
            location_in_response=location,
        )
