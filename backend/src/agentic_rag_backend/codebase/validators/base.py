"""Base validator interface for codebase hallucination detection.

Defines the abstract base class that all validators must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..types import ValidationResult


class BaseValidator(ABC):
    """Abstract base class for codebase validators.

    All validators must implement the validate() method which checks
    a reference (symbol, path, endpoint, etc.) against the codebase
    and returns a ValidationResult.
    """

    @abstractmethod
    def validate(
        self,
        reference: str,
        location: Optional[tuple[int, int]] = None,
        context: Optional[dict] = None,
    ) -> ValidationResult:
        """Validate a reference against the codebase.

        Args:
            reference: The reference to validate (symbol name, path, etc.)
            location: Optional (start, end) position in the response text
            context: Optional additional context for validation

        Returns:
            ValidationResult with validation outcome and suggestions
        """
        pass

    @property
    @abstractmethod
    def validator_type(self) -> str:
        """Return the type identifier for this validator.

        Used for logging and error reporting.

        Returns:
            String identifier (e.g., 'symbol', 'path', 'api')
        """
        pass
