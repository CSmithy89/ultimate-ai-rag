"""File path validator for codebase hallucination detection.

Validates that file path references exist in the repository.
"""

import os
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import structlog

from ..symbol_table import SymbolTable
from ..types import ValidationResult
from .base import BaseValidator

logger = structlog.get_logger(__name__)


class FilePathValidator(BaseValidator):
    """Validator for file path references.

    Checks that referenced file paths exist in the repository.
    Supports both absolute and relative path validation.

    Attributes:
        symbol_table: The SymbolTable with known files
        repo_root: Root path of the repository
    """

    def __init__(
        self,
        symbol_table: SymbolTable,
        repo_root: Optional[str] = None,
    ) -> None:
        """Initialize the file path validator.

        Args:
            symbol_table: SymbolTable containing known file paths
            repo_root: Root path of the repository for filesystem checks
        """
        self._symbol_table = symbol_table
        self._repo_root = repo_root or symbol_table.repo_path

    @property
    def validator_type(self) -> str:
        """Return the validator type identifier."""
        return "path"

    def validate(
        self,
        reference: str,
        location: Optional[tuple[int, int]] = None,
        context: Optional[dict] = None,
    ) -> ValidationResult:
        """Validate a file path reference.

        Args:
            reference: The file path to validate
            location: Optional position in the response text
            context: Optional context (unused for path validation)

        Returns:
            ValidationResult indicating if the path exists
        """
        return self.validate_path(reference, location)

    def validate_path(
        self,
        file_path: str,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate a file path against the repository.

        Checks both the symbol table and filesystem for the path.

        Args:
            file_path: The file path to validate
            location: Optional position in the response text

        Returns:
            ValidationResult with validation outcome and suggestions
        """
        # Normalize the path
        normalized = self._normalize_path(file_path)

        # Check symbol table first (indexed files)
        if self._symbol_table.file_exists(normalized):
            return ValidationResult(
                symbol_name=file_path,
                is_valid=True,
                confidence=1.0,
                reason=f"File '{normalized}' exists in symbol table",
                suggestions=[],
                location_in_response=location,
            )

        # Check filesystem if repo_root is set
        if self._repo_root:
            full_path = Path(self._repo_root) / normalized
            # SECURITY: Prevent path traversal attacks
            try:
                resolved_path = full_path.resolve()
                repo_resolved = Path(self._repo_root).resolve()
                try:
                    resolved_path.relative_to(repo_resolved)
                except ValueError:
                    logger.warning(
                        "path_traversal_blocked",
                        requested_path=file_path,
                        resolved_path=str(resolved_path),
                    )
                    return ValidationResult(
                        symbol_name=file_path,
                        is_valid=False,
                        confidence=1.0,
                        reason=f"Path '{file_path}' is outside repository bounds",
                        suggestions=[],
                        location_in_response=location,
                    )
            except (OSError, ValueError) as e:
                logger.warning("path_resolution_failed", path=file_path, error=str(e))
                return ValidationResult(
                    symbol_name=file_path,
                    is_valid=False,
                    confidence=0.8,
                    reason=f"Invalid path '{file_path}'",
                    suggestions=[],
                    location_in_response=location,
                )
            if full_path.exists():
                return ValidationResult(
                    symbol_name=file_path,
                    is_valid=True,
                    confidence=1.0,
                    reason=f"File '{normalized}' exists in repository",
                    suggestions=[],
                    location_in_response=location,
                )

        # File not found - look for similar paths
        known_files = self._symbol_table.get_all_known_files()
        similar = self._find_similar_paths(normalized, known_files)

        if similar:
            return ValidationResult(
                symbol_name=file_path,
                is_valid=False,
                confidence=0.85,
                reason=f"File '{file_path}' not found. Similar files: {', '.join(similar[:3])}",
                suggestions=similar,
                location_in_response=location,
            )
        else:
            return ValidationResult(
                symbol_name=file_path,
                is_valid=False,
                confidence=0.8,
                reason=f"File '{file_path}' not found in repository",
                suggestions=[],
                location_in_response=location,
            )

    def _normalize_path(self, file_path: str) -> str:
        """Normalize a file path for consistent comparison.

        Removes leading slashes and normalizes separators.

        Args:
            file_path: The path to normalize

        Returns:
            Normalized path string
        """
        # Remove leading slashes and normalize
        path = file_path.lstrip("/\\")
        path = path.replace("\\", "/")

        # Handle common prefixes
        prefixes_to_remove = ["./"]
        for prefix in prefixes_to_remove:
            while path.startswith(prefix):
                path = path[len(prefix):]

        return path

    def _find_similar_paths(
        self,
        path: str,
        known_paths: list[str],
        limit: int = 5,
    ) -> list[str]:
        """Find similar file paths using filename matching.

        Args:
            path: The path to find matches for
            known_paths: List of known file paths
            limit: Maximum number of suggestions

        Returns:
            List of similar paths
        """
        # Extract filename for matching
        filename = os.path.basename(path)

        # Try exact filename match first
        exact_matches = [
            p for p in known_paths
            if os.path.basename(p) == filename
        ]
        if exact_matches:
            return exact_matches[:limit]

        # Try fuzzy matching on filename
        known_filenames = {os.path.basename(p): p for p in known_paths}
        similar_names = get_close_matches(
            filename,
            list(known_filenames.keys()),
            n=limit,
            cutoff=0.6,
        )

        return [known_filenames[n] for n in similar_names]

    def validate_directory(
        self,
        dir_path: str,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate a directory path against the repository.

        Args:
            dir_path: The directory path to validate
            location: Optional position in the response text

        Returns:
            ValidationResult with validation outcome
        """
        normalized = self._normalize_path(dir_path)

        # Check if any known files are in this directory
        known_files = self._symbol_table.get_all_known_files()
        has_files = any(
            f.startswith(normalized + "/") or f == normalized
            for f in known_files
        )

        if has_files:
            return ValidationResult(
                symbol_name=dir_path,
                is_valid=True,
                confidence=1.0,
                reason=f"Directory '{normalized}' contains known files",
                suggestions=[],
                location_in_response=location,
            )

        # Check filesystem
        if self._repo_root:
            full_path = Path(self._repo_root) / normalized
            # SECURITY: Prevent path traversal attacks
            try:
                resolved_path = full_path.resolve()
                repo_resolved = Path(self._repo_root).resolve()
                if not str(resolved_path).startswith(str(repo_resolved)):
                    logger.warning(
                        "path_traversal_blocked",
                        requested_path=dir_path,
                        resolved_path=str(resolved_path),
                    )
                    return ValidationResult(
                        symbol_name=dir_path,
                        is_valid=False,
                        confidence=1.0,
                        reason=f"Path '{dir_path}' is outside repository bounds",
                        suggestions=[],
                        location_in_response=location,
                    )
            except (OSError, ValueError) as e:
                logger.warning("path_resolution_failed", path=dir_path, error=str(e))
                return ValidationResult(
                    symbol_name=dir_path,
                    is_valid=False,
                    confidence=0.8,
                    reason=f"Invalid path '{dir_path}'",
                    suggestions=[],
                    location_in_response=location,
                )
            if full_path.is_dir():
                return ValidationResult(
                    symbol_name=dir_path,
                    is_valid=True,
                    confidence=1.0,
                    reason=f"Directory '{normalized}' exists in repository",
                    suggestions=[],
                    location_in_response=location,
                )

        return ValidationResult(
            symbol_name=dir_path,
            is_valid=False,
            confidence=0.8,
            reason=f"Directory '{dir_path}' not found in repository",
            suggestions=[],
            location_in_response=location,
        )
