"""API endpoint validator for codebase hallucination detection.

Validates that API endpoint references match defined routes.
"""

import re
from typing import Optional

import structlog

from ..symbol_table import SymbolTable
from ..types import SymbolType, ValidationResult
from .base import BaseValidator

logger = structlog.get_logger(__name__)


class APIEndpointValidator(BaseValidator):
    """Validator for API endpoint references.

    Checks that referenced API endpoints (paths and methods) exist
    in the codebase. Can validate against FastAPI route definitions
    or OpenAPI specifications.

    Attributes:
        symbol_table: The SymbolTable containing route definitions
        routes: Cached API route patterns
    """

    def __init__(
        self,
        symbol_table: SymbolTable,
        openapi_spec: Optional[dict] = None,
    ) -> None:
        """Initialize the API endpoint validator.

        Args:
            symbol_table: SymbolTable containing indexed codebase symbols
            openapi_spec: Optional OpenAPI specification dictionary
        """
        self._symbol_table = symbol_table
        self._openapi_spec = openapi_spec
        self._routes: dict[str, list[str]] = {}  # path -> list of methods
        self._route_patterns: list[tuple[re.Pattern, list[str]]] = []

        if openapi_spec:
            self._load_from_openapi(openapi_spec)

    @property
    def validator_type(self) -> str:
        """Return the validator type identifier."""
        return "api"

    def _load_from_openapi(self, spec: dict) -> None:
        """Load API routes from OpenAPI specification.

        Args:
            spec: OpenAPI specification dictionary
        """
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            # Convert OpenAPI path params to regex
            pattern = self._path_to_regex(path)
            method_list = [
                m.upper()
                for m in methods.keys()
                if m.lower() not in ("parameters", "servers", "summary", "description")
            ]
            self._routes[path] = method_list
            self._route_patterns.append((pattern, method_list))

        logger.info(
            "openapi_routes_loaded",
            route_count=len(self._routes),
        )

    def _path_to_regex(self, path: str) -> re.Pattern:
        """Convert an OpenAPI path to a regex pattern.

        Args:
            path: OpenAPI path (e.g., '/users/{id}')

        Returns:
            Compiled regex pattern
        """
        # Escape special regex chars, then convert path params
        escaped = re.escape(path)
        # Convert {param} placeholders to regex groups
        pattern = re.sub(r"\\{[^}]+\\}", r"[^/]+", escaped)
        return re.compile(f"^{pattern}$")

    def add_route(
        self,
        path: str,
        methods: list[str],
    ) -> None:
        """Add an API route for validation.

        Args:
            path: The API path (e.g., '/api/v1/users')
            methods: List of HTTP methods (e.g., ['GET', 'POST'])
        """
        self._routes[path] = [m.upper() for m in methods]
        pattern = self._path_to_regex(path)
        self._route_patterns.append((pattern, [m.upper() for m in methods]))

    def add_routes_from_fastapi_symbols(self) -> int:
        """Extract API routes from FastAPI decorator symbols.

        Scans the symbol table for function symbols with FastAPI
        decorator patterns and extracts route information.

        Returns:
            Number of routes added
        """
        count = 0
        functions = self._symbol_table.get_symbols_by_type(SymbolType.FUNCTION)
        methods = self._symbol_table.get_symbols_by_type(SymbolType.METHOD)

        route_decorators = {
            "@router.get": "GET",
            "@router.post": "POST",
            "@router.put": "PUT",
            "@router.delete": "DELETE",
            "@router.patch": "PATCH",
            "@app.get": "GET",
            "@app.post": "POST",
            "@app.put": "PUT",
            "@app.delete": "DELETE",
            "@app.patch": "PATCH",
        }

        for symbol in functions + methods:
            if symbol.signature:
                # Try to extract route info from signature/docstring
                # This is a simplified heuristic
                for decorator, method in route_decorators.items():
                    if symbol.docstring and decorator in symbol.docstring:
                        # Extract path from decorator
                        path_match = re.search(
                            rf'{decorator}\s*\(\s*["\']([^"\']+)["\']',
                            symbol.docstring,
                        )
                        if path_match:
                            path = path_match.group(1)
                            if path not in self._routes:
                                self._routes[path] = []
                            if method not in self._routes[path]:
                                self._routes[path].append(method)
                                count += 1

        logger.info(
            "fastapi_routes_extracted",
            route_count=count,
        )
        return count

    def validate(
        self,
        reference: str,
        location: Optional[tuple[int, int]] = None,
        context: Optional[dict] = None,
    ) -> ValidationResult:
        """Validate an API endpoint reference.

        Args:
            reference: The API endpoint to validate (path or path+method)
            location: Optional position in the response text
            context: Optional context with 'method' key

        Returns:
            ValidationResult indicating if the endpoint exists
        """
        method = None
        if context and "method" in context:
            method = context["method"].upper()

        return self.validate_endpoint(reference, method, location)

    def validate_endpoint(
        self,
        path: str,
        method: Optional[str] = None,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate an API endpoint path and optional method.

        Args:
            path: The API path to validate
            method: Optional HTTP method to validate
            location: Optional position in the response text

        Returns:
            ValidationResult with validation outcome
        """
        # Normalize the path
        normalized = path.strip()
        if not normalized.startswith("/"):
            normalized = "/" + normalized

        # Check exact match first
        if normalized in self._routes:
            if method:
                if method.upper() in self._routes[normalized]:
                    return ValidationResult(
                        symbol_name=f"{method.upper()} {path}",
                        is_valid=True,
                        confidence=1.0,
                        reason=f"Endpoint '{method.upper()} {normalized}' found",
                        suggestions=[],
                        location_in_response=location,
                    )
                else:
                    valid_methods = self._routes[normalized]
                    return ValidationResult(
                        symbol_name=f"{method.upper()} {path}",
                        is_valid=False,
                        confidence=0.9,
                        reason=f"Path '{normalized}' exists but method '{method}' not allowed. Valid: {', '.join(valid_methods)}",
                        suggestions=[f"{m} {normalized}" for m in valid_methods],
                        location_in_response=location,
                    )
            else:
                return ValidationResult(
                    symbol_name=path,
                    is_valid=True,
                    confidence=1.0,
                    reason=f"Endpoint path '{normalized}' found",
                    suggestions=[],
                    location_in_response=location,
                )

        # Check pattern matches (for parameterized routes)
        for pattern, methods in self._route_patterns:
            if pattern.match(normalized):
                if method:
                    if method.upper() in methods:
                        return ValidationResult(
                            symbol_name=f"{method.upper()} {path}",
                            is_valid=True,
                            confidence=0.95,
                            reason=f"Endpoint '{method.upper()} {normalized}' matches pattern",
                            suggestions=[],
                            location_in_response=location,
                        )
                    else:
                        return ValidationResult(
                            symbol_name=f"{method.upper()} {path}",
                            is_valid=False,
                            confidence=0.9,
                            reason=f"Path '{normalized}' matches pattern but method '{method}' not allowed",
                            suggestions=[f"{m} {normalized}" for m in methods],
                            location_in_response=location,
                        )
                else:
                    return ValidationResult(
                        symbol_name=path,
                        is_valid=True,
                        confidence=0.95,
                        reason=f"Endpoint path '{normalized}' matches pattern",
                        suggestions=[],
                        location_in_response=location,
                    )

        # Not found - suggest similar paths
        similar = self._find_similar_paths(normalized)

        if similar:
            return ValidationResult(
                symbol_name=f"{method + ' ' if method else ''}{path}",
                is_valid=False,
                confidence=0.8,
                reason=f"Endpoint '{normalized}' not found. Similar: {', '.join(similar[:3])}",
                suggestions=similar,
                location_in_response=location,
            )
        else:
            # If no routes are loaded, low confidence in result
            confidence = 0.5 if not self._routes else 0.8
            return ValidationResult(
                symbol_name=f"{method + ' ' if method else ''}{path}",
                is_valid=False,
                confidence=confidence,
                reason=f"Endpoint '{normalized}' not found in API definition",
                suggestions=[],
                location_in_response=location,
            )

    def _find_similar_paths(self, path: str, limit: int = 5) -> list[str]:
        """Find similar API paths.

        Args:
            path: The path to find matches for
            limit: Maximum number of suggestions

        Returns:
            List of similar paths
        """
        from difflib import get_close_matches

        all_paths = list(self._routes.keys())
        return get_close_matches(path, all_paths, n=limit, cutoff=0.6)

    def get_all_routes(self) -> dict[str, list[str]]:
        """Get all registered API routes.

        Returns:
            Dictionary mapping paths to lists of methods
        """
        return dict(self._routes)
