"""Hallucination detector for codebase references in LLM responses.

Provides the main HallucinationDetector class that validates LLM responses
against a codebase symbol table to catch hallucinated code references.
"""

import json
import re
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

import structlog

from .extractor import SymbolExtractor
from .parser import ASTParser
from .symbol_table import SymbolTable
from .types import (
    CodeSymbol,
    HallucinationReport,
    ValidationResult,
    get_language_for_file,
)
from .validators.api_validator import APIEndpointValidator
from .validators.import_validator import ImportValidator
from .validators.path_validator import FilePathValidator
from .validators.symbol_validator import SymbolValidator

logger = structlog.get_logger(__name__)


class DetectorMode(str, Enum):
    """Operating modes for the hallucination detector."""

    WARN = "warn"  # Log warnings but allow response through
    BLOCK = "block"  # Block response if hallucinations exceed threshold


class HallucinationDetector:
    """Detect hallucinated code references in LLM responses.

    Validates that symbol names, file paths, API endpoints, and imports
    referenced in LLM responses actually exist in the indexed codebase.

    The detector can operate in two modes:
    - WARN: Log warnings for hallucinations but allow response through
    - BLOCK: Block responses that exceed the hallucination threshold

    Attributes:
        mode: Operating mode (warn or block)
        threshold: Hallucination ratio threshold for blocking (0.0-1.0)
        confidence_threshold: Minimum confidence for uncertain results
    """

    # Patterns for extracting code references from responses
    _PATTERNS = {
        # Function/method calls: function_name() or object.method()
        "function_call": re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE),
        # Class names: ClassName (PascalCase)
        "class_name": re.compile(r"\b([A-Z][a-zA-Z0-9]*)\b"),
        # File paths: src/module/file.py or ./path/to/file
        "file_path": re.compile(
            r"(?:^|[\s`\"'])([a-zA-Z0-9_./-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md))",
            re.MULTILINE,
        ),
        # API endpoints: /api/v1/resource or GET /users
        "api_endpoint": re.compile(
            r"(?:GET|POST|PUT|DELETE|PATCH)?\s*(/[a-zA-Z0-9_/-]+(?:\{[^}]+\})?)",
            re.IGNORECASE,
        ),
        # Import statements
        "import": re.compile(
            r"(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+)(?!\s+from))",
        ),
        # JavaScript/TypeScript imports: import x from "module" or import "module"
        "import_js": re.compile(
            r'import\s+(?:[^;]*?\s+from\s+)?["\']([^"\']+)["\']',
        ),
        "require_js": re.compile(
            r'require\(\s*["\']([^"\']+)["\']\s*\)',
        ),
        # Qualified names: module.class.method
        "qualified_name": re.compile(r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)\b", re.IGNORECASE),
    }

    # Common words to exclude from symbol validation
    _EXCLUDED_WORDS = {
        "the", "and", "for", "with", "this", "that", "from", "import", "class",
        "def", "return", "if", "else", "elif", "try", "except", "finally",
        "while", "for", "in", "is", "not", "or", "and", "none", "true", "false",
        "async", "await", "yield", "raise", "pass", "break", "continue",
        "lambda", "global", "nonlocal", "assert", "del", "as", "with",
        "http", "https", "api", "json", "data", "error", "status", "code",
        "response", "request", "result", "value", "key", "type", "name",
        "file", "path", "url", "id", "user", "users", "get", "post", "put",
        "delete", "patch", "create", "update", "read", "list", "item",
        "module", "package", "library", "function", "method", "variable",
    }

    def __init__(
        self,
        symbol_table: SymbolTable,
        mode: DetectorMode = DetectorMode.WARN,
        threshold: float = 0.3,
        confidence_threshold: float = 0.7,
        openapi_spec: Optional[dict] = None,
    ) -> None:
        """Initialize the hallucination detector.

        Args:
            symbol_table: SymbolTable containing indexed codebase symbols
            mode: Operating mode (warn or block)
            threshold: Hallucination ratio threshold for blocking (0.0-1.0)
            confidence_threshold: Minimum confidence for uncertain results
            openapi_spec: Optional OpenAPI specification for API validation
        """
        self._symbol_table = symbol_table
        self._mode = mode
        self._threshold = threshold
        self._confidence_threshold = confidence_threshold

        # Initialize validators
        self._symbol_validator = SymbolValidator(symbol_table)
        self._path_validator = FilePathValidator(symbol_table)
        self._api_validator = APIEndpointValidator(symbol_table, openapi_spec)
        self._import_validator = ImportValidator(symbol_table)

        logger.info(
            "hallucination_detector_initialized",
            mode=mode.value,
            threshold=threshold,
            symbol_count=symbol_table.symbol_count(),
            file_count=symbol_table.file_count(),
        )

    @property
    def mode(self) -> DetectorMode:
        """Get the current operating mode."""
        return self._mode

    @mode.setter
    def mode(self, value: DetectorMode) -> None:
        """Set the operating mode."""
        self._mode = value

    @property
    def threshold(self) -> float:
        """Get the hallucination threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the hallucination threshold (0.0-1.0)."""
        self._threshold = max(0.0, min(1.0, value))

    def validate_response(
        self,
        response_text: str,
        tenant_id: Optional[str] = None,
    ) -> HallucinationReport:
        """Validate an LLM response for hallucinated code references.

        Extracts code references from the response text and validates
        each one against the symbol table and validators.

        Args:
            response_text: The LLM response text to validate
            tenant_id: Optional tenant identifier for multi-tenancy

        Returns:
            HallucinationReport with validation results and metrics
        """
        start_time = time.time()
        validation_results: list[ValidationResult] = []
        files_checked: set[str] = set()

        # Extract and validate different types of references
        validation_results.extend(self._validate_symbols(response_text))
        validation_results.extend(self._validate_paths(response_text, files_checked))
        validation_results.extend(self._validate_api_endpoints(response_text))
        validation_results.extend(self._validate_imports(response_text))

        # Calculate metrics
        total = len(validation_results)
        valid = sum(1 for r in validation_results if r.is_valid)
        invalid = sum(1 for r in validation_results if not r.is_valid and r.confidence >= self._confidence_threshold)
        uncertain = total - valid - invalid

        # Calculate confidence score (higher is better)
        if total > 0:
            confidence_score = valid / total
        else:
            confidence_score = 1.0  # No references to validate

        processing_time_ms = int((time.time() - start_time) * 1000)

        report = HallucinationReport(
            total_symbols_checked=total,
            valid_symbols=valid,
            invalid_symbols=invalid,
            uncertain_symbols=uncertain,
            validation_results=validation_results,
            files_checked=sorted(files_checked),
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            response_text=response_text[:1000] if len(response_text) > 1000 else response_text,
            tenant_id=tenant_id,
        )

        # Log based on mode
        if invalid > 0:
            hallucination_ratio = invalid / total if total > 0 else 0
            if hallucination_ratio > self._threshold:
                logger.warning(
                    "hallucinations_detected",
                    mode=self._mode.value,
                    invalid=invalid,
                    total=total,
                    ratio=hallucination_ratio,
                    threshold=self._threshold,
                    should_block=self._mode == DetectorMode.BLOCK,
                )
            else:
                logger.info(
                    "validation_complete",
                    valid=valid,
                    invalid=invalid,
                    uncertain=uncertain,
                    confidence=confidence_score,
                )
        else:
            logger.debug(
                "validation_complete_clean",
                total=total,
                processing_time_ms=processing_time_ms,
            )

        return report

    def should_block(self, report: HallucinationReport) -> bool:
        """Determine if the response should be blocked based on the report.

        Args:
            report: HallucinationReport from validate_response()

        Returns:
            True if mode is BLOCK and hallucination ratio exceeds threshold
        """
        if self._mode != DetectorMode.BLOCK:
            return False

        if report.total_symbols_checked == 0:
            return False

        hallucination_ratio = report.invalid_symbols / report.total_symbols_checked
        return hallucination_ratio > self._threshold

    def _validate_symbols(self, text: str) -> list[ValidationResult]:
        """Extract and validate symbol references from text.

        Args:
            text: The text to extract symbols from

        Returns:
            List of validation results for symbol references
        """
        results = []
        validated: set[str] = set()

        # Extract function calls
        for match in self._PATTERNS["function_call"].finditer(text):
            name = match.group(1)
            if name.lower() not in self._EXCLUDED_WORDS and name not in validated:
                validated.add(name)
                result = self._symbol_validator.validate(
                    name,
                    location=(match.start(1), match.end(1)),
                    context={"expected_type": "function"},
                )
                results.append(result)

        # Extract class names
        for match in self._PATTERNS["class_name"].finditer(text):
            name = match.group(1)
            # Skip common type names and short names
            if (
                len(name) > 2
                and name.lower() not in self._EXCLUDED_WORDS
                and name not in validated
                and not name.isupper()  # Skip constants
            ):
                validated.add(name)
                result = self._symbol_validator.validate(
                    name,
                    location=(match.start(1), match.end(1)),
                    context={"expected_type": "class"},
                )
                results.append(result)

        # Extract qualified names
        for match in self._PATTERNS["qualified_name"].finditer(text):
            name = match.group(1)
            if name not in validated:
                validated.add(name)
                result = self._symbol_validator.validate_qualified(
                    name,
                    location=(match.start(1), match.end(1)),
                )
                results.append(result)

        return results

    def _validate_paths(
        self,
        text: str,
        files_checked: set[str],
    ) -> list[ValidationResult]:
        """Extract and validate file path references from text.

        Args:
            text: The text to extract paths from
            files_checked: Set to collect validated file paths

        Returns:
            List of validation results for file path references
        """
        results = []
        validated: set[str] = set()

        for match in self._PATTERNS["file_path"].finditer(text):
            path = match.group(1)
            if path not in validated:
                validated.add(path)
                files_checked.add(path)
                result = self._path_validator.validate(
                    path,
                    location=(match.start(1), match.end(1)),
                )
                results.append(result)

        return results

    def _validate_api_endpoints(self, text: str) -> list[ValidationResult]:
        """Extract and validate API endpoint references from text.

        Args:
            text: The text to extract endpoints from

        Returns:
            List of validation results for API endpoint references
        """
        results = []
        validated: set[str] = set()

        for match in self._PATTERNS["api_endpoint"].finditer(text):
            endpoint = match.group(1)
            # Skip common path-like text that isn't an API endpoint
            if endpoint.startswith("/") and endpoint not in validated:
                validated.add(endpoint)
                # Try to extract method if present
                full_match = match.group(0).strip()
                method = None
                for m in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    if full_match.upper().startswith(m):
                        method = m
                        break
                result = self._api_validator.validate(
                    endpoint,
                    location=(match.start(1), match.end(1)),
                    context={"method": method} if method else None,
                )
                results.append(result)

        return results

    def _validate_imports(self, text: str) -> list[ValidationResult]:
        """Extract and validate import statements from text.

        Args:
            text: The text to extract imports from

        Returns:
            List of validation results for import references
        """
        results = []
        validated: set[str] = set()

        for match in self._PATTERNS["import_js"].finditer(text):
            module = match.group(1)
            if module and module not in validated:
                validated.add(module)
                result = self._import_validator.validate(
                    module,
                    location=(match.start(1), match.end(1)),
                )
                results.append(result)

        for match in self._PATTERNS["require_js"].finditer(text):
            module = match.group(1)
            if module and module not in validated:
                validated.add(module)
                result = self._import_validator.validate(
                    module,
                    location=(match.start(1), match.end(1)),
                )
                results.append(result)

        for match in self._PATTERNS["import"].finditer(text):
            # Get either from-import or regular import
            module = match.group(1) or match.group(2)
            if module and module not in validated:
                validated.add(module)
                result = self._import_validator.validate(
                    module,
                    location=(match.start(), match.end()),
                )
                results.append(result)

        return results

    def add_api_routes_from_openapi(self, openapi_spec: dict) -> None:
        """Load API routes from an OpenAPI specification.

        Args:
            openapi_spec: OpenAPI specification dictionary
        """
        paths = openapi_spec.get("paths", {})
        for path, methods in paths.items():
            method_list = [
                m.upper()
                for m in methods.keys()
                if m.lower() not in ("parameters", "servers", "summary", "description")
            ]
            self._api_validator.add_route(path, method_list)

    def add_installed_packages(self, packages: list[str]) -> None:
        """Add known installed packages for import validation.

        Args:
            packages: List of package names
        """
        for pkg in packages:
            self._import_validator.add_installed_package(pkg)


def _extract_requirement_name(requirement: str) -> Optional[str]:
    """Extract a package name from a dependency string."""
    entry = requirement.strip()
    if not entry or entry.startswith("#"):
        return None
    if entry.startswith("-r") or entry.startswith("--"):
        return None
    if entry.startswith("-e "):
        entry = entry[3:].strip()

    # Drop environment markers (PEP 508)
    entry = entry.split(";", 1)[0].strip()

    # Handle direct references like "package @ https://..."
    if "@" in entry and "://" in entry:
        entry = entry.split("@", 1)[0].strip()

    # Remove extras and version specifiers
    entry = entry.split("[", 1)[0].strip()
    for sep in ("==", ">=", "<=", "!=", "~=", ">", "<"):
        if sep in entry:
            entry = entry.split(sep, 1)[0].strip()
            break

    return entry or None


def _load_pyproject_dependencies(pyproject_path: Path) -> set[str]:
    """Load dependencies from pyproject.toml."""
    if not pyproject_path.exists():
        return set()

    try:
        import tomllib
    except ImportError:
        logger.warning("tomllib_unavailable", path=str(pyproject_path))
        return set()

    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("pyproject_parse_failed", path=str(pyproject_path), error=str(e))
        return set()

    deps: set[str] = set()

    project = data.get("project", {})
    for entry in project.get("dependencies", []) or []:
        name = _extract_requirement_name(entry)
        if name:
            deps.add(name)

    optional = project.get("optional-dependencies", {}) or {}
    if isinstance(optional, dict):
        for entries in optional.values():
            for entry in entries or []:
                name = _extract_requirement_name(entry)
                if name:
                    deps.add(name)

    poetry = data.get("tool", {}).get("poetry", {}) or {}
    poetry_deps = poetry.get("dependencies", {}) or {}
    if isinstance(poetry_deps, dict):
        for name in poetry_deps.keys():
            if name and name.lower() != "python":
                deps.add(name)

    poetry_groups = poetry.get("group", {}) or {}
    if isinstance(poetry_groups, dict):
        for group in poetry_groups.values():
            group_deps = group.get("dependencies", {}) if isinstance(group, dict) else {}
            for name in group_deps.keys():
                if name and name.lower() != "python":
                    deps.add(name)

    return deps


def _load_requirements(requirements_path: Path) -> set[str]:
    """Load dependencies from a requirements.txt-style file."""
    if not requirements_path.exists():
        return set()

    try:
        content = requirements_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("requirements_read_failed", path=str(requirements_path), error=str(e))
        return set()

    deps: set[str] = set()
    for line in content.splitlines():
        name = _extract_requirement_name(line)
        if name:
            deps.add(name)
    return deps


def _load_package_json(package_json_path: Path) -> set[str]:
    """Load dependencies from package.json."""
    if not package_json_path.exists():
        return set()

    try:
        data = json.loads(package_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("package_json_parse_failed", path=str(package_json_path), error=str(e))
        return set()

    deps: set[str] = set()
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        section_deps = data.get(section, {})
        if isinstance(section_deps, dict):
            deps.update(section_deps.keys())
    return deps


@lru_cache(maxsize=64)
def load_repo_dependencies(repo_path: str) -> set[str]:
    """Load dependencies declared in the repository."""
    if not repo_path:
        return set()

    root = Path(repo_path)
    if not root.exists():
        return set()

    candidate_roots = [
        root,
        root / "backend",
        root / "frontend",
    ]

    deps: set[str] = set()
    for candidate in candidate_roots:
        deps.update(_load_pyproject_dependencies(candidate / "pyproject.toml"))
        deps.update(_load_requirements(candidate / "requirements.txt"))
        deps.update(_load_requirements(candidate / "requirements-dev.txt"))
        deps.update(_load_package_json(candidate / "package.json"))

    return deps


async def index_repository(
    repo_path: str,
    tenant_id: str,
    ignore_patterns: Optional[list[str]] = None,
) -> SymbolTable:
    """Index a repository and create a symbol table.

    Scans all supported source files in the repository, extracts symbols,
    and builds a symbol table for hallucination detection.

    Args:
        repo_path: Path to the repository root
        tenant_id: Tenant identifier for multi-tenancy
        ignore_patterns: Optional list of glob patterns to ignore

    Returns:
        Populated SymbolTable for the repository

    Raises:
        ValueError: If repo_path is invalid or contains path traversal attempts
        FileNotFoundError: If repo_path does not exist
    """
    import aiofiles.os

    from gitignore_parser import parse_gitignore

    # SECURITY: Validate repo_path to prevent path traversal and SSRF
    repo_path_obj = Path(repo_path)

    # Must be absolute path
    if not repo_path_obj.is_absolute():
        raise ValueError(f"repo_path must be an absolute path, got: {repo_path}")

    # Check for path traversal attempts in the input
    if ".." in repo_path:
        raise ValueError(f"repo_path cannot contain '..': {repo_path}")

    # Resolve to canonical path and check it still matches
    try:
        resolved_path = repo_path_obj.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not resolved_path.is_dir():
        raise ValueError(f"repo_path must be a directory: {repo_path}")

    # Use resolved path for all operations
    repo_path = str(resolved_path)

    symbol_table = SymbolTable(tenant_id, repo_path)
    parser = ASTParser()
    extractor = SymbolExtractor(parser)

    # Default ignore patterns
    default_ignore = [
        "node_modules/",
        ".git/",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        "venv/",
        "dist/",
        "build/",
        ".next/",
    ]
    ignore_patterns = (ignore_patterns or []) + default_ignore

    # Try to load .gitignore
    gitignore_path = Path(repo_path) / ".gitignore"
    gitignore_matcher = None
    if gitignore_path.exists():
        try:
            gitignore_matcher = parse_gitignore(str(gitignore_path))
        except Exception as e:
            logger.warning("gitignore_parse_failed", error=str(e))

    def should_ignore(path: Path) -> bool:
        """Check if a path should be ignored."""
        rel_path = str(path.relative_to(repo_path))

        # Check gitignore
        if gitignore_matcher and gitignore_matcher(str(path)):
            return True

        # Check custom patterns
        for pattern in ignore_patterns:
            if pattern.endswith("/"):
                if f"/{pattern[:-1]}/" in f"/{rel_path}/" or rel_path.startswith(pattern[:-1] + "/"):
                    return True
            else:
                import fnmatch
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path.name, pattern):
                    return True

        return False

    async def scan_directory(dir_path: Path) -> None:
        """Recursively scan a directory for source files."""
        try:
            entries = await aiofiles.os.scandir(str(dir_path))
            for entry in entries:
                entry_path = Path(entry.path)

                if should_ignore(entry_path):
                    continue

                if entry.is_file():
                    lang = get_language_for_file(entry.name)
                    if lang:
                        rel_path = str(entry_path.relative_to(repo_path))
                        symbol_table.add_known_file(rel_path)

                        try:
                            symbols = extractor.extract_from_file(str(entry_path))
                            for symbol in symbols:
                                # Update file_path to be relative
                                updated = CodeSymbol(
                                    name=symbol.name,
                                    type=symbol.type,
                                    scope=symbol.scope,
                                    file_path=rel_path,
                                    line_start=symbol.line_start,
                                    line_end=symbol.line_end,
                                    signature=symbol.signature,
                                    parent=symbol.parent,
                                    docstring=symbol.docstring,
                                    qualified_name=symbol.qualified_name,
                                )
                                symbol_table.add(updated)
                        except Exception as e:
                            logger.warning(
                                "file_extraction_failed",
                                file=rel_path,
                                error=str(e),
                            )

                elif entry.is_dir():
                    await scan_directory(entry_path)

        except (OSError, PermissionError) as e:
            logger.warning("directory_scan_failed", path=str(dir_path), error=str(e))

    await scan_directory(Path(repo_path))

    logger.info(
        "repository_indexed",
        repo_path=repo_path,
        tenant_id=tenant_id,
        symbol_count=symbol_table.symbol_count(),
        file_count=symbol_table.file_count(),
    )

    return symbol_table
