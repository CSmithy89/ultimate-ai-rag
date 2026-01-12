#!/usr/bin/env python3
"""RFC 7807 Error Response Linter.

This script analyzes Python source code to enforce RFC 7807 Problem Details
compliance for error responses.

Story: 18-11 - Implement RFC 7807 Linter Rule
Origin: Epic 14 Retrospective recommendation

Checks performed:
1. Direct HTTPException usage (should use AppError subclasses)
2. Custom exception handlers should return proper RFC 7807 format
3. Report any non-compliant error handling patterns

Exit codes:
- 0: No violations found
- 1: Violations found (warnings only, configurable)
- 2: Critical violations found (errors)

Usage:
    python scripts/lint_rfc7807.py [--strict] [--fix-suggestions] [path...]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# RFC 7807 required fields
RFC7807_REQUIRED_FIELDS = {"type", "title", "status", "detail", "instance"}

# AppError subclasses that are allowed (imported from core.errors)
ALLOWED_ERROR_CLASSES = {
    "AppError",
    "ValidationError",
    "InvalidUrlError",
    "JobNotFoundError",
    "TenantRequiredError",
    "CrawlError",
    "DatabaseError",
    "RedisError",
    "InvalidPdfError",
    "FileTooLargeError",
    "PasswordProtectedError",
    "ParseError",
    "StorageError",
    "ExtractionError",
    "EmbeddingError",
    "GraphBuildError",
    "Neo4jError",
    "DeduplicationError",
    "ChunkingError",
    "IngestionError",
    "CodebaseValidationError",
    "CodebaseIndexError",
    "HallucinationError",
    "A2AAgentNotFoundError",
    "A2AAgentUnhealthyError",
    "A2ACapabilityNotFoundError",
    "A2ATaskNotFoundError",
    "A2ATaskTimeoutError",
    "A2ADelegationError",
    "A2ARegistrationError",
    "A2APermissionError",
    "A2AAuthFailedError",
    "A2AServiceUnavailableError",
    "A2ASessionLimitExceededError",
    "A2AMessageLimitExceededError",
    "A2ARateLimitExceededError",
    # Memory errors
    "MemoryNotFoundError",
    "MemoryScopeInvalidError",
    "MemoryLimitExceededError",
}

# Files/patterns to exclude from strict checking
EXCLUDED_PATTERNS = {
    # Third-party protocol implementations that may need HTTPException
    "mcp_server/",
    # Test files may intentionally raise HTTPException
    "tests/",
}


@dataclass
class Violation:
    """Represents an RFC 7807 compliance violation."""

    file: Path
    line: int
    column: int
    message: str
    severity: str = "warning"  # "warning" or "error"
    suggestion: str = ""


@dataclass
class LintResult:
    """Aggregated lint results."""

    violations: list[Violation] = field(default_factory=list)
    files_checked: int = 0
    files_with_violations: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level violations."""
        return any(v.severity == "error" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level violations."""
        return any(v.severity == "warning" for v in self.violations)


class RFC7807Visitor(ast.NodeVisitor):
    """AST visitor that checks for RFC 7807 compliance violations."""

    def __init__(self, file_path: Path, strict: bool = False) -> None:
        self.file_path = file_path
        self.strict = strict
        self.violations: list[Violation] = []
        self._in_exception_handler = False
        self._imports: dict[str, str] = {}  # alias -> full name
        self._from_imports: dict[str, str] = {}  # name -> module

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-import statements."""
        module = node.module or ""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._from_imports[name] = module
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        """Check raise statements for HTTPException usage."""
        if node.exc is None:
            self.generic_visit(node)
            return

        # Check for HTTPException raises
        exc_name = self._get_exception_name(node.exc)

        if exc_name == "HTTPException":
            # Check if file is in excluded patterns
            is_excluded = any(
                pattern in str(self.file_path) for pattern in EXCLUDED_PATTERNS
            )

            severity = "error" if self.strict and not is_excluded else "warning"

            # Try to extract status_code for better suggestion
            suggestion = self._generate_suggestion(node.exc)

            self.violations.append(
                Violation(
                    file=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    message=(
                        "Direct HTTPException usage detected. "
                        "Consider using AppError subclass for RFC 7807 compliance."
                    ),
                    severity=severity,
                    suggestion=suggestion,
                )
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for JSONResponse calls that might be error responses."""
        func_name = self._get_call_name(node)

        if func_name == "JSONResponse":
            # Check if this looks like an error response
            self._check_json_response(node)

        self.generic_visit(node)

    def _get_exception_name(self, node: ast.expr) -> str | None:
        """Extract the exception class name from a raise statement."""
        if isinstance(node, ast.Call):
            return self._get_call_name(node)
        elif isinstance(node, ast.Name):
            return node.id
        return None

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the function/class name from a call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _generate_suggestion(self, node: ast.expr) -> str:
        """Generate a suggestion for replacing HTTPException."""
        status_code: int | None = None

        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "status_code" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, int):
                        status_code = kw.value.value

        if status_code is not None:
            suggestions: dict[int, str] = {
                400: "ValidationError or InvalidUrlError",
                401: "TenantRequiredError or A2AAuthFailedError",
                403: "A2APermissionError",
                404: "JobNotFoundError, A2AAgentNotFoundError, or A2ATaskNotFoundError",
                409: "Consider creating a ConflictError(AppError) subclass",
                413: "FileTooLargeError",
                422: "ValidationError or HallucinationError",
                429: "A2ARateLimitExceededError or A2AMessageLimitExceededError",
                500: "DatabaseError, RedisError, or appropriate domain error",
                503: "A2AServiceUnavailableError or A2AAgentUnhealthyError",
                504: "A2ATaskTimeoutError",
            }
            if status_code in suggestions:
                return f"Consider using: {suggestions[status_code]}"

        return "Consider creating an appropriate AppError subclass"

    def _check_json_response(self, node: ast.Call) -> None:
        """Check if a JSONResponse contains RFC 7807 fields."""
        content_dict = None

        for kw in node.keywords:
            if kw.arg == "content" and isinstance(kw.value, ast.Dict):
                content_dict = kw.value
                break

        if content_dict is None:
            return

        # Check if this looks like an error response (has status_code >= 400)
        for kw in node.keywords:
            if kw.arg == "status_code" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, int) and kw.value.value >= 400:
                    # This is likely an error response, check for RFC 7807 fields
                    found_fields = set()
                    for key in content_dict.keys:
                        if isinstance(key, ast.Constant) and isinstance(
                            key.value, str
                        ):
                            found_fields.add(key.value)

                    missing_fields = RFC7807_REQUIRED_FIELDS - found_fields

                    if missing_fields and "detail" not in found_fields:
                        # Only warn if it doesn't look like a proper RFC 7807 response
                        self.violations.append(
                            Violation(
                                file=self.file_path,
                                line=node.lineno,
                                column=node.col_offset,
                                message=(
                                    f"Error response may be missing RFC 7807 fields: "
                                    f"{', '.join(sorted(missing_fields))}"
                                ),
                                severity="warning",
                                suggestion=(
                                    "Use AppError.to_problem_detail() or ensure "
                                    "response includes: type, title, status, detail, instance"
                                ),
                            )
                        )
                    break


def find_python_files(paths: list[Path]) -> Iterator[Path]:
    """Find all Python files in the given paths."""
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


def lint_file(file_path: Path, strict: bool = False) -> list[Violation]:
    """Lint a single Python file for RFC 7807 compliance."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return [
            Violation(
                file=file_path,
                line=e.lineno or 0,
                column=e.offset or 0,
                message=f"Syntax error: {e.msg}",
                severity="error",
            )
        ]
    except Exception as e:
        return [
            Violation(
                file=file_path,
                line=0,
                column=0,
                message=f"Failed to parse file: {e}",
                severity="error",
            )
        ]

    visitor = RFC7807Visitor(file_path, strict=strict)
    visitor.visit(tree)
    return visitor.violations


def lint_paths(
    paths: list[Path], strict: bool = False, exclude: list[str] | None = None
) -> LintResult:
    """Lint all Python files in the given paths."""
    result = LintResult()
    exclude_patterns = set(exclude or [])

    for file_path in find_python_files(paths):
        # Skip excluded patterns
        if any(pattern in str(file_path) for pattern in exclude_patterns):
            continue

        result.files_checked += 1
        violations = lint_file(file_path, strict=strict)

        if violations:
            result.files_with_violations += 1
            result.violations.extend(violations)

    return result


def format_violation(violation: Violation, show_suggestion: bool = False) -> str:
    """Format a violation for output."""
    severity_prefix = "ERROR" if violation.severity == "error" else "WARNING"
    output = (
        f"{violation.file}:{violation.line}:{violation.column}: "
        f"{severity_prefix}: {violation.message}"
    )

    if show_suggestion and violation.suggestion:
        output += f"\n  Suggestion: {violation.suggestion}"

    return output


def main() -> int:
    """Main entry point for the linter."""
    parser = argparse.ArgumentParser(
        description="RFC 7807 Error Response Linter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("src")],
        help="Paths to lint (default: src)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat HTTPException usage as errors (exit code 2)",
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show suggestions for fixing violations",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Patterns to exclude from linting (can be specified multiple times)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show summary, not individual violations",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "github"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # Ensure paths exist
    for path in args.paths:
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            return 2

    result = lint_paths(args.paths, strict=args.strict, exclude=args.exclude)

    # Output results
    if args.format == "json":
        import json

        output = {
            "files_checked": result.files_checked,
            "files_with_violations": result.files_with_violations,
            "violations": [
                {
                    "file": str(v.file),
                    "line": v.line,
                    "column": v.column,
                    "message": v.message,
                    "severity": v.severity,
                    "suggestion": v.suggestion,
                }
                for v in result.violations
            ],
        }
        print(json.dumps(output, indent=2))
    elif args.format == "github":
        # GitHub Actions annotation format
        for v in result.violations:
            level = "error" if v.severity == "error" else "warning"
            print(f"::{level} file={v.file},line={v.line},col={v.column}::{v.message}")
    else:
        if not args.quiet:
            for violation in result.violations:
                print(format_violation(violation, args.fix_suggestions))

        # Summary
        print(f"\n{'=' * 60}")
        print("RFC 7807 Compliance Report")
        print(f"{'=' * 60}")
        print(f"Files checked: {result.files_checked}")
        print(f"Files with violations: {result.files_with_violations}")
        print(f"Total violations: {len(result.violations)}")

        errors = sum(1 for v in result.violations if v.severity == "error")
        warnings = sum(1 for v in result.violations if v.severity == "warning")
        print(f"  - Errors: {errors}")
        print(f"  - Warnings: {warnings}")

        if result.violations:
            print("\nNote: Use AppError subclasses from core.errors for RFC 7807 compliance.")
            if not args.fix_suggestions:
                print("Run with --fix-suggestions to see recommended fixes.")

    # Exit code
    if result.has_errors:
        return 2
    elif result.has_warnings and args.strict:
        return 1
    elif result.has_warnings:
        return 0  # Warnings only, non-strict mode
    return 0


if __name__ == "__main__":
    sys.exit(main())
