#!/usr/bin/env python3
"""Export OpenAPI schema from FastAPI application.

This script exports the OpenAPI JSON schema without starting the server,
making it suitable for CI/CD pipelines and documentation generation.

Usage:
    # Export to stdout
    python backend/scripts/export_openapi.py

    # Export to file
    python backend/scripts/export_openapi.py --output docs/openapi.json

    # Pretty-print output
    python backend/scripts/export_openapi.py --pretty

    # Export YAML format (requires pyyaml)
    python backend/scripts/export_openapi.py --format yaml --output docs/openapi.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def export_openapi_schema(
    output_path: str | None = None,
    pretty: bool = False,
    format: str = "json",
) -> str | None:
    """Export the OpenAPI schema from the FastAPI application.

    Args:
        output_path: Optional path to write the schema to. If None, prints to stdout.
        pretty: Whether to pretty-print the JSON output.
        format: Output format ('json' or 'yaml').

    Returns:
        The schema string if output_path is None, otherwise None.
    """
    # Set environment variables to skip database connections during import
    os.environ["SKIP_DB_POOL"] = "1"
    os.environ["SKIP_GRAPHITI"] = "1"

    # Import the app after setting environment variables
    from agentic_rag_backend.main import app

    # Get OpenAPI schema
    schema = app.openapi()

    # Format output
    if format == "yaml":
        try:
            import yaml
            output = yaml.dump(schema, default_flow_style=False, allow_unicode=True)
        except ImportError:
            print("Error: pyyaml is required for YAML output. Install with: pip install pyyaml", file=sys.stderr)
            sys.exit(1)
    else:
        indent = 2 if pretty else None
        output = json.dumps(schema, indent=indent, ensure_ascii=False)

    # Write or return
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output)
        print(f"OpenAPI schema exported to: {output_path}", file=sys.stderr)
        return None
    else:
        return output


def main() -> None:
    """CLI entry point for OpenAPI schema export."""
    parser = argparse.ArgumentParser(
        description="Export OpenAPI schema from the Agentic RAG Backend API",
        epilog="Examples:\n"
               "  %(prog)s --pretty\n"
               "  %(prog)s --output docs/openapi.json\n"
               "  %(prog)s --format yaml --output docs/openapi.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "-p", "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    result = export_openapi_schema(
        output_path=args.output,
        pretty=args.pretty,
        format=args.format,
    )

    if result:
        print(result)


if __name__ == "__main__":
    main()
