"""CLI entry point for retrieval quality benchmarks.

Usage:
    uv run benchmark-retrieval --help
    uv run benchmark-retrieval --dataset tests/benchmarks/data/eval_dataset.json
    uv run benchmark-retrieval --k 1 5 10 --strategies vector hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .datasets import create_sample_dataset, load_dataset, save_dataset
from .runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    RetrievalStrategy,
    compare_to_baseline,
    load_baseline,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional list of arguments (uses sys.argv if None)

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="benchmark-retrieval",
        description="Run retrieval quality benchmarks with standard IR metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (sample dataset, k=1,5,10, vector strategy)
  uv run benchmark-retrieval

  # Run with custom dataset
  uv run benchmark-retrieval --dataset tests/benchmarks/data/eval_dataset.json

  # Run with multiple strategies
  uv run benchmark-retrieval --strategies vector graph hybrid

  # Run with custom K values
  uv run benchmark-retrieval --k 1 3 5 10 20

  # Generate sample dataset
  uv run benchmark-retrieval --generate-sample

  # Compare to baseline
  uv run benchmark-retrieval --baseline tests/benchmarks/results/baseline.json

Output:
  Results are saved to tests/benchmarks/results/ in JSON format.
  The JSON contains MRR@K, NDCG@K, Precision@K, Recall@K for each strategy.
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        help="Path to evaluation dataset JSON file. Uses sample dataset if not provided.",
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values for metrics (default: 1 5 10)",
    )

    parser.add_argument(
        "--strategies",
        "-s",
        type=str,
        nargs="+",
        choices=["vector", "graph", "hybrid"],
        default=["vector"],
        help="Retrieval strategies to benchmark (default: vector)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("tests/benchmarks/results"),
        help="Directory to save results (default: tests/benchmarks/results)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run identifier (auto-generated if not provided)",
    )

    parser.add_argument(
        "--tenant-id",
        type=str,
        default="benchmark-tenant",
        help="Tenant ID for multi-tenancy isolation (default: benchmark-tenant)",
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline results for comparison",
    )

    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate and save sample evaluation dataset, then exit",
    )

    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("tests/benchmarks/data/eval_dataset.json"),
        help="Path to save sample dataset (with --generate-sample)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format (for CI integration)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args(args)


async def run_benchmarks(args: argparse.Namespace) -> int:
    """Run the benchmark suite.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    # Handle sample dataset generation
    if args.generate_sample:
        sample = create_sample_dataset()
        save_dataset(sample, args.sample_output)
        if not args.quiet:
            print(f"Sample dataset saved to: {args.sample_output}")
            print(f"  Queries: {sample.num_queries}")
            print(f"  Documents: {sample.num_documents}")
        return 0

    # Load dataset
    if args.dataset:
        if not args.quiet:
            print(f"Loading dataset: {args.dataset}")
        try:
            dataset = load_dataset(args.dataset)
        except FileNotFoundError:
            print(f"Error: Dataset file not found: {args.dataset}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            return 1
    else:
        if not args.quiet:
            print("Using sample dataset (provide --dataset for custom data)")
        dataset = create_sample_dataset()

    if not args.quiet:
        print(f"Dataset: {dataset.name}")
        print(f"  Queries: {dataset.num_queries}")
        print(f"  Documents: {dataset.num_documents}")
        print(f"  Avg relevant/query: {dataset.avg_relevant_per_query:.2f}")

    # Build config
    strategies = [RetrievalStrategy(s) for s in args.strategies]
    config = BenchmarkConfig(
        k_values=sorted(args.k),
        strategies=strategies,
        tenant_id=args.tenant_id,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    if not args.quiet:
        print("\nRunning benchmarks:")
        print(f"  K values: {config.k_values}")
        print(f"  Strategies: {[s.value for s in config.strategies]}")

    # Run benchmarks
    runner = BenchmarkRunner(dataset, config)
    report = await runner.run()

    # Load and compare to baseline if provided
    if args.baseline:
        baseline = load_baseline(args.baseline)
        if baseline:
            comparison = compare_to_baseline(report, baseline)
            report.baseline_comparison = comparison

    # Output results
    if args.json:
        print(report.to_json())
    else:
        print("\n" + "=" * 60)
        print(report.summary())
        print("=" * 60)

        if report.baseline_comparison:
            print("\nBaseline Comparison:")
            for delta in report.baseline_comparison.get("deltas", []):
                print(
                    f"  {delta['strategy']}@{delta['k']}: "
                    f"MRR {delta['mrr_delta']:+.4f}, "
                    f"NDCG {delta['ndcg_delta']:+.4f}"
                )

        output_path = config.output_dir / f"benchmark_{config.run_id}.json"
        print(f"\nResults saved to: {output_path}")

    return 0


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Optional list of arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parsed = parse_args(args)
    return asyncio.run(run_benchmarks(parsed))


if __name__ == "__main__":
    sys.exit(main())
