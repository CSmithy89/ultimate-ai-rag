"""Helpers for benchmark result recording."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

RESULTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "testing"
    / "benchmark-results.jsonl"
)


def record_benchmark(name: str, duration_ms: float, metadata: dict[str, Any] | None = None) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "name": name,
        "duration_ms": round(duration_ms, 2),
        "metadata": metadata or {},
    }
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
