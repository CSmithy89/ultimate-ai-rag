"""Benchmark for ingestion speed (NFR2)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

import pytest

from agentic_rag_backend.indexing.parser import parse_pdf
from tests.benchmarks.utils import record_benchmark

if os.getenv("RUN_BENCHMARKS") != "1":
    pytest.skip("RUN_BENCHMARKS=1 required for benchmark tests", allow_module_level=True)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
MAX_INGESTION_DURATION_MS = 5 * 60 * 1000  # NFR2: 5 minutes


@pytest.mark.asyncio
async def test_ingestion_pdf_benchmark() -> None:
    pdf_path_env = os.getenv("BENCHMARK_PDF_PATH")
    if pdf_path_env:
        pdf_path = Path(pdf_path_env)
    else:
        pdf_path = FIXTURES_DIR / "sample_complex.pdf"

    if not pdf_path.exists():
        pytest.skip("Benchmark PDF not available")

    start = time.perf_counter()
    parsed = parse_pdf(
        pdf_path,
        document_id=uuid4(),
        tenant_id=uuid4(),
    )
    duration_ms = (time.perf_counter() - start) * 1000

    record_benchmark(
        name="nfr2_ingestion_pdf",
        duration_ms=duration_ms,
        metadata={
            "file": str(pdf_path),
            "page_count": parsed.page_count,
            "file_size": parsed.file_size,
        },
    )

    assert duration_ms < MAX_INGESTION_DURATION_MS
