"""Integration tests for ingestion pipeline components."""

from __future__ import annotations

import hashlib
import http.server
import socket
import socketserver
import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.indexing.crawler import CrawlerService
from agentic_rag_backend.indexing.parser import parse_pdf

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _embedding() -> list[float]:
    return [0.01] * 1536


class _SimpleHandler(http.server.BaseHTTPRequestHandler):
    response_body = "<html><head><title>Test</title></head><body>Hello ingestion</body></html>"

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(self.response_body.encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A003
        return


class _TestServer:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._httpd: socketserver.TCPServer | None = None
        self.url: str | None = None

    def __enter__(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            _, port = sock.getsockname()
        self._httpd = socketserver.TCPServer(("127.0.0.1", port), _SimpleHandler)
        self.url = f"http://127.0.0.1:{port}/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2)


@pytest.mark.asyncio
async def test_url_ingestion_stores_document(
    postgres_client: PostgresClient,
    integration_cleanup: str,
) -> None:
    tenant_id = integration_cleanup
    tenant_uuid = UUID(tenant_id)

    with _TestServer() as server:
        crawler = CrawlerService()
        page = await crawler.crawl_page(server.url)
        assert page is not None
        doc_id = await postgres_client.create_document(
            tenant_id=tenant_uuid,
            source_type="url",
            content_hash=page.content_hash,
            source_url=page.url,
        )

    stored = await postgres_client.get_document(doc_id, tenant_uuid)
    assert stored is not None
    assert stored["source_url"] == page.url


@pytest.mark.asyncio
async def test_pdf_parsing_creates_chunks(
    postgres_client: PostgresClient,
    integration_cleanup: str,
) -> None:
    tenant_id = integration_cleanup
    tenant_uuid = UUID(tenant_id)

    sample_pdf = FIXTURES_DIR / "sample_simple.pdf"
    parsed = parse_pdf(sample_pdf)
    assert parsed.page_count >= 1

    doc_id = await postgres_client.create_document(
        tenant_id=tenant_uuid,
        source_type="pdf",
        content_hash=parsed.content_hash,
        filename=parsed.filename,
        file_size=parsed.file_size,
        page_count=parsed.page_count,
        metadata=parsed.metadata.model_dump(),
    )

    content = parsed.to_unified_document().content
    await postgres_client.create_chunk(
        tenant_id=tenant_uuid,
        document_id=doc_id,
        content=content,
        chunk_index=0,
        token_count=25,
        embedding=_embedding(),
    )

    async with postgres_client.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS count FROM chunks WHERE tenant_id = $1 AND embedding IS NOT NULL",
            tenant_uuid,
        )
    assert row["count"] >= 1


@pytest.mark.asyncio
async def test_entity_extraction_writes_graph_nodes(
    neo4j_client: Neo4jClient,
    integration_cleanup: str,
) -> None:
    tenant_id = integration_cleanup
    entity_id = str(uuid4())

    await neo4j_client.create_entity(
        entity_id=entity_id,
        tenant_id=tenant_id,
        name="IngestionEntity",
        entity_type="Concept",
    )

    async with neo4j_client.driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e) AS count",
            tenant_id=tenant_id,
        )
        record = await result.single()
    assert record["count"] >= 1


@pytest.mark.asyncio
async def test_deduplication_prevents_duplicate_documents(
    postgres_client: PostgresClient,
    integration_cleanup: str,
) -> None:
    tenant_id = integration_cleanup
    tenant_uuid = UUID(tenant_id)

    content = "duplicate content"
    content_hash = _content_hash(content)

    await postgres_client.create_document(
        tenant_id=tenant_uuid,
        source_type="text",
        content_hash=content_hash,
    )
    await postgres_client.create_document(
        tenant_id=tenant_uuid,
        source_type="text",
        content_hash=content_hash,
    )

    async with postgres_client.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS count FROM documents WHERE tenant_id = $1 AND content_hash = $2",
            tenant_uuid,
            content_hash,
        )
    assert row["count"] == 1
