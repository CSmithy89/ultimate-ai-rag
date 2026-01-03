"""Tests for migrate_to_graphiti script helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from scripts import migrate_to_graphiti
from agentic_rag_backend.models.documents import parse_document_metadata


class _FakeSettings:
    database_url = "postgresql://localhost/test"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "test"
    openai_api_key = "test"
    openai_base_url = None
    anthropic_api_key = None
    gemini_api_key = None
    voyage_api_key = None
    llm_provider = "openai"
    llm_api_key = "test"
    llm_base_url = None
    # Multi-provider embedding support
    embedding_provider = "openai"
    embedding_api_key = "test"
    embedding_base_url = None
    embedding_model = "text-embedding-ada-002"
    graphiti_embedding_model = "text-embedding-3-small"
    graphiti_llm_model = "gpt-4o-mini"


class _FakeGraphiti:
    async def disconnect(self) -> None:
        return None


async def _async_noop() -> None:
    return None


@pytest.mark.asyncio
async def test_validation_fails_on_relationship_mismatch(monkeypatch) -> None:
    tenant_id = "11111111-1111-1111-1111-111111111111"
    document_id = UUID("22222222-2222-2222-2222-222222222222")

    async def fake_create_graphiti_client(**kwargs):
        return _FakeGraphiti()

    async def fake_get_postgres_client(url):
        return object()

    async def fake_get_neo4j_client(**kwargs):
        return object()

    async def fake_fetch_tenant_ids(postgres):
        return [tenant_id]

    async def fake_fetch_documents(postgres, tenant, limit):
        return [
            {
                "id": document_id,
                "tenant_id": tenant_id,
                "source_type": "text",
                "source_url": None,
                "filename": None,
                "content_hash": None,
                "metadata": {},
            }
        ]

    async def fake_fetch_chunks_for_documents(postgres, tenant, document_ids):
        return {document_id: ["content"]}

    async def fake_count_legacy_entities(neo4j, tenant):
        return 1

    async def fake_count_legacy_relationships(neo4j, tenant):
        return 2

    async def fake_count_graphiti_nodes(neo4j, tenant):
        return 1

    async def fake_count_graphiti_relationships(neo4j, tenant):
        return 0

    async def fake_count_legacy_relationship_types(neo4j, tenant):
        return {"MENTIONS": 2}

    async def fake_count_graphiti_relationship_types(neo4j, tenant):
        return {}

    monkeypatch.setattr(migrate_to_graphiti, "GRAPHITI_AVAILABLE", True)
    monkeypatch.setattr(migrate_to_graphiti, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(migrate_to_graphiti, "create_graphiti_client", fake_create_graphiti_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_postgres_client", fake_get_postgres_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_neo4j_client", fake_get_neo4j_client)
    monkeypatch.setattr(migrate_to_graphiti, "close_postgres_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "close_neo4j_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_tenant_ids", fake_fetch_tenant_ids)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_documents", fake_fetch_documents)
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_fetch_chunks_for_documents",
        fake_fetch_chunks_for_documents,
    )
    monkeypatch.setattr(migrate_to_graphiti, "_count_legacy_entities", fake_count_legacy_entities)
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_count_legacy_relationships",
        fake_count_legacy_relationships,
    )
    monkeypatch.setattr(migrate_to_graphiti, "_count_graphiti_nodes", fake_count_graphiti_nodes)
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_count_graphiti_relationships",
        fake_count_graphiti_relationships,
    )
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_count_legacy_relationship_types",
        fake_count_legacy_relationship_types,
    )
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_count_graphiti_relationship_types",
        fake_count_graphiti_relationship_types,
    )

    result = await migrate_to_graphiti.migrate(
        tenant_id=None,
        limit=None,
        dry_run=True,
        backup_path=None,
        validate=True,
    )

    assert result == 2


@pytest.mark.asyncio
async def test_backup_path_triggers_export(monkeypatch, tmp_path: Path) -> None:
    async def fake_create_graphiti_client(**kwargs):
        return _FakeGraphiti()

    async def fake_get_postgres_client(url):
        return object()

    async def fake_get_neo4j_client(**kwargs):
        return object()

    async def fake_fetch_tenant_ids(postgres):
        return ["11111111-1111-1111-1111-111111111111"]

    async def fake_fetch_documents(postgres, tenant, limit):
        return []

    export_calls = []

    async def fake_export_legacy_graph(neo4j, tenant_id, output_path):
        export_calls.append((tenant_id, output_path))

    monkeypatch.setattr(migrate_to_graphiti, "GRAPHITI_AVAILABLE", True)
    monkeypatch.setattr(migrate_to_graphiti, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(migrate_to_graphiti, "create_graphiti_client", fake_create_graphiti_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_postgres_client", fake_get_postgres_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_neo4j_client", fake_get_neo4j_client)
    monkeypatch.setattr(migrate_to_graphiti, "close_postgres_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "close_neo4j_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_tenant_ids", fake_fetch_tenant_ids)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_documents", fake_fetch_documents)
    monkeypatch.setattr(migrate_to_graphiti, "_export_legacy_graph", fake_export_legacy_graph)

    result = await migrate_to_graphiti.migrate(
        tenant_id=None,
        limit=None,
        dry_run=True,
        backup_path=tmp_path,
        validate=False,
    )

    assert result == 0
    assert export_calls


@pytest.mark.asyncio
async def test_dry_run_skips_ingest(monkeypatch) -> None:
    tenant_id = "11111111-1111-1111-1111-111111111111"
    document_id = UUID("22222222-2222-2222-2222-222222222222")

    async def fake_create_graphiti_client(**kwargs):
        return _FakeGraphiti()

    async def fake_get_postgres_client(url):
        return object()

    async def fake_get_neo4j_client(**kwargs):
        return object()

    async def fake_fetch_tenant_ids(postgres):
        return [tenant_id]

    async def fake_fetch_documents(postgres, tenant, limit):
        return [
            {
                "id": document_id,
                "tenant_id": tenant_id,
                "source_type": "text",
                "source_url": None,
                "filename": None,
                "content_hash": None,
                "metadata": {},
            }
        ]

    async def fake_fetch_chunks_for_documents(postgres, tenant, document_ids):
        return {document_id: ["content"]}

    async def fail_ingest(**kwargs):
        raise AssertionError("ingest should not run during dry run")

    monkeypatch.setattr(migrate_to_graphiti, "GRAPHITI_AVAILABLE", True)
    monkeypatch.setattr(migrate_to_graphiti, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(migrate_to_graphiti, "create_graphiti_client", fake_create_graphiti_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_postgres_client", fake_get_postgres_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_neo4j_client", fake_get_neo4j_client)
    monkeypatch.setattr(migrate_to_graphiti, "close_postgres_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "close_neo4j_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_tenant_ids", fake_fetch_tenant_ids)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_documents", fake_fetch_documents)
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_fetch_chunks_for_documents",
        fake_fetch_chunks_for_documents,
    )
    monkeypatch.setattr(migrate_to_graphiti, "ingest_document_as_episode", fail_ingest)

    result = await migrate_to_graphiti.migrate(
        tenant_id=None,
        limit=None,
        dry_run=True,
        backup_path=None,
        validate=False,
    )

    assert result == 0


@pytest.mark.asyncio
async def test_migration_skips_empty_documents(monkeypatch) -> None:
    tenant_id = "11111111-1111-1111-1111-111111111111"
    document_id = UUID("22222222-2222-2222-2222-222222222222")

    async def fake_create_graphiti_client(**kwargs):
        return _FakeGraphiti()

    async def fake_get_postgres_client(url):
        return object()

    async def fake_get_neo4j_client(**kwargs):
        return object()

    async def fake_fetch_tenant_ids(postgres):
        return [tenant_id]

    async def fake_fetch_documents(postgres, tenant, limit):
        return [
            {
                "id": document_id,
                "tenant_id": tenant_id,
                "source_type": "text",
                "source_url": None,
                "filename": None,
                "content_hash": None,
                "metadata": {},
            }
        ]

    async def fake_fetch_chunks_for_documents(postgres, tenant, document_ids):
        return {document_id: []}

    async def fail_ingest(**kwargs):
        raise AssertionError("ingest should not run for empty documents")

    monkeypatch.setattr(migrate_to_graphiti, "GRAPHITI_AVAILABLE", True)
    monkeypatch.setattr(migrate_to_graphiti, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(migrate_to_graphiti, "create_graphiti_client", fake_create_graphiti_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_postgres_client", fake_get_postgres_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_neo4j_client", fake_get_neo4j_client)
    monkeypatch.setattr(migrate_to_graphiti, "close_postgres_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "close_neo4j_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_tenant_ids", fake_fetch_tenant_ids)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_documents", fake_fetch_documents)
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_fetch_chunks_for_documents",
        fake_fetch_chunks_for_documents,
    )
    monkeypatch.setattr(migrate_to_graphiti, "ingest_document_as_episode", fail_ingest)

    result = await migrate_to_graphiti.migrate(
        tenant_id=None,
        limit=None,
        dry_run=False,
        backup_path=None,
        validate=False,
    )

    assert result == 0


@pytest.mark.asyncio
async def test_migration_handles_invalid_metadata(monkeypatch) -> None:
    tenant_id = "11111111-1111-1111-1111-111111111111"
    document_id = UUID("22222222-2222-2222-2222-222222222222")
    captured_extra = {}

    async def fake_create_graphiti_client(**kwargs):
        return _FakeGraphiti()

    async def fake_get_postgres_client(url):
        return object()

    async def fake_get_neo4j_client(**kwargs):
        return object()

    async def fake_fetch_tenant_ids(postgres):
        return [tenant_id]

    async def fake_fetch_documents(postgres, tenant, limit):
        return [
            {
                "id": document_id,
                "tenant_id": tenant_id,
                "source_type": "text",
                "source_url": None,
                "filename": None,
                "content_hash": None,
                "metadata": "{not-json}",
            }
        ]

    async def fake_fetch_chunks_for_documents(postgres, tenant, document_ids):
        return {document_id: ["content"]}

    async def capture_ingest(*, document, **kwargs):
        captured_extra.update(document.metadata.extra)

    monkeypatch.setattr(migrate_to_graphiti, "GRAPHITI_AVAILABLE", True)
    monkeypatch.setattr(migrate_to_graphiti, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(migrate_to_graphiti, "create_graphiti_client", fake_create_graphiti_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_postgres_client", fake_get_postgres_client)
    monkeypatch.setattr(migrate_to_graphiti, "get_neo4j_client", fake_get_neo4j_client)
    monkeypatch.setattr(migrate_to_graphiti, "close_postgres_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "close_neo4j_client", _async_noop)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_tenant_ids", fake_fetch_tenant_ids)
    monkeypatch.setattr(migrate_to_graphiti, "_fetch_documents", fake_fetch_documents)
    monkeypatch.setattr(
        migrate_to_graphiti,
        "_fetch_chunks_for_documents",
        fake_fetch_chunks_for_documents,
    )
    monkeypatch.setattr(migrate_to_graphiti, "ingest_document_as_episode", capture_ingest)

    result = await migrate_to_graphiti.migrate(
        tenant_id=None,
        limit=None,
        dry_run=False,
        backup_path=None,
        validate=False,
    )

    assert result == 0
    assert captured_extra["legacy_document_id"] == str(document_id)


def test_parse_document_metadata_logs_on_invalid() -> None:
    class FakeLogger:
        def __init__(self) -> None:
            self.calls = []

        def warning(self, event: str, **kwargs) -> None:
            self.calls.append((event, kwargs))

    logger = FakeLogger()
    metadata = parse_document_metadata(
        {"page_count": "not-a-number"},
        extra_fields={"legacy_document_id": "doc-1"},
        log=logger,
        log_context={"legacy_document_id": "doc-1"},
    )

    assert metadata.extra["legacy_document_id"] == "doc-1"
    assert logger.calls
