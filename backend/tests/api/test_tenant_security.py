"""Security tests for cross-tenant data isolation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from agentic_rag_backend.main import app
from agentic_rag_backend.api.routes.ingest import get_postgres
from agentic_rag_backend.api.routes.memories import get_memory_store, get_settings
from agentic_rag_backend.models.ingest import JobStatus, JobStatusEnum, JobType, JobProgress
from agentic_rag_backend.memory import ScopedMemory, MemoryScope

@pytest.fixture
def secure_client(mock_postgres_client):
    """Create a client with mocked postgres that we can control per-test."""
    app.dependency_overrides[get_postgres] = lambda: mock_postgres_client
    
    # Mock settings to enable memory scopes
    mock_settings = MagicMock()
    mock_settings.memory_scopes_enabled = True
    app.dependency_overrides[get_settings] = lambda: mock_settings
    
    # Mock memory store
    mock_store = MagicMock()
    app.dependency_overrides[get_memory_store] = lambda: mock_store
    
    with TestClient(app) as client:
        client.mock_store = mock_store
        yield client
    app.dependency_overrides.clear()

def test_get_job_status_cross_tenant_isolation(secure_client, mock_postgres_client):
    """Verify that a job belonging to Tenant A cannot be accessed by Tenant B."""
    target_job_id = uuid4()
    tenant_a = uuid4()
    tenant_b = uuid4()

    # Setup mock to return the job ONLY if tenant_id matches tenant_a
    async def side_effect_get_job(job_id, tenant_id):
        if str(job_id) == str(target_job_id) and str(tenant_id) == str(tenant_a):
            return JobStatus(
                job_id=target_job_id,
                tenant_id=tenant_a,
                job_type=JobType.CRAWL,
                status=JobStatusEnum.COMPLETED,
                progress=JobProgress(pages_crawled=1),
                created_at=datetime.now(timezone.utc),
            )
        return None

    mock_postgres_client.get_job.side_effect = side_effect_get_job

    # Tenant A should see the job
    response_a = secure_client.get(
        f"/api/v1/ingest/jobs/{target_job_id}",
        params={"tenant_id": str(tenant_a)}
    )
    assert response_a.status_code == 200
    assert response_a.json()["data"]["job_id"] == str(target_job_id)

    # Tenant B should NOT see the job (404 Not Found)
    response_b = secure_client.get(
        f"/api/v1/ingest/jobs/{target_job_id}",
        params={"tenant_id": str(tenant_b)}
    )
    assert response_b.status_code == 404
    # Check that error message contains "not found"
    assert "not found" in response_b.json()["detail"].lower()

def test_get_memory_cross_tenant_isolation(secure_client):
    """Verify that a memory belonging to Tenant A cannot be accessed by Tenant B."""
    target_memory_id = uuid4()
    tenant_a = uuid4()
    tenant_b = uuid4()
    
    mock_store = secure_client.mock_store
    
    # Setup mock to return the memory ONLY if tenant_id matches tenant_a
    async def side_effect_get_memory(memory_id, tenant_id):
        # API layer calls with strings
        if str(memory_id) == str(target_memory_id) and str(tenant_id) == str(tenant_a):
            return ScopedMemory(
                id=target_memory_id,
                tenant_id=tenant_a,
                content="Secret memory",
                scope=MemoryScope.GLOBAL,
                created_at=datetime.now(timezone.utc),
                accessed_at=datetime.now(timezone.utc)
            )
        return None
        
    mock_store.get_memory.side_effect = side_effect_get_memory
    
    # Tenant A should see it
    response_a = secure_client.get(
        f"/api/v1/memories/{target_memory_id}",
        params={"tenant_id": str(tenant_a)}
    )
    assert response_a.status_code == 200, f"Tenant A failed to get its own memory: {response_a.text}"
    assert response_a.json()["data"]["content"] == "Secret memory"
    
    # Tenant B should NOT see it
    response_b = secure_client.get(
        f"/api/v1/memories/{target_memory_id}",
        params={"tenant_id": str(tenant_b)}
    )
    assert response_b.status_code == 404
    assert "not found" in response_b.json()["detail"].lower()

def test_list_jobs_cross_tenant_isolation(secure_client, mock_postgres_client):
    """Verify that listing jobs filters by tenant ID."""
    tenant_a = uuid4()
    tenant_b = uuid4()

    # Mock list_jobs to verify it's called with the correct tenant_id
    mock_postgres_client.list_jobs.return_value = ([], 0)

    # List jobs for Tenant B
    secure_client.get(
        "/api/v1/ingest/jobs",
        params={"tenant_id": str(tenant_b), "limit": 10}
    )

    # Verify postgres was queried with Tenant B's ID
    mock_postgres_client.list_jobs.assert_called_with(
        tenant_id=tenant_b,
        status=None,
        limit=10,
        offset=0
    )
    
    # Ensure it wasn't called with Tenant A (just to be sure)
    call_args = mock_postgres_client.list_jobs.call_args
    assert call_args.kwargs["tenant_id"] != tenant_a