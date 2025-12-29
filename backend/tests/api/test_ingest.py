"""Tests for ingestion API endpoints."""

from uuid import uuid4



class TestCreateCrawlJob:
    """Tests for POST /api/v1/ingest/url endpoint."""

    def test_create_crawl_job_success(self, client, sample_crawl_request):
        """Test successful crawl job creation."""
        response = client.post("/api/v1/ingest/url", json=sample_crawl_request)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "data" in data
        assert "meta" in data

        # Check data fields
        assert "job_id" in data["data"]
        assert data["data"]["status"] == "queued"

        # Check meta fields
        assert "requestId" in data["meta"]
        assert "timestamp" in data["meta"]

    def test_create_crawl_job_invalid_url(self, client, sample_tenant_id):
        """Test crawl job creation with invalid URL."""
        response = client.post(
            "/api/v1/ingest/url",
            json={
                "url": "not-a-valid-url",
                "tenant_id": str(sample_tenant_id),
            },
        )

        # Pydantic validation should catch invalid URL
        assert response.status_code == 422

    def test_create_crawl_job_missing_tenant_id(self, client):
        """Test crawl job creation without tenant_id."""
        response = client.post(
            "/api/v1/ingest/url",
            json={
                "url": "https://docs.example.com",
            },
        )

        assert response.status_code == 422

    def test_create_crawl_job_with_options(self, client, sample_tenant_id):
        """Test crawl job creation with custom options."""
        response = client.post(
            "/api/v1/ingest/url",
            json={
                "url": "https://docs.example.com",
                "tenant_id": str(sample_tenant_id),
                "max_depth": 5,
                "options": {
                    "follow_links": True,
                    "respect_robots_txt": False,
                    "rate_limit": 2.0,
                    "include_patterns": [".*docs.*"],
                    "exclude_patterns": [".*blog.*"],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "queued"

    def test_create_crawl_job_max_depth_validation(self, client, sample_tenant_id):
        """Test max_depth validation (must be between 1 and 10)."""
        # Too high
        response = client.post(
            "/api/v1/ingest/url",
            json={
                "url": "https://docs.example.com",
                "tenant_id": str(sample_tenant_id),
                "max_depth": 15,
            },
        )
        assert response.status_code == 422

        # Too low
        response = client.post(
            "/api/v1/ingest/url",
            json={
                "url": "https://docs.example.com",
                "tenant_id": str(sample_tenant_id),
                "max_depth": 0,
            },
        )
        assert response.status_code == 422


class TestGetJobStatus:
    """Tests for GET /api/v1/ingest/jobs/{job_id} endpoint."""

    def test_get_job_status_success(self, client, sample_job_id, sample_tenant_id):
        """Test successful job status retrieval."""
        response = client.get(
            f"/api/v1/ingest/jobs/{sample_job_id}",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "data" in data
        assert "meta" in data

        # Check job status fields
        assert data["data"]["job_id"] == str(sample_job_id)
        assert data["data"]["tenant_id"] == str(sample_tenant_id)
        assert data["data"]["status"] == "queued"
        assert data["data"]["job_type"] == "crawl"

    def test_get_job_status_not_found(
        self, client, sample_tenant_id, mock_postgres_client
    ):
        """Test job status for non-existent job."""
        # Override the mock to return None
        mock_postgres_client.get_job.return_value = None

        random_job_id = uuid4()
        response = client.get(
            f"/api/v1/ingest/jobs/{random_job_id}",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["title"] == "Job Not Found"

    def test_get_job_status_missing_tenant_id(self, client, sample_job_id):
        """Test job status without tenant_id query param."""
        response = client.get(f"/api/v1/ingest/jobs/{sample_job_id}")

        assert response.status_code == 422


class TestListJobs:
    """Tests for GET /api/v1/ingest/jobs endpoint."""

    def test_list_jobs_success(self, client, sample_tenant_id):
        """Test successful job listing."""
        response = client.get(
            "/api/v1/ingest/jobs",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "jobs" in data["data"]
        assert "total" in data["data"]
        assert "limit" in data["data"]
        assert "offset" in data["data"]

    def test_list_jobs_with_status_filter(self, client, sample_tenant_id):
        """Test job listing with status filter."""
        response = client.get(
            "/api/v1/ingest/jobs",
            params={
                "tenant_id": str(sample_tenant_id),
                "status": "running",
            },
        )

        assert response.status_code == 200

    def test_list_jobs_with_pagination(self, client, sample_tenant_id):
        """Test job listing with pagination."""
        response = client.get(
            "/api/v1/ingest/jobs",
            params={
                "tenant_id": str(sample_tenant_id),
                "limit": 10,
                "offset": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["limit"] == 10
        assert data["data"]["offset"] == 5

    def test_list_jobs_limit_validation(self, client, sample_tenant_id):
        """Test that limit is validated (1-100)."""
        # Too high
        response = client.get(
            "/api/v1/ingest/jobs",
            params={
                "tenant_id": str(sample_tenant_id),
                "limit": 200,
            },
        )
        assert response.status_code == 422

        # Too low
        response = client.get(
            "/api/v1/ingest/jobs",
            params={
                "tenant_id": str(sample_tenant_id),
                "limit": 0,
            },
        )
        assert response.status_code == 422


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
