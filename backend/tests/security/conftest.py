"""Security test fixtures for multi-tenancy enforcement testing.

This module provides shared fixtures for security tests across the
tenant isolation test suite.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

# Test tenant IDs - using deterministic UUIDs for reproducibility
TENANT_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TENANT_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TENANT_C_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"


@pytest.fixture
def tenant_a_id() -> str:
    """Provide tenant A's deterministic ID."""
    return TENANT_A_ID


@pytest.fixture
def tenant_b_id() -> str:
    """Provide tenant B's deterministic ID."""
    return TENANT_B_ID


@pytest.fixture
def tenant_c_id() -> str:
    """Provide tenant C's deterministic ID."""
    return TENANT_C_ID


@pytest.fixture
def random_tenant_id() -> str:
    """Provide a random tenant ID for isolation."""
    return str(uuid4())


@pytest.fixture
def two_tenant_ids() -> tuple[str, str]:
    """Provide two random tenant IDs for cross-tenant testing."""
    return str(uuid4()), str(uuid4())


def _require_integration_env() -> None:
    """Skip tests if integration environment is not available."""
    if os.getenv("INTEGRATION_TESTS") != "1":
        pytest.skip("INTEGRATION_TESTS=1 required for integration security tests")


@pytest.fixture
def require_integration() -> None:
    """Fixture that skips if integration env is not available."""
    _require_integration_env()


# Pytest markers for security test categories
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for security tests."""
    config.addinivalue_line(
        "markers",
        "security: marks tests as security-related (deselect with '-m \"not security\"')",
    )
    config.addinivalue_line(
        "markers",
        "tenant_isolation: marks tests for tenant isolation enforcement",
    )
    config.addinivalue_line(
        "markers",
        "cross_tenant: marks adversarial cross-tenant access tests",
    )
