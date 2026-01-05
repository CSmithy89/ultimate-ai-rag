"""Security test fixtures for multi-tenancy enforcement testing.

This module provides shared fixtures for security tests across the
tenant isolation test suite, including attack simulation payloads
for Story 19-J1.
"""

from __future__ import annotations

import os
from typing import Any
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
    config.addinivalue_line(
        "markers",
        "attack_simulation: marks tests that simulate real attack patterns (Story 19-J1)",
    )
    config.addinivalue_line(
        "markers",
        "injection: marks SQL/Cypher injection attack tests",
    )
    config.addinivalue_line(
        "markers",
        "session_hijacking: marks session hijacking attack tests",
    )


# ============================================================================
# Attack Payload Fixtures for Story 19-J1
# ============================================================================

# Attacker and victim tenant IDs for attack simulation tests
ATTACKER_TENANT_ID = "11111111-1111-1111-1111-111111111111"
VICTIM_TENANT_ID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def attacker_tenant_id() -> str:
    """Provide the attacker's tenant ID for attack simulations."""
    return ATTACKER_TENANT_ID


@pytest.fixture
def victim_tenant_id() -> str:
    """Provide the victim's tenant ID for attack simulations."""
    return VICTIM_TENANT_ID


@pytest.fixture
def sql_injection_payloads() -> list[str]:
    """Common SQL injection attack payloads.

    Based on OWASP SQL Injection Testing Guide.
    """
    return [
        "'; DROP TABLE chunks; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM chunks WHERE tenant_id='victim' --",
        "1; UPDATE chunks SET tenant_id='attacker' --",
        "' OR tenant_id='victim",
        "%27%20OR%20%271%27%3D%271",  # URL encoded
        "\\x27 OR 1=1--",  # Hex encoded
    ]


@pytest.fixture
def cypher_injection_payloads() -> list[str]:
    """Common Cypher (Neo4j) injection attack payloads."""
    return [
        "') RETURN * UNION MATCH (n) RETURN n //",
        "' RETURN n UNION MATCH (m) RETURN m //",
        "' OR 1=1 WITH n MATCH (m) RETURN m //",
        "') WITH n MATCH (m:Secret) RETURN m //",
    ]


@pytest.fixture
def parameter_tampering_payloads() -> list[dict[str, Any]]:
    """Parameter tampering attack payloads."""
    return [
        {"tenant_id": [ATTACKER_TENANT_ID, VICTIM_TENANT_ID]},
        {"tenant_id": {"$ne": ATTACKER_TENANT_ID}},
        {"tenant_id": None},
        {"tenant_id": ""},
        {"tenant_id": VICTIM_TENANT_ID, "__proto__": {"admin": True}},
    ]


@pytest.fixture
def session_hijacking_scenarios() -> list[dict[str, Any]]:
    """Session hijacking attack scenarios."""
    return [
        {
            "name": "stolen_session_id",
            "description": "Use victim's session_id with attacker's tenant",
        },
        {
            "name": "session_enumeration",
            "description": "Enumerate session IDs to find victim sessions",
        },
        {
            "name": "session_fixation",
            "description": "Force victim to use attacker-controlled session",
        },
    ]
