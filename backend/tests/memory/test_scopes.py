"""Unit tests for memory scope hierarchy and validation (Story 20-A1).

Tests the scope hierarchy logic:
- SESSION includes USER and GLOBAL memories
- USER includes GLOBAL memories
- AGENT includes GLOBAL memories
- GLOBAL is the root scope
"""

import pytest

from agentic_rag_backend.memory.models import MemoryScope
from agentic_rag_backend.memory.scopes import (
    SCOPE_HIERARCHY,
    get_parent_scopes,
    get_scopes_to_search,
    is_scope_accessible,
    validate_scope_context,
)


class TestScopeHierarchy:
    """Test the scope hierarchy constants."""

    def test_session_scope_parents(self):
        """SESSION scope should include USER and GLOBAL."""
        parents = SCOPE_HIERARCHY[MemoryScope.SESSION]
        assert MemoryScope.USER in parents
        assert MemoryScope.GLOBAL in parents
        assert len(parents) == 2

    def test_user_scope_parents(self):
        """USER scope should include GLOBAL only."""
        parents = SCOPE_HIERARCHY[MemoryScope.USER]
        assert MemoryScope.GLOBAL in parents
        assert len(parents) == 1

    def test_agent_scope_parents(self):
        """AGENT scope should include GLOBAL only."""
        parents = SCOPE_HIERARCHY[MemoryScope.AGENT]
        assert MemoryScope.GLOBAL in parents
        assert len(parents) == 1

    def test_global_scope_has_no_parents(self):
        """GLOBAL scope should have no parent scopes."""
        parents = SCOPE_HIERARCHY[MemoryScope.GLOBAL]
        assert len(parents) == 0


class TestGetParentScopes:
    """Test the get_parent_scopes function."""

    def test_session_parents(self):
        """get_parent_scopes(SESSION) returns USER and GLOBAL."""
        parents = get_parent_scopes(MemoryScope.SESSION)
        assert parents == [MemoryScope.USER, MemoryScope.GLOBAL]

    def test_user_parents(self):
        """get_parent_scopes(USER) returns GLOBAL."""
        parents = get_parent_scopes(MemoryScope.USER)
        assert parents == [MemoryScope.GLOBAL]

    def test_agent_parents(self):
        """get_parent_scopes(AGENT) returns GLOBAL."""
        parents = get_parent_scopes(MemoryScope.AGENT)
        assert parents == [MemoryScope.GLOBAL]

    def test_global_parents(self):
        """get_parent_scopes(GLOBAL) returns empty list."""
        parents = get_parent_scopes(MemoryScope.GLOBAL)
        assert parents == []


class TestGetScopesToSearch:
    """Test the get_scopes_to_search function."""

    def test_session_with_parents(self):
        """SESSION scope search includes SESSION, USER, GLOBAL."""
        scopes = get_scopes_to_search(MemoryScope.SESSION, include_parent_scopes=True)
        assert scopes == [MemoryScope.SESSION, MemoryScope.USER, MemoryScope.GLOBAL]

    def test_session_without_parents(self):
        """SESSION scope search without parents is SESSION only."""
        scopes = get_scopes_to_search(MemoryScope.SESSION, include_parent_scopes=False)
        assert scopes == [MemoryScope.SESSION]

    def test_user_with_parents(self):
        """USER scope search includes USER and GLOBAL."""
        scopes = get_scopes_to_search(MemoryScope.USER, include_parent_scopes=True)
        assert scopes == [MemoryScope.USER, MemoryScope.GLOBAL]

    def test_agent_with_parents(self):
        """AGENT scope search includes AGENT and GLOBAL."""
        scopes = get_scopes_to_search(MemoryScope.AGENT, include_parent_scopes=True)
        assert scopes == [MemoryScope.AGENT, MemoryScope.GLOBAL]

    def test_global_with_parents(self):
        """GLOBAL scope search is just GLOBAL."""
        scopes = get_scopes_to_search(MemoryScope.GLOBAL, include_parent_scopes=True)
        assert scopes == [MemoryScope.GLOBAL]

    def test_global_without_parents(self):
        """GLOBAL scope without parents is same as with parents."""
        scopes = get_scopes_to_search(MemoryScope.GLOBAL, include_parent_scopes=False)
        assert scopes == [MemoryScope.GLOBAL]


class TestValidateScopeContext:
    """Test the validate_scope_context function."""

    def test_user_scope_with_user_id(self):
        """USER scope is valid with user_id."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.USER,
            user_id="user-123",
            session_id=None,
            agent_id=None,
        )
        assert is_valid is True
        assert error is None

    def test_user_scope_without_user_id(self):
        """USER scope is invalid without user_id."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.USER,
            user_id=None,
            session_id=None,
            agent_id=None,
        )
        assert is_valid is False
        assert "user_id is required" in error

    def test_session_scope_with_session_id(self):
        """SESSION scope is valid with session_id."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.SESSION,
            user_id=None,
            session_id="session-456",
            agent_id=None,
        )
        assert is_valid is True
        assert error is None

    def test_session_scope_without_session_id(self):
        """SESSION scope is invalid without session_id."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.SESSION,
            user_id="user-123",
            session_id=None,
            agent_id=None,
        )
        assert is_valid is False
        assert "session_id is required" in error

    def test_agent_scope_with_agent_id(self):
        """AGENT scope is valid with agent_id."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.AGENT,
            user_id=None,
            session_id=None,
            agent_id="orchestrator-agent",
        )
        assert is_valid is True
        assert error is None

    def test_agent_scope_without_agent_id(self):
        """AGENT scope is invalid without agent_id."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.AGENT,
            user_id=None,
            session_id=None,
            agent_id=None,
        )
        assert is_valid is False
        assert "agent_id is required" in error

    def test_global_scope_no_requirements(self):
        """GLOBAL scope is valid without any context IDs."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.GLOBAL,
            user_id=None,
            session_id=None,
            agent_id=None,
        )
        assert is_valid is True
        assert error is None

    def test_global_scope_with_optional_ids(self):
        """GLOBAL scope is still valid with optional context IDs."""
        is_valid, error = validate_scope_context(
            scope=MemoryScope.GLOBAL,
            user_id="user-123",
            session_id="session-456",
            agent_id="agent-789",
        )
        assert is_valid is True
        assert error is None


class TestIsScopeAccessible:
    """Test the is_scope_accessible function."""

    def test_same_scope_always_accessible(self):
        """Same scope is always accessible."""
        for scope in MemoryScope:
            assert is_scope_accessible(scope, scope, include_parent_scopes=False)
            assert is_scope_accessible(scope, scope, include_parent_scopes=True)

    def test_user_accessible_from_session(self):
        """USER scope is accessible from SESSION with parent scopes."""
        assert is_scope_accessible(
            MemoryScope.USER, MemoryScope.SESSION, include_parent_scopes=True
        )

    def test_user_not_accessible_from_session_without_parents(self):
        """USER scope is not accessible from SESSION without parent scopes."""
        assert not is_scope_accessible(
            MemoryScope.USER, MemoryScope.SESSION, include_parent_scopes=False
        )

    def test_global_accessible_from_all_with_parents(self):
        """GLOBAL scope is accessible from all scopes with parent scopes."""
        for scope in MemoryScope:
            if scope != MemoryScope.GLOBAL:
                assert is_scope_accessible(
                    MemoryScope.GLOBAL, scope, include_parent_scopes=True
                )

    def test_session_not_accessible_from_user(self):
        """SESSION scope is not accessible from USER scope (not a parent)."""
        assert not is_scope_accessible(
            MemoryScope.SESSION, MemoryScope.USER, include_parent_scopes=True
        )

    def test_agent_not_accessible_from_user(self):
        """AGENT scope is not accessible from USER scope (not a parent)."""
        assert not is_scope_accessible(
            MemoryScope.AGENT, MemoryScope.USER, include_parent_scopes=True
        )


class TestMemoryScopeEnum:
    """Test the MemoryScope enum values."""

    def test_scope_values(self):
        """Verify all scope enum values."""
        assert MemoryScope.USER.value == "user"
        assert MemoryScope.SESSION.value == "session"
        assert MemoryScope.AGENT.value == "agent"
        assert MemoryScope.GLOBAL.value == "global"

    def test_scope_count(self):
        """Verify we have exactly 4 scopes."""
        assert len(MemoryScope) == 4

    def test_scope_from_string(self):
        """Verify scopes can be created from string values."""
        assert MemoryScope("user") == MemoryScope.USER
        assert MemoryScope("session") == MemoryScope.SESSION
        assert MemoryScope("agent") == MemoryScope.AGENT
        assert MemoryScope("global") == MemoryScope.GLOBAL

    def test_invalid_scope_raises(self):
        """Verify invalid scope string raises ValueError."""
        with pytest.raises(ValueError):
            MemoryScope("invalid_scope")
