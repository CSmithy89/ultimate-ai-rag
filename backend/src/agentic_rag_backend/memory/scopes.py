"""Scope hierarchy and validation logic for Epic 20 Memory Platform."""

from typing import Optional

from .models import MemoryScope


# Scope hierarchy: maps each scope to its parent scopes for inclusive search
# SESSION includes USER and GLOBAL (user's memories + shared knowledge)
# USER includes GLOBAL (user's memories + shared knowledge)
# AGENT includes GLOBAL (agent's memories + shared knowledge)
# GLOBAL has no parents
SCOPE_HIERARCHY: dict[MemoryScope, list[MemoryScope]] = {
    MemoryScope.SESSION: [MemoryScope.USER, MemoryScope.GLOBAL],
    MemoryScope.USER: [MemoryScope.GLOBAL],
    MemoryScope.AGENT: [MemoryScope.GLOBAL],
    MemoryScope.GLOBAL: [],
}


def get_parent_scopes(scope: MemoryScope) -> list[MemoryScope]:
    """Get parent scopes for a given scope level.

    The scope hierarchy defines which memories should be included when
    searching with include_parent_scopes=True:
    - SESSION: includes USER and GLOBAL memories
    - USER: includes GLOBAL memories
    - AGENT: includes GLOBAL memories
    - GLOBAL: no parent scopes

    Args:
        scope: The starting scope level

    Returns:
        List of parent scopes in hierarchy order (closest first)

    Examples:
        >>> get_parent_scopes(MemoryScope.SESSION)
        [MemoryScope.USER, MemoryScope.GLOBAL]
        >>> get_parent_scopes(MemoryScope.GLOBAL)
        []
    """
    return SCOPE_HIERARCHY.get(scope, [])


def get_scopes_to_search(
    scope: MemoryScope, include_parent_scopes: bool = True
) -> list[MemoryScope]:
    """Get all scopes to search for a query.

    Args:
        scope: The starting scope level
        include_parent_scopes: Whether to include parent scopes

    Returns:
        List of scopes to search, starting with the given scope

    Examples:
        >>> get_scopes_to_search(MemoryScope.SESSION, include_parent_scopes=True)
        [MemoryScope.SESSION, MemoryScope.USER, MemoryScope.GLOBAL]
        >>> get_scopes_to_search(MemoryScope.SESSION, include_parent_scopes=False)
        [MemoryScope.SESSION]
    """
    scopes = [scope]
    if include_parent_scopes:
        scopes.extend(get_parent_scopes(scope))
    return scopes


def validate_scope_context(
    scope: MemoryScope,
    user_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Validate that required context is provided for the given scope.

    Each scope has specific requirements:
    - USER scope: requires user_id
    - SESSION scope: requires session_id (user_id recommended but not required)
    - AGENT scope: requires agent_id
    - GLOBAL scope: no additional context required

    Args:
        scope: Memory scope level
        user_id: User identifier
        session_id: Session identifier
        agent_id: Agent identifier

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Examples:
        >>> validate_scope_context(MemoryScope.USER, "user-123", None, None)
        (True, None)
        >>> validate_scope_context(MemoryScope.USER, None, None, None)
        (False, "user_id is required for USER scope")
    """
    if scope == MemoryScope.USER:
        if not user_id:
            return False, "user_id is required for USER scope"
    elif scope == MemoryScope.SESSION:
        if not session_id:
            return False, "session_id is required for SESSION scope"
    elif scope == MemoryScope.AGENT:
        if not agent_id:
            return False, "agent_id is required for AGENT scope"
    # GLOBAL scope has no additional requirements

    return True, None


def is_scope_accessible(
    target_scope: MemoryScope,
    search_scope: MemoryScope,
    include_parent_scopes: bool = True,
) -> bool:
    """Check if a target scope is accessible from a search scope.

    A scope is accessible if:
    - It equals the search scope, OR
    - It is a parent scope and include_parent_scopes is True

    Args:
        target_scope: The scope of the memory being checked
        search_scope: The scope from which the search is being performed
        include_parent_scopes: Whether to include parent scopes

    Returns:
        True if the target scope is accessible, False otherwise

    Examples:
        >>> is_scope_accessible(MemoryScope.USER, MemoryScope.SESSION, True)
        True
        >>> is_scope_accessible(MemoryScope.USER, MemoryScope.SESSION, False)
        False
        >>> is_scope_accessible(MemoryScope.SESSION, MemoryScope.SESSION, False)
        True
    """
    if target_scope == search_scope:
        return True

    if include_parent_scopes:
        return target_scope in get_parent_scopes(search_scope)

    return False
