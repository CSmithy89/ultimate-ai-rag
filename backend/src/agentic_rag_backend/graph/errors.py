"""Custom exceptions for the graph module (Story 20-B1).

This module defines exceptions specific to community detection operations:
- CommunityNotFoundError: Raised when a community is not found
- CommunityDetectionError: Raised when community detection fails
- GraphTooSmallError: Raised when graph is too small for community detection
"""


class CommunityNotFoundError(Exception):
    """Raised when a community is not found.

    Attributes:
        community_id: The ID of the community that was not found
        tenant_id: The tenant ID for the request
    """

    def __init__(self, community_id: str, tenant_id: str) -> None:
        self.community_id = community_id
        self.tenant_id = tenant_id
        super().__init__(
            f"Community '{community_id}' not found for tenant '{tenant_id}'"
        )


class CommunityDetectionError(Exception):
    """Raised when community detection fails.

    Attributes:
        message: Error description
        tenant_id: The tenant ID for the request
        algorithm: The algorithm that was being used
    """

    def __init__(
        self,
        message: str,
        tenant_id: str,
        algorithm: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.algorithm = algorithm
        super().__init__(f"Community detection failed for tenant '{tenant_id}': {message}")


class GraphTooSmallError(Exception):
    """Raised when the graph is too small for community detection.

    Attributes:
        node_count: Number of nodes in the graph
        min_required: Minimum nodes required
        tenant_id: The tenant ID for the request
    """

    def __init__(
        self,
        node_count: int,
        min_required: int,
        tenant_id: str,
    ) -> None:
        self.node_count = node_count
        self.min_required = min_required
        self.tenant_id = tenant_id
        super().__init__(
            f"Graph too small for community detection: "
            f"{node_count} nodes (minimum {min_required} required) for tenant '{tenant_id}'"
        )
