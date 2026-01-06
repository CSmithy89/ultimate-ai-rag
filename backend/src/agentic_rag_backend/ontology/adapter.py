"""Ontology adapter with feature flag support.

Story 20-E1: Implement Ontology Support

This module provides a feature-flag-aware adapter for ontology operations.
"""

from typing import Optional

import structlog

from .loader import (
    DEFAULT_ONTOLOGY_AUTO_TYPE,
    DEFAULT_ONTOLOGY_SUPPORT_ENABLED,
    OntologyLoader,
    OWLREADY2_AVAILABLE,
)
from .models import (
    LoadedOntology,
    OntologyClass,
    OntologyLoadResult,
    OntologyProperty,
)

logger = structlog.get_logger(__name__)


class OntologyAdapter:
    """Feature-flag-aware adapter for ontology operations.

    This adapter wraps the OntologyLoader and adds feature flag checks
    before performing any operations. When ontology support is disabled,
    operations return empty/None results without errors.

    Example:
        adapter = OntologyAdapter(
            enabled=True,
            auto_type=True,
            default_ontology_path="/path/to/domain.owl",
        )

        # Load ontology for a tenant
        result = await adapter.load_ontology(tenant_id="tenant-123")

        # Get entity type
        entity_type = adapter.get_entity_type("Person", "tenant-123")
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_ONTOLOGY_SUPPORT_ENABLED,
        auto_type: bool = DEFAULT_ONTOLOGY_AUTO_TYPE,
        default_ontology_path: Optional[str] = None,
    ) -> None:
        """Initialize the ontology adapter.

        Args:
            enabled: Whether ontology support is enabled
            auto_type: Whether to auto-type entities during ingestion
            default_ontology_path: Default path to ontology file to load
        """
        self._enabled = enabled
        self._auto_type = auto_type
        self._default_ontology_path = default_ontology_path
        self._loader: Optional[OntologyLoader] = None

        if enabled:
            if not OWLREADY2_AVAILABLE:
                logger.warning(
                    "ontology_enabled_but_owlready2_missing",
                    hint="Ontology support is enabled but owlready2 is not installed. "
                    "Install with: pip install owlready2",
                )
            self._loader = OntologyLoader()

    @property
    def enabled(self) -> bool:
        """Check if ontology support is enabled."""
        return self._enabled

    @property
    def auto_type(self) -> bool:
        """Check if auto-typing is enabled."""
        return self._auto_type

    @property
    def is_available(self) -> bool:
        """Check if ontology support is available (enabled and owlready2 installed)."""
        return self._enabled and OWLREADY2_AVAILABLE

    async def load_ontology(
        self,
        tenant_id: str,
        ontology_path: Optional[str] = None,
    ) -> OntologyLoadResult:
        """Load an ontology for a tenant.

        Args:
            tenant_id: The tenant identifier
            ontology_path: Path to ontology file (uses default if not provided)

        Returns:
            OntologyLoadResult with loaded ontology or error
        """
        if not self._enabled:
            logger.debug("ontology_support_disabled", tenant_id=tenant_id)
            return OntologyLoadResult.failure(
                error="Ontology support is disabled",
                load_time_ms=0.0,
            )

        if self._loader is None:
            return OntologyLoadResult.failure(
                error="Ontology loader not initialized",
                load_time_ms=0.0,
            )

        path = ontology_path or self._default_ontology_path
        if not path:
            return OntologyLoadResult.failure(
                error="No ontology path specified and no default configured",
                load_time_ms=0.0,
            )

        return await self._loader.load_ontology(path, tenant_id)

    def get_entity_type(
        self,
        entity_name: str,
        tenant_id: str,
        case_sensitive: bool = False,
    ) -> Optional[OntologyClass]:
        """Find matching ontology class for an entity.

        Args:
            entity_name: The entity name to type
            tenant_id: The tenant identifier
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            Matching OntologyClass or None
        """
        if not self._enabled or self._loader is None:
            return None

        return self._loader.get_entity_type(
            entity_name=entity_name,
            tenant_id=tenant_id,
            case_sensitive=case_sensitive,
        )

    def get_ontology(self, tenant_id: str) -> Optional[LoadedOntology]:
        """Get the loaded ontology for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            LoadedOntology or None
        """
        if not self._enabled or self._loader is None:
            return None

        return self._loader.get_ontology(tenant_id)

    def list_classes(self, tenant_id: str) -> list[OntologyClass]:
        """List all classes in the tenant's ontology.

        Args:
            tenant_id: The tenant identifier

        Returns:
            List of OntologyClass instances
        """
        if not self._enabled or self._loader is None:
            return []

        return self._loader.list_classes(tenant_id)

    def list_properties(self, tenant_id: str) -> list[OntologyProperty]:
        """List all properties in the tenant's ontology.

        Args:
            tenant_id: The tenant identifier

        Returns:
            List of OntologyProperty instances
        """
        if not self._enabled or self._loader is None:
            return []

        return self._loader.list_properties(tenant_id)

    def unload_ontology(self, tenant_id: str) -> bool:
        """Unload the ontology for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            True if unloaded, False otherwise
        """
        if not self._enabled or self._loader is None:
            return False

        return self._loader.unload_ontology(tenant_id)

    def is_loaded(self, tenant_id: str) -> bool:
        """Check if an ontology is loaded for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            True if loaded
        """
        if not self._enabled or self._loader is None:
            return False

        return self._loader.is_loaded(tenant_id)


def get_ontology_adapter(
    enabled: Optional[bool] = None,
    auto_type: Optional[bool] = None,
    default_ontology_path: Optional[str] = None,
) -> OntologyAdapter:
    """Create an OntologyAdapter from settings or provided values.

    This factory function creates an adapter using configuration from
    settings if not explicitly provided.

    Args:
        enabled: Override for ONTOLOGY_SUPPORT_ENABLED
        auto_type: Override for ONTOLOGY_AUTO_TYPE
        default_ontology_path: Override for ONTOLOGY_PATH

    Returns:
        Configured OntologyAdapter
    """
    # Import here to avoid circular imports
    from ..config import get_settings

    settings = get_settings()

    return OntologyAdapter(
        enabled=enabled if enabled is not None else getattr(
            settings, "ontology_support_enabled", DEFAULT_ONTOLOGY_SUPPORT_ENABLED
        ),
        auto_type=auto_type if auto_type is not None else getattr(
            settings, "ontology_auto_type", DEFAULT_ONTOLOGY_AUTO_TYPE
        ),
        default_ontology_path=default_ontology_path or getattr(
            settings, "ontology_path", None
        ),
    )
