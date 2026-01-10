"""Ontology support for domain-specific entity typing.

Story 20-E1: Implement Ontology Support

This module provides OWL ontology loading and management capabilities,
enabling domain-specific entity typing during knowledge graph ingestion.

Features:
- Load OWL ontologies from files or URLs
- Extract classes and properties for entity typing
- Match entities to ontology classes
- Feature flag support (ONTOLOGY_SUPPORT_ENABLED)
- Tenant isolation for loaded ontologies

Example:
    from agentic_rag_backend.ontology import (
        OntologyLoader,
        OntologyAdapter,
        get_ontology_adapter,
    )

    # Using the adapter (recommended - handles feature flags)
    adapter = get_ontology_adapter()
    if adapter.enabled:
        result = await adapter.load_ontology(
            tenant_id="tenant-123",
            ontology_path="/path/to/domain.owl",
        )
        if result.success:
            entity_type = adapter.get_entity_type("Person", "tenant-123")

    # Using the loader directly (low-level)
    loader = OntologyLoader()
    result = await loader.load_ontology(
        ontology_path="/path/to/domain.owl",
        tenant_id="tenant-123",
    )
"""

from .models import (
    OntologyClass,
    OntologyProperty,
    LoadedOntology,
    OntologyLoadResult,
)
from .loader import (
    OntologyLoader,
    OWLREADY2_AVAILABLE,
    DEFAULT_ONTOLOGY_SUPPORT_ENABLED,
    DEFAULT_ONTOLOGY_AUTO_TYPE,
    MAX_ONTOLOGY_FILE_SIZE_BYTES,
)
from .adapter import (
    OntologyAdapter,
    get_ontology_adapter,
)

__all__ = [
    # Models
    "OntologyClass",
    "OntologyProperty",
    "LoadedOntology",
    "OntologyLoadResult",
    # Loader
    "OntologyLoader",
    "OWLREADY2_AVAILABLE",
    # Adapter
    "OntologyAdapter",
    "get_ontology_adapter",
    # Constants
    "DEFAULT_ONTOLOGY_SUPPORT_ENABLED",
    "DEFAULT_ONTOLOGY_AUTO_TYPE",
    "MAX_ONTOLOGY_FILE_SIZE_BYTES",
]
