"""OWL ontology loading and management.

Story 20-E1: Implement Ontology Support

This module provides functionality to load and parse OWL ontologies
using the owlready2 library.
"""

import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import structlog

from .models import (
    LoadedOntology,
    OntologyClass,
    OntologyLoadResult,
    OntologyProperty,
)

logger = structlog.get_logger(__name__)

# Maximum file size for ontology files (50MB default)
MAX_ONTOLOGY_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Check if owlready2 is available
try:
    from owlready2 import get_ontology, Thing
    from owlready2.prop import ObjectPropertyClass, DataPropertyClass

    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False
    logger.warning(
        "owlready2_not_available",
        hint="Install owlready2 to enable ontology support: pip install owlready2",
    )


class OntologyLoader:
    """Load and manage OWL ontologies.

    This class provides methods to load OWL ontologies from files or URLs,
    extract classes and properties, and match entities to ontology types.

    Ontologies are stored per-tenant to ensure tenant isolation.

    Example:
        loader = OntologyLoader()
        result = await loader.load_ontology(
            ontology_path="/path/to/domain.owl",
            tenant_id="tenant-123",
        )
        if result.success:
            entity_type = loader.get_entity_type("Person", "tenant-123")
    """

    def __init__(self) -> None:
        """Initialize the ontology loader."""
        self._loaded_ontologies: dict[str, LoadedOntology] = {}

    @staticmethod
    def is_available() -> bool:
        """Check if owlready2 is available for ontology loading.

        Returns:
            True if owlready2 is installed and can be imported
        """
        return OWLREADY2_AVAILABLE

    def _validate_ontology_path(self, ontology_path: str) -> tuple[bool, Optional[str]]:
        """Validate the ontology path or URL.

        Args:
            ontology_path: Path to file or URL

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not ontology_path:
            return False, "Ontology path cannot be empty"

        # Check if it's a URL
        parsed = urlparse(ontology_path)
        if parsed.scheme in ("http", "https", "file"):
            # URL format - owlready2 will handle validation
            return True, None

        # Check if it's a local file path
        path = Path(ontology_path)
        if not path.exists():
            return False, f"Ontology file not found: {ontology_path}"

        if not path.is_file():
            return False, f"Ontology path is not a file: {ontology_path}"

        # Check file size
        try:
            file_size = path.stat().st_size
            if file_size > MAX_ONTOLOGY_FILE_SIZE_BYTES:
                return False, (
                    f"Ontology file too large: {file_size} bytes "
                    f"(max: {MAX_ONTOLOGY_FILE_SIZE_BYTES} bytes)"
                )
        except OSError as e:
            return False, f"Cannot read ontology file stats: {e}"

        # Check file extension
        valid_extensions = {".owl", ".rdf", ".xml", ".ttl", ".n3", ".nt"}
        if path.suffix.lower() not in valid_extensions:
            logger.warning(
                "ontology_unusual_extension",
                path=ontology_path,
                extension=path.suffix,
                valid_extensions=list(valid_extensions),
            )

        return True, None

    async def load_ontology(
        self,
        ontology_path: str,
        tenant_id: str,
    ) -> OntologyLoadResult:
        """Load an OWL ontology from file or URL.

        This method parses an OWL ontology and extracts all classes and properties,
        storing them for the specified tenant.

        Args:
            ontology_path: Path to the OWL file or URL
            tenant_id: The tenant identifier for isolation

        Returns:
            OntologyLoadResult with the loaded ontology or error
        """
        start_time = time.time()

        if not OWLREADY2_AVAILABLE:
            return OntologyLoadResult.failure(
                error="owlready2 library is not installed. "
                "Install with: pip install owlready2",
                load_time_ms=0.0,
            )

        # Validate path
        is_valid, error = self._validate_ontology_path(ontology_path)
        if not is_valid:
            elapsed_ms = (time.time() - start_time) * 1000
            return OntologyLoadResult.failure(error=error or "Invalid path", load_time_ms=elapsed_ms)

        try:
            # Load ontology using owlready2 in a thread to avoid blocking the event loop
            import asyncio
            onto_obj = get_ontology(ontology_path)
            onto = await asyncio.to_thread(onto_obj.load)

            # Extract classes
            classes: list[OntologyClass] = []
            for cls in onto.classes():
                # Skip the base Thing class
                if cls is Thing:
                    continue

                # Get parent URIs (excluding Thing)
                parent_uris = []
                for parent in cls.is_a:
                    if hasattr(parent, "iri") and parent is not Thing:
                        parent_uris.append(parent.iri)

                # Get properties for this class
                class_properties = []
                try:
                    for prop in cls.get_properties():
                        if hasattr(prop, "name"):
                            class_properties.append(prop.name)
                except (AttributeError, TypeError) as e:
                    # Some classes may not support get_properties
                    logger.debug(
                        "ontology_class_get_properties_failed",
                        class_name=cls.name,
                        error=str(e),
                    )

                # Get description from rdfs:comment
                description = None
                if hasattr(cls, "comment") and cls.comment:
                    comments = list(cls.comment)
                    if comments:
                        description = str(comments[0])

                ont_class = OntologyClass(
                    uri=cls.iri,
                    name=cls.name,
                    description=description,
                    parent_uris=parent_uris,
                    properties=class_properties,
                )
                classes.append(ont_class)

            # Extract properties
            properties: list[OntologyProperty] = []
            for prop in onto.properties():
                # Determine property type
                is_object = isinstance(prop, ObjectPropertyClass) if OWLREADY2_AVAILABLE else True
                is_data = isinstance(prop, DataPropertyClass) if OWLREADY2_AVAILABLE else False

                # Get domain and range
                domain_uri = None
                range_uri = None

                if hasattr(prop, "domain") and prop.domain:
                    domain_list = list(prop.domain)
                    if domain_list and hasattr(domain_list[0], "iri"):
                        domain_uri = domain_list[0].iri

                if hasattr(prop, "range") and prop.range:
                    range_list = list(prop.range)
                    if range_list:
                        if hasattr(range_list[0], "iri"):
                            range_uri = range_list[0].iri
                        else:
                            # Datatype range
                            range_uri = str(range_list[0])

                # Get description
                description = None
                if hasattr(prop, "comment") and prop.comment:
                    comments = list(prop.comment)
                    if comments:
                        description = str(comments[0])

                ont_prop = OntologyProperty(
                    uri=prop.iri,
                    name=prop.name,
                    domain=domain_uri,
                    range=range_uri,
                    description=description,
                    is_object_property=is_object,
                    is_data_property=is_data,
                )
                properties.append(ont_prop)

            # Create the loaded ontology
            ontology_name = onto.name if onto.name else Path(ontology_path).stem
            loaded = LoadedOntology(
                name=ontology_name,
                tenant_id=tenant_id,
                classes=classes,
                properties=properties,
                source_path=ontology_path,
                metadata={
                    "base_iri": str(onto.base_iri) if onto.base_iri else None,
                },
            )

            # Store for this tenant
            self._loaded_ontologies[tenant_id] = loaded

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "ontology_loaded",
                name=ontology_name,
                classes=len(classes),
                properties=len(properties),
                tenant_id=tenant_id,
                load_time_ms=round(elapsed_ms, 2),
            )

            return OntologyLoadResult(
                ontology=loaded,
                success=True,
                load_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "ontology_load_failed",
                path=ontology_path,
                tenant_id=tenant_id,
                error=str(e),
            )
            return OntologyLoadResult.failure(
                error=f"Failed to load ontology: {e}",
                load_time_ms=elapsed_ms,
            )

    def get_entity_type(
        self,
        entity_name: str,
        tenant_id: str,
        case_sensitive: bool = False,
    ) -> Optional[OntologyClass]:
        """Find matching ontology class for an entity name.

        This method attempts to match an entity name to a loaded ontology class
        using substring matching on class names.

        Args:
            entity_name: The name of the entity to type
            tenant_id: The tenant identifier
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            The matching OntologyClass or None if no match found
        """
        if tenant_id not in self._loaded_ontologies:
            logger.debug(
                "ontology_not_loaded_for_tenant",
                tenant_id=tenant_id,
            )
            return None

        ontology = self._loaded_ontologies[tenant_id]

        # First try exact name match
        for cls in ontology.classes:
            if case_sensitive:
                if cls.name == entity_name:
                    return cls
            else:
                if cls.name.lower() == entity_name.lower():
                    return cls

        # Then try substring match
        for cls in ontology.classes:
            if cls.matches_name(entity_name, case_sensitive=case_sensitive):
                return cls

        return None

    def get_ontology(self, tenant_id: str) -> Optional[LoadedOntology]:
        """Get the loaded ontology for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            The LoadedOntology or None if not loaded
        """
        return self._loaded_ontologies.get(tenant_id)

    def list_classes(self, tenant_id: str) -> list[OntologyClass]:
        """List all classes in the tenant's ontology.

        Args:
            tenant_id: The tenant identifier

        Returns:
            List of OntologyClass instances, empty if no ontology loaded
        """
        ontology = self._loaded_ontologies.get(tenant_id)
        if ontology is None:
            return []
        return ontology.classes.copy()

    def list_properties(self, tenant_id: str) -> list[OntologyProperty]:
        """List all properties in the tenant's ontology.

        Args:
            tenant_id: The tenant identifier

        Returns:
            List of OntologyProperty instances, empty if no ontology loaded
        """
        ontology = self._loaded_ontologies.get(tenant_id)
        if ontology is None:
            return []
        return ontology.properties.copy()

    def unload_ontology(self, tenant_id: str) -> bool:
        """Unload the ontology for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            True if an ontology was unloaded, False if none was loaded
        """
        if tenant_id in self._loaded_ontologies:
            del self._loaded_ontologies[tenant_id]
            logger.info("ontology_unloaded", tenant_id=tenant_id)
            return True
        return False

    def is_loaded(self, tenant_id: str) -> bool:
        """Check if an ontology is loaded for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            True if an ontology is loaded for this tenant
        """
        return tenant_id in self._loaded_ontologies

    def get_loaded_tenant_ids(self) -> list[str]:
        """Get list of tenant IDs that have loaded ontologies.

        Returns:
            List of tenant IDs
        """
        return list(self._loaded_ontologies.keys())


# Default constants for configuration
DEFAULT_ONTOLOGY_SUPPORT_ENABLED = False
DEFAULT_ONTOLOGY_AUTO_TYPE = False
