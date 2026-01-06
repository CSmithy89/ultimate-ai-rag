"""Data models for ontology support.

Story 20-E1: Implement Ontology Support
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class OntologyClass:
    """A class from an OWL ontology.

    Represents a concept/type that can be used to classify entities
    during knowledge graph ingestion.

    Attributes:
        uri: The full URI of the ontology class (e.g., "http://example.org/Person")
        name: The local name of the class (e.g., "Person")
        description: Optional human-readable description from rdfs:comment
        parent_uris: List of parent class URIs (for inheritance hierarchy)
        properties: List of property names applicable to this class
        metadata: Additional metadata from the ontology
    """

    uri: str
    name: str
    description: Optional[str] = None
    parent_uris: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the ontology class after initialization."""
        if not self.uri:
            raise ValueError("OntologyClass uri cannot be empty")
        if not self.name:
            raise ValueError("OntologyClass name cannot be empty")

    def is_subclass_of(self, parent_uri: str) -> bool:
        """Check if this class is a subclass of the given parent.

        Args:
            parent_uri: The URI of the potential parent class

        Returns:
            True if this class has the parent in its parent_uris
        """
        return parent_uri in self.parent_uris

    def matches_name(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if this class name matches the given text.

        Args:
            text: The text to match against the class name
            case_sensitive: Whether to do case-sensitive matching

        Returns:
            True if the class name is found in the text
        """
        if case_sensitive:
            return self.name in text
        return self.name.lower() in text.lower()


@dataclass
class OntologyProperty:
    """A property from an OWL ontology.

    Represents a relationship or attribute that can exist between entities
    or between an entity and a value.

    Attributes:
        uri: The full URI of the property (e.g., "http://example.org/hasParent")
        name: The local name of the property (e.g., "hasParent")
        domain: The class URI that can have this property (subject)
        range: The class URI or datatype that this property points to (object)
        description: Optional human-readable description from rdfs:comment
        is_object_property: True if this is an object property (relates entities)
        is_data_property: True if this is a data property (relates to values)
        metadata: Additional metadata from the ontology
    """

    uri: str
    name: str
    domain: Optional[str] = None
    range: Optional[str] = None
    description: Optional[str] = None
    is_object_property: bool = True
    is_data_property: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the ontology property after initialization."""
        if not self.uri:
            raise ValueError("OntologyProperty uri cannot be empty")
        if not self.name:
            raise ValueError("OntologyProperty name cannot be empty")
        # A property should be either object or data property, not both
        if self.is_object_property and self.is_data_property:
            raise ValueError(
                "OntologyProperty cannot be both object and data property"
            )


@dataclass
class LoadedOntology:
    """A loaded ontology with its classes and properties.

    Represents the complete parsed result of loading an OWL ontology file.

    Attributes:
        name: The name or IRI of the ontology
        tenant_id: The tenant that owns this ontology
        classes: List of OntologyClass instances
        properties: List of OntologyProperty instances
        source_path: The path or URL from which the ontology was loaded
        loaded_at: Timestamp when the ontology was loaded
        metadata: Additional metadata (version, annotations, etc.)
    """

    name: str
    tenant_id: str
    classes: list[OntologyClass] = field(default_factory=list)
    properties: list[OntologyProperty] = field(default_factory=list)
    source_path: Optional[str] = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def class_count(self) -> int:
        """Return the number of classes in this ontology."""
        return len(self.classes)

    @property
    def property_count(self) -> int:
        """Return the number of properties in this ontology."""
        return len(self.properties)

    def get_class_by_name(self, name: str) -> Optional[OntologyClass]:
        """Find a class by its local name.

        Args:
            name: The local name to search for

        Returns:
            The matching OntologyClass or None
        """
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None

    def get_class_by_uri(self, uri: str) -> Optional[OntologyClass]:
        """Find a class by its URI.

        Args:
            uri: The full URI to search for

        Returns:
            The matching OntologyClass or None
        """
        for cls in self.classes:
            if cls.uri == uri:
                return cls
        return None

    def get_property_by_name(self, name: str) -> Optional[OntologyProperty]:
        """Find a property by its local name.

        Args:
            name: The local name to search for

        Returns:
            The matching OntologyProperty or None
        """
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None

    def get_properties_for_class(self, class_uri: str) -> list[OntologyProperty]:
        """Get all properties applicable to a class.

        Args:
            class_uri: The URI of the class

        Returns:
            List of OntologyProperty instances with matching domain
        """
        return [
            prop for prop in self.properties
            if prop.domain == class_uri or prop.domain is None
        ]


@dataclass
class OntologyLoadResult:
    """Result of loading an ontology.

    Attributes:
        ontology: The loaded ontology (None if loading failed)
        success: Whether loading was successful
        error: Error message if loading failed
        load_time_ms: Time taken to load the ontology in milliseconds
    """

    ontology: Optional[LoadedOntology] = None
    success: bool = True
    error: Optional[str] = None
    load_time_ms: float = 0.0

    @staticmethod
    def failure(error: str, load_time_ms: float = 0.0) -> "OntologyLoadResult":
        """Create a failure result.

        Args:
            error: The error message
            load_time_ms: Time spent before failure

        Returns:
            An OntologyLoadResult indicating failure
        """
        return OntologyLoadResult(
            ontology=None,
            success=False,
            error=error,
            load_time_ms=load_time_ms,
        )
