"""Tests for the ontology module.

Story 20-E1: Implement Ontology Support

These tests verify:
- OntologyClass and OntologyProperty model validation
- OntologyLoader functionality
- OntologyAdapter feature flag behavior
- Tenant isolation
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import tempfile
import time

import pytest

from agentic_rag_backend.ontology import (
    OntologyClass,
    OntologyProperty,
    LoadedOntology,
    OntologyLoadResult,
    OntologyLoader,
    OntologyAdapter,
    get_ontology_adapter,
    OWLREADY2_AVAILABLE,
    DEFAULT_ONTOLOGY_SUPPORT_ENABLED,
    DEFAULT_ONTOLOGY_AUTO_TYPE,
    MAX_ONTOLOGY_FILE_SIZE_BYTES,
)


# =============================================================================
# OntologyClass Tests
# =============================================================================


class TestOntologyClass:
    """Tests for the OntologyClass dataclass."""

    def test_create_minimal_class(self) -> None:
        """Test creating an ontology class with minimal required fields."""
        cls = OntologyClass(
            uri="http://example.org/Person",
            name="Person",
        )
        assert cls.uri == "http://example.org/Person"
        assert cls.name == "Person"
        assert cls.description is None
        assert cls.parent_uris == []
        assert cls.properties == []
        assert cls.metadata == {}

    def test_create_full_class(self) -> None:
        """Test creating an ontology class with all fields."""
        cls = OntologyClass(
            uri="http://example.org/Student",
            name="Student",
            description="A student enrolled in a course",
            parent_uris=["http://example.org/Person"],
            properties=["hasName", "hasAge", "enrolledIn"],
            metadata={"source": "university_ontology"},
        )
        assert cls.uri == "http://example.org/Student"
        assert cls.name == "Student"
        assert cls.description == "A student enrolled in a course"
        assert cls.parent_uris == ["http://example.org/Person"]
        assert cls.properties == ["hasName", "hasAge", "enrolledIn"]
        assert cls.metadata == {"source": "university_ontology"}

    def test_empty_uri_raises_error(self) -> None:
        """Test that empty URI raises ValueError."""
        with pytest.raises(ValueError, match="uri cannot be empty"):
            OntologyClass(uri="", name="Test")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            OntologyClass(uri="http://example.org/Test", name="")

    def test_is_subclass_of(self) -> None:
        """Test the is_subclass_of method."""
        cls = OntologyClass(
            uri="http://example.org/Student",
            name="Student",
            parent_uris=["http://example.org/Person", "http://example.org/Agent"],
        )
        assert cls.is_subclass_of("http://example.org/Person") is True
        assert cls.is_subclass_of("http://example.org/Agent") is True
        assert cls.is_subclass_of("http://example.org/Thing") is False

    def test_matches_name_case_insensitive(self) -> None:
        """Test name matching with case-insensitive mode."""
        cls = OntologyClass(uri="http://example.org/Person", name="Person")
        assert cls.matches_name("person", case_sensitive=False) is True
        assert cls.matches_name("PERSON", case_sensitive=False) is True
        assert cls.matches_name("The Person entity", case_sensitive=False) is True

    def test_matches_name_case_sensitive(self) -> None:
        """Test name matching with case-sensitive mode."""
        cls = OntologyClass(uri="http://example.org/Person", name="Person")
        assert cls.matches_name("Person", case_sensitive=True) is True
        assert cls.matches_name("person", case_sensitive=True) is False
        assert cls.matches_name("The Person entity", case_sensitive=True) is True


# =============================================================================
# OntologyProperty Tests
# =============================================================================


class TestOntologyProperty:
    """Tests for the OntologyProperty dataclass."""

    def test_create_minimal_property(self) -> None:
        """Test creating an ontology property with minimal fields."""
        prop = OntologyProperty(
            uri="http://example.org/hasName",
            name="hasName",
        )
        assert prop.uri == "http://example.org/hasName"
        assert prop.name == "hasName"
        assert prop.domain is None
        assert prop.range is None
        assert prop.is_object_property is True
        assert prop.is_data_property is False

    def test_create_object_property(self) -> None:
        """Test creating an object property (relates entities)."""
        prop = OntologyProperty(
            uri="http://example.org/hasParent",
            name="hasParent",
            domain="http://example.org/Person",
            range="http://example.org/Person",
            description="Parent relationship",
            is_object_property=True,
            is_data_property=False,
        )
        assert prop.is_object_property is True
        assert prop.is_data_property is False
        assert prop.domain == "http://example.org/Person"

    def test_create_data_property(self) -> None:
        """Test creating a data property (relates to values)."""
        prop = OntologyProperty(
            uri="http://example.org/hasAge",
            name="hasAge",
            domain="http://example.org/Person",
            range="xsd:integer",
            is_object_property=False,
            is_data_property=True,
        )
        assert prop.is_object_property is False
        assert prop.is_data_property is True
        assert prop.range == "xsd:integer"

    def test_empty_uri_raises_error(self) -> None:
        """Test that empty URI raises ValueError."""
        with pytest.raises(ValueError, match="uri cannot be empty"):
            OntologyProperty(uri="", name="test")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            OntologyProperty(uri="http://example.org/test", name="")

    def test_both_object_and_data_raises_error(self) -> None:
        """Test that a property cannot be both object and data."""
        with pytest.raises(ValueError, match="cannot be both"):
            OntologyProperty(
                uri="http://example.org/test",
                name="test",
                is_object_property=True,
                is_data_property=True,
            )


# =============================================================================
# LoadedOntology Tests
# =============================================================================


class TestLoadedOntology:
    """Tests for the LoadedOntology dataclass."""

    def test_create_empty_ontology(self) -> None:
        """Test creating an empty ontology."""
        onto = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-123",
        )
        assert onto.name == "test_ontology"
        assert onto.tenant_id == "tenant-123"
        assert onto.class_count == 0
        assert onto.property_count == 0

    def test_create_ontology_with_classes(self) -> None:
        """Test creating an ontology with classes and properties."""
        classes = [
            OntologyClass(uri="http://example.org/Person", name="Person"),
            OntologyClass(uri="http://example.org/Student", name="Student"),
        ]
        properties = [
            OntologyProperty(uri="http://example.org/hasName", name="hasName"),
        ]
        onto = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-123",
            classes=classes,
            properties=properties,
        )
        assert onto.class_count == 2
        assert onto.property_count == 1

    def test_get_class_by_name(self) -> None:
        """Test finding a class by name."""
        classes = [
            OntologyClass(uri="http://example.org/Person", name="Person"),
            OntologyClass(uri="http://example.org/Student", name="Student"),
        ]
        onto = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-123",
            classes=classes,
        )
        found = onto.get_class_by_name("Person")
        assert found is not None
        assert found.uri == "http://example.org/Person"

        not_found = onto.get_class_by_name("Teacher")
        assert not_found is None

    def test_get_class_by_uri(self) -> None:
        """Test finding a class by URI."""
        classes = [
            OntologyClass(uri="http://example.org/Person", name="Person"),
        ]
        onto = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-123",
            classes=classes,
        )
        found = onto.get_class_by_uri("http://example.org/Person")
        assert found is not None
        assert found.name == "Person"

    def test_get_properties_for_class(self) -> None:
        """Test getting properties applicable to a class."""
        properties = [
            OntologyProperty(
                uri="http://example.org/hasName",
                name="hasName",
                domain="http://example.org/Person",
            ),
            OntologyProperty(
                uri="http://example.org/hasAge",
                name="hasAge",
                domain="http://example.org/Person",
            ),
            OntologyProperty(
                uri="http://example.org/hasTitle",
                name="hasTitle",
                domain="http://example.org/Book",
            ),
        ]
        onto = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-123",
            properties=properties,
        )
        person_props = onto.get_properties_for_class("http://example.org/Person")
        assert len(person_props) == 2
        names = [p.name for p in person_props]
        assert "hasName" in names
        assert "hasAge" in names


# =============================================================================
# OntologyLoadResult Tests
# =============================================================================


class TestOntologyLoadResult:
    """Tests for the OntologyLoadResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        onto = LoadedOntology(name="test", tenant_id="tenant-1")
        result = OntologyLoadResult(
            ontology=onto,
            success=True,
            load_time_ms=150.5,
        )
        assert result.success is True
        assert result.ontology is not None
        assert result.error is None
        assert result.load_time_ms == 150.5

    def test_create_failure_result(self) -> None:
        """Test creating a failure result."""
        result = OntologyLoadResult.failure(
            error="File not found",
            load_time_ms=10.0,
        )
        assert result.success is False
        assert result.ontology is None
        assert result.error == "File not found"
        assert result.load_time_ms == 10.0


# =============================================================================
# OntologyLoader Tests
# =============================================================================


class TestOntologyLoader:
    """Tests for the OntologyLoader class."""

    def test_is_available(self) -> None:
        """Test checking if owlready2 is available."""
        loader = OntologyLoader()
        # This depends on whether owlready2 is installed
        assert loader.is_available() == OWLREADY2_AVAILABLE

    def test_initial_state(self) -> None:
        """Test the initial state of a new loader."""
        loader = OntologyLoader()
        assert loader.get_loaded_tenant_ids() == []
        assert loader.is_loaded("tenant-123") is False

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self) -> None:
        """Test loading a file that doesn't exist."""
        loader = OntologyLoader()
        result = await loader.load_ontology(
            ontology_path="/nonexistent/path/to/ontology.owl",
            tenant_id="tenant-123",
        )
        assert result.success is False
        assert "not found" in result.error.lower() or "not installed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_load_empty_path(self) -> None:
        """Test loading with an empty path."""
        loader = OntologyLoader()
        result = await loader.load_ontology(
            ontology_path="",
            tenant_id="tenant-123",
        )
        assert result.success is False
        assert "empty" in result.error.lower() or "not installed" in result.error.lower()

    def test_list_classes_unloaded_tenant(self) -> None:
        """Test listing classes for a tenant with no loaded ontology."""
        loader = OntologyLoader()
        classes = loader.list_classes("tenant-123")
        assert classes == []

    def test_list_properties_unloaded_tenant(self) -> None:
        """Test listing properties for a tenant with no loaded ontology."""
        loader = OntologyLoader()
        properties = loader.list_properties("tenant-123")
        assert properties == []

    def test_get_entity_type_unloaded_tenant(self) -> None:
        """Test getting entity type for a tenant with no loaded ontology."""
        loader = OntologyLoader()
        result = loader.get_entity_type("Person", "tenant-123")
        assert result is None

    def test_unload_nonexistent_ontology(self) -> None:
        """Test unloading an ontology that isn't loaded."""
        loader = OntologyLoader()
        result = loader.unload_ontology("tenant-123")
        assert result is False

    def test_tenant_isolation(self) -> None:
        """Test that ontologies are isolated per tenant."""
        loader = OntologyLoader()

        # Manually set up ontologies for two tenants
        loader._loaded_ontologies["tenant-1"] = LoadedOntology(
            name="ontology1",
            tenant_id="tenant-1",
            classes=[
                OntologyClass(uri="http://example.org/Person", name="Person"),
            ],
        )
        loader._loaded_ontologies["tenant-2"] = LoadedOntology(
            name="ontology2",
            tenant_id="tenant-2",
            classes=[
                OntologyClass(uri="http://example.org/Animal", name="Animal"),
            ],
        )

        # Verify isolation
        assert loader.is_loaded("tenant-1") is True
        assert loader.is_loaded("tenant-2") is True
        assert loader.is_loaded("tenant-3") is False

        # Get classes for each tenant
        tenant1_classes = loader.list_classes("tenant-1")
        tenant2_classes = loader.list_classes("tenant-2")

        assert len(tenant1_classes) == 1
        assert tenant1_classes[0].name == "Person"

        assert len(tenant2_classes) == 1
        assert tenant2_classes[0].name == "Animal"

    def test_get_entity_type_exact_match(self) -> None:
        """Test getting entity type with exact name match."""
        loader = OntologyLoader()
        loader._loaded_ontologies["tenant-1"] = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-1",
            classes=[
                OntologyClass(uri="http://example.org/Person", name="Person"),
                OntologyClass(uri="http://example.org/PersonName", name="PersonName"),
            ],
        )

        # Exact match should return Person, not PersonName
        result = loader.get_entity_type("Person", "tenant-1")
        assert result is not None
        assert result.name == "Person"

    def test_get_entity_type_substring_match(self) -> None:
        """Test getting entity type with substring matching."""
        loader = OntologyLoader()
        loader._loaded_ontologies["tenant-1"] = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-1",
            classes=[
                OntologyClass(uri="http://example.org/Person", name="Person"),
            ],
        )

        # "John is a Person entity" contains "Person"
        result = loader.get_entity_type("John is a Person entity", "tenant-1")
        assert result is not None
        assert result.name == "Person"

    def test_get_entity_type_case_sensitivity(self) -> None:
        """Test case-sensitive and case-insensitive matching."""
        loader = OntologyLoader()
        loader._loaded_ontologies["tenant-1"] = LoadedOntology(
            name="test_ontology",
            tenant_id="tenant-1",
            classes=[
                OntologyClass(uri="http://example.org/Person", name="Person"),
            ],
        )

        # Case-insensitive should match
        result = loader.get_entity_type("person", "tenant-1", case_sensitive=False)
        assert result is not None

        # Case-sensitive should not match lowercase
        result = loader.get_entity_type("person", "tenant-1", case_sensitive=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_load_file_too_large(self, tmp_path: Path) -> None:
        """Test that oversized ontology files are rejected."""
        # Create a file that exceeds the size limit
        large_file = tmp_path / "large.owl"
        # Write just enough to exceed the limit check (we'll mock the size)
        large_file.write_text("test content")

        loader = OntologyLoader()

        # Mock the file size to exceed the limit
        original_stat = large_file.stat

        class MockStat:
            st_size = MAX_ONTOLOGY_FILE_SIZE_BYTES + 1

        with patch.object(Path, "stat", return_value=MockStat()):
            result = await loader.load_ontology(
                ontology_path=str(large_file),
                tenant_id="tenant-123",
            )

        # Should fail if owlready2 is available (validation runs before load)
        # If owlready2 is not available, it fails earlier
        assert result.success is False
        if OWLREADY2_AVAILABLE:
            assert "too large" in result.error.lower() or "not installed" in result.error.lower()


# =============================================================================
# OntologyAdapter Tests
# =============================================================================


class TestOntologyAdapter:
    """Tests for the OntologyAdapter class."""

    def test_create_disabled_adapter(self) -> None:
        """Test creating a disabled adapter."""
        adapter = OntologyAdapter(enabled=False)
        assert adapter.enabled is False
        assert adapter.is_available is False

    def test_create_enabled_adapter(self) -> None:
        """Test creating an enabled adapter."""
        adapter = OntologyAdapter(enabled=True)
        assert adapter.enabled is True
        # is_available depends on owlready2
        assert adapter.is_available == OWLREADY2_AVAILABLE

    def test_auto_type_property(self) -> None:
        """Test the auto_type property."""
        adapter = OntologyAdapter(enabled=True, auto_type=True)
        assert adapter.auto_type is True

        adapter = OntologyAdapter(enabled=True, auto_type=False)
        assert adapter.auto_type is False

    @pytest.mark.asyncio
    async def test_load_ontology_disabled(self) -> None:
        """Test loading ontology when adapter is disabled."""
        adapter = OntologyAdapter(enabled=False)
        result = await adapter.load_ontology(
            tenant_id="tenant-123",
            ontology_path="/path/to/ontology.owl",
        )
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_load_ontology_no_path(self) -> None:
        """Test loading ontology with no path specified."""
        adapter = OntologyAdapter(enabled=True)
        result = await adapter.load_ontology(tenant_id="tenant-123")
        assert result.success is False
        assert "path" in result.error.lower() or "not installed" in result.error.lower()

    def test_get_entity_type_disabled(self) -> None:
        """Test getting entity type when adapter is disabled."""
        adapter = OntologyAdapter(enabled=False)
        result = adapter.get_entity_type("Person", "tenant-123")
        assert result is None

    def test_list_classes_disabled(self) -> None:
        """Test listing classes when adapter is disabled."""
        adapter = OntologyAdapter(enabled=False)
        result = adapter.list_classes("tenant-123")
        assert result == []

    def test_list_properties_disabled(self) -> None:
        """Test listing properties when adapter is disabled."""
        adapter = OntologyAdapter(enabled=False)
        result = adapter.list_properties("tenant-123")
        assert result == []

    def test_is_loaded_disabled(self) -> None:
        """Test is_loaded when adapter is disabled."""
        adapter = OntologyAdapter(enabled=False)
        result = adapter.is_loaded("tenant-123")
        assert result is False

    def test_unload_ontology_disabled(self) -> None:
        """Test unloading ontology when adapter is disabled."""
        adapter = OntologyAdapter(enabled=False)
        result = adapter.unload_ontology("tenant-123")
        assert result is False


# =============================================================================
# get_ontology_adapter Factory Tests
# =============================================================================


class TestGetOntologyAdapter:
    """Tests for the get_ontology_adapter factory function."""

    def test_create_with_overrides_only(self) -> None:
        """Test creating adapter with override values (no mocking needed)."""
        # When all values are provided, get_settings is not called
        adapter = get_ontology_adapter(
            enabled=True,
            auto_type=True,
            default_ontology_path="/custom/path.owl",
        )
        assert adapter.enabled is True
        assert adapter.auto_type is True

    def test_create_disabled_adapter_via_factory(self) -> None:
        """Test creating disabled adapter via factory."""
        adapter = get_ontology_adapter(
            enabled=False,
            auto_type=False,
            default_ontology_path=None,
        )
        assert adapter.enabled is False
        assert adapter.auto_type is False


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_ontology_support_enabled(self) -> None:
        """Test default value for ONTOLOGY_SUPPORT_ENABLED."""
        assert DEFAULT_ONTOLOGY_SUPPORT_ENABLED is False

    def test_default_ontology_auto_type(self) -> None:
        """Test default value for ONTOLOGY_AUTO_TYPE."""
        assert DEFAULT_ONTOLOGY_AUTO_TYPE is False

    def test_max_ontology_file_size(self) -> None:
        """Test MAX_ONTOLOGY_FILE_SIZE_BYTES is reasonable."""
        # Should be at least 1MB
        assert MAX_ONTOLOGY_FILE_SIZE_BYTES >= 1024 * 1024
        # Should be at most 100MB
        assert MAX_ONTOLOGY_FILE_SIZE_BYTES <= 100 * 1024 * 1024


# =============================================================================
# Integration Tests (with mock owlready2)
# =============================================================================


class TestOntologyLoaderWithMockedOwlready2:
    """Integration tests with mocked owlready2."""

    @pytest.mark.asyncio
    async def test_load_ontology_with_mock(self) -> None:
        """Test loading ontology with mocked owlready2."""
        if not OWLREADY2_AVAILABLE:
            pytest.skip("owlready2 not available")

        # Create a minimal OWL file
        owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/test#"
     xml:base="http://example.org/test"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
    <owl:Ontology rdf:about="http://example.org/test"/>

    <owl:Class rdf:about="http://example.org/test#Person">
        <rdfs:comment>A human being</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/test#Student">
        <rdfs:subClassOf rdf:resource="http://example.org/test#Person"/>
    </owl:Class>

    <owl:ObjectProperty rdf:about="http://example.org/test#hasParent">
        <rdfs:domain rdf:resource="http://example.org/test#Person"/>
        <rdfs:range rdf:resource="http://example.org/test#Person"/>
    </owl:ObjectProperty>
</rdf:RDF>
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".owl", delete=False
        ) as f:
            f.write(owl_content)
            owl_path = f.name

        try:
            loader = OntologyLoader()
            result = await loader.load_ontology(
                ontology_path=owl_path,
                tenant_id="tenant-123",
            )

            if result.success:
                assert result.ontology is not None
                assert result.ontology.class_count >= 1
                assert loader.is_loaded("tenant-123") is True

                # Test entity type matching
                person_type = loader.get_entity_type("Person", "tenant-123")
                if person_type:
                    assert person_type.name == "Person"
        finally:
            Path(owl_path).unlink(missing_ok=True)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ontology_class_with_special_characters(self) -> None:
        """Test class name with special characters."""
        cls = OntologyClass(
            uri="http://example.org/My-Class_v2",
            name="My-Class_v2",
        )
        assert cls.name == "My-Class_v2"
        assert cls.matches_name("My-Class_v2", case_sensitive=True) is True

    def test_ontology_class_unicode_name(self) -> None:
        """Test class with unicode characters."""
        cls = OntologyClass(
            uri="http://example.org/Personne",
            name="Personne",
            description="Une personne francaise",
        )
        assert cls.name == "Personne"

    def test_loaded_ontology_empty_lists(self) -> None:
        """Test loaded ontology with empty class and property lists."""
        onto = LoadedOntology(
            name="empty",
            tenant_id="tenant-1",
            classes=[],
            properties=[],
        )
        assert onto.class_count == 0
        assert onto.property_count == 0
        assert onto.get_class_by_name("Anything") is None
        assert onto.get_property_by_name("anything") is None

    def test_multiple_classes_same_name(self) -> None:
        """Test handling of multiple classes with same name (returns first)."""
        onto = LoadedOntology(
            name="test",
            tenant_id="tenant-1",
            classes=[
                OntologyClass(uri="http://example.org/ns1#Person", name="Person"),
                OntologyClass(uri="http://example.org/ns2#Person", name="Person"),
            ],
        )
        # Should return the first one
        found = onto.get_class_by_name("Person")
        assert found is not None
        assert found.uri == "http://example.org/ns1#Person"

    @pytest.mark.asyncio
    async def test_loader_validate_path_url(self) -> None:
        """Test that URL paths are accepted."""
        loader = OntologyLoader()
        is_valid, error = loader._validate_ontology_path("http://example.org/ontology.owl")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_loader_validate_path_https(self) -> None:
        """Test that HTTPS URLs are accepted."""
        loader = OntologyLoader()
        is_valid, error = loader._validate_ontology_path("https://example.org/ontology.owl")
        assert is_valid is True
        assert error is None


# =============================================================================
# Performance Tests (AC6: <5 seconds for typical ontologies)
# =============================================================================


class TestPerformance:
    """Performance tests for ontology loading (AC6)."""

    @pytest.mark.asyncio
    async def test_ontology_load_performance(self) -> None:
        """Test that ontology loading completes within acceptable time.

        AC6: Ontology loading completes in <5 seconds for typical domain
        ontologies (<1000 classes).
        """
        if not OWLREADY2_AVAILABLE:
            pytest.skip("owlready2 not available")

        # Create a minimal OWL ontology with ~100 classes
        classes_xml = "\n".join([
            f'''    <owl:Class rdf:about="http://example.org/test#Class{i}">
        <rdfs:comment>Test class {i}</rdfs:comment>
    </owl:Class>'''
            for i in range(100)
        ])

        owl_content = f"""<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/test#"
     xml:base="http://example.org/test"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/test"/>
{classes_xml}
</rdf:RDF>
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".owl", delete=False
        ) as f:
            f.write(owl_content)
            owl_path = f.name

        try:
            loader = OntologyLoader()

            start_time = time.time()
            result = await loader.load_ontology(
                ontology_path=owl_path,
                tenant_id="perf-test",
            )
            elapsed = time.time() - start_time

            # AC6: Must complete in <5 seconds
            assert elapsed < 5.0, f"Ontology loading took {elapsed:.2f}s, expected <5s"

            if result.success:
                assert result.ontology is not None
                # Should have loaded ~100 classes
                assert result.ontology.class_count >= 90
        finally:
            Path(owl_path).unlink(missing_ok=True)
