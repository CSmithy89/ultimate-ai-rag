"""Tests for Graphiti client wrapper and entity types."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.models.entity_types import (
    TechnicalConcept,
    CodePattern,
    APIEndpoint,
    ConfigurationOption,
    ENTITY_TYPES,
    EDGE_TYPE_MAPPINGS,
    get_edge_types,
)


class TestEntityTypes:
    """Tests for custom entity type definitions."""

    def test_technical_concept_defaults(self):
        """TechnicalConcept should have correct default values."""
        concept = TechnicalConcept(name="FastAPI", group_id="test-group")
        assert concept.name == "FastAPI"
        assert concept.domain == "general"
        assert concept.complexity == "intermediate"
        assert concept.category is None

    def test_technical_concept_custom_values(self):
        """TechnicalConcept should accept custom field values."""
        concept = TechnicalConcept(
            name="GraphQL",
            group_id="test-group",
            domain="backend",
            complexity="advanced",
            category="api",
        )
        assert concept.domain == "backend"
        assert concept.complexity == "advanced"
        assert concept.category == "api"

    def test_code_pattern_defaults(self):
        """CodePattern should have correct default values."""
        pattern = CodePattern(name="Singleton", group_id="test-group")
        assert pattern.name == "Singleton"
        assert pattern.language == "python"
        assert pattern.pattern_type == "implementation"
        assert pattern.use_case is None

    def test_code_pattern_custom_values(self):
        """CodePattern should accept custom field values."""
        pattern = CodePattern(
            name="Observer",
            group_id="test-group",
            language="typescript",
            pattern_type="design",
            use_case="event handling",
        )
        assert pattern.language == "typescript"
        assert pattern.pattern_type == "design"
        assert pattern.use_case == "event handling"

    def test_api_endpoint_defaults(self):
        """APIEndpoint should have correct default values."""
        endpoint = APIEndpoint(name="GetUsers", group_id="test-group")
        assert endpoint.name == "GetUsers"
        assert endpoint.method == "GET"
        assert endpoint.path == "/"
        assert endpoint.version is None
        assert endpoint.auth_required is True

    def test_api_endpoint_custom_values(self):
        """APIEndpoint should accept custom field values."""
        endpoint = APIEndpoint(
            name="CreateUser",
            group_id="test-group",
            method="POST",
            path="/api/v1/users",
            version="v1",
            auth_required=False,
        )
        assert endpoint.method == "POST"
        assert endpoint.path == "/api/v1/users"
        assert endpoint.version == "v1"
        assert endpoint.auth_required is False

    def test_configuration_option_defaults(self):
        """ConfigurationOption should have correct default values."""
        config = ConfigurationOption(name="DATABASE_URL", group_id="test-group")
        assert config.name == "DATABASE_URL"
        assert config.config_type == "environment"
        assert config.default_value is None
        assert config.required is False
        assert config.sensitive is False

    def test_configuration_option_custom_values(self):
        """ConfigurationOption should accept custom field values."""
        config = ConfigurationOption(
            name="API_KEY",
            group_id="test-group",
            config_type="environment",
            default_value=None,
            required=True,
            sensitive=True,
        )
        assert config.config_type == "environment"
        assert config.required is True
        assert config.sensitive is True

    def test_entity_types_registry(self):
        """ENTITY_TYPES should contain all custom entity types."""
        assert len(ENTITY_TYPES) == 4
        assert TechnicalConcept in ENTITY_TYPES
        assert CodePattern in ENTITY_TYPES
        assert APIEndpoint in ENTITY_TYPES
        assert ConfigurationOption in ENTITY_TYPES


class TestEdgeTypeMappings:
    """Tests for edge type mapping configuration."""

    def test_technical_concept_to_technical_concept(self):
        """Should have edge types for TechnicalConcept pairs."""
        edges = get_edge_types("TechnicalConcept", "TechnicalConcept")
        assert "DEPENDS_ON" in edges
        assert "RELATED_TO" in edges
        assert "EXTENDS" in edges

    def test_technical_concept_to_code_pattern(self):
        """Should have edge types for TechnicalConcept to CodePattern."""
        edges = get_edge_types("TechnicalConcept", "CodePattern")
        assert "IMPLEMENTED_BY" in edges
        assert "USES_PATTERN" in edges

    def test_code_pattern_to_code_pattern(self):
        """Should have edge types for CodePattern pairs."""
        edges = get_edge_types("CodePattern", "CodePattern")
        assert "COMPOSES_WITH" in edges
        assert "VARIANT_OF" in edges

    def test_api_endpoint_to_technical_concept(self):
        """Should have edge types for APIEndpoint to TechnicalConcept."""
        edges = get_edge_types("APIEndpoint", "TechnicalConcept")
        assert "EXPOSES" in edges
        assert "REQUIRES" in edges

    def test_configuration_option_to_technical_concept(self):
        """Should have edge types for ConfigurationOption to TechnicalConcept."""
        edges = get_edge_types("ConfigurationOption", "TechnicalConcept")
        assert "CONFIGURES" in edges
        assert "ENABLES" in edges

    def test_unmapped_pair_returns_defaults(self):
        """Unmapped entity pairs should return default edge types."""
        edges = get_edge_types("UnknownType", "AnotherType")
        assert "RELATED_TO" in edges
        assert "REFERENCES" in edges

    def test_edge_type_mappings_structure(self):
        """EDGE_TYPE_MAPPINGS should have correct structure."""
        for key, value in EDGE_TYPE_MAPPINGS.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], str)
            assert isinstance(value, list)
            assert all(isinstance(edge, str) for edge in value)


class TestGraphitiClient:
    """Tests for GraphitiClient wrapper."""

    @pytest.fixture
    def mock_graphiti_imports(self):
        """Mock graphiti_core imports."""
        with patch.dict("sys.modules", {
            "graphiti_core": MagicMock(),
            "graphiti_core.llm_client": MagicMock(),
            "graphiti_core.embedder": MagicMock(),
        }):
            yield

    def test_client_initialization(self, mock_graphiti_imports):
        """GraphitiClient should initialize with correct parameters."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        client = GraphitiClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            openai_api_key="sk-test",
        )
        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        # API key is private - not exposed as public attribute
        assert client.is_connected is False

    def test_client_not_connected_raises(self, mock_graphiti_imports):
        """Accessing client before connect should raise RuntimeError."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        client = GraphitiClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            openai_api_key="sk-test",
        )
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_client_connect(self, mock_graphiti_imports):
        """GraphitiClient should connect successfully."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        with patch("agentic_rag_backend.db.graphiti.Graphiti") as mock_graphiti, \
             patch("agentic_rag_backend.db.graphiti.OpenAIClient"), \
             patch("agentic_rag_backend.db.graphiti.OpenAIEmbedder"):

            mock_graphiti.return_value = MagicMock()

            client = GraphitiClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                openai_api_key="sk-test",
            )
            await client.connect()

            assert client.is_connected is True
            mock_graphiti.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_disconnect(self, mock_graphiti_imports):
        """GraphitiClient should disconnect cleanly."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        with patch("agentic_rag_backend.db.graphiti.Graphiti") as mock_graphiti, \
             patch("agentic_rag_backend.db.graphiti.OpenAIClient"), \
             patch("agentic_rag_backend.db.graphiti.OpenAIEmbedder"):

            mock_instance = MagicMock()
            mock_instance.driver = MagicMock()
            mock_instance.driver.close = AsyncMock()
            mock_graphiti.return_value = mock_instance

            client = GraphitiClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                openai_api_key="sk-test",
            )
            await client.connect()
            await client.disconnect()

            assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_build_indices(self, mock_graphiti_imports):
        """GraphitiClient should build indices successfully."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        with patch("agentic_rag_backend.db.graphiti.Graphiti") as mock_graphiti, \
             patch("agentic_rag_backend.db.graphiti.OpenAIClient"), \
             patch("agentic_rag_backend.db.graphiti.OpenAIEmbedder"):

            mock_instance = MagicMock()
            mock_instance.build_indices_and_constraints = AsyncMock()
            mock_graphiti.return_value = mock_instance

            client = GraphitiClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                openai_api_key="sk-test",
            )
            await client.connect()
            await client.build_indices()

            mock_instance.build_indices_and_constraints.assert_called_once()

    def test_get_entity_types(self, mock_graphiti_imports):
        """GraphitiClient should return configured entity types."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        client = GraphitiClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            openai_api_key="sk-test",
        )
        entity_types = client.get_entity_types()

        assert len(entity_types) == 4
        assert TechnicalConcept in entity_types

    def test_get_edge_type_mappings(self, mock_graphiti_imports):
        """GraphitiClient should return edge type mappings."""
        from agentic_rag_backend.db.graphiti import GraphitiClient

        client = GraphitiClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            openai_api_key="sk-test",
        )
        mappings = client.get_edge_type_mappings()

        assert isinstance(mappings, dict)
        assert ("TechnicalConcept", "TechnicalConcept") in mappings
