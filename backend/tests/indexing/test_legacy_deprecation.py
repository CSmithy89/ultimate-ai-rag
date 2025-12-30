"""Tests for legacy module deprecation and migration to Graphiti."""

from unittest.mock import patch


class TestLegacyModuleDocumentation:
    """Tests verifying legacy modules have deprecation documentation."""

    def test_embeddings_has_deprecation_warning_code(self):
        """Embeddings module should have deprecation warning code."""
        import agentic_rag_backend.indexing.embeddings as embeddings
        import inspect
        
        source = inspect.getsource(embeddings)
        assert "DeprecationWarning" in source
        assert "deprecated" in source.lower()

    def test_entity_extractor_has_deprecation_warning_code(self):
        """Entity extractor module should have deprecation warning code."""
        import agentic_rag_backend.indexing.entity_extractor as entity_extractor
        import inspect
        
        source = inspect.getsource(entity_extractor)
        assert "DeprecationWarning" in source
        assert "deprecated" in source.lower()

    def test_graph_builder_has_deprecation_warning_code(self):
        """Graph builder module should have deprecation warning code."""
        import agentic_rag_backend.indexing.graph_builder as graph_builder
        import inspect
        
        source = inspect.getsource(graph_builder)
        assert "DeprecationWarning" in source
        assert "deprecated" in source.lower()

    def test_indexing_init_has_migration_notes(self):
        """Indexing __init__ should have migration notes."""
        import agentic_rag_backend.indexing as indexing
        
        docstring = indexing.__doc__
        assert docstring is not None
        assert "DEPRECATED" in docstring
        assert "graphiti_ingestion" in docstring.lower()


class TestMigrationToGraphiti:
    """Tests for migration path to Graphiti-based functions."""

    def test_graphiti_ingestion_available(self):
        """Should be able to import Graphiti ingestion functions."""
        from agentic_rag_backend.indexing.graphiti_ingestion import (
            ingest_document_as_episode,
            ingest_with_backend_routing,
        )

        assert callable(ingest_document_as_episode)
        assert callable(ingest_with_backend_routing)

    def test_indexing_init_exports_graphiti_functions(self):
        """Indexing package should export Graphiti functions."""
        from agentic_rag_backend.indexing import (
            ingest_document_as_episode,
            ingest_with_backend_routing,
        )

        assert callable(ingest_document_as_episode)
        assert callable(ingest_with_backend_routing)

    def test_ingestion_backend_config_available(self):
        """Config should have ingestion_backend setting."""
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test",
            "DATABASE_URL": "postgresql://test",
            "NEO4J_URI": "bolt://localhost",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost",
        }):
            from agentic_rag_backend.config import load_settings
            load_settings.cache_clear() if hasattr(load_settings, 'cache_clear') else None
            settings = load_settings()
            
            assert hasattr(settings, 'ingestion_backend')
            assert settings.ingestion_backend in ['graphiti', 'legacy']

    def test_episode_entity_types_exported(self):
        """EPISODE_ENTITY_TYPES should be exported."""
        from agentic_rag_backend.indexing import EPISODE_ENTITY_TYPES
        
        assert isinstance(EPISODE_ENTITY_TYPES, list)
        assert len(EPISODE_ENTITY_TYPES) == 4  # TechnicalConcept, CodePattern, etc.


class TestLegacyAndNewCoexistence:
    """Tests for parallel operation of legacy and Graphiti code."""

    def test_both_backends_can_be_imported(self):
        """Should be able to import both legacy and new modules."""
        # New Graphiti-based module
        from agentic_rag_backend.indexing.graphiti_ingestion import (
            ingest_document_as_episode,
        )
        
        # Legacy modules still importable
        from agentic_rag_backend.indexing.embeddings import EmbeddingGenerator
        
        # Verify both work
        assert callable(ingest_document_as_episode)
        assert EmbeddingGenerator is not None

    def test_episode_ingestion_result_has_required_fields(self):
        """EpisodeIngestionResult should have all required fields."""
        from agentic_rag_backend.indexing.graphiti_ingestion import (
            EpisodeIngestionResult,
        )

        result = EpisodeIngestionResult(
            document_id="doc-123",
            tenant_id="45645645-6456-4564-5645-645645645645",
            episode_uuid="ep-789",
            entities_extracted=5,
            edges_created=3,
            processing_time_ms=1000,
        )

        assert result.document_id == "doc-123"
        assert result.tenant_id == "45645645-6456-4564-5645-645645645645"
        assert result.episode_uuid == "ep-789"
        assert result.entities_extracted == 5
        assert result.edges_created == 3
        assert result.processing_time_ms == 1000

    def test_all_exports_in_init(self):
        """All expected exports should be in __all__."""
        from agentic_rag_backend.indexing import __all__
        
        # Graphiti functions
        assert "ingest_document_as_episode" in __all__
        assert "ingest_with_backend_routing" in __all__
        assert "EpisodeIngestionResult" in __all__
        assert "EPISODE_ENTITY_TYPES" in __all__
        
        # Legacy functions still exported for backwards compatibility
        assert "EmbeddingGenerator" in __all__
        assert "EntityExtractor" in __all__
        assert "GraphBuilder" in __all__
