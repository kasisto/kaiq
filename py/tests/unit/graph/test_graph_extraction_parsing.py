"""
Unit tests for graph extraction XML parsing with lxml recovery
and batched embedding calls.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.main.services.graph_service import GraphService
from shared.abstractions.document import DocumentChunk
from shared.abstractions.graph import Entity, Relationship


def _make_chunk(doc_id: uuid.UUID, chunk_id: uuid.UUID, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        owner_id=uuid.uuid4(),
        collection_ids=[],
        data=text,
        metadata={},
    )


def _make_service() -> GraphService:
    """Create a GraphService with mocked providers."""
    config = MagicMock()
    config.app.fast_llm = "openai/gpt-5.4-mini"
    config.database.graph_creation_settings.generation_config = None

    providers = MagicMock()
    providers.database.config.graph_creation_settings.generation_config = None

    service = GraphService.__new__(GraphService)
    service.config = config
    service.providers = providers
    return service


class TestXMLParsingWithRecovery:
    """Tests for lxml recover=True XML parsing."""

    @pytest.mark.asyncio
    async def test_valid_xml_parses_correctly(self):
        service = _make_service()
        service.providers.embedding.async_get_embeddings = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        )

        doc_id = uuid.uuid4()
        chunks = [_make_chunk(doc_id, uuid.uuid4(), "test")]

        xml = """
        <entity name="Widget">
            <type>Product</type>
            <description>A useful widget</description>
        </entity>
        <relationship>
            <source>Widget</source>
            <target>Factory</target>
            <type>Produced By</type>
            <description>Widget is produced by Factory</description>
            <weight>5.0</weight>
        </relationship>
        """
        entities, rels = await service._parse_graph_search_results_extraction_xml(
            xml, chunks
        )
        assert len(entities) == 1
        assert entities[0].name == "Widget"
        assert entities[0].category == "Product"
        assert len(rels) == 1
        assert rels[0].subject == "Widget"
        assert rels[0].object == "Factory"

    @pytest.mark.asyncio
    async def test_truncated_xml_recovers_complete_entities(self):
        """lxml recover=True should salvage entities before the truncation."""
        service = _make_service()
        service.providers.embedding.async_get_embeddings = AsyncMock(
            return_value=[[0.1, 0.2]]
        )

        doc_id = uuid.uuid4()
        chunks = [_make_chunk(doc_id, uuid.uuid4(), "test")]

        # XML truncated mid-second entity
        xml = """
        <entity name="Complete">
            <type>Thing</type>
            <description>Fully formed</description>
        </entity>
        <entity name="Truncated">
            <type>Thing</type>
            <description>This gets cut off mid-sent
        """
        entities, rels = await service._parse_graph_search_results_extraction_xml(
            xml, chunks
        )
        # Should recover at least the first complete entity
        assert len(entities) >= 1
        assert entities[0].name == "Complete"

    @pytest.mark.asyncio
    async def test_malformed_attributes_recovered(self):
        """lxml should handle minor XML malformations."""
        service = _make_service()
        service.providers.embedding.async_get_embeddings = AsyncMock(
            return_value=[[0.1, 0.2]]
        )

        doc_id = uuid.uuid4()
        chunks = [_make_chunk(doc_id, uuid.uuid4(), "test")]

        # Missing closing > on description — lxml recovers
        xml = """
        <entity name="Widget">
            <type>Product</type>
            <description>A widget</description>
        </entity>
        """
        entities, _ = await service._parse_graph_search_results_extraction_xml(
            xml, chunks
        )
        assert len(entities) == 1


class TestBatchedEmbeddings:
    """Tests that embedding calls are batched, not per-entity."""

    @pytest.mark.asyncio
    async def test_single_embedding_call_for_multiple_entities(self):
        """All descriptions should be embedded in one batch call."""
        service = _make_service()
        mock_embeddings = AsyncMock(
            return_value=[[0.1], [0.2], [0.3], [0.4], [0.5]]
        )
        service.providers.embedding.async_get_embeddings = mock_embeddings

        doc_id = uuid.uuid4()
        chunks = [_make_chunk(doc_id, uuid.uuid4(), "test")]

        xml = """
        <entity name="A"><type>T</type><description>desc A</description></entity>
        <entity name="B"><type>T</type><description>desc B</description></entity>
        <entity name="C"><type>T</type><description>desc C</description></entity>
        <relationship>
            <source>A</source><target>B</target><type>R</type>
            <description>A to B</description><weight>1.0</weight>
        </relationship>
        <relationship>
            <source>B</source><target>C</target><type>R</type>
            <description>B to C</description><weight>1.0</weight>
        </relationship>
        """
        entities, rels = await service._parse_graph_search_results_extraction_xml(
            xml, chunks
        )
        assert len(entities) == 3
        assert len(rels) == 2
        # Should be exactly ONE call to async_get_embeddings (batched)
        assert mock_embeddings.call_count == 1
        # With all 5 descriptions (3 entities + 2 relationships)
        call_args = mock_embeddings.call_args[0][0]
        assert len(call_args) == 5

    @pytest.mark.asyncio
    async def test_empty_xml_no_embedding_call(self):
        """No entities or relationships should not call embeddings."""
        service = _make_service()
        mock_embeddings = AsyncMock(return_value=[])
        service.providers.embedding.async_get_embeddings = mock_embeddings

        doc_id = uuid.uuid4()
        chunks = [_make_chunk(doc_id, uuid.uuid4(), "test")]

        xml = ""
        entities, rels = await service._parse_graph_search_results_extraction_xml(
            xml, chunks
        )
        assert len(entities) == 0
        assert len(rels) == 0
        assert mock_embeddings.call_count == 0


class TestGetGraphGenConfig:
    """Tests for _get_graph_gen_config helper."""

    def test_returns_config_with_model_when_set(self):
        service = _make_service()
        gen_config = MagicMock()
        gen_config.model = "openai/gpt-5.4-mini"
        service.providers.database.config.graph_creation_settings.generation_config = gen_config

        result = service._get_graph_gen_config()
        assert result.model == "openai/gpt-5.4-mini"

    def test_falls_back_to_fast_llm_when_model_none(self):
        from shared.abstractions.llm import GenerationConfig

        service = _make_service()
        gen_config = GenerationConfig(model=None, max_tokens_to_sample=4096)
        service.providers.database.config.graph_creation_settings.generation_config = gen_config

        result = service._get_graph_gen_config()
        assert result.model == "openai/gpt-5.4-mini"
        assert result.max_tokens_to_sample == 4096

    def test_creates_default_when_no_config(self):
        service = _make_service()
        service.providers.database.config.graph_creation_settings.generation_config = None

        result = service._get_graph_gen_config()
        assert result.model == "openai/gpt-5.4-mini"


class TestStoreExtractionsParallel:
    """Tests that entity/relationship storage uses concurrent inserts."""

    @pytest.mark.asyncio
    async def test_entities_inserted_concurrently(self):
        """asyncio.gather should be used for entity inserts."""
        from shared.abstractions.graph import GraphExtraction

        service = _make_service()

        entity_ids = [uuid.uuid4() for _ in range(3)]
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return MagicMock(id=entity_ids[idx], name=kwargs["name"])

        service.providers.database.graphs_handler.entities.create = mock_create
        service.providers.database.graphs_handler.relationships.create = AsyncMock()

        doc_id = uuid.uuid4()
        extraction = GraphExtraction(
            entities=[
                Entity(name="A", parent_id=doc_id, description="a", chunk_ids=[]),
                Entity(name="B", parent_id=doc_id, description="b", chunk_ids=[]),
                Entity(name="C", parent_id=doc_id, description="c", chunk_ids=[]),
            ],
            relationships=[
                Relationship(
                    subject="A", object="B", predicate="links",
                    parent_id=doc_id, chunk_ids=[],
                ),
            ],
        )

        await service.store_graph_search_results_extractions([extraction])
        # All 3 entities should have been created
        assert call_count == 3
        # Relationship should have been created
        service.providers.database.graphs_handler.relationships.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_relationships_with_missing_entity_ids_skipped(self):
        """Relationships referencing unknown entities should be skipped."""
        from shared.abstractions.graph import GraphExtraction

        service = _make_service()

        async def mock_create(**kwargs):
            return MagicMock(id=uuid.uuid4(), name=kwargs["name"])

        service.providers.database.graphs_handler.entities.create = mock_create
        service.providers.database.graphs_handler.relationships.create = AsyncMock()

        doc_id = uuid.uuid4()
        extraction = GraphExtraction(
            entities=[
                Entity(name="A", parent_id=doc_id, description="a", chunk_ids=[]),
            ],
            relationships=[
                Relationship(
                    subject="A", object="UNKNOWN", predicate="links",
                    parent_id=doc_id, chunk_ids=[],
                ),
            ],
        )

        await service.store_graph_search_results_extractions([extraction])
        # Relationship should be skipped (UNKNOWN not in entity map)
        service.providers.database.graphs_handler.relationships.create.assert_not_called()
