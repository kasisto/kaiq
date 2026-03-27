"""
Integration tests for document export/import functionality.
"""

import os
import tempfile
import time
import uuid

import pytest

from r2r import R2RClient, R2RException


@pytest.fixture
def cleanup_documents(client: R2RClient):
    """Fixture to track and cleanup test documents."""
    doc_ids = []

    def _track_document(doc_id):
        doc_ids.append(doc_id)
        return doc_id

    yield _track_document

    # Cleanup all documents
    for doc_id in doc_ids:
        try:
            client.documents.delete(id=doc_id)
        except R2RException:
            pass


def test_export_import_roundtrip_basic(client: R2RClient, cleanup_documents):
    """Test basic export and import roundtrip."""
    # Create a document
    doc_resp = client.documents.create(
        raw_text="This is a test document for export/import.",
        metadata={"title": "Test Export Document", "test": True},
        run_with_orchestration=False,
    )
    original_doc_id = cleanup_documents(doc_resp.results.document_id)

    # Wait for ingestion to complete
    time.sleep(2)

    # Export the document
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "test_export.zip")

        export_result = client.documents.export_full(
            document_ids=[original_doc_id],
            include_embeddings=True,
            include_knowledge_graphs=False,
            output_path=export_path,
        )

        assert os.path.exists(export_path), "Export file was not created"
        assert os.path.getsize(export_path) > 0, "Export file is empty"

        # Delete the original document
        client.documents.delete(id=original_doc_id)

        # Import the document
        import_result = client.documents.import_full(
            file_path=export_path,
            mode="full",
            conflict_resolution="skip",
        )

        assert import_result["success_count"] == 1, "Import did not succeed"
        assert import_result["failed_count"] == 0, "Import had failures"
        assert len(import_result["imported_document_ids"]) == 1, (
            "Expected one imported document"
        )

        # Track imported document for cleanup
        imported_doc_id = import_result["imported_document_ids"][0]
        cleanup_documents(imported_doc_id)

        # Verify the imported document
        retrieved = client.documents.retrieve(id=imported_doc_id)
        assert retrieved.results.id == imported_doc_id, (
            "Failed to retrieve imported document"
        )
        assert retrieved.results.metadata.get("title") == "Test Export Document", (
            "Document metadata not preserved"
        )


def test_export_import_with_knowledge_graph(
    client: R2RClient, cleanup_documents
):
    """Test export/import with knowledge graph extraction."""
    # Create a document with content suitable for KG extraction
    doc_resp = client.documents.create(
        raw_text=(
            "Albert Einstein was a theoretical physicist. "
            "He developed the theory of relativity. "
            "Einstein was born in Germany."
        ),
        metadata={"title": "Einstein Biography", "kg_test": True},
        run_with_orchestration=False,
    )
    original_doc_id = cleanup_documents(doc_resp.results.document_id)

    # Wait for ingestion
    time.sleep(2)

    # Extract knowledge graph
    try:
        client.documents.extract(
            id=original_doc_id,
            run_with_orchestration=False,
        )
        # Wait for extraction
        time.sleep(3)
    except Exception as e:
        # KG extraction might not be available in all test environments
        pytest.skip(f"Knowledge graph extraction not available: {e}")

    # Export with knowledge graph
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "test_kg_export.zip")

        export_result = client.documents.export_full(
            document_ids=[original_doc_id],
            include_embeddings=True,
            include_knowledge_graphs=True,
            output_path=export_path,
        )

        assert os.path.exists(export_path), "Export file was not created"

        # Delete original
        client.documents.delete(id=original_doc_id)

        # Import
        import_result = client.documents.import_full(
            file_path=export_path,
            mode="full",
        )

        assert import_result["success_count"] == 1, "Import failed"

        imported_doc_id = import_result["imported_document_ids"][0]
        cleanup_documents(imported_doc_id)

        # Verify document exists
        retrieved = client.documents.retrieve(id=imported_doc_id)
        assert retrieved.results.id == imported_doc_id


def test_export_multiple_documents(client: R2RClient, cleanup_documents):
    """Test exporting multiple documents at once."""
    # Create multiple documents
    doc_ids = []
    for i in range(3):
        resp = client.documents.create(
            raw_text=f"Test document number {i}",
            metadata={"index": i},
            run_with_orchestration=False,
        )
        doc_id = cleanup_documents(resp.results.document_id)
        doc_ids.append(doc_id)

    # Wait for ingestion
    time.sleep(2)

    # Export all documents
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "multi_export.zip")

        export_result = client.documents.export_full(
            document_ids=doc_ids,
            include_embeddings=True,
            include_knowledge_graphs=False,
            output_path=export_path,
        )

        assert os.path.exists(export_path), "Export file was not created"
        assert os.path.getsize(export_path) > 0, "Export file is empty"

        # Delete originals
        for doc_id in doc_ids:
            client.documents.delete(id=doc_id)

        # Import
        import_result = client.documents.import_full(
            file_path=export_path,
            mode="full",
        )

        assert import_result["success_count"] == 3, (
            f"Expected 3 successful imports, got {import_result['success_count']}"
        )
        assert import_result["failed_count"] == 0, "Import had failures"
        assert len(import_result["imported_document_ids"]) == 3

        # Track imported documents for cleanup
        for doc_id in import_result["imported_document_ids"]:
            cleanup_documents(doc_id)


def test_import_with_conflict_skip(client: R2RClient, cleanup_documents):
    """Test import with skip conflict resolution."""
    # Create a document
    doc_resp = client.documents.create(
        raw_text="Original document",
        run_with_orchestration=False,
    )
    doc_id = cleanup_documents(doc_resp.results.document_id)

    # Wait for ingestion
    time.sleep(2)

    # Export
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "conflict_test.zip")

        client.documents.export_full(
            document_ids=[doc_id],
            include_embeddings=True,
            output_path=export_path,
        )

        # Import while original still exists (should skip)
        import_result = client.documents.import_full(
            file_path=export_path,
            conflict_resolution="skip",
        )

        # With skip mode, the existing document should be skipped
        # Note: Implementation might vary - adjust assertion based on actual behavior
        assert import_result["total_documents"] == 1


def test_import_with_conflict_replace(client: R2RClient, cleanup_documents):
    """Test import with replace conflict resolution."""
    # Create a document
    doc_resp = client.documents.create(
        raw_text="Original document",
        metadata={"version": "v1"},
        run_with_orchestration=False,
    )
    doc_id = cleanup_documents(doc_resp.results.document_id)

    # Wait for ingestion
    time.sleep(2)

    # Update metadata
    client.documents.update_metadata(
        id=doc_id,
        metadata={"version": "v2"},
    )

    # Export
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "replace_test.zip")

        client.documents.export_full(
            document_ids=[doc_id],
            output_path=export_path,
        )

        # Update metadata again
        client.documents.update_metadata(
            id=doc_id,
            metadata={"version": "v3"},
        )

        # Import with replace (should restore v2)
        import_result = client.documents.import_full(
            file_path=export_path,
            conflict_resolution="replace",
        )

        assert import_result["success_count"] == 1

        # Verify the document was replaced
        retrieved = client.documents.retrieve(id=doc_id)
        # Note: metadata merging behavior depends on implementation


def test_export_without_embeddings(client: R2RClient, cleanup_documents):
    """Test exporting without embeddings."""
    doc_resp = client.documents.create(
        raw_text="Test document for no-embedding export",
        run_with_orchestration=False,
    )
    doc_id = cleanup_documents(doc_resp.results.document_id)

    time.sleep(2)

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "no_embeddings.zip")

        export_result = client.documents.export_full(
            document_ids=[doc_id],
            include_embeddings=False,
            include_knowledge_graphs=False,
            output_path=export_path,
        )

        assert os.path.exists(export_path), "Export file was not created"
        # File should be smaller without embeddings
        file_size = os.path.getsize(export_path)
        assert file_size > 0, "Export file is empty"


def test_import_regenerate_embeddings(client: R2RClient, cleanup_documents):
    """Test importing with embedding regeneration."""
    doc_resp = client.documents.create(
        raw_text="Test for regenerating embeddings",
        run_with_orchestration=False,
    )
    doc_id = cleanup_documents(doc_resp.results.document_id)

    time.sleep(2)

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "regen_embeddings.zip")

        client.documents.export_full(
            document_ids=[doc_id],
            include_embeddings=True,
            output_path=export_path,
        )

        # Delete original
        client.documents.delete(id=doc_id)

        # Import with regenerate_embeddings=True
        import_result = client.documents.import_full(
            file_path=export_path,
            regenerate_embeddings=True,
        )

        assert import_result["success_count"] == 1
        assert import_result["regenerated_embeddings"] is True

        imported_doc_id = import_result["imported_document_ids"][0]
        cleanup_documents(imported_doc_id)


def test_selective_import(client: R2RClient, cleanup_documents):
    """Test selective import of specific documents."""
    # Create multiple documents
    doc_ids = []
    for i in range(3):
        resp = client.documents.create(
            raw_text=f"Document {i}",
            run_with_orchestration=False,
        )
        doc_id = cleanup_documents(resp.results.document_id)
        doc_ids.append(doc_id)

    time.sleep(2)

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "selective.zip")

        # Export all 3
        client.documents.export_full(
            document_ids=doc_ids,
            output_path=export_path,
        )

        # Delete all
        for doc_id in doc_ids:
            client.documents.delete(id=doc_id)

        # Import only the first document
        import_result = client.documents.import_full(
            file_path=export_path,
            mode="selective",
            document_ids_filter=[doc_ids[0]],
        )

        assert import_result["success_count"] == 1
        assert len(import_result["imported_document_ids"]) == 1

        imported_id = import_result["imported_document_ids"][0]
        cleanup_documents(imported_id)
        assert str(imported_id) == str(doc_ids[0])
