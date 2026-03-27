# Document Export/Import Feature

## Overview

The document export/import feature enables users to export documents along with their knowledge graphs and all relevant data in a zipped format. This allows users to import documents without having to reingest them, preserving all metadata, embeddings, chunks, and knowledge graph entities/relationships.

## Key Features

- **Complete Data Preservation**: Export documents with all associated data including:
  - Document metadata and original files
  - Text chunks with embeddings
  - Knowledge graph entities and relationships
  - Collection metadata

- **Zero Re-ingestion**: Import documents without re-parsing or re-processing
- **Flexible Import Modes**: Full, metadata-only, or selective import
- **Conflict Resolution**: Handle existing documents with skip, replace, or error strategies
- **Portable Format**: Standard ZIP format with JSON/JSONL and NumPy arrays

## Architecture

### Components Implemented

1. **Data Models** ([py/shared/abstractions/export_import.py](py/shared/abstractions/export_import.py))
   - `ExportConfig`: Configuration for export operations
   - `ImportConfig`: Configuration for import operations
   - `ExportManifest`: Manifest file structure
   - `ImportResult`: Import operation results
   - `ImportMode`: Enum for import modes (FULL, METADATA_ONLY, SELECTIVE)
   - `ConflictResolution`: Enum for conflict handling (SKIP, REPLACE, ERROR)

2. **Services**
   - `ExportService` ([py/core/main/services/export_service.py](py/core/main/services/export_service.py)): Handles document export
   - `ImportService` ([py/core/main/services/import_service.py](py/core/main/services/import_service.py)): Handles document import

3. **API Endpoints** ([py/core/main/api/v3/documents_router.py](py/core/main/api/v3/documents_router.py))
   - `POST /documents/export-full`: Export documents with knowledge graphs
   - `POST /documents/import-full`: Import documents from export package

4. **SDK Methods**
   - Async SDK: [py/sdk/asnyc_methods/documents.py](py/sdk/asnyc_methods/documents.py)
   - Sync SDK: [py/sdk/sync_methods/documents.py](py/sdk/sync_methods/documents.py)

5. **Integration Tests** ([py/tests/integration/test_export_import.py](py/tests/integration/test_export_import.py))

## Export Package Structure

```
export_<timestamp>.zip
├── manifest.json                    # Export metadata and index
├── schema_version.txt               # Export format version (3.0)
├── documents/
│   ├── <doc_id_1>/
│   │   ├── metadata.json           # Document metadata
│   │   ├── file.<ext>              # Original file
│   │   ├── chunks.jsonl            # All chunks (JSONL format)
│   │   ├── embeddings/
│   │   │   ├── summary.npy         # Summary embedding (NumPy)
│   │   │   └── chunks.npy          # Chunk embeddings (NumPy)
│   │   └── knowledge_graph/
│   │       ├── entities.jsonl      # Document entities
│   │       ├── relationships.jsonl # Document relationships
│   │       └── embeddings/
│   │           ├── entities.npy    # Entity embeddings
│   │           └── relationships.npy # Relationship embeddings
│   └── ...
└── collections/
    └── <collection_id>/
        └── metadata.json            # Collection metadata
```

## Usage Examples

### Python SDK

#### Export Documents

```python
from r2r import R2RClient

client = R2RClient()

# Export single document
export_bytes = client.documents.export_full(
    document_ids=["doc-uuid-1"],
    include_embeddings=True,
    include_knowledge_graphs=True,
    output_path="export.zip"  # Optional: saves to file
)

# Export multiple documents
export_bytes = client.documents.export_full(
    document_ids=["doc-uuid-1", "doc-uuid-2", "doc-uuid-3"],
    include_embeddings=True,
    include_knowledge_graphs=True,
    include_collections=True,
    output_path="multi_export.zip"
)

# Export without embeddings (smaller file size)
export_bytes = client.documents.export_full(
    document_ids=["doc-uuid-1"],
    include_embeddings=False,
    include_knowledge_graphs=True,
    output_path="no_embeddings.zip"
)
```

#### Import Documents

```python
# Full import (default)
result = client.documents.import_full(
    file_path="export.zip",
    mode="full",
    conflict_resolution="skip"
)

print(f"Imported {result['success_count']} documents")
print(f"Skipped {result['skipped_count']} conflicts")
print(f"Failed {result['failed_count']} documents")

# Import with conflict replacement
result = client.documents.import_full(
    file_path="export.zip",
    conflict_resolution="replace"  # Replace existing documents
)

# Selective import (only specific documents)
result = client.documents.import_full(
    file_path="export.zip",
    mode="selective",
    document_ids_filter=["doc-uuid-1", "doc-uuid-2"]
)

# Import to a different collection
result = client.documents.import_full(
    file_path="export.zip",
    target_collection_id="new-collection-uuid"
)

# Import with embedding regeneration
result = client.documents.import_full(
    file_path="export.zip",
    regenerate_embeddings=True  # Regenerate embeddings instead of using exported ones
)
```

### REST API

#### Export Endpoint

```bash
curl -X POST "http://localhost:7272/v3/documents/export-full" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc-uuid-1", "doc-uuid-2"],
    "include_embeddings": true,
    "include_knowledge_graphs": true,
    "include_collections": true
  }' \
  --output export.zip
```

#### Import Endpoint

```bash
curl -X POST "http://localhost:7272/v3/documents/import-full" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@export.zip" \
  -F "mode=full" \
  -F "conflict_resolution=skip"
```

## Configuration Options

### Export Configuration

- **document_ids** (required): List of document UUIDs to export
- **include_embeddings** (default: true): Include vector embeddings in export
- **include_knowledge_graphs** (default: true): Include KG entities and relationships
- **include_collections** (default: true): Include collection metadata

### Import Configuration

- **mode** (default: "full"): Import mode
  - `full`: Import everything including embeddings
  - `metadata_only`: Skip embeddings (useful for regenerating with different model)
  - `selective`: Only import specific documents (requires document_ids_filter)

- **conflict_resolution** (default: "skip"): How to handle existing documents
  - `skip`: Skip documents that already exist
  - `replace`: Delete and replace existing documents
  - `error`: Fail if any conflicts are found

- **regenerate_embeddings** (default: false): Force regeneration of embeddings even if included in export

- **target_collection_id** (optional): Override collection for all imported documents

- **document_ids_filter** (optional): List of document UUIDs to import (for selective mode)

## Manifest File Format

The `manifest.json` file contains metadata about the export:

```json
{
  "export_version": "1.0",
  "export_timestamp": "2026-03-26T12:00:00Z",
  "export_source": "kaiq-r2r",
  "schema_version": "3.0",
  "total_documents": 2,
  "documents": [
    {
      "id": "uuid-1",
      "title": "Document Title",
      "type": "pdf",
      "size_bytes": 102400,
      "has_embeddings": true,
      "has_knowledge_graph": true,
      "chunk_count": 42,
      "entity_count": 15,
      "relationship_count": 28,
      "collections": ["collection-uuid-1"]
    }
  ],
  "collections": [
    {
      "id": "collection-uuid-1",
      "name": "My Collection",
      "description": "Collection description",
      "document_count": 5
    }
  ],
  "embedding_model": "openai/text-embedding-3-small",
  "embedding_dimensions": 1536,
  "includes_embeddings": true,
  "includes_knowledge_graphs": true
}
```

## Use Cases

### 1. Backup and Restore

Export documents for backup purposes and restore them without reingestion:

```python
# Backup
all_docs = client.documents.list(limit=1000)
doc_ids = [doc.id for doc in all_docs.results]
client.documents.export_full(
    document_ids=doc_ids,
    output_path="backup_2026_03_26.zip"
)

# Restore
client.documents.import_full(
    file_path="backup_2026_03_26.zip",
    conflict_resolution="skip"
)
```

### 2. Environment Migration

Move documents between development, staging, and production:

```python
# Export from dev environment
dev_client = R2RClient("https://dev.example.com")
dev_client.documents.export_full(
    document_ids=selected_docs,
    output_path="migration.zip"
)

# Import to production
prod_client = R2RClient("https://prod.example.com")
prod_client.documents.import_full(
    file_path="migration.zip",
    conflict_resolution="error"  # Fail if conflicts
)
```

### 3. Knowledge Base Sharing

Share curated documents with knowledge graphs between teams:

```python
# Team A exports their knowledge base
team_a_client.documents.export_full(
    document_ids=curated_docs,
    include_knowledge_graphs=True,
    output_path="team_a_kb.zip"
)

# Team B imports to their collection
team_b_client.documents.import_full(
    file_path="team_a_kb.zip",
    target_collection_id="team-b-shared-kb"
)
```

### 4. Embedding Model Migration

Re-embed documents with a different embedding model:

```python
# Export without embeddings
client.documents.export_full(
    document_ids=all_docs,
    include_embeddings=False,
    output_path="docs_no_embeddings.zip"
)

# Update embedding configuration to new model
# Then import with regeneration
client.documents.import_full(
    file_path="docs_no_embeddings.zip",
    regenerate_embeddings=True,
    conflict_resolution="replace"
)
```

## Error Handling

The import process provides detailed error information:

```python
result = client.documents.import_full(file_path="export.zip")

# Check results
print(f"Success: {result['success_count']}")
print(f"Skipped: {result['skipped_count']}")  # Documents already existing (SKIP mode)
print(f"Failed: {result['failed_count']}")

if result["failed_count"] > 0:
    for error in result["errors"]:
        print(f"Document {error['document_id']} ({error['document_title']}) failed:")
        print(f"  Type: {error['error_type']}")
        print(f"  Message: {error['error_message']}")
```

### Conflict Resolution Behavior

When importing documents that already exist:

- **SKIP mode**: Existing documents are filtered out before import. They are counted in `skipped_count` and their IDs are in `skipped_document_ids`. No errors are raised.

- **REPLACE mode**: Existing documents are deleted first, then re-imported with all new data (chunks, embeddings, knowledge graphs). The `skipped_count` will be 0.

- **ERROR mode**: Import immediately fails with a 409 status code and a user-friendly error message listing conflicting document IDs. The message suggests using SKIP or REPLACE mode instead.

## Validation and Compatibility

### Schema Version Validation

The import service validates that the export schema version is compatible with the current system:

- Export format version: `1.0`
- Database schema version: `3.0`

### Embedding Dimension Validation

When importing embeddings, the system validates that embedding dimensions match:

```python
# This will fail if dimensions don't match
try:
    result = client.documents.import_full(file_path="export.zip")
except Exception as e:
    print(f"Import failed: {e}")
    # Use regenerate_embeddings=True to fix
    result = client.documents.import_full(
        file_path="export.zip",
        regenerate_embeddings=True
    )
```

## Performance Considerations

### Export Performance

- Exports are streamed to handle large datasets
- Embeddings are stored in efficient NumPy binary format
- ZIP compression reduces file size

### Import Performance

- Bulk inserts are used for chunks and entities
- Transactions ensure atomicity
- Large imports may take time depending on data volume

### File Size Estimation

Approximate export file sizes:
- 1 document with 100 chunks: ~1-5 MB (with embeddings)
- 1 document with 100 chunks: ~500 KB (without embeddings)
- Knowledge graph adds: ~100-500 KB per document (depending on entity count)

## Security and Authorization

- Users can only export documents they have access to
- Import operations create documents with the importing user as owner
- Collection access is validated during both export and import

## Testing

Run integration tests:

```bash
pytest py/tests/integration/test_export_import.py -v
```

Test coverage includes:
- Basic export/import roundtrip
- Multiple document export
- Knowledge graph preservation
- Conflict resolution strategies
- Selective import
- Embedding regeneration
- Error handling

## Troubleshooting

### Common Issues

**Issue**: "Embedding dimension mismatch"
```
Solution: Use regenerate_embeddings=True or update embedding configuration
```

**Issue**: "Document conflicts found"
```
Solution: Use conflict_resolution="replace" or "skip" instead of "error"
```

**Issue**: "Invalid export: manifest.json not found"
```
Solution: Ensure the ZIP file is a valid export package
```

**Issue**: "Access denied to document"
```
Solution: Verify user has permission to access the documents being exported
```

## Limitations

- Collection-level graph data is not included in Phase 1 (document-scoped only)
- Large exports (>10,000 documents) may require streaming optimization
- Embedding regeneration requires the embedding service to be available

## Future Enhancements

Planned improvements:
- Collection-level graph export/import
- Incremental exports (only new/changed documents)
- Async job support for large exports
- Export/import audit logging
- Compression level configuration
- Progress tracking for large imports

## API Response Format

### Import Response Structure

The import endpoint returns results in the following format:

```json
{
  "results": {
    "success_count": 5,
    "skipped_count": 2,
    "failed_count": 0,
    "total_documents": 7,
    "imported_document_ids": ["uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5"],
    "skipped_document_ids": ["uuid-6", "uuid-7"],
    "errors": [],
    "collections_created": ["collection-uuid-1"],
    "regenerated_embeddings": false
  }
}
```

**Note**: The SDK automatically unwraps the `results` object for convenience, but direct API calls will receive this nested structure.

## API Reference

See the full API documentation at `/docs` endpoint for detailed parameter descriptions and response schemas.
