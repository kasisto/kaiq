"""
Abstractions for document export and import functionality.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import Field

from shared.abstractions.base import R2RSerializable


class ImportMode(str, Enum):
    """Mode for importing documents."""

    FULL = "full"  # Import everything including embeddings
    METADATA_ONLY = "metadata_only"  # Skip embeddings
    SELECTIVE = "selective"  # Use document_ids_filter


class ConflictResolution(str, Enum):
    """Strategy for handling conflicts during import."""

    SKIP = "skip"  # Skip conflicting documents
    REPLACE = "replace"  # Replace existing documents
    ERROR = "error"  # Fail on conflicts


class ExportManifest(R2RSerializable):
    """Manifest file for document export."""

    export_version: str = Field(
        default="1.0", description="Export format version"
    )
    export_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Export creation time"
    )
    export_source: str = Field(
        default="kaiq-r2r", description="Export source system"
    )
    schema_version: str = Field(
        default="3.0", description="Database schema version"
    )
    total_documents: int = Field(
        default=0, description="Total number of documents"
    )
    documents: list["ExportDocumentInfo"] = Field(
        default_factory=list, description="Document metadata list"
    )
    collections: list["ExportCollectionInfo"] = Field(
        default_factory=list, description="Collection metadata list"
    )
    embedding_model: Optional[str] = Field(
        default=None, description="Embedding model used"
    )
    embedding_dimensions: Optional[int] = Field(
        default=None, description="Embedding vector dimensions"
    )
    includes_embeddings: bool = Field(
        default=True, description="Whether embeddings are included"
    )
    includes_knowledge_graphs: bool = Field(
        default=True, description="Whether knowledge graphs are included"
    )


class ExportDocumentInfo(R2RSerializable):
    """Document information in export manifest."""

    id: UUID = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    type: str = Field(..., description="Document type")
    size_bytes: int = Field(..., description="File size in bytes")
    has_embeddings: bool = Field(
        default=False, description="Whether document has embeddings"
    )
    has_knowledge_graph: bool = Field(
        default=False, description="Whether document has knowledge graph"
    )
    chunk_count: int = Field(default=0, description="Number of chunks")
    entity_count: int = Field(default=0, description="Number of entities")
    relationship_count: int = Field(
        default=0, description="Number of relationships"
    )
    collections: list[UUID] = Field(
        default_factory=list, description="Collection IDs"
    )


class ExportCollectionInfo(R2RSerializable):
    """Collection information in export manifest."""

    id: UUID = Field(..., description="Collection ID")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(
        default=None, description="Collection description"
    )
    document_count: int = Field(
        default=0, description="Number of documents in collection"
    )


class ImportError(R2RSerializable):
    """Error information for failed imports."""

    document_id: Optional[UUID] = Field(
        default=None, description="Document ID that failed"
    )
    document_title: Optional[str] = Field(
        default=None, description="Document title that failed"
    )
    error_type: str = Field(..., description="Error type")
    error_message: str = Field(..., description="Error message")


class ImportResult(R2RSerializable):
    """Result of document import operation."""

    success_count: int = Field(
        default=0, description="Number of successfully imported documents"
    )
    skipped_count: int = Field(
        default=0, description="Number of skipped documents"
    )
    failed_count: int = Field(
        default=0, description="Number of failed documents"
    )
    total_documents: int = Field(
        default=0, description="Total documents in import"
    )
    imported_document_ids: list[UUID] = Field(
        default_factory=list, description="Successfully imported document IDs"
    )
    skipped_document_ids: list[UUID] = Field(
        default_factory=list, description="Skipped document IDs"
    )
    errors: list[ImportError] = Field(
        default_factory=list, description="Import errors"
    )
    collections_created: list[UUID] = Field(
        default_factory=list, description="Created collection IDs"
    )
    regenerated_embeddings: bool = Field(
        default=False, description="Whether embeddings were regenerated"
    )


class ExportConfig(R2RSerializable):
    """Configuration for document export."""

    document_ids: list[UUID] = Field(
        ..., description="Document IDs to export"
    )
    include_embeddings: bool = Field(
        default=True, description="Include embeddings in export"
    )
    include_knowledge_graphs: bool = Field(
        default=True, description="Include knowledge graphs in export"
    )
    include_collections: bool = Field(
        default=True, description="Include collection metadata"
    )
    export_format_version: str = Field(
        default="1.0", description="Export format version"
    )


class ImportConfig(R2RSerializable):
    """Configuration for document import."""

    mode: ImportMode = Field(
        default=ImportMode.FULL, description="Import mode"
    )
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.SKIP,
        description="Conflict resolution strategy",
    )
    regenerate_embeddings: bool = Field(
        default=False, description="Force regeneration of embeddings"
    )
    target_collection_id: Optional[UUID] = Field(
        default=None, description="Override collection ID"
    )
    document_ids_filter: Optional[list[UUID]] = Field(
        default=None, description="Only import specific documents"
    )
    validate_schema: bool = Field(
        default=True, description="Validate schema compatibility"
    )
