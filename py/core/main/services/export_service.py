"""
Export service for exporting documents with knowledge graphs.
"""

import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

from core.base import R2RException
from shared.abstractions import (
    ExportCollectionInfo,
    ExportConfig,
    ExportDocumentInfo,
    ExportManifest,
    StoreType,
)

from ..abstractions import R2RProviders
from ..config import R2RConfig

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting documents with all associated data."""

    def __init__(
        self,
        config: R2RConfig,
        providers: R2RProviders,
    ) -> None:
        self.config = config
        self.providers = providers

    async def export_documents_with_graphs(
        self,
        export_config: ExportConfig,
        user_id: UUID,
    ) -> tuple[str, io.BytesIO, int]:
        """
        Export documents with all associated data in a ZIP format.

        Args:
            export_config: Export configuration
            user_id: User performing the export

        Returns:
            tuple of (zip_filename, zip_file_io, size_bytes)
        """
        logger.info(
            f"Starting export of {len(export_config.document_ids)} documents for user {user_id}"
        )

        # Validate documents exist and user has access
        await self._validate_documents_access(
            export_config.document_ids, user_id
        )

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            # Collect manifest data
            manifest_documents = []
            manifest_collections = {}

            for doc_id in export_config.document_ids:
                try:
                    doc_info = await self._export_single_document(
                        zip_file=zip_file,
                        document_id=doc_id,
                        include_embeddings=export_config.include_embeddings,
                        include_knowledge_graphs=export_config.include_knowledge_graphs,
                        include_collections=export_config.include_collections,
                    )
                    manifest_documents.append(doc_info)

                    # Collect collection info
                    if export_config.include_collections:
                        for coll_id in doc_info.collections:
                            if coll_id not in manifest_collections:
                                coll_info = await self._get_collection_info(
                                    coll_id
                                )
                                if coll_info:
                                    manifest_collections[coll_id] = coll_info

                except Exception as e:
                    logger.error(
                        f"Failed to export document {doc_id}: {str(e)}"
                    )
                    raise R2RException(
                        status_code=500,
                        message=f"Failed to export document {doc_id}: {str(e)}",
                    )

            # Create manifest
            manifest = ExportManifest(
                export_version=export_config.export_format_version,
                export_timestamp=datetime.utcnow(),
                total_documents=len(manifest_documents),
                documents=manifest_documents,
                collections=list(manifest_collections.values()),
                embedding_model=self.config.embedding.provider,
                embedding_dimensions=self.config.embedding.base_dimension,
                includes_embeddings=export_config.include_embeddings,
                includes_knowledge_graphs=export_config.include_knowledge_graphs,
            )

            # Write manifest
            zip_file.writestr(
                "manifest.json",
                json.dumps(manifest.model_dump(), indent=2, default=str),
            )

            # Write schema version
            zip_file.writestr("schema_version.txt", "3.0")

        # Get size and prepare for return
        zip_buffer.seek(0)
        size = len(zip_buffer.getvalue())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"documents_export_{timestamp}.zip"

        logger.info(
            f"Export completed: {filename} ({size} bytes, {len(manifest_documents)} documents)"
        )

        return filename, zip_buffer, size

    async def _validate_documents_access(
        self, document_ids: list[UUID], user_id: UUID
    ) -> None:
        """Validate that all documents exist and user has access."""
        docs = await self.providers.database.documents_handler.get_documents_overview(
            offset=0,
            limit=len(document_ids),
            filter_document_ids=document_ids,
        )

        if len(docs["results"]) != len(document_ids):
            found_ids = {doc.id for doc in docs["results"]}
            missing_ids = set(document_ids) - found_ids
            raise R2RException(
                status_code=404,
                message=f"Documents not found: {missing_ids}",
            )

        # Check user has access to all documents
        for doc in docs["results"]:
            if doc.owner_id != user_id:
                raise R2RException(
                    status_code=403,
                    message=f"Access denied to document {doc.id}",
                )

    async def _export_single_document(
        self,
        zip_file: zipfile.ZipFile,
        document_id: UUID,
        include_embeddings: bool,
        include_knowledge_graphs: bool,
        include_collections: bool,
    ) -> ExportDocumentInfo:
        """Export a single document with all its data."""
        logger.debug(f"Exporting document {document_id}")

        # Get document metadata
        doc = await self.providers.database.documents_handler.get_documents_overview(
            offset=0, limit=1, filter_document_ids=[document_id]
        )
        if not doc["results"]:
            raise R2RException(
                status_code=404, message=f"Document {document_id} not found"
            )

        doc_info = doc["results"][0]
        doc_dir = f"documents/{document_id}"

        # Export document metadata
        metadata = {
            "id": str(doc_info.id),
            "collection_ids": [str(cid) for cid in doc_info.collection_ids],
            "owner_id": str(doc_info.owner_id),
            "type": doc_info.document_type,
            "metadata": doc_info.metadata,
            "title": doc_info.title,
            "summary": doc_info.summary,
            "version": doc_info.version,
            "size_in_bytes": doc_info.size_in_bytes,
            "ingestion_status": doc_info.ingestion_status,
            "extraction_status": doc_info.extraction_status,
            "created_at": str(doc_info.created_at),
            "updated_at": str(doc_info.updated_at),
            "total_tokens": doc_info.total_tokens,
        }
        zip_file.writestr(
            f"{doc_dir}/metadata.json",
            json.dumps(metadata, indent=2, default=str),
        )

        # Export original file
        try:
            file_name, file_content, _ = (
                await self.providers.file.retrieve_file(document_id)
            )
            file_ext = Path(file_name).suffix or ".bin"
            zip_file.writestr(
                f"{doc_dir}/file{file_ext}", file_content.read()
            )
        except Exception as e:
            logger.warning(
                f"Could not export file for document {document_id}: {e}"
            )

        # Export chunks
        chunks_data = []
        chunk_count = 0
        chunk_embeddings = []

        chunks = await self.providers.database.chunks_handler.list_document_chunks(
            document_id=document_id, offset=0, limit=10000, include_vectors=include_embeddings
        )

        for chunk in chunks["results"]:
            chunk_obj = {
                "id": str(chunk["id"]),
                "document_id": str(chunk["document_id"]),
                "owner_id": str(chunk["owner_id"]),
                "collection_ids": [str(cid) for cid in chunk.get("collection_ids", [])],
                "text": chunk.get("text"),
                "metadata": chunk.get("metadata", {}),
            }
            chunks_data.append(chunk_obj)
            chunk_count += 1

            if include_embeddings and chunk.get("vector"):
                chunk_embeddings.append(chunk["vector"])

        # Write chunks as JSONL
        if chunks_data:
            chunks_jsonl = "\n".join(
                [json.dumps(c, default=str) for c in chunks_data]
            )
            zip_file.writestr(f"{doc_dir}/chunks.jsonl", chunks_jsonl)

        # Write embeddings
        if include_embeddings:
            embeddings_dir = f"{doc_dir}/embeddings"

            # Summary embedding
            if doc_info.summary_embedding:
                summary_emb_bytes = io.BytesIO()
                np.save(summary_emb_bytes, np.array(doc_info.summary_embedding))
                zip_file.writestr(
                    f"{embeddings_dir}/summary.npy",
                    summary_emb_bytes.getvalue(),
                )

            # Chunk embeddings
            if chunk_embeddings:
                chunk_emb_bytes = io.BytesIO()
                np.save(chunk_emb_bytes, np.array(chunk_embeddings))
                zip_file.writestr(
                    f"{embeddings_dir}/chunks.npy", chunk_emb_bytes.getvalue()
                )

        # Export knowledge graph
        entity_count = 0
        relationship_count = 0

        if include_knowledge_graphs:
            # Export entities
            entities_data = []
            entity_embeddings = []

            try:
                entities_list, _ = await self.providers.database.graphs_handler.entities.get(
                    parent_id=document_id, store_type=StoreType.DOCUMENTS, offset=0, limit=10000, include_embeddings=include_embeddings
                )

                for entity in entities_list:
                    entity_obj = {
                        "id": str(entity.id),
                        "name": entity.name,
                        "category": entity.category,
                        "description": entity.description,
                        "chunk_ids": [str(cid) for cid in (entity.chunk_ids or [])],
                        "metadata": entity.metadata or {},
                    }
                    entities_data.append(entity_obj)
                    entity_count += 1

                    if include_embeddings and hasattr(entity, 'description_embedding') and entity.description_embedding:
                        entity_embeddings.append(entity.description_embedding)

                if entities_data:
                    kg_dir = f"{doc_dir}/knowledge_graph"
                    entities_jsonl = "\n".join(
                        [json.dumps(e, default=str) for e in entities_data]
                    )
                    zip_file.writestr(f"{kg_dir}/entities.jsonl", entities_jsonl)

                    if include_embeddings and entity_embeddings:
                        entity_emb_bytes = io.BytesIO()
                        np.save(entity_emb_bytes, np.array(entity_embeddings))
                        zip_file.writestr(
                            f"{kg_dir}/embeddings/entities.npy",
                            entity_emb_bytes.getvalue(),
                        )
            except Exception as e:
                logger.warning(
                    f"Could not export entities for document {document_id}: {e}"
                )

            # Export relationships
            relationships_data = []
            relationship_embeddings = []

            try:
                relationships_list, _ = await self.providers.database.graphs_handler.relationships.get(
                    parent_id=document_id, store_type=StoreType.DOCUMENTS, offset=0, limit=10000, include_metadata=True
                )

                for rel in relationships_list:
                    rel_obj = {
                        "id": str(rel.id),
                        "subject": rel.subject,
                        "predicate": rel.predicate,
                        "object": rel.object,
                        "description": rel.description,
                        "subject_id": str(rel.subject_id) if rel.subject_id else None,
                        "object_id": str(rel.object_id) if rel.object_id else None,
                        "weight": rel.weight,
                        "chunk_ids": [str(cid) for cid in (rel.chunk_ids or [])],
                        "metadata": rel.metadata or {},
                    }
                    relationships_data.append(rel_obj)
                    relationship_count += 1

                    if include_embeddings and hasattr(rel, 'description_embedding') and rel.description_embedding:
                        relationship_embeddings.append(rel.description_embedding)

                if relationships_data:
                    kg_dir = f"{doc_dir}/knowledge_graph"
                    relationships_jsonl = "\n".join(
                        [json.dumps(r, default=str) for r in relationships_data]
                    )
                    zip_file.writestr(
                        f"{kg_dir}/relationships.jsonl", relationships_jsonl
                    )

                    if include_embeddings and relationship_embeddings:
                        rel_emb_bytes = io.BytesIO()
                        np.save(rel_emb_bytes, np.array(relationship_embeddings))
                        zip_file.writestr(
                            f"{kg_dir}/embeddings/relationships.npy",
                            rel_emb_bytes.getvalue(),
                        )
            except Exception as e:
                logger.warning(
                    f"Could not export relationships for document {document_id}: {e}"
                )

        # Create document info for manifest
        export_doc_info = ExportDocumentInfo(
            id=doc_info.id,
            title=doc_info.title or "Untitled",
            type=doc_info.document_type,
            size_bytes=doc_info.size_in_bytes or 0,
            has_embeddings=include_embeddings and len(chunk_embeddings) > 0,
            has_knowledge_graph=include_knowledge_graphs
            and (entity_count > 0 or relationship_count > 0),
            chunk_count=chunk_count,
            entity_count=entity_count,
            relationship_count=relationship_count,
            collections=doc_info.collection_ids,
        )

        logger.debug(
            f"Exported document {document_id}: {chunk_count} chunks, "
            f"{entity_count} entities, {relationship_count} relationships"
        )

        return export_doc_info

    async def _get_collection_info(
        self, collection_id: UUID
    ) -> Optional[ExportCollectionInfo]:
        """Get collection information."""
        try:
            collections = await self.providers.database.collections_handler.get_collections_overview(
                offset=0, limit=1, filter_collection_ids=[collection_id]
            )

            if collections["results"]:
                coll = collections["results"][0]
                return ExportCollectionInfo(
                    id=coll.id,
                    name=coll.name,
                    description=coll.description,
                    document_count=coll.document_count or 0,
                )
        except Exception as e:
            logger.warning(
                f"Could not get collection info for {collection_id}: {e}"
            )

        return None
