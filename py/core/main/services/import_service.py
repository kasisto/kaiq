"""
Import service for importing documents with knowledge graphs.
"""

import io
import json
import logging
import zipfile
from datetime import datetime
from typing import Optional
from uuid import UUID

import numpy as np

from core.base import (
    DocumentResponse,
    DocumentType,
    GraphExtractionStatus,
    IngestionStatus,
    R2RException,
)
from core.base.api.models import User
from shared.abstractions import (
    ConflictResolution,
    ExportManifest,
    ImportConfig,
    ImportError,
    ImportMode,
    ImportResult,
    StoreType,
)
from shared.abstractions.vector import Vector, VectorEntry

from ..abstractions import R2RProviders
from ..config import R2RConfig

logger = logging.getLogger(__name__)


class ImportService:
    """Service for importing documents from export packages."""

    def __init__(
        self,
        config: R2RConfig,
        providers: R2RProviders,
    ) -> None:
        self.config = config
        self.providers = providers

    async def import_documents_from_export(
        self,
        zip_file: io.BytesIO,
        import_config: ImportConfig,
        user: User,
    ) -> ImportResult:
        """
        Import documents from export ZIP without reingestion.

        Args:
            zip_file: ZIP file containing exported documents
            import_config: Import configuration
            user: User performing the import

        Returns:
            ImportResult with success/failure counts and error details
        """
        logger.info(f"Starting import for user {user.id}")

        result = ImportResult(total_documents=0)

        try:
            # Open and validate ZIP
            with zipfile.ZipFile(zip_file, "r") as zf:
                # Parse and validate manifest
                manifest = await self._parse_and_validate_manifest(
                    zf, import_config
                )
                result.total_documents = manifest.total_documents

                # Filter documents if selective import
                documents_to_import = manifest.documents
                if (
                    import_config.mode == ImportMode.SELECTIVE
                    and import_config.document_ids_filter
                ):
                    documents_to_import = [
                        d
                        for d in manifest.documents
                        if d.id in import_config.document_ids_filter
                    ]
                    result.total_documents = len(documents_to_import)

                # Check for conflicts and get set of existing IDs to skip
                existing_ids_to_skip = await self._check_conflicts(
                    documents_to_import, import_config.conflict_resolution
                )

                # Update skipped count and track skipped IDs
                result.skipped_count = len(existing_ids_to_skip)
                result.skipped_document_ids = list(existing_ids_to_skip)

                # Create or validate collections
                if import_config.target_collection_id:
                    # All documents go to target collection
                    target_collection_id = import_config.target_collection_id
                else:
                    # Create collections from manifest
                    for coll_info in manifest.collections:
                        try:
                            await self._create_or_get_collection(
                                coll_info, user.id
                            )
                            result.collections_created.append(coll_info.id)
                        except Exception as e:
                            logger.warning(
                                f"Could not create collection {coll_info.id}: {e}"
                            )

                # Import each document (skip existing ones if in SKIP mode)
                for doc_info in documents_to_import:
                    # Skip documents that already exist (SKIP mode)
                    if doc_info.id in existing_ids_to_skip:
                        logger.debug(f"Skipping existing document {doc_info.id}")
                        continue
                    try:
                        await self._import_single_document(
                            zf=zf,
                            doc_info=doc_info,
                            manifest=manifest,
                            import_config=import_config,
                            user=user,
                            target_collection_id=import_config.target_collection_id,
                        )
                        result.success_count += 1
                        result.imported_document_ids.append(doc_info.id)
                    except Exception as e:
                        logger.error(
                            f"Failed to import document {doc_info.id}: {str(e)}"
                        )
                        result.failed_count += 1
                        result.errors.append(
                            ImportError(
                                document_id=doc_info.id,
                                document_title=doc_info.title,
                                error_type=type(e).__name__,
                                error_message=str(e),
                            )
                        )

                result.regenerated_embeddings = (
                    import_config.regenerate_embeddings
                )

        except R2RException:
            # Re-raise R2RException with its original status code (e.g., 409 for conflicts)
            raise
        except Exception as e:
            logger.error(f"Import failed: {str(e)}")
            raise R2RException(
                status_code=500, message=f"Import failed: {str(e)}"
            )

        logger.info(
            f"Import completed: {result.success_count} succeeded, "
            f"{result.failed_count} failed, {result.skipped_count} skipped"
        )

        return result

    async def _parse_and_validate_manifest(
        self, zf: zipfile.ZipFile, import_config: ImportConfig
    ) -> ExportManifest:
        """Parse and validate the manifest file."""
        try:
            manifest_data = zf.read("manifest.json")
            manifest_dict = json.loads(manifest_data)
            manifest = ExportManifest(**manifest_dict)
        except KeyError:
            raise R2RException(
                status_code=400, message="Invalid export: manifest.json not found"
            )
        except json.JSONDecodeError:
            raise R2RException(
                status_code=400, message="Invalid export: corrupt manifest.json"
            )

        # Validate schema version
        if import_config.validate_schema:
            try:
                schema_version = zf.read("schema_version.txt").decode("utf-8").strip()
                if schema_version != "3.0":
                    logger.warning(
                        f"Schema version mismatch: export={schema_version}, expected=3.0"
                    )
            except KeyError:
                logger.warning("schema_version.txt not found in export")

        # Validate embedding dimensions if embeddings are included
        if manifest.includes_embeddings and not import_config.regenerate_embeddings:
            if manifest.embedding_dimensions != self.config.embedding.base_dimension:
                raise R2RException(
                    status_code=400,
                    message=f"Embedding dimension mismatch: "
                    f"export={manifest.embedding_dimensions}, "
                    f"system={self.config.embedding.base_dimension}. "
                    f"Use regenerate_embeddings=True to regenerate.",
                )

        logger.info(
            f"Validated manifest: {manifest.total_documents} documents, "
            f"version {manifest.export_version}"
        )

        return manifest

    async def _check_conflicts(
        self,
        documents: list,
        conflict_resolution: ConflictResolution,
    ) -> set[UUID]:
        """Check for document ID conflicts and return set of existing IDs to skip."""
        document_ids = [d.id for d in documents]

        existing_docs = await self.providers.database.documents_handler.get_documents_overview(
            offset=0, limit=len(document_ids), filter_document_ids=document_ids
        )

        existing_ids = set()
        if existing_docs["results"]:
            existing_ids = {doc.id for doc in existing_docs["results"]}

            if conflict_resolution == ConflictResolution.ERROR:
                raise R2RException(
                    status_code=409,
                    message=f"Document conflicts found: {list(existing_ids)}",
                )
            elif conflict_resolution == ConflictResolution.SKIP:
                logger.info(
                    f"Will skip {len(existing_ids)} existing documents"
                )
            elif conflict_resolution == ConflictResolution.REPLACE:
                logger.info(
                    f"Will replace {len(existing_ids)} existing documents"
                )
                # Delete existing documents
                for doc_id in existing_ids:
                    await self.providers.database.documents_handler.delete(
                        doc_id
                    )
                # After replacement, don't skip these IDs
                existing_ids = set()

        return existing_ids

    async def _create_or_get_collection(
        self, coll_info, owner_id: UUID
    ) -> UUID:
        """Create or get existing collection."""
        try:
            # Try to get existing collection by name
            collections = await self.providers.database.collections_handler.get_collections_overview(
                offset=0, limit=1, filter_names=[coll_info.name]
            )

            if collections["results"]:
                return collections["results"][0].id

            # Create new collection
            new_collection = (
                await self.providers.database.collections_handler.create(
                    owner_id=owner_id,
                    name=coll_info.name,
                    description=coll_info.description,
                )
            )
            return new_collection.id

        except Exception as e:
            logger.error(f"Error creating/getting collection: {e}")
            raise

    async def _import_single_document(
        self,
        zf: zipfile.ZipFile,
        doc_info,
        manifest: ExportManifest,
        import_config: ImportConfig,
        user: User,
        target_collection_id: Optional[UUID],
    ) -> None:
        """Import a single document with all its data."""
        logger.debug(f"Importing document {doc_info.id}")

        doc_dir = f"documents/{doc_info.id}"

        # Read document metadata
        metadata_path = f"{doc_dir}/metadata.json"
        try:
            metadata_json = zf.read(metadata_path)
            doc_metadata = json.loads(metadata_json)
        except KeyError:
            raise R2RException(
                status_code=400,
                message=f"Missing metadata for document {doc_info.id}",
            )

        # Override collection if target specified
        collection_ids = (
            [target_collection_id]
            if target_collection_id
            else [UUID(cid) for cid in doc_metadata["collection_ids"]]
        )

        # Create document entry
        doc_response = DocumentResponse(
            id=UUID(doc_metadata["id"]),
            collection_ids=collection_ids,
            owner_id=user.id,  # Use importing user as owner
            document_type=doc_metadata["type"],
            metadata=doc_metadata.get("metadata", {}),
            title=doc_metadata.get("title", "Untitled"),
            summary=doc_metadata.get("summary"),
            version=doc_metadata.get("version", "v0"),
            size_in_bytes=doc_metadata.get("size_in_bytes", 0),
            ingestion_status=IngestionStatus.SUCCESS,
            extraction_status=GraphExtractionStatus(
                doc_metadata.get("extraction_status", "pending")
            ),
            created_at=datetime.fromisoformat(
                doc_metadata["created_at"].replace("Z", "+00:00")
            )
            if doc_metadata.get("created_at")
            else datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_tokens=doc_metadata.get("total_tokens", 0),
        )

        # Store document metadata
        await self.providers.database.documents_handler.upsert_documents_overview(
            doc_response
        )

        # Import original file
        file_paths = [
            name for name in zf.namelist() if name.startswith(f"{doc_dir}/file")
        ]
        if file_paths:
            file_path = file_paths[0]
            file_content = zf.read(file_path)
            file_name = doc_metadata.get("title", "file") + file_path.split("file")[1]

            await self.providers.file.store_file(
                document_id=doc_info.id,
                file_name=file_name,
                file_content=io.BytesIO(file_content),
                file_type=doc_metadata["type"],
            )

        # Import chunks
        chunks_path = f"{doc_dir}/chunks.jsonl"
        chunk_embeddings = None

        try:
            chunks_data = zf.read(chunks_path).decode("utf-8")
            chunks = [json.loads(line) for line in chunks_data.strip().split("\n")]

            # Load chunk embeddings if present
            if (
                not import_config.regenerate_embeddings
                and manifest.includes_embeddings
            ):
                embeddings_path = f"{doc_dir}/embeddings/chunks.npy"
                try:
                    emb_data = zf.read(embeddings_path)
                    chunk_embeddings = np.load(io.BytesIO(emb_data))
                except KeyError:
                    logger.warning(
                        f"Chunk embeddings not found for document {doc_info.id}"
                    )

            # Insert chunks with embeddings
            await self._insert_chunks(
                chunks=chunks,
                embeddings=chunk_embeddings,
                collection_ids=collection_ids,
                regenerate=import_config.regenerate_embeddings,
            )

        except KeyError:
            logger.warning(f"No chunks found for document {doc_info.id}")

        # Import summary embedding
        if (
            not import_config.regenerate_embeddings
            and manifest.includes_embeddings
        ):
            summary_emb_path = f"{doc_dir}/embeddings/summary.npy"
            try:
                summary_emb_data = zf.read(summary_emb_path)
                summary_embedding = np.load(io.BytesIO(summary_emb_data)).tolist()

                # Update document with summary embedding
                await self.providers.database.documents_handler.upsert_documents_overview(
                    DocumentResponse(
                        id=doc_info.id,
                        collection_ids=collection_ids,
                        owner_id=user.id,
                        document_type=doc_metadata["type"],
                        metadata=doc_metadata.get("metadata", {}),
                        title=doc_metadata.get("title", "Untitled"),
                        summary=doc_metadata.get("summary"),
                        summary_embedding=summary_embedding,
                        version=doc_metadata.get("version", "v0"),
                        size_in_bytes=doc_metadata.get("size_in_bytes", 0),
                        ingestion_status=IngestionStatus.SUCCESS,
                        extraction_status=GraphExtractionStatus(
                            doc_metadata.get("extraction_status", "pending")
                        ),
                        created_at=doc_response.created_at,
                        updated_at=datetime.utcnow(),
                        total_tokens=doc_metadata.get("total_tokens", 0),
                    )
                )
            except KeyError:
                logger.debug(f"No summary embedding for document {doc_info.id}")

        # Import knowledge graph
        if manifest.includes_knowledge_graphs and doc_info.has_knowledge_graph:
            await self._import_knowledge_graph(
                zf=zf,
                doc_dir=doc_dir,
                doc_id=doc_info.id,
                import_config=import_config,
            )

        logger.debug(f"Successfully imported document {doc_info.id}")

    async def _insert_chunks(
        self,
        chunks: list[dict],
        embeddings: Optional[np.ndarray],
        collection_ids: list[UUID],
        regenerate: bool,
    ) -> None:
        """Insert chunks into database with embeddings."""
        # Get embedding dimension from config
        embedding_dim = self.config.embedding.base_dimension

        for idx, chunk_data in enumerate(chunks):
            embedding = None
            if not regenerate and embeddings is not None:
                embedding = embeddings[idx].tolist()
            elif regenerate or embeddings is None:
                # Create placeholder embedding with zeros
                embedding = [0.0] * embedding_dim

            # Create vector object
            vector = Vector(
                data=embedding,
                length=len(embedding),
            )

            # Create chunk entry
            entry = VectorEntry(
                id=UUID(chunk_data["id"]),
                document_id=UUID(chunk_data["document_id"]),
                owner_id=UUID(chunk_data["owner_id"]),
                collection_ids=collection_ids,
                text=chunk_data["text"],
                metadata=chunk_data.get("metadata", {}),
                vector=vector,
            )

            await self.providers.database.chunks_handler.upsert(entry)

        # If regenerating, trigger embedding generation
        if regenerate:
            logger.info(
                f"Regenerating embeddings for {len(chunks)} chunks"
            )
            # Note: This would need to be implemented to queue embedding jobs
            # For now, just log the requirement

    async def _import_knowledge_graph(
        self,
        zf: zipfile.ZipFile,
        doc_dir: str,
        doc_id: UUID,
        import_config: ImportConfig,
    ) -> None:
        """Import knowledge graph entities and relationships."""
        kg_dir = f"{doc_dir}/knowledge_graph"

        # Import entities
        entities_path = f"{kg_dir}/entities.jsonl"
        try:
            entities_data = zf.read(entities_path).decode("utf-8")
            entities = [
                json.loads(line) for line in entities_data.strip().split("\n")
            ]

            # Load entity embeddings if present
            entity_embeddings = None
            if not import_config.regenerate_embeddings:
                try:
                    entity_emb_path = f"{kg_dir}/embeddings/entities.npy"
                    entity_emb_data = zf.read(entity_emb_path)
                    entity_embeddings = np.load(io.BytesIO(entity_emb_data))
                except KeyError:
                    logger.debug(f"No entity embeddings for document {doc_id}")

            # Insert entities
            for idx, entity_data in enumerate(entities):
                embedding = None
                if entity_embeddings is not None:
                    embedding = entity_embeddings[idx].tolist()

                await self.providers.database.graphs_handler.entities.create(
                    name=entity_data["name"],
                    category=entity_data.get("category"),
                    description=entity_data.get("description"),
                    description_embedding=embedding,
                    parent_id=doc_id,
                    store_type=StoreType.DOCUMENTS,
                    chunk_ids=[UUID(cid) for cid in entity_data.get("chunk_ids", [])],
                    metadata=entity_data.get("metadata", {}),
                )

        except KeyError:
            logger.debug(f"No entities found for document {doc_id}")

        # Import relationships
        relationships_path = f"{kg_dir}/relationships.jsonl"
        try:
            relationships_data = zf.read(relationships_path).decode("utf-8")
            relationships = [
                json.loads(line)
                for line in relationships_data.strip().split("\n")
            ]

            # Load relationship embeddings if present
            relationship_embeddings = None
            if not import_config.regenerate_embeddings:
                try:
                    rel_emb_path = f"{kg_dir}/embeddings/relationships.npy"
                    rel_emb_data = zf.read(rel_emb_path)
                    relationship_embeddings = np.load(io.BytesIO(rel_emb_data))
                except KeyError:
                    logger.debug(
                        f"No relationship embeddings for document {doc_id}"
                    )

            # Insert relationships
            for idx, rel_data in enumerate(relationships):
                embedding = None
                if relationship_embeddings is not None:
                    embedding = relationship_embeddings[idx].tolist()

                subject_id = UUID(rel_data["subject_id"]) if rel_data.get("subject_id") else None
                object_id = UUID(rel_data["object_id"]) if rel_data.get("object_id") else None

                await self.providers.database.graphs_handler.relationships.create(
                    subject=rel_data["subject"],
                    subject_id=subject_id,
                    predicate=rel_data["predicate"],
                    object=rel_data["object"],
                    object_id=object_id,
                    description=rel_data.get("description"),
                    description_embedding=embedding,
                    parent_id=doc_id,
                    store_type=StoreType.DOCUMENTS,
                    weight=rel_data.get("weight", 1.0),
                    chunk_ids=[UUID(cid) for cid in rel_data.get("chunk_ids", [])],
                    metadata=rel_data.get("metadata", {}),
                )

        except KeyError:
            logger.debug(f"No relationships found for document {doc_id}")
