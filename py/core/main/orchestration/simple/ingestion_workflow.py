#py/core/main/orchestration/simple/ingestion_workflow.py
import logging
from uuid import UUID

from fastapi import HTTPException
from litellm import AuthenticationError

from core.base import (
    DocumentChunk,
    GraphExtractionStatus,
    R2RException,
)
from core.utils import (
    generate_default_user_collection_id,
    generate_extraction_id,
    num_tokens,
)

from ...services import IngestionService
from ..hatchet.ingestion_workflow import (  # type: ignore[attr-defined]
    _assign_doc_to_collection,
    _maybe_enrich_chunks,
    should_skip_graph_extraction,
)

logger = logging.getLogger(__name__)


def simple_ingestion_factory(service: IngestionService):
    async def ingest_files(input_data):
        document_info = None
        try:
            from core.base import IngestionStatus
            from core.main import IngestionServiceAdapter

            parsed_data = IngestionServiceAdapter.parse_ingest_file_input(
                input_data
            )

            document_info = service.create_document_info_from_file(
                parsed_data["document_id"],
                parsed_data["user"],
                parsed_data["file_data"]["filename"],
                parsed_data["metadata"],
                parsed_data["version"],
                parsed_data["size_in_bytes"],
            )

            await service.update_document_status(
                document_info, status=IngestionStatus.PARSING
            )

            ingestion_config = parsed_data["ingestion_config"]
            extractions_generator = service.parse_file(
                document_info=document_info,
                ingestion_config=ingestion_config,
            )
            extractions = [
                extraction.model_dump()
                async for extraction in extractions_generator
            ]

            # 2) Sum tokens
            total_tokens = 0
            for chunk_dict in extractions:
                text_data = chunk_dict["data"]
                if not isinstance(text_data, str):
                    text_data = text_data.decode("utf-8", errors="ignore")
                total_tokens += num_tokens(text_data)
            document_info.total_tokens = total_tokens

            if not ingestion_config.get("skip_document_summary", False):
                await service.update_document_status(
                    document_info=document_info,
                    status=IngestionStatus.AUGMENTING,
                )
                await service.augment_document_info(document_info, extractions)

            await service.update_document_status(
                document_info, status=IngestionStatus.EMBEDDING
            )
            embedding_generator = service.embed_document(extractions)
            embeddings = [
                embedding.model_dump()
                async for embedding in embedding_generator
            ]

            await service.update_document_status(
                document_info, status=IngestionStatus.STORING
            )
            storage_generator = service.store_embeddings(embeddings)
            async for _ in storage_generator:
                pass

            await service.finalize_ingestion(document_info)

            await service.update_document_status(
                document_info, status=IngestionStatus.SUCCESS
            )

            collection_ids = parsed_data.get("collection_ids")

            try:
                if not collection_ids:
                    collection_id = generate_default_user_collection_id(
                        document_info.owner_id
                    )
                    await _assign_doc_to_collection(
                        service, document_info, collection_id,
                    )
                else:
                    for collection_id in collection_ids:
                        try:
                            name = document_info.title or "N/A"
                            await service.providers.database.collections_handler.create_collection(
                                owner_id=document_info.owner_id,
                                name=name, description="",
                                collection_id=collection_id,
                            )
                            await service.providers.database.graphs_handler.create(
                                collection_id=collection_id,
                                name=name, description="",
                                graph_id=collection_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not create collection: %s", e,
                            )
                        await _assign_doc_to_collection(
                            service, document_info, collection_id,
                        )
            except Exception as e:
                logger.error(
                    "Error assigning document to collection: %s", e,
                )

            # Chunk enrichment
            await _maybe_enrich_chunks(
                service, document_info, ingestion_config,
            )

            # Automatic extraction
            if service.providers.ingestion.config.automatic_extraction:
                doc_type = document_info.document_type.value
                skip_types = (
                    service.providers.ingestion.config.skip_graph_extraction_for_types
                )
                skip_graph = should_skip_graph_extraction(
                    doc_type=doc_type,
                    skip_types=skip_types,
                    ingestion_config=ingestion_config,
                    document_id=str(document_info.id),
                )

                if not skip_graph:
                    logger.warning(
                        "Automatic extraction not yet implemented for `simple` "
                        "ingestion workflows."
                    )
                else:
                    # Mark extraction as complete when skipping
                    await service.providers.database.documents_handler.set_workflow_status(
                        id=document_info.id,
                        status_type="extraction_status",
                        status=GraphExtractionStatus.SUCCESS,
                    )
                    logger.info(
                        "Graph extraction skipped for document %s",
                        document_info.id,
                    )

        except AuthenticationError as e:
            if document_info is not None:
                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.FAILED,
                    metadata={"failure": str(e)},
                )
            raise R2RException(
                status_code=401,
                message="Authentication error: Invalid API key or credentials.",
            ) from e
        except Exception as e:
            if document_info is not None:
                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.FAILED,
                    metadata={"failure": str(e)},
                )
            if isinstance(e, R2RException):
                raise
            raise HTTPException(
                status_code=500, detail=f"Error during ingestion: {str(e)}"
            ) from e

    async def ingest_chunks(input_data):
        document_info = None
        try:
            from core.base import IngestionStatus
            from core.main import IngestionServiceAdapter

            parsed_data = IngestionServiceAdapter.parse_ingest_chunks_input(
                input_data
            )

            document_info = await service.ingest_chunks_ingress(**parsed_data)

            await service.update_document_status(
                document_info, status=IngestionStatus.EMBEDDING
            )
            document_id = document_info.id

            collection_ids = parsed_data.get("collection_ids") or []
            if isinstance(collection_ids, str):
                collection_ids = [collection_ids]
            collection_ids = [UUID(id_str) for id_str in collection_ids]

            extractions = [
                DocumentChunk(
                    id=(
                        generate_extraction_id(document_id, i)
                        if chunk.id is None
                        else chunk.id
                    ),
                    document_id=document_id,
                    collection_ids=collection_ids,
                    owner_id=document_info.owner_id,
                    data=chunk.text,
                    metadata=parsed_data["metadata"],
                ).model_dump()
                for i, chunk in enumerate(parsed_data["chunks"])
            ]

            embedding_generator = service.embed_document(extractions)
            embeddings = [
                embedding.model_dump()
                async for embedding in embedding_generator
            ]

            await service.update_document_status(
                document_info, status=IngestionStatus.STORING
            )
            storage_generator = service.store_embeddings(embeddings)
            async for _ in storage_generator:
                pass

            await service.finalize_ingestion(document_info)

            await service.update_document_status(
                document_info, status=IngestionStatus.SUCCESS
            )

            try:
                if not collection_ids:
                    collection_id = generate_default_user_collection_id(
                        document_info.owner_id
                    )
                    await _assign_doc_to_collection(
                        service, document_info, collection_id,
                    )
                else:
                    for collection_id in collection_ids:
                        try:
                            name = document_info.title or "N/A"
                            await service.providers.database.collections_handler.create_collection(
                                owner_id=document_info.owner_id,
                                name=name, description="",
                                collection_id=collection_id,
                            )
                            await service.providers.database.graphs_handler.create(
                                collection_id=collection_id,
                                name=name, description="",
                                graph_id=collection_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not create collection: %s", e,
                            )
                        await _assign_doc_to_collection(
                            service, document_info, collection_id,
                        )

            except R2RException:
                raise
            except Exception as e:
                logger.error(
                    "Error assigning document to collection: %s", e,
                )

            if service.providers.ingestion.config.automatic_extraction:
                raise R2RException(
                    status_code=501,
                    message="Automatic extraction not yet implemented "
                    "for `simple` ingestion workflows.",
                )

        except Exception as e:
            if document_info is not None:
                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.FAILED,
                    metadata={"failure": str(e)},
                )
            if isinstance(e, R2RException):
                raise
            raise HTTPException(
                status_code=500,
                detail=f"Error during chunk ingestion: {str(e)}",
            ) from e

    async def create_chunks(input_data):
        from core.main import IngestionServiceAdapter

        try:
            parsed_data = IngestionServiceAdapter.parse_create_chunks_input(
                input_data
            )
            document_uuid = (
                UUID(parsed_data["document_id"])
                if isinstance(parsed_data["document_id"], str)
                else parsed_data["document_id"]
            )

            created_chunks = await service.create_chunks_ingress(
                document_id=document_uuid,
                chunks=parsed_data["chunks"],
                user=parsed_data["user"],
            )

            return created_chunks

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during chunk creation: {str(e)}",
            ) from e

    async def update_chunk(input_data):
        from core.main import IngestionServiceAdapter

        try:
            parsed_data = IngestionServiceAdapter.parse_update_chunk_input(
                input_data
            )
            document_uuid = (
                UUID(parsed_data["document_id"])
                if isinstance(parsed_data["document_id"], str)
                else parsed_data["document_id"]
            )
            extraction_uuid = (
                UUID(parsed_data["id"])
                if isinstance(parsed_data["id"], str)
                else parsed_data["id"]
            )

            await service.update_chunk_ingress(
                document_id=document_uuid,
                chunk_id=extraction_uuid,
                text=parsed_data.get("text"),
                user=parsed_data["user"],
                metadata=parsed_data.get("metadata"),
                collection_ids=parsed_data.get("collection_ids"),
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during chunk update: {str(e)}",
            ) from e

    async def create_vector_index(input_data):
        try:
            from core.main import IngestionServiceAdapter

            parsed_data = (
                IngestionServiceAdapter.parse_create_vector_index_input(
                    input_data
                )
            )

            await service.providers.database.chunks_handler.create_index(
                **parsed_data
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during vector index creation: {str(e)}",
            ) from e

    async def delete_vector_index(input_data):
        try:
            from core.main import IngestionServiceAdapter

            parsed_data = (
                IngestionServiceAdapter.parse_delete_vector_index_input(
                    input_data
                )
            )

            await service.providers.database.chunks_handler.delete_index(
                **parsed_data
            )

            return {"status": "Vector index deleted successfully."}

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during vector index deletion: {str(e)}",
            ) from e

    return {
        "ingest-files": ingest_files,
        "ingest-chunks": ingest_chunks,
        "create-chunks": create_chunks,
        "update-chunk": update_chunk,
        "create-vector-index": create_vector_index,
        "delete-vector-index": delete_vector_index,
    }
