import asyncio
import json
import logging
import os
import uuid
from datetime import timedelta
from typing import Any, Optional
from uuid import UUID

from fastapi import HTTPException
from hatchet_sdk import (
    ConcurrencyExpression,
    ConcurrencyLimitStrategy,
    Context,
    Hatchet,
)
from litellm import AuthenticationError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.base import (
    DocumentChunk,
    GraphConstructionStatus,
    GraphExtractionStatus,
    IngestionStatus,
    OrchestrationProvider,
    generate_extraction_id,
)
from core.base.abstractions import DocumentResponse, R2RException
from core.utils import (
    generate_default_user_collection_id,
    num_tokens,
    update_settings_from_dict,
)

from ...services import IngestionService, IngestionServiceAdapter

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Pure-logic helper (no Hatchet dependency)
# ---------------------------------------------------------------------------

def extract_user_id(data: dict) -> str:
    """Extract user_id from a JSON user string for CEL concurrency."""
    try:
        user = data.get("user", "")
        if isinstance(user, str) and user:
            user = json.loads(user)
        if isinstance(user, dict):
            return str(user.get("id", str(uuid.uuid4())))
    except Exception:
        pass
    return str(uuid.uuid4())


def should_skip_graph_extraction(
    doc_type: str,
    skip_types: list[str],
    ingestion_config: dict,
    document_id: str,
) -> bool:
    """Determine if graph extraction should be skipped for a document."""
    skip_graph = False

    if doc_type in skip_types:
        skip_graph = True
        logger.info(
            "Auto-skipping graph extraction for %s (document_id=%s)",
            doc_type, document_id,
        )

    if ingestion_config.get("skip_graph_extraction", False):
        skip_graph = True
        logger.info(
            "Skipping graph extraction per user request "
            "(document_id=%s)", document_id,
        )

    return skip_graph


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class IngestFilesInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_data: dict = Field(default_factory=dict)
    document_id: str = ""
    collection_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    ingestion_config: dict = Field(default_factory=dict)
    user: str = ""
    size_in_bytes: int = 0
    version: str = "v0"
    user_id: str = ""  # hoisted for CEL concurrency

    @model_validator(mode="before")
    @classmethod
    def hoist_user_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("user_id"):
            data["user_id"] = extract_user_id(data)
        return data


class IngestChunksInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str = ""
    chunks: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    ingestion_config: dict = Field(default_factory=dict)
    user: str = ""
    collection_ids: list[str] = Field(default_factory=list)


class UpdateChunkInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str = ""
    id: str = ""
    text: Optional[str] = None
    metadata: Optional[dict] = None
    user: str = ""
    collection_ids: Optional[list[str]] = None


class CreateVectorIndexInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    table_name: str = ""
    index_method: str = ""
    index_measure: str = ""
    index_name: str = ""
    index_column: str = ""
    index_arguments: dict = Field(default_factory=dict)
    concurrently: bool = True


class DeleteVectorIndexInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    index_name: str = ""
    table_name: str = ""
    concurrently: bool = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def hatchet_ingestion_factory(
    hatchet: Hatchet,
    service: IngestionService,
    config: Any,
    provider: OrchestrationProvider,
) -> dict[str, Any]:
    """Create v1 Hatchet workflow objects for all ingestion workflows."""

    # ======================================================================
    # ingest-files
    # ======================================================================
    ingest_files_wf = hatchet.workflow(
        name="ingest-files",
        input_validator=IngestFilesInput,
        concurrency=ConcurrencyExpression(
            expression="input.user_id",
            max_runs=config.ingestion_concurrency_limit,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @ingest_files_wf.task(retries=0, execution_timeout=timedelta(minutes=60))
    async def parse(input: IngestFilesInput, ctx: Context) -> dict:
        try:
            logger.info("Initiating ingestion workflow, step: parse")
            input_data = input.model_dump()
            parsed_data = (
                IngestionServiceAdapter.parse_ingest_file_input(input_data)
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
                document_info, status=IngestionStatus.PARSING,
            )

            ingestion_config = parsed_data["ingestion_config"] or {}
            extractions = [
                e async for e in service.parse_file(
                    document_info, ingestion_config
                )
            ]

            # Sum tokens
            total_tokens = 0
            for chunk in extractions:
                text_data = chunk.data
                if not isinstance(text_data, str):
                    text_data = text_data.decode("utf-8", errors="ignore")
                total_tokens += num_tokens(text_data)
            document_info.total_tokens = total_tokens

            extraction_dicts = [e.to_dict() for e in extractions]

            if not ingestion_config.get("skip_document_summary", False):
                await service.update_document_status(
                    document_info, status=IngestionStatus.AUGMENTING,
                )
                await service.augment_document_info(
                    document_info, extraction_dicts,
                )

            await service.update_document_status(
                document_info, status=IngestionStatus.EMBEDDING,
            )

            embeddings = [
                emb async for emb in service.embed_document(
                    extraction_dicts
                )
            ]

            await service.update_document_status(
                document_info, status=IngestionStatus.STORING,
            )

            async for _ in service.store_embeddings(embeddings):
                pass

            await service.finalize_ingestion(document_info)
            await service.update_document_status(
                document_info, status=IngestionStatus.SUCCESS,
            )

            # Assign to collections
            collection_ids = input.collection_ids
            if not collection_ids:
                coll_id = generate_default_user_collection_id(
                    document_info.owner_id
                )
                await _assign_doc_to_collection(
                    service, document_info, coll_id,
                )
            else:
                for cid_str in collection_ids:
                    coll_id = UUID(cid_str)
                    await _assign_doc_to_collection(
                        service, document_info, coll_id,
                    )

            # Chunk enrichment
            await _maybe_enrich_chunks(
                service, document_info, ingestion_config,
            )

            # Graph extraction
            if service.providers.ingestion.config.automatic_extraction:
                await _maybe_extract_graph(
                    service, provider, document_info,
                    ingestion_config, input_data,
                )

            return {
                "status": "Successfully finalized ingestion",
                "document_info": document_info.to_dict(),
            }

        except AuthenticationError:
            raise R2RException(
                status_code=401,
                message="Authentication error: Invalid API key "
                "or credentials.",
            ) from None
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during ingestion: {str(e)}",
            ) from e

    @ingest_files_wf.on_failure_task()
    async def on_failure_ingest_files(
        input: IngestFilesInput, ctx: Context,
    ) -> None:
        await _handle_ingestion_failure(
            service, input.document_id, ctx,
        )

    # ======================================================================
    # ingest-chunks
    # ======================================================================
    ingest_chunks_wf = hatchet.workflow(
        name="ingest-chunks",
        input_validator=IngestChunksInput,
    )

    @ingest_chunks_wf.task(execution_timeout=timedelta(minutes=60))
    async def ingest(input: IngestChunksInput, ctx: Context) -> dict:
        input_data = input.model_dump()
        parsed_data = (
            IngestionServiceAdapter.parse_ingest_chunks_input(input_data)
        )

        document_info = await service.ingest_chunks_ingress(**parsed_data)
        await service.update_document_status(
            document_info, status=IngestionStatus.EMBEDDING,
        )
        document_id = document_info.id

        extractions = [
            DocumentChunk(
                id=generate_extraction_id(document_id, i),
                document_id=document_id,
                collection_ids=[],
                owner_id=document_info.owner_id,
                data=chunk.text,
                metadata=parsed_data["metadata"],
            ).to_dict()
            for i, chunk in enumerate(parsed_data["chunks"])
        ]

        total_tokens = 0
        for chunk in extractions:
            text_data = chunk["data"]
            if not isinstance(text_data, str):
                text_data = text_data.decode("utf-8", errors="ignore")
            total_tokens += num_tokens(text_data)
        document_info.total_tokens = total_tokens

        return {
            "status": "Successfully ingested chunks",
            "extractions": extractions,
            "document_info": document_info.to_dict(),
        }

    @ingest_chunks_wf.task(
        parents=[ingest], execution_timeout=timedelta(minutes=60),
    )
    async def embed(input: IngestChunksInput, ctx: Context) -> dict:
        doc_dict = ctx.task_output(ingest)["document_info"]
        document_info = DocumentResponse(**doc_dict)
        extractions = ctx.task_output(ingest)["extractions"]

        embeddings = [
            emb.model_dump()
            async for emb in service.embed_document(extractions)
        ]

        await service.update_document_status(
            document_info, status=IngestionStatus.STORING,
        )

        async for _ in service.store_embeddings(embeddings):
            pass

        return {
            "status": "Successfully embedded and stored chunks",
            "document_info": document_info.to_dict(),
        }

    @ingest_chunks_wf.task(
        parents=[embed], execution_timeout=timedelta(minutes=60),
    )
    async def finalize(input: IngestChunksInput, ctx: Context) -> dict:
        doc_dict = ctx.task_output(embed)["document_info"]
        document_info = DocumentResponse(**doc_dict)

        await service.finalize_ingestion(document_info)
        await service.update_document_status(
            document_info, status=IngestionStatus.SUCCESS,
        )

        try:
            collection_ids = input.collection_ids
            if not collection_ids:
                coll_id = generate_default_user_collection_id(
                    document_info.owner_id
                )
                await _assign_doc_to_collection(
                    service, document_info, coll_id,
                )
            else:
                for cid_str in collection_ids:
                    coll_id = UUID(cid_str)
                    try:
                        name = document_info.title or "N/A"
                        await service.providers.database.collections_handler.create_collection(
                            owner_id=document_info.owner_id,
                            name=name, description="",
                            collection_id=coll_id,
                        )
                        await service.providers.database.graphs_handler.create(
                            collection_id=coll_id,
                            name=name, description="",
                            graph_id=coll_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not create collection: %s", e,
                        )

                    await _assign_doc_to_collection(
                        service, document_info, coll_id,
                    )
        except Exception as e:
            logger.error(
                "Error assigning document to collection: %s", e,
            )

        # Trigger automatic extraction if enabled
        if service.providers.ingestion.config.automatic_extraction:
            await _maybe_extract_graph(
                service, provider, document_info,
                input.ingestion_config, input.model_dump(),
            )

        return {
            "status": "Successfully finalized ingestion",
            "document_info": document_info.to_dict(),
        }

    @ingest_chunks_wf.on_failure_task()
    async def on_failure_ingest_chunks(
        input: IngestChunksInput, ctx: Context,
    ) -> None:
        await _handle_ingestion_failure(
            service, input.document_id, ctx,
        )

    # ======================================================================
    # update-chunk
    # ======================================================================
    update_chunk_wf = hatchet.workflow(
        name="update-chunk", input_validator=UpdateChunkInput,
    )

    @update_chunk_wf.task(execution_timeout=timedelta(minutes=60))
    async def update_chunk(
        input: UpdateChunkInput, ctx: Context,
    ) -> dict:
        try:
            input_data = input.model_dump()
            parsed_data = (
                IngestionServiceAdapter.parse_update_chunk_input(
                    input_data
                )
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

            return {
                "message": "Chunk update completed successfully.",
                "document_ids": [str(document_uuid)],
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during chunk update: {str(e)}",
            ) from e

    @update_chunk_wf.on_failure_task()
    async def on_failure_update_chunk(
        input: UpdateChunkInput, ctx: Context,
    ) -> None:
        pass

    # ======================================================================
    # create-vector-index
    # ======================================================================
    create_vector_index_wf = hatchet.workflow(
        name="create-vector-index",
        input_validator=CreateVectorIndexInput,
    )

    @create_vector_index_wf.task(
        execution_timeout=timedelta(minutes=60),
    )
    async def create_vector_index(
        input: CreateVectorIndexInput, ctx: Context,
    ) -> dict:
        input_data = input.model_dump()
        parsed_data = (
            IngestionServiceAdapter.parse_create_vector_index_input(
                input_data
            )
        )
        await service.providers.database.chunks_handler.create_index(
            **parsed_data
        )
        return {"status": "Vector index creation queued successfully."}

    # ======================================================================
    # delete-vector-index
    # ======================================================================
    delete_vector_index_wf = hatchet.workflow(
        name="delete-vector-index",
        input_validator=DeleteVectorIndexInput,
    )

    @delete_vector_index_wf.task(
        execution_timeout=timedelta(minutes=10),
    )
    async def delete_vector_index(
        input: DeleteVectorIndexInput, ctx: Context,
    ) -> dict:
        input_data = input.model_dump()
        parsed_data = (
            IngestionServiceAdapter.parse_delete_vector_index_input(
                input_data
            )
        )
        await service.providers.database.chunks_handler.delete_index(
            **parsed_data
        )
        return {"status": "Vector index deleted successfully."}

    # ------------------------------------------------------------------
    return {
        "ingest-files": ingest_files_wf,
        "ingest-chunks": ingest_chunks_wf,
        "update-chunk": update_chunk_wf,
        "create-vector-index": create_vector_index_wf,
        "delete-vector-index": delete_vector_index_wf,
    }


# ---------------------------------------------------------------------------
# Shared helpers (keep task functions readable)
# ---------------------------------------------------------------------------

async def _mark_graph_outdated(service: IngestionService, collection_id: UUID) -> None:
    """Mark both graph statuses as outdated for a collection."""
    db = service.providers.database
    await db.documents_handler.set_workflow_status(
        id=collection_id,
        status_type="graph_sync_status",
        status=GraphConstructionStatus.OUTDATED,
    )
    await db.documents_handler.set_workflow_status(
        id=collection_id,
        status_type="graph_cluster_status",
        status=GraphConstructionStatus.OUTDATED,
    )


async def _assign_doc_to_collection(
    service: IngestionService,
    document_info: DocumentResponse,
    collection_id: UUID,
) -> None:
    """Assign document + chunks to a collection and mark graph outdated."""
    db = service.providers.database
    await db.collections_handler.assign_document_to_collection_relational(
        document_id=document_info.id,
        collection_id=collection_id,
    )
    await db.chunks_handler.assign_document_chunks_to_collection(
        document_id=document_info.id,
        collection_id=collection_id,
    )
    await _mark_graph_outdated(service, collection_id)


async def _maybe_enrich_chunks(
    service: IngestionService,
    document_info: DocumentResponse,
    ingestion_config: dict,
) -> None:
    """Run chunk enrichment if configured."""
    server_settings = getattr(
        service.providers.ingestion.config,
        "chunk_enrichment_settings",
        None,
    )
    if not server_settings:
        return

    settings = update_settings_from_dict(
        server_settings,
        ingestion_config.get("chunk_enrichment_settings", {}) or {},
    )

    if not settings.enable_chunk_enrichment:
        return

    logger.info("Enriching document with contextual chunks")

    document_info = (
        await service.providers.database.documents_handler.get_documents_overview(
            offset=0, limit=1,
            filter_user_ids=[document_info.owner_id],
            filter_document_ids=[document_info.id],
        )
    )["results"][0]

    await service.update_document_status(
        document_info, status=IngestionStatus.ENRICHING,
    )
    await service.chunk_enrichment(
        document_id=document_info.id,
        document_summary=document_info.summary,
        chunk_enrichment_settings=settings,
    )
    await service.update_document_status(
        document_info, status=IngestionStatus.SUCCESS,
    )


async def _maybe_extract_graph(
    service: IngestionService,
    provider: OrchestrationProvider,
    document_info: DocumentResponse,
    ingestion_config: dict,
    input_data: dict,
) -> None:
    """Trigger graph extraction if automatic and not skipped."""
    doc_type = document_info.document_type.value
    skip_types = (
        service.providers.ingestion.config
        .skip_graph_extraction_for_types
    )
    if should_skip_graph_extraction(
        doc_type=doc_type,
        skip_types=skip_types,
        ingestion_config=ingestion_config,
        document_id=str(document_info.id),
    ):
        await service.providers.database.documents_handler.set_workflow_status(
            id=document_info.id,
            status_type="extraction_status",
            status=GraphExtractionStatus.SUCCESS,
        )
        logger.info(
            "Graph extraction skipped for document %s (type=%s)",
            document_info.id, doc_type,
        )
        return

    extract_input = {
        "document_id": str(document_info.id),
        "graph_creation_settings": (
            service.providers.database.config
            .graph_creation_settings.model_dump_json()
        ),
        "user": input_data["user"],
    }

    graph_wf = provider.get_workflow("graph-extraction")
    ref = await graph_wf.aio_run_no_wait(extract_input)
    await asyncio.wait_for(
        ref.aio_result(),
        timeout=float(
            os.environ.get("GRAPH_EXTRACTION_TIMEOUT", "3600")
        ),
    )


async def _handle_ingestion_failure(
    service: IngestionService,
    document_id: str,
    ctx: Context,
) -> None:
    """Shared failure handler for ingestion workflows."""
    if not document_id:
        logger.error(
            "No document id in workflow input to mark a failure."
        )
        return

    try:
        docs = (
            await service.providers.database.documents_handler.get_documents_overview(
                offset=0, limit=1,
                filter_document_ids=[document_id],
            )
        )["results"]

        if not docs:
            logger.error(
                "Document %s not found in database to mark failure.",
                document_id,
            )
            return

        document_info = docs[0]
        if document_info.ingestion_status != IngestionStatus.SUCCESS:
            await service.update_document_status(
                document_info,
                status=IngestionStatus.FAILED,
                metadata={"failure": str(ctx.task_run_errors)},
            )
    except Exception as e:
        logger.error(
            "Failed to update document status for %s: %s",
            document_id, e,
        )
