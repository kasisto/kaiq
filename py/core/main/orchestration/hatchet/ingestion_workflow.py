# type: ignore
"""
Hatchet workflows for document ingestion.
"""
import asyncio
import json
import logging
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from hatchet_sdk import Context, ConcurrencyExpression, ConcurrencyLimitStrategy
from fastapi import HTTPException
from litellm import AuthenticationError

from core.base import (
    DocumentChunk,
    GraphConstructionStatus,
    IngestionStatus,
    OrchestrationConfig,
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

if TYPE_CHECKING:
    from hatchet_sdk import Hatchet

logger = logging.getLogger()


# ============================================================================
# Pydantic Input Models
# ============================================================================

class UserInput(BaseModel):
    """User model for workflow inputs"""
    id: str
    email: Optional[str] = None
    # Add other user fields as needed

class IngestFileInput(BaseModel):
    """Input model for ingest-files workflow"""
    document_id: Optional[UUID] = None
    user: dict  # Will be parsed to User internally
    file_data: dict
    metadata: Optional[dict] = {}
    ingestion_config: Optional[dict] = {}
    collection_ids: Optional[list[str]] = []
    version: Optional[str] = None
    size_in_bytes: int

class IngestChunksInput(BaseModel):
    """Input model for ingest-chunks workflow"""
    document_id: UUID
    user: dict  # Will be parsed to User internally
    chunks: list[dict]
    metadata: Optional[dict] = {}
    collection_ids: Optional[list[str]] = []
    id: Optional[str] = None

class UpdateChunkInput(BaseModel):
    """Input model for update-chunk workflow"""
    document_id: UUID
    id: UUID  # chunk ID
    user: dict  # Will be parsed to User internally
    text: Optional[str] = None
    metadata: Optional[dict] = {}
    collection_ids: Optional[list[str]] = []

class VectorIndexCreateInput(BaseModel):
    """Input model for create-vector-index workflow"""
    table_name: str
    index_method: str
    index_measure: str
    index_name: str
    index_column: str
    index_arguments: dict
    concurrently: bool = True

class VectorIndexDeleteInput(BaseModel):
    """Input model for delete-vector-index workflow"""
    index_name: str
    table_name: Optional[str] = None
    concurrently: bool = True


# ============================================================================
# Workflow Factory
# ============================================================================

def hatchet_ingestion_factory(
    hatchet: "Hatchet",
    service: IngestionService,
    config: OrchestrationConfig
) -> list:
    """
    Create ingestion workflows.

    Returns a list of workflow objects to register with the worker.
    """
    workflows = []

    # ========================================================================
    # 1. ingest-files workflow (with concurrency control)
    # ========================================================================
    ingest_files_wf = hatchet.workflow(
        name="ingest-files",
        concurrency=ConcurrencyExpression(
            expression="input.request.document_id",  # Group by document ID (user is a JSON string, can't access .id)
            max_runs=config.ingestion_concurrency_limit,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        )
    )

    @ingest_files_wf.task(
        retries=0,
        execution_timeout=timedelta(minutes=60)
    )
    async def parse_file(input: dict, ctx: Context) -> dict:
        """Main parse task for file ingestion"""
        try:
            logger.info("🚀 [PARSE STEP] ========================================")
            logger.info("🚀 [PARSE STEP] Starting parse step execution")
            logger.info(f"🚀 [PARSE STEP] Workflow run ID: {ctx.workflow_run_id()}")
            logger.info("🚀 [PARSE STEP] ========================================")

            # Parse input using existing adapter
            # Extract request from wrapped input (API passes {"request": workflow_input})
            workflow_input = input.get("request", input)
            parsed_data = IngestionServiceAdapter.parse_ingest_file_input(workflow_input)

            logger.info(f"🚀 [PARSE STEP] Document ID: {parsed_data.get('document_id')}")
            logger.info(f"🚀 [PARSE STEP] Collection IDs: {parsed_data.get('collection_ids')}")

            # Create document info
            document_info = service.create_document_info_from_file(
                parsed_data["document_id"],
                parsed_data["user"],
                parsed_data["file_data"]["filename"],
                parsed_data["metadata"],
                parsed_data["version"],
                parsed_data["size_in_bytes"],
            )

            await service.update_document_status(
                document_info,
                status=IngestionStatus.PARSING,
            )

            # Parse the file
            ingestion_config = parsed_data["ingestion_config"] or {}
            extractions_generator = service.parse_file(
                document_info, ingestion_config
            )

            extractions = []
            async for extraction in extractions_generator:
                extractions.append(extraction)

            # Sum tokens
            total_tokens = 0
            for chunk in extractions:
                text_data = chunk.data
                if not isinstance(text_data, str):
                    text_data = text_data.decode("utf-8", errors="ignore")
                total_tokens += num_tokens(text_data)
            document_info.total_tokens = total_tokens

            # Document summary if not skipped
            if not ingestion_config.get("skip_document_summary", False):
                await service.update_document_status(
                    document_info, status=IngestionStatus.AUGMENTING
                )
                await service.augment_document_info(
                    document_info,
                    [extraction.to_dict() for extraction in extractions],
                )

            # Embedding phase
            await service.update_document_status(
                document_info,
                status=IngestionStatus.EMBEDDING,
            )

            embedding_generator = service.embed_document(
                [extraction.to_dict() for extraction in extractions]
            )

            embeddings = []
            async for embedding in embedding_generator:
                embeddings.append(embedding)

            # Storage phase
            await service.update_document_status(
                document_info,
                status=IngestionStatus.STORING,
            )

            storage_generator = service.store_embeddings(embeddings)
            async for _ in storage_generator:
                pass

            # Finalize
            await service.finalize_ingestion(document_info)

            await service.update_document_status(
                document_info,
                status=IngestionStatus.SUCCESS,
            )

            # Handle collection assignments
            collection_ids = parsed_data.get("collection_ids")
            if not collection_ids:
                collection_id = generate_default_user_collection_id(
                    document_info.owner_id
                )
                await service.providers.database.collections_handler.assign_document_to_collection_relational(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )
                await service.providers.database.chunks_handler.assign_document_chunks_to_collection(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.OUTDATED,
                )
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_cluster_status",
                    status=GraphConstructionStatus.OUTDATED,
                )
            else:
                for collection_id_str in collection_ids:
                    collection_id = UUID(collection_id_str)
                    try:
                        name = document_info.title or "N/A"
                        description = ""
                        await service.providers.database.collections_handler.create_collection(
                            owner_id=document_info.owner_id,
                            name=name,
                            description=description,
                            collection_id=collection_id,
                        )
                        await service.providers.database.graphs_handler.create(
                            collection_id=collection_id,
                            name=name,
                            description=description,
                            graph_id=collection_id,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Warning, could not create collection with error: {str(e)}"
                        )

                    await service.providers.database.collections_handler.assign_document_to_collection_relational(
                        document_id=document_info.id,
                        collection_id=collection_id,
                    )
                    await service.providers.database.chunks_handler.assign_document_chunks_to_collection(
                        document_id=document_info.id,
                        collection_id=collection_id,
                    )
                    await service.providers.database.documents_handler.set_workflow_status(
                        id=collection_id,
                        status_type="graph_sync_status",
                        status=GraphConstructionStatus.OUTDATED,
                    )
                    await service.providers.database.documents_handler.set_workflow_status(
                        id=collection_id,
                        status_type="graph_cluster_status",
                        status=GraphConstructionStatus.OUTDATED,
                    )

            # Handle chunk enrichment if enabled
            chunk_enrichment_settings = None
            if server_chunk_enrichment_settings := getattr(
                service.providers.ingestion.config,
                "chunk_enrichment_settings",
                None,
            ):
                chunk_enrichment_settings = update_settings_from_dict(
                    server_chunk_enrichment_settings,
                    ingestion_config.get("chunk_enrichment_settings", {})
                    or {},
                )

            if chunk_enrichment_settings and chunk_enrichment_settings.enable_chunk_enrichment:
                logger.info("Enriching document with contextual chunks")

                document_info = (
                    await service.providers.database.documents_handler.get_documents_overview(
                        offset=0,
                        limit=1,
                        filter_user_ids=[document_info.owner_id],
                        filter_document_ids=[document_info.id],
                    )
                )["results"][0]

                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.ENRICHING,
                )

                await service.chunk_enrichment(
                    document_id=document_info.id,
                    document_summary=document_info.summary,
                    chunk_enrichment_settings=chunk_enrichment_settings,
                )

                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.SUCCESS,
                )

            # Automatic graph extraction if enabled
            if service.providers.ingestion.config.automatic_extraction:
                extract_input = {
                    "document_id": str(document_info.id),
                    "graph_creation_settings": service.providers.database.config.graph_creation_settings.model_dump_json(),
                    "user": input["user"],
                }

                # TODO: Spawn graph-extraction workflow when it's ready
                # For now, try the workflow name
                try:
                    # Note: spawn_workflow might not exist, need to check SDK docs
                    # This might need to be replaced with hatchet.aio_run()
                    pass
                except Exception as e:
                    logger.warning(f"Could not spawn graph extraction: {e}")

            return {
                "status": "Successfully finalized ingestion",
                "document_info": document_info.to_dict(),
            }

        except AuthenticationError:
            raise R2RException(
                status_code=401,
                message="Authentication error: Invalid API key or credentials.",
            ) from None
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during ingestion: {str(e)}",
            ) from e

    @ingest_files_wf.on_failure_task()
    async def on_ingest_failure(input: dict, ctx: Context) -> dict:
        """Handle failure for ingest-files workflow"""
        # Extract request from wrapped input
        workflow_input = input.get("request", input)
        document_id = workflow_input.get("document_id")

        if not document_id:
            logger.error("No document id found in workflow input to mark failure.")
            return {"status": "failure_handled"}

        try:
            documents_overview = (
                await service.providers.database.documents_handler.get_documents_overview(
                    offset=0,
                    limit=1,
                    filter_document_ids=[document_id],
                )
            )["results"]

            if not documents_overview:
                logger.error(
                    f"Document with id {document_id} not found in database to mark failure."
                )
                return {"status": "failure_handled"}

            document_info = documents_overview[0]

            if document_info.ingestion_status != IngestionStatus.SUCCESS:
                # Get error info from context
                error_info = str(ctx.task_run_errors) if hasattr(ctx, 'task_run_errors') else "Unknown error"
                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.FAILED,
                    metadata={"failure": error_info},
                )
        except Exception as e:
            logger.error(f"Failed to update document status for {document_id}: {e}")

        return {"status": "failure_handled"}

    workflows.append(ingest_files_wf)

    # ========================================================================
    # 2. ingest-chunks workflow (multi-step DAG)
    # ========================================================================
    ingest_chunks_wf = hatchet.workflow(
        name="ingest-chunks"
    )

    # Step 1: Ingest
    @ingest_chunks_wf.task(execution_timeout=timedelta(minutes=60))
    async def ingest_chunks(input: dict, ctx: Context) -> dict:
        """Ingest chunks step"""
        # Extract request from wrapped input (API passes {"request": workflow_input})
        workflow_input = input.get("request", input)
        parsed_data = IngestionServiceAdapter.parse_ingest_chunks_input(workflow_input)

        document_info = await service.ingest_chunks_ingress(**parsed_data)

        await service.update_document_status(
            document_info, status=IngestionStatus.EMBEDDING
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

        # Sum tokens
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

    # Step 2: Embed (depends on ingest)
    @ingest_chunks_wf.task(
        parents=[ingest_chunks],
        execution_timeout=timedelta(minutes=60)
    )
    async def embed_chunks(input: dict, ctx: Context) -> dict:
        """Embed chunks step"""
        # Access parent output
        ingest_output = ctx.task_output(ingest_chunks)
        document_info_dict = ingest_output["document_info"]
        document_info = DocumentResponse(**document_info_dict)
        extractions = ingest_output["extractions"]

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

        return {
            "status": "Successfully embedded and stored chunks",
            "document_info": document_info.to_dict(),
        }

    # Step 3: Finalize (depends on embed)
    @ingest_chunks_wf.task(
        parents=[embed_chunks],
        execution_timeout=timedelta(minutes=60)
    )
    async def finalize_chunks(input: dict, ctx: Context) -> dict:
        """Finalize chunks ingestion"""
        embed_output = ctx.task_output(embed_chunks)
        document_info_dict = embed_output["document_info"]
        document_info = DocumentResponse(**document_info_dict)

        await service.finalize_ingestion(document_info)
        await service.update_document_status(
            document_info, status=IngestionStatus.SUCCESS
        )

        # Handle collection assignments
        workflow_input = input.get("request", input)
        parsed_data = IngestionServiceAdapter.parse_ingest_chunks_input(workflow_input)
        collection_ids = parsed_data.get("collection_ids", [])

        if collection_ids:
            for collection_id_str in collection_ids:
                collection_id = UUID(collection_id_str)
                await service.providers.database.collections_handler.assign_document_to_collection_relational(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )

        return {
            "status": "Successfully finalized chunk ingestion",
            "document_info": document_info.to_dict(),
            "task_id": ctx.workflow_run_id(),
        }

    @ingest_chunks_wf.on_failure_task()
    async def on_chunks_failure(input: dict, ctx: Context) -> dict:
        """Handle failure for ingest-chunks workflow"""
        # Extract request from wrapped input
        workflow_input = input.get("request", input)
        document_id = workflow_input.get("document_id")
        if document_id:
            try:
                documents_overview = (
                    await service.providers.database.documents_handlers.get_documents_overview(
                        offset=0,
                        limit=1,
                        filter_document_ids=[document_id],
                    )
                )["results"]

                if documents_overview:
                    document_info = documents_overview[0]
                    if document_info.ingestion_status != IngestionStatus.SUCCESS:
                        await service.update_document_status(
                            document_info, status=IngestionStatus.FAILED
                        )
            except Exception as e:
                logger.error(f"Failed to update document status: {e}")

        return {"status": "failure_handled"}

    workflows.append(ingest_chunks_wf)

    # ========================================================================
    # 3. update-chunk workflow (simple single-step)
    # ========================================================================
    update_chunk_wf = hatchet.workflow(
        name="update-chunk"
    )

    @update_chunk_wf.task(execution_timeout=timedelta(minutes=60))
    async def update_chunk(input: dict, ctx: Context) -> dict:
        """Update a single chunk"""
        workflow_input = input.get("request", input)
        parsed_data = IngestionServiceAdapter.parse_update_chunk_input(workflow_input)

        document_uuid = parsed_data["document_id"]
        extraction_uuid = parsed_data["id"]

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
            "task_id": ctx.workflow_run_id(),
            "document_ids": [str(document_uuid)],
        }

    @update_chunk_wf.on_failure_task()
    async def on_update_failure(input: dict, ctx: Context) -> dict:
        """Handle failure for update-chunk workflow"""
        logger.error(f"Update chunk failed: {ctx.task_run_errors if hasattr(ctx, 'task_run_errors') else 'Unknown'}")
        return {"status": "failure_handled"}

    workflows.append(update_chunk_wf)

    # ========================================================================
    # 4. create-vector-index workflow
    # ========================================================================
    create_index_wf = hatchet.workflow(
        name="create-vector-index"
    )

    @create_index_wf.task(execution_timeout=timedelta(minutes=60))
    async def create_vector_index(input: dict, ctx: Context) -> dict:
        """Create a vector index"""
        workflow_input = input.get("request", input)
        parsed_data = IngestionServiceAdapter.parse_create_vector_index_input(workflow_input)

        await service.providers.database.chunks_handler.create_index(**parsed_data)

        return {
            "status": "Vector index creation queued successfully.",
        }

    workflows.append(create_index_wf)

    # ========================================================================
    # 5. delete-vector-index workflow
    # ========================================================================
    delete_index_wf = hatchet.workflow(
        name="delete-vector-index"
    )

    @delete_index_wf.task(execution_timeout=timedelta(minutes=10))
    async def delete_vector_index(input: dict, ctx: Context) -> dict:
        """Delete a vector index"""
        workflow_input = input.get("request", input)
        parsed_data = IngestionServiceAdapter.parse_delete_vector_index_input(workflow_input)

        await service.providers.database.chunks_handler.delete_index(**parsed_data)

        return {"status": "Vector index deleted successfully."}

    workflows.append(delete_index_wf)

    return workflows