# type: ignore
"""
Hatchet workflows for graph operations.
"""
import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from pydantic import BaseModel
from hatchet_sdk import Context, ConcurrencyExpression, ConcurrencyLimitStrategy

from core.base import (
    GraphConstructionStatus,
    OrchestrationConfig,
)

from ...services import GraphService

if TYPE_CHECKING:
    from hatchet_sdk import Hatchet

logger = logging.getLogger()


class GraphExtractionInput(BaseModel):
    """Input model for graph-extraction workflow"""
    document_id: Optional[UUID] = None
    collection_id: Optional[UUID] = None
    graph_creation_settings: str  # JSON string
    user: dict

class GraphClusteringInput(BaseModel):
    """Input model for graph-clustering workflow"""
    collection_id: UUID
    graph_id: UUID
    graph_enrichment_settings: dict

class GraphCommunitySummarizationInput(BaseModel):
    """Input model for graph-community-summarization workflow"""
    offset: int
    limit: int
    graph_id: Optional[UUID] = None
    collection_id: Optional[UUID] = None
    graph_enrichment_settings: dict

class GraphDeduplicationInput(BaseModel):
    """Input model for graph-deduplication workflow"""
    document_id: UUID


def hatchet_graph_search_results_factory(
    hatchet: "Hatchet",
    service: GraphService,
    config: OrchestrationConfig
) -> list:
    """
    Create graph workflows.

    Returns a list of workflow objects to register with the worker.
    """
    workflows = []

    # ========================================================================
    # 1. graph-extraction workflow (with concurrency control)
    # ========================================================================
    graph_extraction_wf = hatchet.workflow(
        name="graph-extraction",
        concurrency=ConcurrencyExpression(
            expression="input.request.user.id",  # Group by user ID
            max_runs=config.graph_search_results_concurrency_limit,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        )
    )

    @graph_extraction_wf.task(
        retries=0,
        execution_timeout=timedelta(minutes=60)
    )
    async def extract_graph(input: dict, ctx: Context) -> dict:
        """Extract entities and relationships from documents"""
        try:
            logger.info(f"Starting graph extraction for input: {input}")

            # Extract request from wrapped input (API passes {"request": workflow_input})
            workflow_input = input.get("request", input)

            # Parse graph creation settings from JSON string
            graph_creation_settings = json.loads(workflow_input["graph_creation_settings"])

            # Determine if processing collection or document
            collection_id = workflow_input.get("collection_id")
            document_id = workflow_input.get("document_id")

            if collection_id:
                collection_id = UUID(collection_id) if isinstance(collection_id, str) else collection_id
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.PROCESSING,
                )

                await service.process_collection(
                    collection_id=collection_id,
                    graph_creation_settings=graph_creation_settings
                )

                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.SUCCESS,
                )

                return {
                    "message": f"Successfully processed collection {collection_id}",
                    "task_id": ctx.workflow_run_id(),
                }

            elif document_id:
                document_id = UUID(document_id) if isinstance(document_id, str) else document_id
                await service.process_graph(
                    document_id=document_id,
                    graph_creation_settings=graph_creation_settings
                )

                return {
                    "message": f"Successfully processed document {document_id}",
                    "task_id": ctx.workflow_run_id(),
                }

            else:
                raise ValueError("Either collection_id or document_id must be provided")

        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            # Update status to failed
            if collection_id:
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.FAILED,
                )
            raise

    @graph_extraction_wf.on_failure_task()
    async def on_extraction_failure(input: dict, ctx: Context) -> dict:
        """Handle failure for graph-extraction workflow"""
        # Extract request from wrapped input
        workflow_input = input.get("request", input)
        collection_id = workflow_input.get("collection_id")
        if collection_id:
            try:
                await service.providers.database.documents_handler.set_workflow_status(
                    id=UUID(collection_id),
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.FAILED,
                )
            except Exception as e:
                logger.error(f"Failed to update workflow status: {e}")

        return {"status": "failure_handled"}

    workflows.append(graph_extraction_wf)

    # ========================================================================
    # 2. graph-clustering workflow
    # ========================================================================
    graph_clustering_wf = hatchet.workflow(
        name="graph-clustering"
    )

    @graph_clustering_wf.task(execution_timeout=timedelta(minutes=60))
    async def cluster_graph(input: dict, ctx: Context) -> dict:
        """Perform graph clustering"""
        try:
            # Extract request from wrapped input
            workflow_input = input.get("request", input)

            collection_id = UUID(workflow_input["collection_id"]) if isinstance(workflow_input["collection_id"], str) else workflow_input["collection_id"]
            graph_id = UUID(workflow_input["graph_id"]) if isinstance(workflow_input["graph_id"], str) else workflow_input["graph_id"]
            graph_enrichment_settings = workflow_input["graph_enrichment_settings"]

            await service.providers.database.documents_handler.set_workflow_status(
                id=collection_id,
                status_type="graph_cluster_status",
                status=GraphConstructionStatus.PROCESSING,
            )

            # Perform clustering
            num_clusters = await service.cluster_graph(
                graph_id=graph_id,
                graph_enrichment_settings=graph_enrichment_settings
            )

            await service.providers.database.documents_handler.set_workflow_status(
                id=collection_id,
                status_type="graph_cluster_status",
                status=GraphConstructionStatus.SUCCESS,
            )

            # Trigger community summarization if successful
            # TODO: Spawn graph-community-summarization workflow

            return {
                "message": f"Graph clustered successfully with {num_clusters} clusters",
                "task_id": ctx.workflow_run_id(),
                "num_clusters": num_clusters,
            }

        except Exception as e:
            logger.error(f"Graph clustering failed: {e}")
            if collection_id:
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_cluster_status",
                    status=GraphConstructionStatus.FAILED,
                )
            raise

    @graph_clustering_wf.on_failure_task()
    async def on_clustering_failure(input: dict, ctx: Context) -> dict:
        """Handle failure for graph-clustering workflow"""
        collection_id = input.get("collection_id")
        if collection_id:
            try:
                await service.providers.database.documents_handler.set_workflow_status(
                    id=UUID(collection_id),
                    status_type="graph_cluster_status",
                    status=GraphConstructionStatus.FAILED,
                )
            except Exception as e:
                logger.error(f"Failed to update workflow status: {e}")

        return {"status": "failure_handled"}

    workflows.append(graph_clustering_wf)

    # ========================================================================
    # 3. graph-community-summarization workflow
    # ========================================================================
    graph_summary_wf = hatchet.workflow(
        name="graph-community-summarization"
    )

    @graph_summary_wf.task(execution_timeout=timedelta(minutes=60))
    async def summarize_communities(input: dict, ctx: Context) -> dict:
        """Generate summaries for graph communities"""
        try:
            # Extract request from wrapped input
            workflow_input = input.get("request", input)

            offset = workflow_input["offset"]
            limit = workflow_input["limit"]
            graph_id = workflow_input.get("graph_id")
            collection_id = workflow_input.get("collection_id")
            graph_enrichment_settings = workflow_input["graph_enrichment_settings"]

            if graph_id:
                graph_id = UUID(graph_id) if isinstance(graph_id, str) else graph_id
            if collection_id:
                collection_id = UUID(collection_id) if isinstance(collection_id, str) else collection_id

            # Perform community summarization
            community_summary = await service.create_communities_summary(
                offset=offset,
                limit=limit,
                graph_id=graph_id,
                collection_id=collection_id,
                graph_enrichment_settings=graph_enrichment_settings
            )

            return {
                "result": f"Successfully ran graph_search_results community summary for communities {offset} to {offset + len(community_summary)}",
                "task_id": ctx.workflow_run_id(),
            }

        except Exception as e:
            logger.error(f"Community summarization failed: {e}")
            raise

    workflows.append(graph_summary_wf)

    # ========================================================================
    # 4. graph-deduplication workflow
    # ========================================================================
    graph_dedup_wf = hatchet.workflow(
        name="graph-deduplication"
    )

    @graph_dedup_wf.task(execution_timeout=timedelta(minutes=60))
    async def deduplicate_graph(input: dict, ctx: Context) -> dict:
        """Deduplicate graph entities"""
        try:
            # Extract request from wrapped input
            workflow_input = input.get("request", input)

            document_id = UUID(workflow_input["document_id"]) if isinstance(workflow_input["document_id"], str) else workflow_input["document_id"]

            # Perform deduplication
            await service.deduplicate_entities(document_id=document_id)

            return {
                "message": f"Successfully deduplicated graph for document {document_id}",
                "task_id": ctx.workflow_run_id(),
            }

        except Exception as e:
            logger.error(f"Graph deduplication failed: {e}")
            raise

    workflows.append(graph_dedup_wf)

    return workflows