import asyncio
import copy
import json
import logging
import math
import os
import time
import uuid
from datetime import timedelta
from typing import Any, Optional

from hatchet_sdk import (
    ConcurrencyExpression,
    Context,
    Hatchet,
    TriggerWorkflowOptions,
)
from pydantic import BaseModel, ConfigDict

from core import GenerationConfig
from core.base import OrchestrationProvider, R2RException
from core.base.abstractions import (
    GraphConstructionStatus,
    GraphExtractionStatus,
)
from core.utils import convert_nonserializable_objects
from core.utils.otel_setup import (
    create_duration_histogram,
    get_meter,
    set_tenant_context,
)

from ...services import GraphService

logger = logging.getLogger(__name__)

# ── OTel Metrics for Graph Workflows (Tier 3 #14) ──
_meter = get_meter("r2r.graph")
_graph_entities = _meter.create_counter(
    "r2r.graph.entities_total",
    unit="{entity}",
    description="Total graph entities extracted",
)
_graph_extraction_duration = create_duration_histogram(
    _meter,
    "r2r.graph.extraction_duration_seconds",
    description="Graph extraction step duration",
)
_graph_extraction_status = _meter.create_counter(
    "r2r.graph.extraction_status_total",
    unit="{extraction}",
    description="Graph extraction outcomes by status (success/failed/skipped)",
)


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class GraphExtractionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: Optional[str] = None
    collection_id: Optional[str] = None
    graph_creation_settings: Any = None
    user: Optional[str] = None
    # Tier 1 #1: Tenant context propagated from ingestion workflow
    org_id: Optional[str] = None
    tenant_id: Optional[str] = None


class GraphClusteringInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    collection_id: Optional[str] = None
    graph_id: Optional[str] = None
    graph_enrichment_settings: Any = None
    user: Optional[str] = None
    org_id: Optional[str] = None
    tenant_id: Optional[str] = None


class GraphCommunitySummarizationInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    offset: int = 0
    limit: int = 100
    graph_id: Optional[str] = None
    collection_id: Optional[str] = None
    graph_enrichment_settings: Any = None
    org_id: Optional[str] = None
    tenant_id: Optional[str] = None


class GraphDeduplicationInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: Optional[str] = None
    org_id: Optional[str] = None
    tenant_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers (shared with simple workflows -- pure logic, no Hatchet)
# ---------------------------------------------------------------------------

def _set_tenant_context_from_input(
    input_obj: BaseModel,
) -> None:
    """Extract org_id/tenant_id from workflow input and set ContextVars.

    Graph workflow inputs now carry org_id/tenant_id propagated from the
    ingestion workflow that spawned them. This populates tenant context for
    all downstream OTel metrics and spans.
    """
    org_id = getattr(input_obj, "org_id", None) or ""
    tenant_id = getattr(input_obj, "tenant_id", None) or ""
    set_tenant_context(
        org_id=str(org_id),
        tenant_id=str(tenant_id),
    )


def get_input_data_dict(input_data: dict, fast_llm: str = "") -> dict:
    """Parse raw input dicts -- coerce UUIDs and generation config."""
    for key, value in input_data.items():
        if value is None:
            continue

        if key in ("document_id", "collection_id", "graph_id"):
            input_data[key] = (
                uuid.UUID(value)
                if not isinstance(value, uuid.UUID)
                else value
            )

        if key in (
            "graph_creation_settings",
            "graph_enrichment_settings",
        ):
            input_data[key] = (
                json.loads(value)
                if not isinstance(value, dict)
                else value
            )
            if "generation_config" in input_data[key]:
                gen_cfg = input_data[key]["generation_config"]
                if isinstance(gen_cfg, dict):
                    input_data[key]["generation_config"] = (
                        GenerationConfig(**gen_cfg)
                    )
                elif not isinstance(gen_cfg, GenerationConfig):
                    input_data[key]["generation_config"] = (
                        GenerationConfig()
                    )
                input_data[key]["generation_config"].model = (
                    input_data[key]["generation_config"].model
                    or fast_llm
                )

    return input_data


def _get_tenant_attrs(input_obj: BaseModel) -> dict[str, str]:
    """Build standard tenant attributes for graph metrics."""
    return {
        "org_id": str(getattr(input_obj, "org_id", None) or ""),
        "tenant_id": str(getattr(input_obj, "tenant_id", None) or ""),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def hatchet_graph_search_results_factory(
    hatchet: Hatchet,
    service: GraphService,
    config: Any,
    provider: OrchestrationProvider,
) -> dict[str, Any]:
    """Create v1 Hatchet workflow objects for all graph workflows."""

    fast_llm = getattr(
        getattr(service, "config", None), "app", None
    )
    fast_llm = getattr(fast_llm, "fast_llm", "") if fast_llm else ""

    # ======================================================================
    # graph-extraction
    # ======================================================================
    graph_extraction_wf = hatchet.workflow(
        name="graph-extraction",
        input_validator=GraphExtractionInput,
        concurrency=ConcurrencyExpression.from_int(
            config.graph_search_results_concurrency_limit
        ),
    )

    @graph_extraction_wf.task(
        retries=1, execution_timeout=timedelta(minutes=360),
    )
    async def extraction(
        input: GraphExtractionInput, ctx: Context,
    ) -> dict:
        # Tier 1 #1: Set tenant context from workflow input
        _set_tenant_context_from_input(input)
        _start = time.monotonic()

        request = input.model_dump()
        input_data = get_input_data_dict(request, fast_llm)
        document_id = input_data.get("document_id")
        collection_id = input_data.get("collection_id")
        _attrs = _get_tenant_attrs(input)

        if collection_id and not document_id:
            # Fan-out: spawn one workflow per document in collection
            document_ids = (
                await service.get_document_ids_for_create_graph(
                    collection_id=collection_id,
                    **input_data["graph_creation_settings"],
                )
            )
            refs = []
            for doc_id in document_ids:
                child_data = copy.deepcopy(input_data)
                child_data["collection_id"] = str(
                    child_data["collection_id"]
                )
                child_data["document_id"] = str(doc_id)
                # Propagate tenant context to child workflows
                child_data["org_id"] = input.org_id or ""
                child_data["tenant_id"] = input.tenant_id or ""
                ref = await graph_extraction_wf.aio_run_no_wait(
                    convert_nonserializable_objects(child_data),
                    options=TriggerWorkflowOptions(
                        key=str(doc_id),
                    ),
                )
                refs.append(ref)

            results = await asyncio.gather(
                *[r.aio_result() for r in refs],
                return_exceptions=True,
            )
            failed = [r for r in results if isinstance(r, Exception)]
            for r in failed:
                logger.error(
                    "Graph extraction child workflow failed: %s", r,
                )
            if len(failed) == len(results) and results:
                raise RuntimeError(
                    f"All {len(results)} graph extraction "
                    f"workflows failed for collection {collection_id}"
                )
            return {
                "result": (
                    "successfully submitted graph extraction "
                    f"for collection {collection_id}"
                ),
                "document_id": str(collection_id),
                "fan_out": True,
            }

        # Single-document extraction
        await service.providers.database.documents_handler.set_workflow_status(
            id=document_id,
            status_type="extraction_status",
            status=GraphExtractionStatus.PROCESSING,
        )
        extractions = []
        _status = "failed"
        try:
            async for ext in service.graph_search_results_extraction(
                document_id=document_id,
                **input_data["graph_creation_settings"],
            ):
                logger.info(
                    "Graph extraction: %d entities, %d relationships "
                    "(document_id=%s)",
                    len(ext.entities),
                    len(ext.relationships),
                    document_id,
                )
                extractions.append(ext)
                # Tier 3 #14: Count entities
                try:
                    _graph_entities.add(
                        len(ext.entities), _attrs
                    )
                except Exception:
                    pass

            await service.store_graph_search_results_extractions(
                extractions
            )
            _status = "success"
        except Exception:
            _status = "failed"
            raise
        finally:
            # Tier 3 #14: Record extraction duration and status
            try:
                _elapsed = time.monotonic() - _start
                _graph_extraction_duration.record(_elapsed, _attrs)
                _graph_extraction_status.add(
                    1, {**_attrs, "status": _status}
                )
            except Exception:
                pass

        logger.info(
            "Successfully ran graph extraction for document %s",
            document_id,
        )
        return {
            "result": (
                "successfully ran graph extraction "
                f"for document {document_id}"
            ),
            "document_id": str(document_id),
        }

    @graph_extraction_wf.task(
        retries=1,
        execution_timeout=timedelta(minutes=360),
        parents=[extraction],
    )
    async def entity_description(
        input: GraphExtractionInput, ctx: Context,
    ) -> dict:
        # Tier 1 #1: Set tenant context from workflow input
        _set_tenant_context_from_input(input)

        # After a collection fan-out, the parent's entity_description
        # fires with the original (collection-level) input where
        # document_id is None.  Each child handles its own entity
        # description, so we no-op here.
        extraction_out = ctx.task_output(extraction)
        if extraction_out.get("fan_out"):
            return {"result": "skipped -- handled by fan-out children"}

        input_data = get_input_data_dict(
            input.model_dump(), fast_llm,
        )
        document_id = input_data.get("document_id")

        await service.graph_search_results_entity_description(
            document_id=document_id,
            **input_data["graph_creation_settings"],
        )
        logger.info(
            "Successfully ran entity description for document %s",
            document_id,
        )

        auto_dedup = (
            service.providers.database.config
            .graph_creation_settings.automatic_deduplication
        )
        if auto_dedup:
            try:
                dedup_wf = provider.get_workflow("graph-deduplication")
                ref = await dedup_wf.aio_run_no_wait(
                    {
                        "document_id": str(document_id),
                        "org_id": input.org_id or "",
                        "tenant_id": input.tenant_id or "",
                    },
                )
                await asyncio.wait_for(
                    ref.aio_result(),
                    timeout=float(
                        os.environ.get("GRAPH_DEDUP_TIMEOUT", "3600")
                    ),
                )
            except Exception as e:
                logger.error(
                    "Auto-dedup failed for document %s (extraction "
                    "itself succeeded): %s", document_id, e,
                )

        # Mark extraction complete so the UI shows the correct status
        # instead of staying stuck at "processing" / "pending".
        doc_uuid = document_id if isinstance(document_id, uuid.UUID) else uuid.UUID(document_id)
        await service.providers.database.documents_handler.set_workflow_status(
            id=doc_uuid,
            status_type="extraction_status",
            status=GraphExtractionStatus.SUCCESS,
        )

        return {
            "result": (
                "successfully ran entity description "
                f"for document {document_id}"
            ),
        }

    @graph_extraction_wf.on_failure_task()
    async def on_failure_extraction(
        input: GraphExtractionInput, ctx: Context,
    ) -> None:
        _set_tenant_context_from_input(input)
        document_id = input.document_id
        if not document_id:
            logger.info(
                "No document id in workflow input to mark failure."
            )
            return
        # Record failure metric
        try:
            _graph_extraction_status.add(
                1,
                {**_get_tenant_attrs(input), "status": "failed"},
            )
        except Exception:
            pass
        try:
            doc_uuid = document_id if isinstance(document_id, uuid.UUID) else uuid.UUID(document_id)
            await service.providers.database.documents_handler.set_workflow_status(
                id=doc_uuid,
                status_type="extraction_status",
                status=GraphExtractionStatus.FAILED,
            )
            logger.info(
                "Updated extraction status for %s to FAILED",
                document_id,
            )
        except Exception as e:
            logger.error(
                "Failed to update document status for %s: %s",
                document_id, e,
            )

    # ======================================================================
    # graph-clustering
    # ======================================================================
    graph_clustering_wf = hatchet.workflow(
        name="graph-clustering",
        input_validator=GraphClusteringInput,
    )

    @graph_clustering_wf.task(
        retries=1,
        execution_timeout=timedelta(minutes=360),
    )
    async def clustering(
        input: GraphClusteringInput, ctx: Context,
    ) -> dict:
        _set_tenant_context_from_input(input)
        logger.info("Running Graph Clustering")
        input_data = get_input_data_dict(
            input.model_dump(), fast_llm,
        )
        collection_id = input_data.get("collection_id")
        graph_id = input_data.get("graph_id")

        workflow_status = (
            await service.providers.database.documents_handler.get_workflow_status(
                id=collection_id,
                status_type="graph_cluster_status",
            )
        )
        if workflow_status == GraphConstructionStatus.SUCCESS:
            raise R2RException(
                "Communities have already been built for this "
                "collection. To build again, first reset the graph.",
                400,
            )

        try:
            result = (
                await service.graph_search_results_clustering(
                    collection_id=collection_id,
                    graph_id=graph_id,
                    **input_data["graph_enrichment_settings"],
                )
            )
            num_communities = result["num_communities"][0]
            if num_communities == 0:
                raise R2RException("No communities found", 400)

            return {"result": result}
        except Exception as e:
            await service.providers.database.documents_handler.set_workflow_status(
                id=collection_id,
                status_type="graph_cluster_status",
                status=GraphConstructionStatus.FAILED,
            )
            raise e

    @graph_clustering_wf.task(
        retries=1,
        execution_timeout=timedelta(minutes=360),
        parents=[clustering],
    )
    async def community_summary(
        input: GraphClusteringInput, ctx: Context,
    ) -> dict:
        _set_tenant_context_from_input(input)
        input_data = get_input_data_dict(
            input.model_dump(), fast_llm,
        )
        collection_id = input_data.get("collection_id")
        graph_id = input_data.get("graph_id")

        num_communities = ctx.task_output(clustering)[
            "result"
        ]["num_communities"][0]

        parallel = min(100, num_communities)
        total_workflows = math.ceil(num_communities / parallel)

        logger.info(
            "Spawning %d community summary workflows for "
            "%d communities",
            total_workflows, num_communities,
        )

        summarization_wf = provider.get_workflow(
            "graph-community-summarization"
        )

        refs = []
        for i in range(total_workflows):
            offset = i * parallel
            limit = min(parallel, num_communities - offset)

            ref = await summarization_wf.aio_run_no_wait(
                {
                    "offset": offset,
                    "limit": limit,
                    "graph_id": (
                        str(graph_id) if graph_id else None
                    ),
                    "collection_id": (
                        str(collection_id)
                        if collection_id
                        else None
                    ),
                    "graph_enrichment_settings": convert_nonserializable_objects(
                        input_data["graph_enrichment_settings"]
                    ),
                    "org_id": input.org_id or "",
                    "tenant_id": input.tenant_id or "",
                },
                options=TriggerWorkflowOptions(
                    key=f"{i}/{total_workflows}_community_summary",
                ),
            )
            refs.append(ref)

        results = await asyncio.gather(
            *[r.aio_result() for r in refs],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            logger.error(
                "Community summary: %d/%d workflows failed",
                len(failed), len(results),
            )
            for f in failed:
                logger.error("Community summary error: %s", f)

        if len(failed) == len(results) and results:
            raise RuntimeError(
                f"All {len(results)} community summary workflows "
                f"failed for collection {collection_id}"
            )

        logger.info(
            "Completed %d/%d community summary workflows",
            len(results) - len(failed), len(results),
        )

        # Update statuses -- only set SUCCESS when no failures
        document_ids = (
            await service.providers.database.documents_handler.get_document_ids_by_status(
                status_type="extraction_status",
                status=GraphExtractionStatus.SUCCESS,
                collection_id=collection_id,
            )
        )
        await service.providers.database.documents_handler.set_workflow_status(
            id=document_ids,
            status_type="extraction_status",
            status=GraphExtractionStatus.ENRICHED,
        )
        final_status = (
            GraphConstructionStatus.SUCCESS
            if not failed
            else GraphConstructionStatus.FAILED
        )
        await service.providers.database.documents_handler.set_workflow_status(
            id=collection_id,
            status_type="graph_cluster_status",
            status=final_status,
        )

        if failed:
            return {
                "result": (
                    f"Partial failure: {len(failed)}/{len(results)} "
                    "community summary workflows failed"
                ),
            }

        return {
            "result": (
                "Successfully completed enrichment with "
                f"{len(results)} summary workflows"
            ),
        }

    @graph_clustering_wf.on_failure_task()
    async def on_failure_clustering(
        input: GraphClusteringInput, ctx: Context,
    ) -> None:
        _set_tenant_context_from_input(input)
        collection_id = input.collection_id
        if collection_id:
            try:
                await service.providers.database.documents_handler.set_workflow_status(
                    id=uuid.UUID(collection_id),
                    status_type="graph_cluster_status",
                    status=GraphConstructionStatus.FAILED,
                )
            except Exception as e:
                logger.error(
                    "Failed to update cluster status for %s: %s",
                    collection_id, e,
                )

    # ======================================================================
    # graph-community-summarization
    # ======================================================================
    graph_community_summarization_wf = hatchet.workflow(
        name="graph-community-summarization",
        input_validator=GraphCommunitySummarizationInput,
        concurrency=ConcurrencyExpression.from_int(
            config.graph_search_results_concurrency_limit
        ),
    )

    @graph_community_summarization_wf.task(
        retries=1, execution_timeout=timedelta(minutes=360),
    )
    async def summarize_communities(
        input: GraphCommunitySummarizationInput, ctx: Context,
    ) -> dict:
        _set_tenant_context_from_input(input)
        start_time = time.time()
        input_data = get_input_data_dict(
            input.model_dump(), fast_llm,
        )

        base_args = {
            k: v
            for k, v in input_data.items()
            if k != "graph_enrichment_settings"
        }
        enrichment_args = input_data.get(
            "graph_enrichment_settings", {}
        )
        merged_args = {**base_args, **enrichment_args}

        summary = (
            await service.graph_search_results_community_summary(
                **merged_args
            )
        )
        logger.info(
            "Community summary for communities %d to %d "
            "in %.2f seconds",
            input_data["offset"],
            input_data["offset"] + len(summary),
            time.time() - start_time,
        )
        return {
            "result": (
                "successfully ran community summary for "
                f"communities {input_data['offset']} to "
                f"{input_data['offset'] + len(summary)}"
            ),
        }

    # ======================================================================
    # graph-deduplication
    # ======================================================================
    graph_deduplication_wf = hatchet.workflow(
        name="graph-deduplication",
        input_validator=GraphDeduplicationInput,
        concurrency=ConcurrencyExpression.from_int(
            config.graph_search_results_concurrency_limit
        ),
    )

    @graph_deduplication_wf.task(
        retries=1, execution_timeout=timedelta(minutes=360),
    )
    async def deduplicate(
        input: GraphDeduplicationInput, ctx: Context,
    ) -> dict:
        _set_tenant_context_from_input(input)
        start_time = time.time()
        input_data = get_input_data_dict(
            input.model_dump(), fast_llm,
        )
        document_id = input_data.get("document_id")

        await service.deduplicate_document_entities(
            document_id=document_id,
        )
        logger.info(
            "Deduplication for document %s in %.2f seconds",
            document_id, time.time() - start_time,
        )
        return {
            "result": (
                "Successfully ran deduplication "
                f"for document {document_id}"
            ),
        }

    # ------------------------------------------------------------------
    return {
        "graph-extraction": graph_extraction_wf,
        "graph-clustering": graph_clustering_wf,
        "graph-community-summarization": (
            graph_community_summarization_wf
        ),
        "graph-deduplication": graph_deduplication_wf,
    }
