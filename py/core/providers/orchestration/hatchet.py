import asyncio
import logging
import os
from typing import Any, Optional

from core.base import OrchestrationConfig, OrchestrationProvider, Workflow

logger = logging.getLogger(__name__)


def _on_worker_done(task: asyncio.Task) -> None:
    """Exit the process when the Hatchet worker dies unexpectedly.

    This triggers a container restart so the worker recovers
    automatically instead of running without task processing.
    """
    if task.cancelled():
        logger.info("Hatchet worker task cancelled (clean shutdown)")
        return
    if exc := task.exception():
        logger.error(
            "Hatchet worker died unexpectedly: %s — exiting so "
            "the container restarts", exc,
        )
    else:
        logger.warning(
            "Hatchet worker task completed unexpectedly — exiting "
            "so the container restarts"
        )
    os._exit(1)


class HatchetOrchestrationProvider(OrchestrationProvider):
    def __init__(self, config: OrchestrationConfig):
        super().__init__(config)
        try:
            from hatchet_sdk import Hatchet
        except ImportError:
            raise ImportError(
                "Hatchet SDK not installed. Please install it with "
                "`pip install hatchet-sdk`."
            ) from None

        self.orchestrator = Hatchet()
        self.config: OrchestrationConfig = config
        self.messages: dict[str, str] = {}
        self._workflows: dict[str, Any] = {}
        self._worker_name: str = "r2r-worker"
        self._worker_slots: int = config.max_runs
        self._worker_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Worker lifecycle (deferred — v1 needs workflows at construction)
    # ------------------------------------------------------------------

    def get_worker(
        self, name: str, max_runs: Optional[int] = None,
    ) -> Any:
        self._worker_name = name
        if max_runs is not None:
            self._worker_slots = max_runs
        return self  # placeholder — real worker created in start_worker

    async def start_worker(self):
        if not self._workflows:
            raise ValueError(
                "No workflows registered. "
                "Call register_workflows() before start_worker()."
            )

        self.worker = self.orchestrator.worker(
            self._worker_name,
            slots=self._worker_slots,
            workflows=list(self._workflows.values()),
        )
        # WORKAROUND (hatchet-sdk 1.29.3): The SDK has no public API
        # to start a worker inside an existing event loop.
        #   - worker.start() creates its own loop and blocks (unusable
        #     inside Uvicorn).
        #   - WorkerStartOptions(loop=...) is deprecated and ignored.
        #   - worker._aio_start() is the internal async entry point
        #     but requires worker.loop to be pre-set (otherwise the
        #     action runner raises RuntimeError).
        # If the SDK is upgraded, verify _aio_start and worker.loop
        # still exist — they are private API and may change.
        try:
            self.worker.loop = asyncio.get_running_loop()
            self._worker_task = asyncio.create_task(
                self.worker._aio_start()
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Hatchet SDK internals changed — worker could not "
                "start. Check hatchet-sdk version compatibility."
            ) from exc
        self._worker_task.add_done_callback(_on_worker_done)

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    def register_workflows(
        self, workflow: Workflow, service: Any, messages: dict,
    ) -> None:
        self.messages.update(messages)

        logger.info(
            "Registering workflows for %s with messages %s.",
            workflow, messages,
        )

        if workflow == Workflow.INGESTION:
            from core.main.orchestration.hatchet.ingestion_workflow import (
                hatchet_ingestion_factory,
            )
            wfs = hatchet_ingestion_factory(
                self.orchestrator, service, self.config, self,
            )
            self._workflows.update(wfs)

        elif workflow == Workflow.GRAPH:
            from core.main.orchestration.hatchet.graph_workflow import (
                hatchet_graph_search_results_factory,
            )
            wfs = hatchet_graph_search_results_factory(
                self.orchestrator, service, self.config, self,
            )
            self._workflows.update(wfs)

    # ------------------------------------------------------------------
    # Workflow lookup (for child spawning from within tasks)
    # ------------------------------------------------------------------

    def get_workflow(self, name: str) -> Any:
        wf = self._workflows.get(name)
        if wf is None:
            raise ValueError(f"Workflow '{name}' not registered yet.")
        return wf

    # ------------------------------------------------------------------
    # Trigger a workflow run
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        workflow_name: str,
        parameters: dict,
        options: dict,
        *args,
        **kwargs,
    ) -> Any:
        wf = self._workflows.get(workflow_name)
        if wf is None:
            raise ValueError(f"Workflow '{workflow_name}' not found.")

        input_data = dict(parameters.get("request", {}))

        # Hoist user_id for CEL concurrency expressions.  The routers
        # pass the user as a JSON string; the CEL expression needs
        # ``input.user_id`` at the top level.
        if "user" in input_data and "user_id" not in input_data:
            from core.main.orchestration.hatchet.ingestion_workflow import (
                extract_user_id,
            )
            input_data["user_id"] = extract_user_id(input_data)

        from hatchet_sdk import TriggerWorkflowOptions

        trigger_opts = None
        if options and options.get("additional_metadata"):
            trigger_opts = TriggerWorkflowOptions(
                additional_metadata=options["additional_metadata"],
            )

        ref = await wf.aio_run_no_wait(
            input_data, options=trigger_opts,
        )

        task_id = getattr(ref, "workflow_run_id", None) or str(ref)
        return {
            "task_id": str(task_id),
            "message": self.messages.get(
                workflow_name, "Workflow queued successfully."
            ),
        }
