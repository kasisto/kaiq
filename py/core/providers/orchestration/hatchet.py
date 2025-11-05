# FIXME: Once the Hatchet workflows are type annotated, remove the type: ignore comments
import asyncio
import logging
from typing import Any, Callable, Optional

from core.base import OrchestrationConfig, OrchestrationProvider, Workflow

logger = logging.getLogger()


class HatchetOrchestrationProvider(OrchestrationProvider):
    def __init__(self, config: OrchestrationConfig):
        super().__init__(config)
        try:
            from hatchet_sdk import ClientConfig, Hatchet
        except ImportError:
            raise ImportError(
                "Hatchet SDK not installed. Please install it using `pip install hatchet-sdk`."
            ) from None
        root_logger = logging.getLogger()

        import os
        # Get Hatchet connection details from environment
        # host_port is for gRPC (7070), server_url is for HTTP REST API (8080)
        hatchet_host_port = os.getenv("HATCHET_CLIENT_HOST_PORT", "hatchet-engine:7070")
        hatchet_server_url = os.getenv("HATCHET_CLIENT_SERVER_URL", "http://hatchet-engine:8080")

        self.orchestrator = Hatchet(
            config=ClientConfig(
                logger=root_logger,
                host_port=hatchet_host_port,
                server_url=hatchet_server_url,
            ),
        )
        self.root_logger = root_logger
        self.config: OrchestrationConfig = config
        self.messages: dict[str, str] = {}

    def workflow(self, *args, **kwargs) -> Callable:
        return self.orchestrator.workflow(*args, **kwargs)

    def get_worker(self, name: str, max_runs: Optional[int] = None) -> Any:
        if not max_runs:
            max_runs = self.config.max_runs
        self.worker = self.orchestrator.worker(name, max_runs)  # type: ignore
        return self.worker

    # Concurrency is handled at the workflow level using ConcurrencyExpression

    async def start_worker(self):
        if not self.worker:
            raise ValueError(
                "Worker not initialized. Call get_worker() first."
            )

        # Run worker in a separate thread to avoid event loop conflicts
        # The worker's start() method needs its own event loop
        import threading
        worker_thread = threading.Thread(target=self.worker.start, daemon=True)
        worker_thread.start()

    async def run_workflow(
        self,
        workflow_name: str,
        parameters: dict,
        options: dict,
        *args,
        **kwargs,
    ) -> Any:
        # The parameters are already wrapped by the API routers
        # Use the runs.create method following hatchet-sdk 1.20+ signature:
        # create(workflow_name: str, input: dict, additional_metadata: dict | None = None, priority: int | None = None)
        additional_metadata = options.get("additional_metadata") if options else None
        priority = options.get("priority") if options else None

        workflow_run = self.orchestrator.runs.create(
            workflow_name=workflow_name,
            input=parameters,
            additional_metadata=additional_metadata,
            priority=priority,
        )

        # Extract workflow_run_id from the response (V1WorkflowRun object)
        # The response has a metadata field with an id field containing the UUID
        task_id = str(workflow_run.metadata.id)

        return {
            "task_id": task_id,
            "message": self.messages.get(
                workflow_name, "Workflow queued successfully."
            ),
        }

    def register_workflows(
        self, workflow: Workflow, service: Any, messages: dict
    ) -> None:
        self.messages.update(messages)

        logger.info(
            f"Registering workflows for {workflow} with messages {messages}."
        )
        if workflow == Workflow.INGESTION:
            from core.main.orchestration.hatchet.ingestion_workflow import (  # type: ignore
                hatchet_ingestion_factory,
            )

            workflows = hatchet_ingestion_factory(self.orchestrator, service, self.config)
            if self.worker:
                for wf in workflows:
                    self.worker.register_workflow(wf)

        elif workflow == Workflow.GRAPH:
            from core.main.orchestration.hatchet.graph_workflow import (  # type: ignore
                hatchet_graph_search_results_factory,
            )

            workflows = hatchet_graph_search_results_factory(self.orchestrator, service, self.config)
            if self.worker:
                for wf in workflows:
                    self.worker.register_workflow(wf)
