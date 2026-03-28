import asyncio
import logging
import random
import time
from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

from litellm import AuthenticationError

from core.base.abstractions import VectorQuantizationSettings

from ..abstractions import (
    ChunkSearchResult,
)
from .base import Provider, ProviderConfig

from core.utils.otel_setup import get_meter, get_tenant_context

logger = logging.getLogger()

# ── OTel Metrics (lazy, module-level singletons) ──
_meter = get_meter("r2r.embedding")
# Word count approximation: we use len(text.split()) because we don't have
# access to the model's tokenizer. This under-counts vs real BPE tokens but
# is directionally useful for cost attribution.
_embedding_words = _meter.create_counter(
    "r2r.embedding.words_total",
    unit="{word}",
    description="Approximate word count sent to embedding model (not true tokens)",
)
_embedding_duration = _meter.create_histogram(
    "r2r.embedding.duration_seconds",
    unit="s",
    description="Embedding request duration",
)
_embedding_batch_size = _meter.create_histogram(
    "r2r.embedding.batch_size",
    unit="{text}",
    description="Number of texts per embedding batch request",
)


class EmbeddingConfig(ProviderConfig):
    provider: str
    base_model: str
    base_dimension: int | float
    rerank_model: Optional[str] = None
    rerank_url: Optional[str] = None
    batch_size: int = 1
    concurrent_request_limit: int = 256
    max_retries: int = 3
    initial_backoff: float = 1
    max_backoff: float = 64.0
    quantization_settings: VectorQuantizationSettings = (
        VectorQuantizationSettings()
    )

    def validate_config(self) -> None:
        if self.provider not in self.supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    @property
    def supported_providers(self) -> list[str]:
        return ["litellm", "openai", "ollama"]


class EmbeddingProvider(Provider):
    class Step(Enum):
        BASE = 1
        RERANK = 2

    def __init__(self, config: EmbeddingConfig):
        if not isinstance(config, EmbeddingConfig):
            raise ValueError(
                "EmbeddingProvider must be initialized with a `EmbeddingConfig`."
            )
        logger.info(f"Initializing EmbeddingProvider with config {config}.")

        super().__init__(config)
        self.config: EmbeddingConfig = config
        self.semaphore = asyncio.Semaphore(config.concurrent_request_limit)
        self.current_requests = 0

    def _record_metrics_from_task(
        self, elapsed: float, task: dict[str, Any]
    ) -> None:
        """Record duration, word-count, and batch-size metrics."""
        try:
            # Resolve texts from either "texts" (list) or "text" (single)
            texts: list[str] = task.get(
                "texts", [task["text"]] if "text" in task else []
            )
            word_count = sum(len(t.split()) for t in texts)
            model = getattr(self.config, "base_model", "unknown")

            ctx = get_tenant_context()
            attrs = {
                "org_id": ctx.get("org_id", ""),
                "tenant_id": ctx.get("tenant_id", ""),
                "gen_ai.request.model": model,
            }
            _embedding_duration.record(elapsed, attrs)
            _embedding_words.add(word_count, attrs)
            if len(texts) > 1:
                _embedding_batch_size.record(len(texts), attrs)
        except Exception:
            logger.debug(
                "Failed to record embedding metrics", exc_info=True
            )

    async def _execute_with_backoff_async(self, task: dict[str, Any]):
        retries = 0
        backoff = self.config.initial_backoff
        _start = time.monotonic()
        while retries < self.config.max_retries:
            try:
                async with self.semaphore:
                    result = await self._execute_task(task)
                self._record_metrics_from_task(
                    time.monotonic() - _start, task
                )
                return result
            except AuthenticationError:
                raise
            except Exception as e:
                logger.warning(
                    f"Request failed (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                if retries == self.config.max_retries:
                    raise
                await asyncio.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, self.config.max_backoff)

    def _execute_with_backoff_sync(self, task: dict[str, Any]):
        retries = 0
        backoff = self.config.initial_backoff
        _start = time.monotonic()
        while retries < self.config.max_retries:
            try:
                result = self._execute_task_sync(task)
                self._record_metrics_from_task(
                    time.monotonic() - _start, task
                )
                return result
            except AuthenticationError:
                raise
            except Exception as e:
                logger.warning(
                    f"Request failed (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                if retries == self.config.max_retries:
                    raise
                time.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, self.config.max_backoff)

    @abstractmethod
    async def _execute_task(self, task: dict[str, Any]):
        pass

    @abstractmethod
    def _execute_task_sync(self, task: dict[str, Any]):
        pass

    async def async_get_embedding(
        self,
        text: str,
        stage: Step = Step.BASE,
    ):
        task = {
            "text": text,
            "stage": stage,
        }
        return await self._execute_with_backoff_async(task)

    def get_embedding(
        self,
        text: str,
        stage: Step = Step.BASE,
    ):
        task = {
            "text": text,
            "stage": stage,
        }
        return self._execute_with_backoff_sync(task)

    async def async_get_embeddings(
        self,
        texts: list[str],
        stage: Step = Step.BASE,
    ):
        task = {
            "texts": texts,
            "stage": stage,
        }
        return await self._execute_with_backoff_async(task)

    def get_embeddings(
        self,
        texts: list[str],
        stage: Step = Step.BASE,
    ) -> list[list[float]]:
        task = {
            "texts": texts,
            "stage": stage,
        }
        return self._execute_with_backoff_sync(task)

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[ChunkSearchResult],
        stage: Step = Step.RERANK,
        limit: int = 10,
    ):
        pass

    @abstractmethod
    async def arerank(
        self,
        query: str,
        results: list[ChunkSearchResult],
        stage: Step = Step.RERANK,
        limit: int = 10,
    ):
        pass
