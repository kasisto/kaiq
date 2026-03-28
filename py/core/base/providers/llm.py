import asyncio
import logging
import random
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Generator, Optional

from litellm import AuthenticationError

from core.base.abstractions import (
    GenerationConfig,
    LLMChatCompletion,
    LLMChatCompletionChunk,
)

from .base import Provider, ProviderConfig

from core.utils.otel_setup import (
    classify_error,
    create_duration_histogram,
    create_histogram_with_buckets,
    get_meter,
    get_tenant_context,
)

logger = logging.getLogger()

# ── OTel Metrics (lazy, module-level singletons) ──
_meter = get_meter("r2r.llm")
_llm_tokens = _meter.create_counter(
    "r2r.llm.tokens_total",
    unit="{token}",
    description="Total LLM tokens consumed",
)
_llm_duration = create_duration_histogram(
    _meter,
    "r2r.llm.duration_seconds",
    description="LLM completion request duration",
)
_llm_retries = _meter.create_counter(
    "r2r.llm.retries_total",
    unit="{retry}",
    description="Total LLM request retries",
)
_llm_failures = _meter.create_counter(
    "r2r.llm.failures_total",
    unit="{failure}",
    description="Total LLM request failures by error type",
)
_llm_concurrent = _meter.create_up_down_counter(
    "r2r.llm.concurrent_requests",
    unit="{request}",
    description="Current number of in-flight LLM requests",
)
_llm_ttft = create_histogram_with_buckets(
    _meter,
    "r2r.llm.time_to_first_token_seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    unit="s",
    description="Time from request start to first content token (streaming)",
)


def _infer_gen_ai_system(model: str) -> str:
    """Infer the gen_ai.system attribute from the model name.

    Follows OpenTelemetry GenAI semantic conventions. The value indicates
    which upstream provider is serving the request.
    """
    model_lower = model.lower()
    if model_lower.startswith("anthropic/") or "claude" in model_lower:
        return "anthropic"
    if model_lower.startswith("openai/") or model_lower.startswith("gpt"):
        return "openai"
    # Default: litellm is the routing layer for all other models
    return "litellm"


def _build_llm_attrs(
    model: str,
    ctx: dict[str, str],
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build standard LLM metric attributes."""
    attrs = {
        "org_id": ctx.get("org_id", ""),
        "tenant_id": ctx.get("tenant_id", ""),
        "gen_ai.request.model": model,
        "gen_ai.system": _infer_gen_ai_system(model),
    }
    if extra:
        attrs.update(extra)
    return attrs


class CompletionConfig(ProviderConfig):
    provider: Optional[str] = None
    generation_config: Optional[GenerationConfig] = None
    concurrent_request_limit: int = 256
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 64.0
    request_timeout: float = 15.0

    def validate_config(self) -> None:
        if not self.provider:
            raise ValueError("Provider must be set.")
        if self.provider not in self.supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    @property
    def supported_providers(self) -> list[str]:
        return ["anthropic", "litellm", "openai", "r2r"]


class CompletionProvider(Provider):
    def __init__(self, config: CompletionConfig) -> None:
        if not isinstance(config, CompletionConfig):
            raise ValueError(
                "CompletionProvider must be initialized with a `CompletionConfig`."
            )
        logger.info(f"Initializing CompletionProvider with config: {config}")
        super().__init__(config)
        self.config: CompletionConfig = config
        self.semaphore = asyncio.Semaphore(config.concurrent_request_limit)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.concurrent_request_limit
        )

    async def _execute_with_backoff_async(
        self,
        task: dict[str, Any],
        apply_timeout: bool = False,
    ):
        retries = 0
        backoff = self.config.initial_backoff
        while retries < self.config.max_retries:
            try:
                # Track concurrency: increment before acquire, decrement
                # after release (in finally).
                try:
                    _llm_concurrent.add(1)
                except Exception:
                    pass
                try:
                    # A semaphore allows us to limit concurrent requests
                    async with self.semaphore:
                        if not apply_timeout:
                            return await self._execute_task(task)

                        try:  # Use asyncio.wait_for to set a timeout for the request
                            return await asyncio.wait_for(
                                self._execute_task(task),
                                timeout=self.config.request_timeout,
                            )
                        except asyncio.TimeoutError as e:
                            raise TimeoutError(
                                f"Request timed out after {self.config.request_timeout} seconds"
                            ) from e
                finally:
                    try:
                        _llm_concurrent.add(-1)
                    except Exception:
                        pass
            except AuthenticationError:
                raise
            except Exception as e:
                logger.warning(
                    f"Request failed (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                # Record retry metric
                try:
                    ctx = get_tenant_context()
                    _model = (
                        task.get("generation_config")
                        and task["generation_config"].model
                    ) or "unknown"
                    _llm_retries.add(
                        1,
                        _build_llm_attrs(_model, ctx),
                    )
                except Exception:
                    pass
                if retries == self.config.max_retries:
                    # Record failure with error type
                    try:
                        ctx = get_tenant_context()
                        _model = (
                            task.get("generation_config")
                            and task["generation_config"].model
                        ) or "unknown"
                        _llm_failures.add(
                            1,
                            _build_llm_attrs(
                                _model, ctx,
                                {"error_type": classify_error(e)},
                            ),
                        )
                    except Exception:
                        pass
                    raise
                await asyncio.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, self.config.max_backoff)

    async def _execute_with_backoff_async_stream(
        self, task: dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        retries = 0
        backoff = self.config.initial_backoff
        while retries < self.config.max_retries:
            try:
                try:
                    _llm_concurrent.add(1)
                except Exception:
                    pass
                try:
                    async with self.semaphore:
                        async for chunk in await self._execute_task(task):
                            yield chunk
                    return  # Successful completion of the stream
                finally:
                    try:
                        _llm_concurrent.add(-1)
                    except Exception:
                        pass
            except AuthenticationError:
                raise
            except Exception as e:
                logger.warning(
                    f"Streaming request failed (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                try:
                    ctx = get_tenant_context()
                    _model = (
                        task.get("generation_config")
                        and task["generation_config"].model
                    ) or "unknown"
                    _llm_retries.add(
                        1,
                        _build_llm_attrs(_model, ctx),
                    )
                except Exception:
                    pass
                if retries == self.config.max_retries:
                    try:
                        ctx = get_tenant_context()
                        _model = (
                            task.get("generation_config")
                            and task["generation_config"].model
                        ) or "unknown"
                        _llm_failures.add(
                            1,
                            _build_llm_attrs(
                                _model, ctx,
                                {"error_type": classify_error(e)},
                            ),
                        )
                    except Exception:
                        pass
                    raise
                await asyncio.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, self.config.max_backoff)

    def _execute_with_backoff_sync(
        self,
        task: dict[str, Any],
        apply_timeout: bool = False,
    ):
        retries = 0
        backoff = self.config.initial_backoff
        while retries < self.config.max_retries:
            if not apply_timeout:
                return self._execute_task_sync(task)

            try:
                try:
                    _llm_concurrent.add(1)
                except Exception:
                    pass
                try:
                    future = self.thread_pool.submit(self._execute_task_sync, task)
                    return future.result(timeout=self.config.request_timeout)
                finally:
                    try:
                        _llm_concurrent.add(-1)
                    except Exception:
                        pass
            except TimeoutError as e:
                raise TimeoutError(
                    f"Request timed out after {self.config.request_timeout} seconds"
                ) from e
            except Exception as e:
                logger.warning(
                    f"Request failed (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                try:
                    ctx = get_tenant_context()
                    _model = (
                        task.get("generation_config")
                        and task["generation_config"].model
                    ) or "unknown"
                    _llm_retries.add(
                        1,
                        _build_llm_attrs(_model, ctx),
                    )
                except Exception:
                    pass
                if retries == self.config.max_retries:
                    try:
                        ctx = get_tenant_context()
                        _model = (
                            task.get("generation_config")
                            and task["generation_config"].model
                        ) or "unknown"
                        _llm_failures.add(
                            1,
                            _build_llm_attrs(
                                _model, ctx,
                                {"error_type": classify_error(e)},
                            ),
                        )
                    except Exception:
                        pass
                    raise
                time.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, self.config.max_backoff)

    def _execute_with_backoff_sync_stream(
        self, task: dict[str, Any]
    ) -> Generator[Any, None, None]:
        retries = 0
        backoff = self.config.initial_backoff
        while retries < self.config.max_retries:
            try:
                try:
                    _llm_concurrent.add(1)
                except Exception:
                    pass
                try:
                    yield from self._execute_task_sync(task)
                    return  # Successful completion of the stream
                finally:
                    try:
                        _llm_concurrent.add(-1)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(
                    f"Streaming request failed (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                try:
                    ctx = get_tenant_context()
                    _model = (
                        task.get("generation_config")
                        and task["generation_config"].model
                    ) or "unknown"
                    _llm_retries.add(
                        1,
                        _build_llm_attrs(_model, ctx),
                    )
                except Exception:
                    pass
                if retries == self.config.max_retries:
                    try:
                        ctx = get_tenant_context()
                        _model = (
                            task.get("generation_config")
                            and task["generation_config"].model
                        ) or "unknown"
                        _llm_failures.add(
                            1,
                            _build_llm_attrs(
                                _model, ctx,
                                {"error_type": classify_error(e)},
                            ),
                        )
                    except Exception:
                        pass
                    raise
                time.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, self.config.max_backoff)

    @abstractmethod
    async def _execute_task(self, task: dict[str, Any]):
        pass

    @abstractmethod
    def _execute_task_sync(self, task: dict[str, Any]):
        pass

    async def aget_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        apply_timeout: bool = False,
        **kwargs,
    ) -> LLMChatCompletion:
        _start = time.monotonic()
        task = {
            "messages": messages,
            "generation_config": generation_config,
            "kwargs": kwargs,
        }
        response = await self._execute_with_backoff_async(
            task=task, apply_timeout=apply_timeout
        )
        completion = LLMChatCompletion(**response.dict())

        # Record LLM metrics
        try:
            _elapsed = time.monotonic() - _start
            ctx = get_tenant_context()
            _model = generation_config.model or "unknown"
            _base_attrs = _build_llm_attrs(_model, ctx)
            _llm_duration.record(_elapsed, _base_attrs)
            if completion.usage:
                if completion.usage.prompt_tokens is not None:
                    _llm_tokens.add(
                        completion.usage.prompt_tokens,
                        {**_base_attrs, "token_type": "prompt"},
                    )
                if completion.usage.completion_tokens is not None:
                    _llm_tokens.add(
                        completion.usage.completion_tokens,
                        {**_base_attrs, "token_type": "completion"},
                    )
                # Cached token tracking
                try:
                    _cached_read = 0
                    _cached_creation = 0
                    # OpenAI: prompt_tokens_details.cached_tokens
                    _details = getattr(
                        completion.usage,
                        "prompt_tokens_details",
                        None,
                    )
                    if _details:
                        _ct = getattr(_details, "cached_tokens", 0)
                        if _ct:
                            _cached_read = _ct
                    # Anthropic: cache_read_input_tokens
                    _cr = getattr(
                        completion.usage,
                        "cache_read_input_tokens",
                        0,
                    )
                    if _cr:
                        _cached_read = _cr
                    # Anthropic: cache_creation_input_tokens
                    _cc = getattr(
                        completion.usage,
                        "cache_creation_input_tokens",
                        0,
                    )
                    if _cc:
                        _cached_creation = _cc
                    if _cached_read:
                        _llm_tokens.add(
                            _cached_read,
                            {**_base_attrs, "token_type": "cached_read"},
                        )
                    if _cached_creation:
                        _llm_tokens.add(
                            _cached_creation,
                            {
                                **_base_attrs,
                                "token_type": "cached_creation",
                            },
                        )
                except Exception:
                    pass
        except Exception:
            logger.debug(
                "Failed to record LLM metrics", exc_info=True
            )

        return completion

    async def aget_completion_stream(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> AsyncGenerator[LLMChatCompletionChunk, None]:
        _start = time.monotonic()
        _total_prompt = 0
        _total_completion = 0
        _first_token_recorded = False
        _cached_read_tokens = 0
        _cached_creation_tokens = 0

        generation_config.stream = True
        task = {
            "messages": messages,
            "generation_config": generation_config,
            "kwargs": kwargs,
        }
        async for chunk in self._execute_with_backoff_async_stream(task):
            # Track usage from chunks (many providers include it in
            # the final chunk when stream_options={"include_usage": True})
            if hasattr(chunk, "usage") and chunk.usage:
                if (
                    hasattr(chunk.usage, "prompt_tokens")
                    and chunk.usage.prompt_tokens is not None
                ):
                    _total_prompt = chunk.usage.prompt_tokens
                if (
                    hasattr(chunk.usage, "completion_tokens")
                    and chunk.usage.completion_tokens is not None
                ):
                    _total_completion = chunk.usage.completion_tokens
                # Cached token tracking (final chunk)
                try:
                    _details = getattr(
                        chunk.usage, "prompt_tokens_details", None
                    )
                    if _details:
                        _ct = getattr(_details, "cached_tokens", 0)
                        if _ct:
                            _cached_read_tokens = _ct
                    _cr = getattr(
                        chunk.usage, "cache_read_input_tokens", 0
                    )
                    if _cr:
                        _cached_read_tokens = _cr
                    _cc = getattr(
                        chunk.usage, "cache_creation_input_tokens", 0
                    )
                    if _cc:
                        _cached_creation_tokens = _cc
                except Exception:
                    pass

            # TTFT: record time to first content token
            if not _first_token_recorded:
                try:
                    _has_content = False
                    if isinstance(chunk, dict):
                        _has_content = bool(
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                    elif (
                        hasattr(chunk, "choices")
                        and chunk.choices
                        and len(chunk.choices) > 0
                    ):
                        _delta = getattr(
                            chunk.choices[0], "delta", None
                        )
                        if _delta and getattr(_delta, "content", None):
                            _has_content = True
                    if _has_content:
                        _ttft = time.monotonic() - _start
                        ctx = get_tenant_context()
                        _model = generation_config.model or "unknown"
                        _llm_ttft.record(
                            _ttft,
                            _build_llm_attrs(_model, ctx),
                        )
                        _first_token_recorded = True
                except Exception:
                    _first_token_recorded = True

            if isinstance(chunk, dict):
                yield LLMChatCompletionChunk(**chunk)
                continue

            if chunk.choices and len(chunk.choices) > 0:
                chunk.choices[0].finish_reason = (
                    chunk.choices[0].finish_reason
                    if chunk.choices[0].finish_reason != ""
                    else None
                )  # handle error output conventions
                chunk.choices[0].finish_reason = (
                    chunk.choices[0].finish_reason
                    if chunk.choices[0].finish_reason != "eos"
                    else "stop"
                )  # hardcode `eos` to `stop` for consistency
                try:
                    yield LLMChatCompletionChunk(**(chunk.dict()))
                except Exception as e:
                    logger.error(f"Error parsing chunk: {e}")
                    yield LLMChatCompletionChunk(**(chunk.as_dict()))

        # Record metrics after stream is exhausted
        try:
            _elapsed = time.monotonic() - _start
            ctx = get_tenant_context()
            _model = generation_config.model or "unknown"
            _base_attrs = _build_llm_attrs(_model, ctx)
            _llm_duration.record(_elapsed, _base_attrs)
            if _total_prompt:
                _llm_tokens.add(
                    _total_prompt,
                    {**_base_attrs, "token_type": "prompt"},
                )
            if _total_completion:
                _llm_tokens.add(
                    _total_completion,
                    {**_base_attrs, "token_type": "completion"},
                )
            if _cached_read_tokens:
                _llm_tokens.add(
                    _cached_read_tokens,
                    {**_base_attrs, "token_type": "cached_read"},
                )
            if _cached_creation_tokens:
                _llm_tokens.add(
                    _cached_creation_tokens,
                    {**_base_attrs, "token_type": "cached_creation"},
                )
        except Exception:
            logger.debug(
                "Failed to record streaming LLM metrics",
                exc_info=True,
            )

    def get_completion_stream(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> Generator[LLMChatCompletionChunk, None, None]:
        # Tier 2 #6: Sync streaming with duration + token tracking
        _start = time.monotonic()
        _total_prompt = 0
        _total_completion = 0

        generation_config.stream = True
        task = {
            "messages": messages,
            "generation_config": generation_config,
            "kwargs": kwargs,
        }
        for chunk in self._execute_with_backoff_sync_stream(task):
            # Track usage from chunks if present
            if hasattr(chunk, "usage") and chunk.usage:
                if (
                    hasattr(chunk.usage, "prompt_tokens")
                    and chunk.usage.prompt_tokens is not None
                ):
                    _total_prompt = chunk.usage.prompt_tokens
                if (
                    hasattr(chunk.usage, "completion_tokens")
                    and chunk.usage.completion_tokens is not None
                ):
                    _total_completion = chunk.usage.completion_tokens

            yield LLMChatCompletionChunk(**chunk.dict())

        # Record metrics after sync stream is exhausted
        try:
            _elapsed = time.monotonic() - _start
            ctx = get_tenant_context()
            _model = generation_config.model or "unknown"
            _base_attrs = _build_llm_attrs(_model, ctx)
            _llm_duration.record(_elapsed, _base_attrs)
            if _total_prompt:
                _llm_tokens.add(
                    _total_prompt,
                    {**_base_attrs, "token_type": "prompt"},
                )
            if _total_completion:
                _llm_tokens.add(
                    _total_completion,
                    {**_base_attrs, "token_type": "completion"},
                )
        except Exception:
            logger.debug(
                "Failed to record sync streaming LLM metrics",
                exc_info=True,
            )
