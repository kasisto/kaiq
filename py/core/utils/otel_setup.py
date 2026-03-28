"""
OpenTelemetry Multi-Tenant Context for R2R

Adds organization/tenant context to all traces, spans, and logs so that
telemetry data can be filtered per-tenant in Grafana/Jaeger/Loki.

Three mechanisms:
1. HTTP middleware: extracts org/tenant from Kong headers into ContextVars
2. TenantSpanProcessor: propagates org/tenant attributes to ALL child spans
3. TenantLogFilter: injects org/tenant into every log record via ContextVar

Auto-instrumentation (via opentelemetry-instrument CLI) handles:
    - TracerProvider and MeterProvider creation
    - FastAPI, database, HTTP client instrumentation
    - Automatic metric generation

Usage:
    from core.utils.otel_setup import setup_opentelemetry

    app = FastAPI()
    setup_opentelemetry(app, "r2r")
"""

import logging
import os
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, Request

# OpenTelemetry is optional - gracefully degrade when not installed (e.g., in tests)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import SpanProcessor
    OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore[assignment]
    SpanProcessor = object  # type: ignore[assignment,misc]
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Tenant ContextVars (shared between middleware, logs, and spans) ──

_org_id_var: ContextVar[str] = ContextVar("org_id", default="")
_tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")
_user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def get_tenant_context() -> dict[str, str]:
    """Get current tenant context from ContextVars."""
    return {
        "org_id": _org_id_var.get(),
        "tenant_id": _tenant_id_var.get(),
        "user_id": _user_id_var.get(),
    }


def set_tenant_context(
    org_id: str = "", tenant_id: str = "", user_id: str = "",
) -> None:
    """Manually set tenant ContextVars (for non-HTTP contexts like Hatchet workers)."""
    _org_id_var.set(org_id)
    _tenant_id_var.set(tenant_id)
    _user_id_var.set(user_id)


def set_tenant_context_from_metadata(metadata: dict) -> None:
    """Extract org/tenant from document metadata and set ContextVars.

    Kaigentic's knowledge-service injects org_id and tenant_id into document
    metadata. This function reads them and populates the ContextVars so that
    metrics recorded in Hatchet worker steps have tenant attribution.
    """
    set_tenant_context(
        org_id=str(metadata.get("org_id", "")),
        tenant_id=str(metadata.get("tenant_id", "")),
        user_id=str(metadata.get("owner", "")),
    )


# ── Log Filter (injects org/tenant into every log record) ──

class TenantLogFilter(logging.Filter):
    """Injects org_id, tenant_id, user_id into every log record.

    Works with any log formatter that references %(org_id)s, %(tenant_id)s,
    %(user_id)s. Also works with JSON/structured loggers that read record
    attributes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.org_id = _org_id_var.get()  # type: ignore[attr-defined]
        record.tenant_id = _tenant_id_var.get()  # type: ignore[attr-defined]
        record.user_id = _user_id_var.get()  # type: ignore[attr-defined]
        return True


# ── SpanProcessor (propagates tenant attrs to ALL child spans) ──

class TenantSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Copies org/tenant from ContextVar to every span on start.

    This ensures ALL spans (not just the root HTTP span) carry tenant
    attributes, making them independently queryable in Grafana/Jaeger.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if span is None or not hasattr(span, "set_attribute"):
            return
        org_id = _org_id_var.get()
        tenant_id = _tenant_id_var.get()
        user_id = _user_id_var.get()
        if org_id:
            span.set_attribute("kaigentic.org_id", org_id)
        if tenant_id:
            span.set_attribute("kaigentic.tenant_id", tenant_id)
        if user_id:
            span.set_attribute("kaigentic.user_id", user_id)

    def on_end(self, span: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ── NoOp implementations (when OTel not installed) ──

class NoOpSpan:
    """A no-op span that does nothing but provides the span interface."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NoOpTracer:
    """A no-op tracer that returns NoOpSpan instances."""

    def start_as_current_span(self, name: str, **kwargs):
        return NoOpSpan()

    def start_span(self, name: str, **kwargs):
        return NoOpSpan()


class NoOpMeter:
    """A no-op meter that returns no-op instruments."""

    def create_counter(self, name: str, **kwargs):
        return NoOpCounter()

    def create_histogram(self, name: str, **kwargs):
        return NoOpHistogram()

    def create_up_down_counter(self, name: str, **kwargs):
        return NoOpCounter()

    def create_observable_gauge(self, name: str, **kwargs):
        return NoOpGauge()


class NoOpGauge:
    """A no-op observable gauge that does nothing.

    Observable gauges use callbacks (not direct record calls), so this
    no-op class needs no methods -- the callback is simply never invoked.
    """
    pass


class NoOpCounter:
    """A no-op counter that does nothing."""

    def add(self, amount: Any, attributes: Any = None) -> None:
        pass


class NoOpHistogram:
    """A no-op histogram that does nothing."""

    def record(self, value: Any, attributes: Any = None) -> None:
        pass


# ── Setup ──

def setup_opentelemetry(
    app: FastAPI,
    service_name: str,
) -> None:
    """
    Add OpenTelemetry tenant context to traces, spans, and logs.

    1. Installs HTTP middleware that extracts org/tenant from Kong headers
       into ContextVars and the root span.
    2. Registers a TenantSpanProcessor that propagates tenant attrs to all
       child spans (DB queries, LLM calls, embeddings, etc.).
    3. Installs a TenantLogFilter that injects org/tenant into all log records.

    NOTE: This assumes auto-instrumentation is enabled via opentelemetry-instrument CLI.
    All provider setup, exporters, and instrumentation are handled automatically.

    Args:
        app: FastAPI application instance
        service_name: Service name (e.g., "r2r")
    """
    # Always install the log filter (works even without OTel)
    root_logger = logging.getLogger()
    if not any(isinstance(f, TenantLogFilter) for f in root_logger.filters):
        root_logger.addFilter(TenantLogFilter())
        logger.info("Tenant log filter installed for %s", service_name)

    # Determine whether OTel tracing is active
    otel_tracing_active = False
    if OTEL_AVAILABLE:
        otel_enabled = os.getenv("OTEL_ENABLED", "false").lower()
        if otel_enabled == "true":
            otel_tracing_active = True
        elif otel_enabled != "false":
            raise ValueError(
                f"Invalid OTEL_ENABLED value: '{otel_enabled}'. "
                "Must be 'true' or 'false'."
            )

    # Register span processor only when OTel tracing is active
    if otel_tracing_active:
        logger.info("Setting up OpenTelemetry tenant context for %s", service_name)
        try:
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "add_span_processor"):
                tracer_provider.add_span_processor(TenantSpanProcessor())
                logger.info("TenantSpanProcessor registered")
        except Exception:
            logger.debug("Could not register TenantSpanProcessor", exc_info=True)
    else:
        logger.info("OpenTelemetry tracing not active for %s", service_name)

    # Always add middleware to populate ContextVars from Kong headers.
    # The log filter needs populated ContextVars even without OTel tracing.
    @app.middleware("http")
    async def add_tenant_context(request: Request, call_next):
        """
        Extract tenant context from Kong-injected headers, set ContextVars,
        and optionally add to the current trace span.
        """
        # Set ContextVars (used by log filter and span processor)
        org_id = request.headers.get("X-Organization-Id", "")
        tenant_id = request.headers.get("X-Tenant-Id", "")
        user_id = request.headers.get("X-User-Id", "")

        org_token = _org_id_var.set(org_id)
        tenant_token = _tenant_id_var.set(tenant_id)
        user_token = _user_id_var.set(user_id)

        try:
            # Explicitly set attributes on the root span because the
            # TenantSpanProcessor fires on_start *before* the middleware has
            # populated the ContextVars.  Without this, the root HTTP span
            # would be missing tenant attributes even though all child spans
            # (created later) pick them up via the SpanProcessor.
            if otel_tracing_active:
                span = trace.get_current_span()
                if span.is_recording():
                    if org_id:
                        span.set_attribute("kaigentic.org_id", org_id)
                    if tenant_id:
                        span.set_attribute("kaigentic.tenant_id", tenant_id)
                    if user_id:
                        span.set_attribute("kaigentic.user_id", user_id)

            response = await call_next(request)
            return response
        finally:
            # Reset ContextVars to prevent leaking between requests
            _org_id_var.reset(org_token)
            _tenant_id_var.reset(tenant_token)
            _user_id_var.reset(user_token)

    logger.info("Tenant context middleware installed for %s", service_name)


def get_tracer(name: str):
    """Get a tracer for manual span creation."""
    if not OTEL_AVAILABLE:
        return NoOpTracer()
    return trace.get_tracer(name)


def get_meter(name: str):
    """Get a meter for custom metrics creation."""
    if not OTEL_AVAILABLE:
        return NoOpMeter()
    from opentelemetry import metrics
    return metrics.get_meter(name)
