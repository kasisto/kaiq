"""
OpenTelemetry Multi-Tenant Context Middleware for Kaigentic Services

This module adds multi-tenant context to OpenTelemetry traces when using auto-instrumentation.

Auto-instrumentation (via opentelemetry-instrument CLI) handles:
    - TracerProvider and MeterProvider creation
    - FastAPI instrumentation
    - Database client instrumentation
    - HTTP client instrumentation
    - Automatic metric generation

This module only adds:
    - Custom middleware to inject org_id/tenant_id from Kong headers into spans

Usage:
    from core.utils.otel_setup import setup_opentelemetry

    app = FastAPI()
    setup_opentelemetry(app, "r2r")
"""

import logging
import os

from fastapi import FastAPI, Request
from opentelemetry import trace

logger = logging.getLogger(__name__)


def setup_opentelemetry(
    app: FastAPI,
    service_name: str,
) -> None:
    """
    Add OpenTelemetry tenant context middleware for multi-tenant tracing.

    NOTE: This assumes auto-instrumentation is enabled via opentelemetry-instrument CLI.
    All provider setup, exporters, and instrumentation are handled automatically.

    Args:
        app: FastAPI application instance
        service_name: Service name (e.g., "r2r")

    Environment Variables:
        OTEL_ENABLED: Enable/disable ("true" or "false", default: true)

    Auto-instrumentation environment variables (handled by opentelemetry-instrument):
        OTEL_SERVICE_NAME: Service name
        OTEL_SERVICE_VERSION: Service version
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Tempo endpoint (gRPC)
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: OpenTelemetry Collector endpoint (gRPC)
        OTEL_METRIC_EXPORT_INTERVAL: Metric export interval in ms
        OTEL_TRACES_SAMPLER: Sampler type
        OTEL_TRACES_SAMPLER_ARG: Sampling rate
        ENVIRONMENT: Deployment environment
    """
    # Check if OpenTelemetry is enabled
    otel_enabled = os.getenv("OTEL_ENABLED", "true").lower()
    if otel_enabled == "false":
        logger.info(f"OpenTelemetry disabled for {service_name}")
        return

    if otel_enabled != "true":
        raise ValueError(f"Invalid OTEL_ENABLED value: '{otel_enabled}'. Must be 'true' or 'false'.")

    logger.info(
        f"Setting up OpenTelemetry tenant context middleware for {service_name}. "
        f"Providers and instrumentation handled by auto-instrumentation."
    )

    # Add middleware to extract tenant context from Kong headers
    @app.middleware("http")
    async def add_tenant_context_to_traces(request: Request, call_next):
        """
        Extract tenant context from Kong-injected headers and add to trace span attributes.
        This ensures all metrics and traces are tagged with tenant information.
        """
        span = trace.get_current_span()

        if span.is_recording():
            # Extract Kong headers
            org_id = request.headers.get("X-Organization-Id")
            tenant_id = request.headers.get("X-Tenant-Id")
            user_id = request.headers.get("X-User-Id")
            user_roles = request.headers.get("X-User-Roles", "")

            # Add tenant context as span attributes
            if org_id:
                span.set_attribute("org_id", org_id)
                span.set_attribute("kaigentic.org_id", org_id)

            if tenant_id:
                span.set_attribute("tenant_id", tenant_id)
                span.set_attribute("kaigentic.tenant_id", tenant_id)

            if user_id:
                span.set_attribute("user_id", user_id)
                span.set_attribute("kaigentic.user_id", user_id)

            if user_roles:
                span.set_attribute("user_roles", user_roles)
                span.set_attribute("kaigentic.user_roles", user_roles)

            # Add service-specific attributes
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("http.method", request.method)

        response = await call_next(request)
        return response

    logger.info(f"OpenTelemetry tenant context middleware configured for {service_name}")


def get_tracer(name: str):
    """
    Get a tracer for manual span creation.

    Args:
        name: Name of the tracer (typically __name__)

    Returns:
        OpenTelemetry tracer instance

    Example:
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("custom.attribute", "value")
            # Do work
    """
    return trace.get_tracer(name)


def get_meter(name: str):
    """
    Get a meter for custom metrics creation.

    Args:
        name: Name of the meter (typically __name__)

    Returns:
        OpenTelemetry meter instance

    Example:
        from opentelemetry import metrics

        meter = get_meter(__name__)
        counter = meter.create_counter("my_counter")
        counter.add(1, {"key": "value"})
    """
    from opentelemetry import metrics
    return metrics.get_meter(name)
