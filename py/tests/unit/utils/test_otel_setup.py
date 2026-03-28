#
# Copyright (c) 2025 Kasisto, Inc.
# All rights reserved.
#
# This source code is proprietary and confidential.
#

"""Tests for OpenTelemetry multi-tenant context setup."""

import logging
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from core.utils.otel_setup import (
    TenantLogFilter,
    TenantSpanProcessor,
    _org_id_var,
    _tenant_id_var,
    _user_id_var,
    get_tenant_context,
    set_tenant_context,
    set_tenant_context_from_metadata,
    get_tracer,
    get_meter,
    NoOpTracer,
    NoOpMeter,
    NoOpSpan,
    setup_opentelemetry,
)


class TestTenantLogFilter:
    """Tests for log record injection."""

    def test_injects_org_tenant_into_log_record(self):
        """Org/tenant from ContextVar appear on every log record."""
        token_org = _org_id_var.set("acme")
        token_tenant = _tenant_id_var.set("tenant-1")
        token_user = _user_id_var.set("user-42")
        try:
            record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
            f = TenantLogFilter()
            f.filter(record)

            assert record.org_id == "acme"
            assert record.tenant_id == "tenant-1"
            assert record.user_id == "user-42"
        finally:
            _org_id_var.reset(token_org)
            _tenant_id_var.reset(token_tenant)
            _user_id_var.reset(token_user)

    def test_defaults_to_empty_string(self):
        """When no ContextVar is set, fields default to empty string."""
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f = TenantLogFilter()
        f.filter(record)

        assert record.org_id == ""
        assert record.tenant_id == ""
        assert record.user_id == ""

    def test_always_returns_true(self):
        """Filter never suppresses log records."""
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert TenantLogFilter().filter(record) is True


class TestTenantSpanProcessor:
    """Tests for span attribute propagation."""

    def test_sets_attributes_on_span(self):
        """Span gets org/tenant/user from ContextVar."""
        token_org = _org_id_var.set("acme")
        token_tenant = _tenant_id_var.set("tenant-1")
        token_user = _user_id_var.set("user-42")
        try:
            span = MagicMock()
            processor = TenantSpanProcessor()
            processor.on_start(span)

            span.set_attribute.assert_any_call("kaigentic.org_id", "acme")
            span.set_attribute.assert_any_call("kaigentic.tenant_id", "tenant-1")
            span.set_attribute.assert_any_call("kaigentic.user_id", "user-42")
        finally:
            _org_id_var.reset(token_org)
            _tenant_id_var.reset(token_tenant)
            _user_id_var.reset(token_user)

    def test_skips_empty_values(self):
        """Does not set attributes for empty ContextVar values."""
        span = MagicMock()
        processor = TenantSpanProcessor()
        processor.on_start(span)

        span.set_attribute.assert_not_called()

    def test_handles_none_span(self):
        """Does not crash on None span."""
        processor = TenantSpanProcessor()
        processor.on_start(None)  # Should not raise

    def test_force_flush_returns_true(self):
        assert TenantSpanProcessor().force_flush() is True


class TestGetTenantContext:
    """Tests for get_tenant_context helper."""

    def test_returns_current_context(self):
        token_org = _org_id_var.set("org-x")
        token_tenant = _tenant_id_var.set("tenant-y")
        token_user = _user_id_var.set("user-z")
        try:
            ctx = get_tenant_context()
            assert ctx == {"org_id": "org-x", "tenant_id": "tenant-y", "user_id": "user-z"}
        finally:
            _org_id_var.reset(token_org)
            _tenant_id_var.reset(token_tenant)
            _user_id_var.reset(token_user)

    def test_returns_defaults_when_unset(self):
        ctx = get_tenant_context()
        assert ctx == {"org_id": "", "tenant_id": "", "user_id": ""}


class TestNoOps:
    """Tests for NoOp implementations when OTel is not installed."""

    def test_noop_tracer_returns_noop_span(self):
        tracer = NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, NoOpSpan)
        span = tracer.start_span("test")
        assert isinstance(span, NoOpSpan)

    def test_noop_span_context_manager(self):
        with NoOpSpan() as span:
            span.set_attribute("key", "value")  # Should not raise
            assert span.is_recording() is False

    def test_noop_meter_returns_noop_instruments(self):
        meter = NoOpMeter()
        counter = meter.create_counter("c")
        counter.add(1)  # Should not raise
        histogram = meter.create_histogram("h")
        histogram.record(1.0)  # Should not raise

    def test_get_tracer_without_otel(self):
        with patch("core.utils.otel_setup.OTEL_AVAILABLE", False):
            tracer = get_tracer("test")
            assert isinstance(tracer, NoOpTracer)

    def test_get_meter_without_otel(self):
        with patch("core.utils.otel_setup.OTEL_AVAILABLE", False):
            meter = get_meter("test")
            assert isinstance(meter, NoOpMeter)


class TestSetupOpentelemetry:
    """Tests for setup_opentelemetry function."""

    def test_installs_log_filter_always(self):
        """Log filter is installed even when OTel is not available."""
        from fastapi import FastAPI
        app = FastAPI()
        root_logger = logging.getLogger()
        initial_filters = len(root_logger.filters)

        with patch("core.utils.otel_setup.OTEL_AVAILABLE", False):
            setup_opentelemetry(app, "test-service")

        assert len(root_logger.filters) > initial_filters
        # Clean up
        root_logger.filters = [f for f in root_logger.filters if not isinstance(f, TenantLogFilter)]

    def test_skips_otel_when_disabled(self):
        """Span processor not registered when OTEL_ENABLED=false, but
        ContextVar middleware is still installed (needed by the log filter)."""
        from fastapi import FastAPI
        app = FastAPI()
        initial_middleware = len(app.user_middleware)

        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            setup_opentelemetry(app, "test-service")

        # Middleware IS added (populates ContextVars for log filter)
        assert len(app.user_middleware) == initial_middleware + 1
        # Clean up log filter
        logging.getLogger().filters = [f for f in logging.getLogger().filters if not isinstance(f, TenantLogFilter)]

    def test_raises_on_invalid_otel_enabled(self):
        """Raises ValueError for invalid OTEL_ENABLED value."""
        from fastapi import FastAPI
        app = FastAPI()

        with (
            patch.dict("os.environ", {"OTEL_ENABLED": "maybe"}),
            patch("core.utils.otel_setup.OTEL_AVAILABLE", True),
        ):
            with pytest.raises(ValueError, match="Invalid OTEL_ENABLED"):
                setup_opentelemetry(app, "test-service")

        # Clean up
        logging.getLogger().filters = [f for f in logging.getLogger().filters if not isinstance(f, TenantLogFilter)]

    def test_registers_span_processor_when_enabled(self):
        """SpanProcessor registered when OTel is enabled."""
        from fastapi import FastAPI
        app = FastAPI()

        mock_provider = MagicMock()
        mock_provider.add_span_processor = MagicMock()

        with (
            patch.dict("os.environ", {"OTEL_ENABLED": "true"}),
            patch("core.utils.otel_setup.OTEL_AVAILABLE", True),
            patch("core.utils.otel_setup.trace") as mock_trace,
        ):
            mock_trace.get_tracer_provider.return_value = mock_provider
            mock_trace.get_current_span.return_value = MagicMock(is_recording=lambda: False)
            setup_opentelemetry(app, "test-service")

        mock_provider.add_span_processor.assert_called_once()
        arg = mock_provider.add_span_processor.call_args[0][0]
        assert isinstance(arg, TenantSpanProcessor)

        # Clean up
        logging.getLogger().filters = [f for f in logging.getLogger().filters if not isinstance(f, TenantLogFilter)]


class TestMiddlewareIntegration:
    """Tests for the tenant context middleware via TestClient."""

    @pytest.fixture
    def app_with_otel(self):
        """Create a FastAPI app with OTel setup and a test endpoint."""
        from fastapi import FastAPI
        app = FastAPI()

        # Manually install just the log filter and ContextVar middleware
        # (skip actual OTel span processor since no real TracerProvider)
        root_logger = logging.getLogger()
        root_logger.addFilter(TenantLogFilter())

        @app.middleware("http")
        async def tenant_middleware(request, call_next):
            from core.utils.otel_setup import _org_id_var, _tenant_id_var, _user_id_var
            org_token = _org_id_var.set(request.headers.get("X-Organization-Id", ""))
            tenant_token = _tenant_id_var.set(request.headers.get("X-Tenant-Id", ""))
            user_token = _user_id_var.set(request.headers.get("X-User-Id", ""))
            try:
                response = await call_next(request)
                return response
            finally:
                _org_id_var.reset(org_token)
                _tenant_id_var.reset(tenant_token)
                _user_id_var.reset(user_token)

        @app.get("/test")
        async def test_endpoint():
            ctx = get_tenant_context()
            return ctx

        yield app
        # Clean up
        root_logger.filters = [f for f in root_logger.filters if not isinstance(f, TenantLogFilter)]

    def test_middleware_sets_context_from_headers(self, app_with_otel):
        """Kong headers are extracted into ContextVars and returned."""
        from fastapi.testclient import TestClient
        client = TestClient(app_with_otel)

        response = client.get("/test", headers={
            "X-Organization-Id": "acme",
            "X-Tenant-Id": "tenant-1",
            "X-User-Id": "user-42",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] == "acme"
        assert data["tenant_id"] == "tenant-1"
        assert data["user_id"] == "user-42"

    def test_middleware_resets_context_between_requests(self, app_with_otel):
        """ContextVars are reset after each request — no leaking."""
        from fastapi.testclient import TestClient
        client = TestClient(app_with_otel)

        # First request with headers
        client.get("/test", headers={"X-Organization-Id": "acme"})

        # Second request WITHOUT headers
        response = client.get("/test")
        data = response.json()
        assert data["org_id"] == ""
        assert data["tenant_id"] == ""
        assert data["user_id"] == ""

    def test_middleware_defaults_to_empty_without_headers(self, app_with_otel):
        """No Kong headers = empty context."""
        from fastapi.testclient import TestClient
        client = TestClient(app_with_otel)

        response = client.get("/test")
        data = response.json()
        assert data == {"org_id": "", "tenant_id": "", "user_id": ""}


class TestSetTenantContext:
    """Tests for set_tenant_context helper (Hatchet worker support)."""

    def test_sets_all_fields(self):
        """All three ContextVars are set."""
        set_tenant_context(org_id="org-1", tenant_id="t-2", user_id="u-3")
        try:
            ctx = get_tenant_context()
            assert ctx == {
                "org_id": "org-1",
                "tenant_id": "t-2",
                "user_id": "u-3",
            }
        finally:
            # Reset to defaults
            set_tenant_context()

    def test_defaults_to_empty_strings(self):
        """Calling with no args sets empty strings."""
        set_tenant_context(org_id="org-1", tenant_id="t-2", user_id="u-3")
        set_tenant_context()
        ctx = get_tenant_context()
        assert ctx == {"org_id": "", "tenant_id": "", "user_id": ""}

    def test_partial_set(self):
        """Only specified fields are changed, others default to empty."""
        set_tenant_context(org_id="org-only")
        try:
            ctx = get_tenant_context()
            assert ctx["org_id"] == "org-only"
            assert ctx["tenant_id"] == ""
            assert ctx["user_id"] == ""
        finally:
            set_tenant_context()


class TestSetTenantContextFromMetadata:
    """Tests for set_tenant_context_from_metadata (document metadata)."""

    def test_extracts_all_fields(self):
        """org_id, tenant_id, and owner are extracted from metadata."""
        metadata = {
            "org_id": "acme-corp",
            "tenant_id": "tenant-42",
            "owner": "user-99",
        }
        set_tenant_context_from_metadata(metadata)
        try:
            ctx = get_tenant_context()
            assert ctx == {
                "org_id": "acme-corp",
                "tenant_id": "tenant-42",
                "user_id": "user-99",
            }
        finally:
            set_tenant_context()

    def test_empty_metadata(self):
        """Empty dict results in empty context strings."""
        set_tenant_context_from_metadata({})
        ctx = get_tenant_context()
        assert ctx == {"org_id": "", "tenant_id": "", "user_id": ""}

    def test_missing_keys_default_to_empty(self):
        """Keys not present in metadata default to empty string."""
        set_tenant_context_from_metadata({"org_id": "org-x"})
        try:
            ctx = get_tenant_context()
            assert ctx["org_id"] == "org-x"
            assert ctx["tenant_id"] == ""
            assert ctx["user_id"] == ""
        finally:
            set_tenant_context()

    def test_non_string_values_are_coerced(self):
        """Non-string values (e.g. UUID) are coerced to string."""
        import uuid
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        metadata = {"org_id": uid, "tenant_id": 123}
        set_tenant_context_from_metadata(metadata)
        try:
            ctx = get_tenant_context()
            assert ctx["org_id"] == str(uid)
            assert ctx["tenant_id"] == "123"
        finally:
            set_tenant_context()
