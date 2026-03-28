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
    get_tracer,
    get_meter,
    NoOpTracer,
    NoOpMeter,
    NoOpSpan,
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
