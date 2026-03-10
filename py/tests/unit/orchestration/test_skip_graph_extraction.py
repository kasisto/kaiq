"""
Unit tests for should_skip_graph_extraction function.

Tests cover all branching paths:
- Raw Excel in skip list -> skip
- Semantic Excel in skip list -> don't skip
- PDF not in skip list -> don't skip
- Manual skip override -> skip
- Both conditions -> skip
- Empty skip_types -> don't skip
"""
import pytest

from core.main.orchestration.hatchet.ingestion_workflow import (
    should_skip_graph_extraction,
)


class TestShouldSkipGraphExtraction:
    """Tests for the should_skip_graph_extraction helper function."""

    def test_raw_xlsx_in_skip_list_returns_true(self):
        """Raw xlsx in skip_types should be skipped."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx", "xls"],
            ingestion_config={},
            document_id="test-doc-1",
        )
        assert result is True

    def test_semantic_xlsx_in_skip_list_returns_false(self):
        """Semantic xlsx in skip_types should NOT be skipped."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx", "xls"],
            ingestion_config={"parser_overrides": {"xlsx": "semantic"}},
            document_id="test-doc-2",
        )
        assert result is False

    def test_pdf_not_in_skip_list_returns_false(self):
        """PDF not in skip_types should NOT be skipped."""
        result = should_skip_graph_extraction(
            doc_type="pdf",
            skip_types=["xlsx", "xls"],
            ingestion_config={},
            document_id="test-doc-3",
        )
        assert result is False

    def test_manual_skip_override_returns_true(self):
        """Manual skip_graph_extraction flag should skip."""
        result = should_skip_graph_extraction(
            doc_type="pdf",
            skip_types=[],
            ingestion_config={"skip_graph_extraction": True},
            document_id="test-doc-4",
        )
        assert result is True

    def test_both_conditions_returns_true(self):
        """Both auto-skip and manual override should skip."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={"skip_graph_extraction": True},
            document_id="test-doc-5",
        )
        assert result is True

    def test_empty_skip_types_returns_false(self):
        """Empty skip_types list should NOT skip (unless manual override)."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=[],
            ingestion_config={},
            document_id="test-doc-6",
        )
        assert result is False

    def test_xls_in_skip_list_returns_true(self):
        """Legacy XLS format in skip_types should be skipped."""
        result = should_skip_graph_extraction(
            doc_type="xls",
            skip_types=["xlsx", "xls"],
            ingestion_config={},
            document_id="test-doc-7",
        )
        assert result is True

    def test_missing_parser_overrides_key(self):
        """Missing parser_overrides key should default to empty dict."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={"other_key": "value"},
            document_id="test-doc-8",
        )
        assert result is True

    def test_advanced_parser_override_still_skips(self):
        """Non-semantic parser override should still skip."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={"parser_overrides": {"xlsx": "advanced"}},
            document_id="test-doc-9",
        )
        assert result is True

    def test_manual_skip_false_does_not_override_auto_skip(self):
        """Explicit skip_graph_extraction=False should NOT override auto-skip."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={"skip_graph_extraction": False},
            document_id="test-doc-10",
        )
        # Auto-skip still applies because xlsx is in skip_types and no semantic parser
        assert result is True

    def test_semantic_parser_with_manual_skip_still_skips(self):
        """Manual skip should override semantic parser allowance."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={
                "parser_overrides": {"xlsx": "semantic"},
                "skip_graph_extraction": True,
            },
            document_id="test-doc-11",
        )
        # Manual skip takes precedence
        assert result is True

    def test_semantic_via_extra_parsers_returns_false(self):
        """Semantic xlsx via extra_parsers config should NOT be skipped."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx", "xls"],
            ingestion_config={"extra_parsers": {"xlsx": ["semantic"]}},
            document_id="test-doc-12",
        )
        assert result is False

    def test_non_semantic_extra_parser_still_skips(self):
        """Non-semantic extra_parser should still skip."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={"extra_parsers": {"xlsx": ["advanced"]}},
            document_id="test-doc-13",
        )
        assert result is True
