"""
Unit tests for should_skip_graph_extraction function.

Tests cover all branching paths:
- Excel (xlsx/xls) in skip list -> always skip
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

    def test_xlsx_in_skip_list_returns_true(self):
        """xlsx in skip_types should always be skipped."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx", "xls"],
            ingestion_config={},
            document_id="test-doc-1",
        )
        assert result is True

    def test_xlsx_with_parser_overrides_still_skips(self):
        """xlsx in skip_types should be skipped regardless of parser config."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx", "xls"],
            ingestion_config={"parser_overrides": {"xlsx": "semantic"}},
            document_id="test-doc-2",
        )
        assert result is True

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

    def test_irrelevant_config_keys_ignored(self):
        """Irrelevant ingestion_config keys should not affect skip logic."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx"],
            ingestion_config={"other_key": "value"},
            document_id="test-doc-8",
        )
        assert result is True

    def test_xlsx_with_extra_parsers_still_skips(self):
        """xlsx in skip_types should be skipped regardless of extra_parsers."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=["xlsx", "xls"],
            ingestion_config={"extra_parsers": {"xlsx": ["semantic"]}},
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
        assert result is True

    def test_semantic_parser_with_manual_skip_still_skips(self):
        """Manual skip_graph_extraction=True skips even when semantic parser is configured."""
        result = should_skip_graph_extraction(
            doc_type="xlsx",
            skip_types=[],  # No auto-skip for xlsx
            ingestion_config={
                "skip_graph_extraction": True,
                "extra_parsers": {"xlsx": ["semantic"]},
            },
            document_id="test-doc-11",
        )
        assert result is True
