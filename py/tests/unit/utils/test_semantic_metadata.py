"""Unit tests for semantic metadata parsing utility."""
import base64
import json

import pytest

from core.utils.semantic_metadata import parse_semantic_metadata


class TestParseSemanticMetadata:
    """Tests for parse_semantic_metadata function."""

    def test_parse_valid_base64_metadata(self):
        """Test parsing valid base64-encoded metadata."""
        metadata = {
            "use_page_content": True,
            "page_content": "Raw sheet data here",
            "page_title": "Sales Report",
            "chunker_type": "xlsx_semantic",
        }
        metadata_b64 = base64.b64encode(
            json.dumps(metadata).encode("utf-8")
        ).decode("ascii")

        text = f"Semantic description here\n\n[XLSX_SEMANTIC_METADATA_B64]{metadata_b64}[/XLSX_SEMANTIC_METADATA_B64]"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        assert clean_text == "Semantic description here"
        assert parsed_metadata["use_page_content"] is True
        assert parsed_metadata["page_content"] == "Raw sheet data here"
        assert parsed_metadata["page_title"] == "Sales Report"
        assert parsed_metadata["chunker_type"] == "xlsx_semantic"

    def test_parse_no_metadata(self):
        """Test parsing text with no metadata block."""
        text = "Just regular text without any metadata"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        assert clean_text == "Just regular text without any metadata"
        assert parsed_metadata == {}

    def test_parse_invalid_base64(self):
        """Test parsing with invalid base64 content."""
        text = "Some text [XLSX_SEMANTIC_METADATA_B64]not-valid-base64!!![/XLSX_SEMANTIC_METADATA_B64]"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        # Should return original text and empty metadata on parse failure
        assert parsed_metadata == {}

    def test_parse_invalid_json(self):
        """Test parsing with valid base64 but invalid JSON."""
        invalid_json = base64.b64encode(b"not valid json").decode("ascii")
        text = f"Some text [XLSX_SEMANTIC_METADATA_B64]{invalid_json}[/XLSX_SEMANTIC_METADATA_B64]"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        assert parsed_metadata == {}

    def test_parse_preserves_special_characters(self):
        """Test that special characters in metadata are preserved."""
        metadata = {
            "page_content": 'Data with "quotes" and [brackets] and\nnewlines',
            "page_title": "Sheet with <special> chars",
        }
        metadata_b64 = base64.b64encode(
            json.dumps(metadata).encode("utf-8")
        ).decode("ascii")

        text = f"Description\n[XLSX_SEMANTIC_METADATA_B64]{metadata_b64}[/XLSX_SEMANTIC_METADATA_B64]"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        assert 'Data with "quotes"' in parsed_metadata["page_content"]
        assert "[brackets]" in parsed_metadata["page_content"]
        assert "\n" in parsed_metadata["page_content"]

    def test_parse_truncated_flag(self):
        """Test that page_content_truncated flag is preserved."""
        metadata = {
            "page_content": "Truncated content...",
            "page_content_truncated": True,
        }
        metadata_b64 = base64.b64encode(
            json.dumps(metadata).encode("utf-8")
        ).decode("ascii")

        text = f"Description [XLSX_SEMANTIC_METADATA_B64]{metadata_b64}[/XLSX_SEMANTIC_METADATA_B64]"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        assert parsed_metadata["page_content_truncated"] is True

    def test_parse_removes_leading_newlines(self):
        """Test that leading newlines before metadata block are removed."""
        metadata = {"page_title": "Test"}
        metadata_b64 = base64.b64encode(
            json.dumps(metadata).encode("utf-8")
        ).decode("ascii")

        text = f"Description\n\n\n[XLSX_SEMANTIC_METADATA_B64]{metadata_b64}[/XLSX_SEMANTIC_METADATA_B64]"

        clean_text, parsed_metadata = parse_semantic_metadata(text)

        assert clean_text == "Description"
        assert not clean_text.endswith("\n")
