# tests/unit/document/test_xlsx_semantic_parser.py
"""
Unit tests for XLSXSemanticParser - LLM-based semantic chunking for Excel files.

Tests cover:
- Excel file detection
- Sheet splitting from markdown
- Semantic chunk generation
- Metadata formatting
- Error handling and fallbacks
"""
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.base.abstractions import GenerationConfig, LLMChatCompletion
from core.base.providers import IngestionConfig
from core.parsers.structured.xlsx_semantic_parser import (
    XLSXSemanticParser,
    XLSSemanticParser,
)


@pytest.fixture
def mock_ingestion_config():
    """Create a mock ingestion config."""
    config = MagicMock(spec=IngestionConfig)
    return config


@pytest.fixture
def mock_database_provider():
    """Create a mock database provider."""
    return MagicMock()


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider that returns semantic descriptions."""
    provider = MagicMock()

    # Create a mock response that mimics LLMChatCompletion
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Budget data showing quarterly expenses by department with totals"
            )
        )
    ]

    # Make aget_completion return the mock response
    provider.aget_completion = AsyncMock(return_value=mock_response)

    return provider


@pytest.fixture
def parser(mock_ingestion_config, mock_database_provider, mock_llm_provider):
    """Create a parser instance with mocked dependencies."""
    return XLSXSemanticParser(
        config=mock_ingestion_config,
        database_provider=mock_database_provider,
        llm_provider=mock_llm_provider,
    )


class TestSheetSplitting:
    """Tests for _split_by_sheets method."""

    def test_split_single_sheet(self, parser):
        """Test splitting markdown with a single sheet."""
        markdown = """## Sheet1

| Col1 | Col2 |
|------|------|
| A    | B    |
"""
        sheets = parser._split_by_sheets(markdown)

        assert len(sheets) == 1
        assert "Sheet1" in sheets
        assert "| Col1 | Col2 |" in sheets["Sheet1"]

    def test_split_multiple_sheets(self, parser):
        """Test splitting markdown with multiple sheets."""
        markdown = """## Sales Data

| Product | Q1 | Q2 |
|---------|----|-------|
| Widget  | 100| 150|

## Expenses

| Category | Amount |
|----------|--------|
| Marketing| 5000   |
"""
        sheets = parser._split_by_sheets(markdown)

        assert len(sheets) == 2
        assert "Sales Data" in sheets
        assert "Expenses" in sheets
        assert "Widget" in sheets["Sales Data"]
        assert "Marketing" in sheets["Expenses"]

    def test_split_no_headers(self, parser):
        """Test fallback when no ## headers found."""
        markdown = """Just some content without headers
| A | B |
|---|---|
| 1 | 2 |
"""
        sheets = parser._split_by_sheets(markdown)

        # Should create a default "Sheet1" entry
        assert len(sheets) == 1
        assert "Sheet1" in sheets

    def test_split_empty_sheets_skipped(self, parser):
        """Test that empty sheets are skipped."""
        markdown = """## HasContent

| Data |
|------|
| Yes  |

## EmptySheet

## AnotherWithContent

| More |
|------|
| Data |
"""
        sheets = parser._split_by_sheets(markdown)

        # EmptySheet should be skipped (no content after header)
        assert "HasContent" in sheets
        assert "AnotherWithContent" in sheets


class TestSemanticChunkGeneration:
    """Tests for LLM-based semantic chunk generation."""

    @pytest.mark.asyncio
    async def test_generate_semantic_chunk_success(self, parser):
        """Test successful semantic chunk generation."""
        sheet_name = "Budget"
        content = "| Dept | Q1 | Q2 |\n|------|----|----|\n| Sales | 100 | 150 |"

        result = await parser._generate_semantic_chunk(sheet_name, content)

        assert result is not None
        assert "Budget" in result or "quarterly" in result.lower()
        # Verify LLM was called
        parser.llm_provider.aget_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_semantic_chunk_llm_failure(
        self, parser, mock_llm_provider
    ):
        """Test fallback when LLM fails."""
        # Make LLM raise an exception
        mock_llm_provider.aget_completion = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await parser._generate_semantic_chunk("TestSheet", "content")

        # Should return fallback description
        assert result is not None
        assert "TestSheet" in result
        assert "tabular" in result.lower() or "spreadsheet" in result.lower()

    @pytest.mark.asyncio
    async def test_content_truncation_for_llm(self, parser):
        """Test that large content is truncated before sending to LLM."""
        # Create content larger than max_content_for_llm
        large_content = "x" * 20000

        await parser._generate_semantic_chunk("BigSheet", large_content)

        # Verify the call was made
        call_args = parser.llm_provider.aget_completion.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0])
        user_message = messages[1]["content"]

        # Should be truncated
        assert len(user_message) < 20000
        assert "[content truncated]" in user_message


class TestMetadataFormatting:
    """Tests for chunk metadata formatting."""

    def test_format_chunk_with_metadata(self, parser):
        """Test metadata block formatting with base64 encoding."""
        result = parser._format_chunk_with_metadata(
            semantic_description="Budget summary",
            page_content="| A | B |",
            page_title="Sheet1",
        )

        # Should contain semantic description
        assert "Budget summary" in result

        # Should contain base64-encoded metadata block
        assert "[XLSX_SEMANTIC_METADATA_B64]" in result
        assert "[/XLSX_SEMANTIC_METADATA_B64]" in result

        # Extract and verify metadata JSON from base64
        import re

        match = re.search(
            r"\[XLSX_SEMANTIC_METADATA_B64\]([A-Za-z0-9+/=]+)"
            r"\[/XLSX_SEMANTIC_METADATA_B64\]",
            result,
        )
        assert match is not None

        # Decode base64 and parse JSON
        metadata_json = base64.b64decode(match.group(1)).decode("utf-8")
        metadata = json.loads(metadata_json)
        assert metadata["use_page_content"] is True
        assert metadata["page_content"] == "| A | B |"
        assert metadata["page_title"] == "Sheet1"
        assert metadata["chunker_type"] == "xlsx_semantic"

    def test_format_chunk_handles_special_characters(self, parser):
        """Test that special characters in content are safely encoded."""
        result = parser._format_chunk_with_metadata(
            semantic_description="Test description",
            page_content='Content with "quotes" and [brackets] and \n newlines',
            page_title="Sheet1",
        )

        # Should be valid and parseable
        import re

        match = re.search(
            r"\[XLSX_SEMANTIC_METADATA_B64\]([A-Za-z0-9+/=]+)"
            r"\[/XLSX_SEMANTIC_METADATA_B64\]",
            result,
        )
        assert match is not None

        metadata_json = base64.b64decode(match.group(1)).decode("utf-8")
        metadata = json.loads(metadata_json)
        assert 'Content with "quotes"' in metadata["page_content"]
        assert "[brackets]" in metadata["page_content"]


class TestFullIngestion:
    """Integration-style tests for the full ingest flow."""

    @pytest.mark.asyncio
    async def test_ingest_with_mocked_markitdown(self, parser):
        """Test full ingestion with mocked MarkItDown."""
        # Mock MarkItDown conversion
        mock_md_result = MagicMock()
        mock_md_result.text_content = """## Sales

| Product | Revenue |
|---------|---------|
| A       | 1000    |

## Costs

| Item | Amount |
|------|--------|
| Rent | 500    |
"""
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_md_result

        with patch.object(parser, "_md_converter", mock_converter):
            # Mock the file writing
            with patch("tempfile.NamedTemporaryFile"):
                with patch("os.path.exists", return_value=True):
                    with patch("os.unlink"):
                        chunks = []
                        async for chunk in parser.ingest(b"fake excel bytes"):
                            chunks.append(chunk)

        # Should have 2 chunks (one per sheet)
        assert len(chunks) == 2

        # Each chunk should have base64-encoded metadata block
        for chunk in chunks:
            assert "[XLSX_SEMANTIC_METADATA_B64]" in chunk

    @pytest.mark.asyncio
    async def test_ingest_empty_content(self, parser):
        """Test ingestion with empty content."""
        mock_md_result = MagicMock()
        mock_md_result.text_content = ""
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_md_result

        with patch.object(parser, "_md_converter", mock_converter):
            with patch("tempfile.NamedTemporaryFile"):
                with patch("os.path.exists", return_value=True):
                    with patch("os.unlink"):
                        chunks = []
                        async for chunk in parser.ingest(b"empty excel"):
                            chunks.append(chunk)

        # Should have no chunks
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_ingest_rejects_string_input(self, parser):
        """Test that string input is rejected."""
        with pytest.raises(ValueError, match="bytes format"):
            async for _ in parser.ingest("not bytes"):
                pass

    @pytest.mark.asyncio
    async def test_ingest_rejects_oversized_files(self, parser):
        """Test that files exceeding size limit are rejected."""
        from core.parsers.structured.xlsx_semantic_parser import MAX_FILE_SIZE_BYTES

        # Create data larger than the limit
        oversized_data = b"x" * (MAX_FILE_SIZE_BYTES + 1)

        with pytest.raises(ValueError, match="exceeds maximum"):
            async for _ in parser.ingest(oversized_data):
                pass


class TestXLSSemanticParser:
    """Tests for legacy XLS format parser."""

    def test_xls_parser_inherits_from_xlsx(self):
        """Test that XLSSemanticParser inherits from XLSXSemanticParser."""
        assert issubclass(XLSSemanticParser, XLSXSemanticParser)

    @pytest.mark.asyncio
    async def test_xls_uses_xls_extension(
        self, mock_ingestion_config, mock_database_provider, mock_llm_provider
    ):
        """Test that XLS parser uses .xls file extension."""
        parser = XLSSemanticParser(
            config=mock_ingestion_config,
            database_provider=mock_database_provider,
            llm_provider=mock_llm_provider,
        )

        mock_md_result = MagicMock()
        mock_md_result.text_content = "## Sheet1\n| A |\n|---|\n| 1 |"
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_md_result

        with patch.object(parser, "_md_converter", mock_converter):
            with patch(
                "tempfile.NamedTemporaryFile"
            ) as mock_temp:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=None)
                mock_file.name = "/tmp/test.xls"
                mock_temp.return_value = mock_file

                with patch("os.path.exists", return_value=True):
                    with patch("os.unlink"):
                        async for _ in parser.ingest(b"xls bytes"):
                            pass

                # Verify .xls extension was used
                mock_temp.assert_called_with(suffix=".xls", delete=False)
