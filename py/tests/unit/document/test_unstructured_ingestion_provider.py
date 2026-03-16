# tests/unit/document/test_unstructured_ingestion_provider.py
"""
Unit tests for UnstructuredIngestionProvider SemanticParsingLimitExceeded fallback paths.

Covers the two branches triggered when the semantic parser exceeds its limits:
  - Branch A: non-empty markdown_content → chunk inline with RecursiveCharacterTextSplitter
  - Branch B: empty markdown_content → re-parse via parse_fallback with DocumentType.value string
"""
import uuid
from unittest.mock import MagicMock

import pytest

from core.base import DocumentType
from core.base.abstractions import Document
from core.base.providers.ingestion import SemanticParsingLimitExceeded
from core.providers.ingestion.unstructured.base import (
    FallbackElement,
    UnstructuredIngestionProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document(doc_type: DocumentType = DocumentType.XLSX) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.owner_id = uuid.uuid4()
    doc.collection_ids = [uuid.uuid4()]
    doc.document_type = doc_type
    doc.metadata = {}
    return doc


def _make_failing_semantic_parser(reason: str, markdown_content: str) -> MagicMock:
    """Async-generator parser that immediately raises SemanticParsingLimitExceeded."""
    async def _ingest(*args, **kwargs):
        raise SemanticParsingLimitExceeded(reason=reason, markdown_content=markdown_content)
        yield  # pragma: no cover

    parser = MagicMock()
    parser.ingest = _ingest
    return parser


def _make_success_parser(text: str) -> MagicMock:
    """Async-generator parser that yields a single string chunk."""
    async def _ingest(*args, **kwargs):
        yield {"content": text}

    parser = MagicMock()
    parser.ingest = _ingest
    return parser


def _make_provider(
    semantic_parser: MagicMock,
    fallback_parser: MagicMock,
) -> UnstructuredIngestionProvider:
    """Build a minimal UnstructuredIngestionProvider without __init__."""
    provider = object.__new__(UnstructuredIngestionProvider)
    provider.config = MagicMock()
    provider.config.to_ingestion_request.return_value = {
        "new_after_n_chars": 1500,
        "overlap": 64,
    }
    provider.parsers = {
        "semantic_xlsx": semantic_parser,
        # "xlsx" key is used by parse_fallback when branch B triggers
        "xlsx": fallback_parser,
    }
    provider._ensure_parser_initialized = MagicMock(return_value="semantic_xlsx")
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_INGESTION_CONFIG: dict = {"extra_parsers": {"xlsx": ["semantic"]}}


class TestUnstructuredSemanticFallback:
    """SemanticParsingLimitExceeded fallback paths in UnstructuredIngestionProvider.parse()."""

    @pytest.mark.asyncio
    async def test_branch_a_chunks_precomputed_markdown_inline(self):
        """Branch A: non-empty markdown_content is split and yielded as DocumentChunks."""
        # Use short chunk_size so the splitter actually creates at least one chunk
        markdown = "# Sheet1\n| Column A | Column B |\n|----------|----------|\n| value1 | value2 |"
        provider = _make_provider(
            _make_failing_semantic_parser(reason="too many sheets", markdown_content=markdown),
            _make_success_parser("unused"),
        )
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, _INGESTION_CONFIG)]

        assert len(chunks) >= 1, "At least one DocumentChunk should be produced from markdown"
        # All chunks carry the document id
        assert all(c.document_id == document.id for c in chunks)
        # Content originates from the markdown (all text should be substrings of markdown)
        combined = " ".join(c.data for c in chunks)
        assert "Sheet1" in combined or "value1" in combined

    @pytest.mark.asyncio
    async def test_branch_a_does_not_invoke_fallback_parser(self):
        """Branch A must not call the plain 'xlsx' parser when markdown is available."""
        fallback_parser = _make_success_parser("should not appear")
        call_count = 0
        original_ingest = fallback_parser.ingest

        async def counting_ingest(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            async for item in original_ingest(*args, **kwargs):
                yield item

        fallback_parser.ingest = counting_ingest

        provider = _make_provider(
            _make_failing_semantic_parser(
                reason="too many sheets",
                markdown_content="# Sheet\n| A |\n|---|\n| 1 |",
            ),
            fallback_parser,
        )
        document = _make_document()

        _ = [c async for c in provider.parse(b"fake.xlsx", document, _INGESTION_CONFIG)]

        assert call_count == 0, "Fallback parser must not be invoked when markdown_content is non-empty"

    @pytest.mark.asyncio
    async def test_branch_b_reparses_via_document_type_value_string(self):
        """Branch B: parse_fallback is called with parser_name=document_type.value (str, not enum)."""
        fallback_text = "recovered text from raw bytes"
        provider = _make_provider(
            _make_failing_semantic_parser(reason="chars exceeded", markdown_content=""),
            _make_success_parser(fallback_text),
        )
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, _INGESTION_CONFIG)]

        assert len(chunks) >= 1
        # Verify parse_fallback was called with the string value "xlsx", not the enum
        # (this is confirmed by parsers["xlsx"] being invoked — if the enum were passed
        #  and used as dict key, it would raise KeyError since the key is the string "xlsx")
        assert any(fallback_text in c.data for c in chunks), (
            "Fallback text from raw bytes re-parse should appear in output chunks"
        )

    @pytest.mark.asyncio
    async def test_branch_b_chunk_carries_document_metadata(self):
        """Chunks from branch B carry document_id and owner_id."""
        provider = _make_provider(
            _make_failing_semantic_parser(reason="chars exceeded", markdown_content=""),
            _make_success_parser("some content"),
        )
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, _INGESTION_CONFIG)]

        assert chunks[0].document_id == document.id
        assert chunks[0].owner_id == document.owner_id

    @pytest.mark.asyncio
    async def test_branch_a_respects_chunk_size_config(self):
        """Branch A uses new_after_n_chars from config (1500 default, not 2048)."""
        # Provide a very small chunk_size to force multiple splits
        provider = _make_provider(
            _make_failing_semantic_parser(
                reason="too many sheets",
                markdown_content="word " * 400,  # 2000 chars — exceeds 1500
            ),
            _make_success_parser("unused"),
        )
        # Override the config to use a tiny chunk size to guarantee splitting
        provider.config.to_ingestion_request.return_value = {
            "new_after_n_chars": 100,
            "overlap": 0,
        }
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, _INGESTION_CONFIG)]

        # With chunk_size=100 and 2000 chars of content, expect multiple chunks
        assert len(chunks) > 1, (
            "Large markdown should be split into multiple chunks with small new_after_n_chars"
        )

    @pytest.mark.asyncio
    async def test_elements_marked_partitioned_by_unstructured(self):
        """All output chunks carry partitioned_by_unstructured=True in metadata."""
        provider = _make_provider(
            _make_failing_semantic_parser(
                reason="too many sheets",
                markdown_content="# Sheet1\n| A | B |\n|---|---|\n| 1 | 2 |",
            ),
            _make_success_parser("unused"),
        )
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, _INGESTION_CONFIG)]

        assert all(c.metadata.get("partitioned_by_unstructured") is True for c in chunks)
