# tests/unit/document/test_r2r_ingestion_provider.py
"""
Unit tests for R2RIngestionProvider SemanticParsingLimitExceeded fallback paths.

Covers the two branches that trigger when semantic parsing limits are exceeded:
  - Branch A: pre-computed markdown_content is non-empty → use it directly
  - Branch B: markdown_content is empty → re-parse from raw bytes via default parser
"""
import uuid
from unittest.mock import MagicMock

import pytest

from core.base import DocumentType
from core.base.abstractions import Document
from core.base.providers.ingestion import SemanticParsingLimitExceeded
from core.providers.ingestion.r2r.base import R2RIngestionProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document(doc_type: DocumentType = DocumentType.XLSX) -> Document:
    """Minimal Document with required fields."""
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.owner_id = uuid.uuid4()
    doc.collection_ids = [uuid.uuid4()]
    doc.document_type = doc_type
    doc.metadata = {}
    return doc


def _make_failing_semantic_parser(reason: str, markdown_content: str) -> MagicMock:
    """Returns a mock parser whose ingest() async-generator immediately raises SemanticParsingLimitExceeded."""
    async def _ingest(*args, **kwargs):
        raise SemanticParsingLimitExceeded(reason=reason, markdown_content=markdown_content)
        yield  # pragma: no cover — makes this an async generator

    parser = MagicMock()
    parser.ingest = _ingest
    return parser


def _make_success_parser(text: str) -> MagicMock:
    """Returns a mock parser whose ingest() yields a single string."""
    async def _ingest(*args, **kwargs):
        yield text

    parser = MagicMock()
    parser.ingest = _ingest
    return parser


def _make_provider(semantic_parser, fallback_parser) -> R2RIngestionProvider:
    """Build a minimal R2RIngestionProvider without running __init__."""
    provider = object.__new__(R2RIngestionProvider)
    provider.parsers = {
        "semantic_xlsx": semantic_parser,
        DocumentType.XLSX: fallback_parser,
    }
    provider._ensure_parser_initialized = MagicMock(return_value="semantic_xlsx")
    # chunk() splits text into a list; return the text unchanged for test clarity
    provider.chunk = MagicMock(side_effect=lambda text, _cfg: [text])
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestR2RSemanticFallback:
    """SemanticParsingLimitExceeded fallback paths in R2RIngestionProvider.parse()."""

    _INGESTION_CONFIG: dict = {"extra_parsers": {"xlsx": ["semantic"]}}

    @pytest.mark.asyncio
    async def test_branch_a_uses_precomputed_markdown_when_nonempty(self):
        """When markdown_content is non-empty, the fallback uses it directly."""
        markdown = "# Sheet1\n| A | B |\n|---|---|\n| 1 | 2 |"
        semantic_parser = _make_failing_semantic_parser(
            reason="too many sheets", markdown_content=markdown
        )
        fallback_parser = _make_success_parser("raw bytes fallback")
        provider = _make_provider(semantic_parser, fallback_parser)
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, self._INGESTION_CONFIG)]

        assert len(chunks) == 1, "Expected exactly one DocumentChunk from pre-computed markdown"
        assert chunks[0].data == markdown
        # Fallback parser must NOT be called when markdown_content is available
        with pytest.raises(AssertionError):
            fallback_parser.ingest.assert_called()

    @pytest.mark.asyncio
    async def test_branch_a_chunk_contains_document_metadata(self):
        """DocumentChunk produced from pre-computed markdown carries document_id and owner_id."""
        markdown = "# BigSheet\n| col |\n|-----|\n| val |"
        document = _make_document()
        provider = _make_provider(
            _make_failing_semantic_parser(reason="chars exceeded", markdown_content=markdown),
            _make_success_parser("unused"),
        )

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, self._INGESTION_CONFIG)]

        assert chunks[0].document_id == document.id
        assert chunks[0].owner_id == document.owner_id

    @pytest.mark.asyncio
    async def test_branch_b_reparses_from_raw_bytes_when_markdown_empty(self):
        """When markdown_content is empty, raw bytes are re-parsed via the default parser."""
        fallback_text = "recovered content from raw bytes"
        semantic_parser = _make_failing_semantic_parser(
            reason="too many sheets", markdown_content=""
        )
        fallback_parser = _make_success_parser(fallback_text)
        provider = _make_provider(semantic_parser, fallback_parser)
        document = _make_document()

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, self._INGESTION_CONFIG)]

        assert len(chunks) == 1
        assert chunks[0].data == fallback_text

    @pytest.mark.asyncio
    async def test_branch_b_chunk_carries_correct_document_ids(self):
        """Raw-bytes fallback chunk still carries document_id and owner_id."""
        document = _make_document()
        provider = _make_provider(
            _make_failing_semantic_parser(reason="too many sheets", markdown_content=""),
            _make_success_parser("some fallback"),
        )

        chunks = [c async for c in provider.parse(b"fake.xlsx", document, self._INGESTION_CONFIG)]

        assert chunks[0].document_id == document.id
        assert chunks[0].owner_id == document.owner_id

    @pytest.mark.asyncio
    async def test_branch_a_does_not_call_fallback_parser(self):
        """Branch A must not call the default document-type parser at all."""
        markdown = "# Sheet1\n| A | B |\n|---|---|\n| 1 | 2 |"
        fallback_parser = _make_success_parser("should not appear")
        provider = _make_provider(
            _make_failing_semantic_parser(reason="too many sheets", markdown_content=markdown),
            fallback_parser,
        )
        document = _make_document()
        # Track calls to the fallback parser's ingest
        call_count = 0
        original_ingest = fallback_parser.ingest

        async def counting_ingest(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            async for item in original_ingest(*args, **kwargs):
                yield item

        fallback_parser.ingest = counting_ingest

        _ = [c async for c in provider.parse(b"fake.xlsx", document, self._INGESTION_CONFIG)]

        assert call_count == 0, "Fallback parser ingest must not be called when markdown_content is non-empty"
