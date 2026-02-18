# type: ignore
"""
Excel Semantic Parser - LLM-based semantic chunking for Excel files.

This parser implements a hybrid retrieval strategy:
1. Converts Excel to Markdown using MarkItDown (preserves table structure)
2. Splits by sheets (## headers) - each sheet becomes a "page"
3. Uses LLM to create semantic descriptions (searchable summaries)
4. Stores raw page content in chunk metadata for retrieval enrichment

The key insight: semantic chunks are "pointers" to data, not the data itself.
During retrieval, the semantic match is enriched with full raw page content.

Usage:
    Configure via extra_parsers: {"xlsx": ["semantic"]}

Metadata flags set by this parser:
    - use_page_content: True (signals retrieval enrichment needed)
    - page_content: Full raw markdown of the sheet
    - page_title: Sheet name
    - chunker_type: "xlsx_semantic"
"""
import base64
import contextlib
import json
import logging
import os
import re
import tempfile
from typing import AsyncGenerator

from core.base.abstractions import GenerationConfig
from core.base.parsers.base_parser import AsyncParser
from core.base.providers import (
    CompletionProvider,
    DatabaseProvider,
    IngestionConfig,
)

logger = logging.getLogger(__name__)

# Maximum number of sheets to process (LLM cost control)
MAX_SHEETS_PER_FILE = int(os.getenv("XLSX_SEMANTIC_MAX_SHEETS", "50"))

# Maximum file size in bytes (100MB default) to prevent memory issues
MAX_FILE_SIZE_BYTES = int(os.getenv("XLSX_SEMANTIC_MAX_FILE_SIZE_MB", "100")) * 1024 * 1024


@contextlib.contextmanager
def safe_temp_file(data: bytes, suffix: str):
    """Context manager for safe temporary file handling with guaranteed cleanup."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # Best effort cleanup

# Default prompt for semantic chunk generation
DEFAULT_SEMANTIC_PROMPT = """Analyze the following spreadsheet content and create a semantic description that:

1. Summarizes what type of data this represents (budgets, metrics, inventory, etc.)
2. Lists key column headers and their meaning
3. Notes important values, ranges, patterns, or trends
4. Identifies time periods, categories, or entities present
5. Includes searchable keywords and terms for retrieval

IMPORTANT: Prefer including actual raw data values over approximations or generalizations.
For accurate retrieval, include specific numbers, names, dates, and exact values from the data.
Avoid vague summaries like "various amounts" - instead include actual figures like "$5,000, $12,500".

Keep the description information-rich with concrete data points.
This description will be used for semantic search, so include terms users might search for.

Sheet Name: {sheet_name}

Content:
{content}

Return ONLY the semantic description, no other text."""


class XLSXSemanticParser(AsyncParser[str | bytes]):
    """
    Excel parser with LLM-based semantic chunking for hybrid retrieval.

    Creates semantic descriptions (pointers) while storing raw page content
    in metadata. During retrieval, the semantic match is enriched with the
    full raw content, preserving Excel's interconnected data relationships.

    This approach solves the problem of traditional chunking destroying
    Excel's table structure, formulas, and cross-references.
    """

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider=None,  # Accept but ignore for consistency
    ):
        self.config = config
        self.database_provider = database_provider
        self.llm_provider = llm_provider

        # Initialize MarkItDown lazily
        self._md_converter = None

        # Get model from config or use default
        # The llm_provider has its own config with model settings
        self.semantic_prompt = DEFAULT_SEMANTIC_PROMPT

        # Max content size to send to LLM (characters)
        self.max_content_for_llm = 15000

        logger.info("XLSXSemanticParser initialized")

    @property
    def md_converter(self):
        """Lazy initialization of MarkItDown converter."""
        if self._md_converter is None:
            try:
                from markitdown import MarkItDown

                self._md_converter = MarkItDown()
                logger.debug("MarkItDown converter initialized")
            except ImportError as e:
                logger.error(
                    "MarkItDown not installed. Install with: pip install markitdown"
                )
                raise ImportError(
                    "MarkItDown is required for XLSXSemanticParser. "
                    "Install with: pip install markitdown"
                ) from e
        return self._md_converter

    async def ingest(
        self, data: bytes, *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Ingest Excel file and yield semantic chunks with raw page content.

        Each yielded chunk is a string containing the semantic description.
        The chunk metadata (page_content, use_page_content, etc.) is passed
        through kwargs and handled by the ingestion pipeline.

        For this parser, we yield dict-like structures that the pipeline
        will process appropriately.
        """
        if isinstance(data, str):
            raise ValueError("Excel data must be in bytes format.")

        # Validate file size to prevent memory issues
        file_size = len(data)
        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            file_mb = file_size / (1024 * 1024)
            logger.error(
                f"Excel file too large: {file_mb:.1f}MB exceeds limit of {max_mb}MB. "
                f"Set XLSX_SEMANTIC_MAX_FILE_SIZE_MB to increase limit."
            )
            raise ValueError(
                f"Excel file size ({file_mb:.1f}MB) exceeds maximum allowed "
                f"({max_mb}MB). Use standard parser or increase limit."
            )

        # Convert Excel to Markdown
        markdown_content = await self._convert_to_markdown(data)

        if not markdown_content:
            logger.warning("No content extracted from Excel file")
            return

        # Split by sheets (## headers)
        sheets = self._split_by_sheets(markdown_content)

        if not sheets:
            logger.warning("No sheets found in Excel file")
            # Fallback: yield entire content as single chunk
            yield markdown_content
            return

        total_sheets = len(sheets)
        logger.info(f"XLSXSemanticParser: Processing {total_sheets} sheets")

        if total_sheets > MAX_SHEETS_PER_FILE:
            logger.warning(
                f"Excel file has {total_sheets} sheets, limiting to "
                f"{MAX_SHEETS_PER_FILE} (set XLSX_SEMANTIC_MAX_SHEETS to change)"
            )

        # Process each sheet (with limit for cost control)
        sheets_processed = 0
        for sheet_name, sheet_content in sheets.items():
            if sheets_processed >= MAX_SHEETS_PER_FILE:
                logger.warning(
                    f"Reached sheet limit ({MAX_SHEETS_PER_FILE}), "
                    f"skipping remaining {total_sheets - sheets_processed} sheets"
                )
                break

            if not sheet_content.strip():
                logger.debug(f"Skipping empty sheet: {sheet_name}")
                continue

            sheets_processed += 1
            logger.info(
                f"Processing sheet {sheets_processed}/{min(total_sheets, MAX_SHEETS_PER_FILE)}: "
                f"'{sheet_name}' ({len(sheet_content)} chars)"
            )

            # Generate semantic description via LLM
            semantic_description = await self._generate_semantic_chunk(
                sheet_name, sheet_content
            )

            if semantic_description:
                # Yield the semantic description as the chunk text
                # The raw content is included in a special format that
                # the ingestion pipeline will parse into metadata
                #
                # Format: [XLSX_SEMANTIC] markers for downstream processing
                chunk_with_metadata = self._format_chunk_with_metadata(
                    semantic_description=semantic_description,
                    page_content=sheet_content,
                    page_title=sheet_name,
                )
                yield chunk_with_metadata
            else:
                # Fallback: yield raw content if LLM fails
                logger.warning(
                    f"LLM chunking failed for sheet '{sheet_name}', "
                    "using raw content"
                )
                yield sheet_content

    async def _convert_to_markdown(self, data: bytes) -> str:
        """Convert Excel bytes to Markdown using MarkItDown."""
        try:
            with safe_temp_file(data, ".xlsx") as tmp_path:
                result = self.md_converter.convert(tmp_path)
                markdown_content = result.text_content

                logger.debug(
                    f"Converted Excel to Markdown: {len(markdown_content)} chars"
                )
                return markdown_content

        except Exception as e:
            logger.error(f"MarkItDown conversion failed: {e}")
            raise

    def _split_by_sheets(self, markdown: str) -> dict[str, str]:
        """
        Split markdown content by ## headers (Excel sheets).

        MarkItDown converts each Excel sheet to a section with ## header.
        This method extracts each sheet as a separate "page" for processing.

        Returns:
            Dict mapping sheet names to their markdown content
        """
        sheets = {}

        # Split by ## headers (each sheet in Excel becomes ## SheetName)
        sections = re.split(r"(?:\n|^)(##\s+[^\n]+)", markdown)

        # Process header-content pairs
        for i in range(1, len(sections), 2):
            header = sections[i]
            sheet_name = header.strip().replace("##", "").strip()

            if i + 1 < len(sections):
                content = sections[i + 1].strip()
                if content:  # Only include non-empty sheets
                    sheets[sheet_name] = content

        # If no ## headers found, treat entire content as single sheet
        if not sheets and markdown.strip():
            sheets["Sheet1"] = markdown.strip()

        return sheets

    async def _generate_semantic_chunk(
        self, sheet_name: str, content: str
    ) -> str | None:
        """
        Use LLM to generate semantic description of sheet content.

        The semantic description serves as a searchable "pointer" to the
        actual data. It includes key terms, data types, and patterns
        that users might search for.

        Args:
            sheet_name: Name of the Excel sheet
            content: Raw markdown content of the sheet

        Returns:
            Semantic description string, or None if generation fails
        """
        try:
            # Truncate content if too large for LLM context
            truncated_content = content[: self.max_content_for_llm]
            if len(content) > self.max_content_for_llm:
                truncated_content += "\n... [content truncated]"

            prompt = self.semantic_prompt.format(
                sheet_name=sheet_name, content=truncated_content
            )

            # Create generation config for this request
            # Use fast_llm from app config for efficiency
            model = getattr(self.config, "app", None)
            model = getattr(model, "fast_llm", None) if model else None
            model = model or os.getenv("XLSX_SEMANTIC_MODEL", "openai/gpt-4.1")

            generation_config = GenerationConfig(
                model=model,
                temperature=0.1,  # Low temperature for consistent output
                max_tokens_to_sample=500,
            )

            # Use the injected LLM provider
            response = await self.llm_provider.aget_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at analyzing spreadsheet data "
                            "and creating concise, searchable descriptions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                generation_config=generation_config,
            )

            semantic_text = response.choices[0].message.content
            if semantic_text:
                semantic_text = semantic_text.strip()
                logger.debug(
                    f"Generated semantic chunk for '{sheet_name}': "
                    f"{len(semantic_text)} chars"
                )
                return semantic_text

            return None

        except Exception as e:
            logger.error(
                f"LLM semantic chunking failed for sheet '{sheet_name}': {e}"
            )
            # Return a simple fallback description
            return (
                f"Spreadsheet data from sheet '{sheet_name}' "
                f"containing tabular information."
            )

    def _format_chunk_with_metadata(
        self,
        semantic_description: str,
        page_content: str,
        page_title: str,
    ) -> str:
        """
        Format chunk with embedded metadata for downstream processing.

        The ingestion pipeline will extract this metadata and store it
        appropriately with the chunk.

        Format uses base64-encoded JSON block for safe parsing (avoids regex
        escaping issues with special characters in content).
        """
        # Create metadata block as JSON
        metadata = {
            "use_page_content": True,
            "page_content": page_content,
            "page_title": page_title,
            "chunker_type": "xlsx_semantic",
        }

        # Base64 encode the JSON to avoid escaping issues with special characters
        metadata_json = json.dumps(metadata)
        metadata_b64 = base64.b64encode(metadata_json.encode("utf-8")).decode("ascii")

        # Format: semantic text followed by base64-encoded metadata block
        # The metadata block uses special markers for parsing
        chunk = (
            f"{semantic_description}\n\n"
            f"[XLSX_SEMANTIC_METADATA_B64]{metadata_b64}[/XLSX_SEMANTIC_METADATA_B64]"
        )

        return chunk


class XLSSemanticParser(XLSXSemanticParser):
    """
    XLS (legacy Excel) parser with LLM-based semantic chunking.

    Inherits from XLSXSemanticParser - MarkItDown handles both formats.
    """

    async def _convert_to_markdown(self, data: bytes) -> str:
        """Convert XLS bytes to Markdown using MarkItDown."""
        try:
            with safe_temp_file(data, ".xls") as tmp_path:
                result = self.md_converter.convert(tmp_path)
                return result.text_content

        except Exception as e:
            logger.error(f"MarkItDown XLS conversion failed: {e}")
            raise
