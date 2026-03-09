"""
Shared utilities for parsing semantic metadata from chunk text.

The XLSX semantic parser embeds metadata in base64-encoded JSON blocks within
chunk text. This module provides a single source of truth for parsing that format.
"""
import base64
import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_semantic_metadata(text: str) -> tuple[str, dict]:
    """
    Parse and strip embedded semantic metadata from chunk text.

    The semantic parser embeds metadata in format:
    [XLSX_SEMANTIC_METADATA_B64]<base64-encoded-json>[/XLSX_SEMANTIC_METADATA_B64]

    Args:
        text: Chunk text that may contain embedded metadata

    Returns:
        Tuple of (clean_text, metadata_dict)
        - clean_text: Text with metadata block removed
        - metadata_dict: Parsed metadata, or empty dict if none found
    """
    metadata = {}

    # Try base64 format: [XLSX_SEMANTIC_METADATA_B64]...[/XLSX_SEMANTIC_METADATA_B64]
    metadata_match = re.search(
        r"\[XLSX_SEMANTIC_METADATA_B64\]([A-Za-z0-9+/=]+)"
        r"\[/XLSX_SEMANTIC_METADATA_B64\]",
        text,
    )

    if metadata_match:
        try:
            metadata_json = base64.b64decode(metadata_match.group(1)).decode("utf-8")
            metadata = json.loads(metadata_json)
            # Remove metadata block from text
            text = re.sub(
                r"\n*\[XLSX_SEMANTIC_METADATA_B64\]"
                r"[A-Za-z0-9+/=]+\[/XLSX_SEMANTIC_METADATA_B64\]",
                "",
                text,
            ).strip()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse base64 semantic metadata: {e}")

    return text, metadata
