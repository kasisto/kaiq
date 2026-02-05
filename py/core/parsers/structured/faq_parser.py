# type: ignore
"""
FAQ Parser - Custom parser for FAQ (Q&A) documents.

This parser handles CSV files containing Question/Answer pairs and preserves
the Q&A structure by yielding each pair as a structured chunk with clear markers.
This ensures the Q&A relationship is maintained through the ingestion pipeline.
"""
import csv
import logging
from io import StringIO
from typing import AsyncGenerator

from core.base.parsers.base_parser import AsyncParser
from core.base.providers import (
    CompletionProvider,
    DatabaseProvider,
    IngestionConfig,
)

logger = logging.getLogger()


class FAQParser(AsyncParser[str | bytes]):
    """
    A parser for FAQ (Q&A) CSV data that preserves question-answer structure.

    Expected CSV format:
        Question,Answer
        "What is X?","X is..."
        "How do I Y?","You can Y by..."

    Each Q&A pair is yielded as a separate chunk with structured markers:
        [FAQ_QUESTION]question text[/FAQ_QUESTION]
        [FAQ_ANSWER]answer text[/FAQ_ANSWER]

    This allows downstream components to easily parse and display Q&A pairs.
    """

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider=None,  # Accept but ignore for consistency with other parsers
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config

    async def ingest(
        self, data: str | bytes, *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Ingest FAQ CSV data and yield structured Q&A chunks.

        Each Q&A pair becomes a separate chunk with clear markers that can be
        parsed by the frontend to display formatted Q&A content.
        """
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        # Early return for empty files
        if not data.strip():
            logger.warning("Empty FAQ content received")
            return

        qa_pairs = self._parse_csv(data)

        if not qa_pairs:
            logger.warning("No Q&A pairs found in FAQ content")
            # Fall back to raw content if no pairs found
            yield data
            return

        logger.info(f"FAQ Parser: Found {len(qa_pairs)} Q&A pairs")

        # Yield each Q&A pair as a separate structured chunk
        for pair in qa_pairs:
            question = pair['question'].strip()
            answer = pair['answer'].strip()

            if question and answer:
                # Format with clear markers for frontend parsing
                chunk = f"[FAQ_QUESTION]{question}[/FAQ_QUESTION]\n[FAQ_ANSWER]{answer}[/FAQ_ANSWER]"
                yield chunk

    def _parse_csv(self, content: str) -> list[dict[str, str]]:
        """Parse CSV content into list of Q&A pairs."""
        qa_pairs = []

        try:
            reader = csv.DictReader(StringIO(content))

            for row in reader:
                # Handle various column name formats (case-insensitive)
                question = (
                    row.get('Question') or
                    row.get('question') or
                    row.get('Q') or
                    row.get('q') or
                    ''
                ).strip()

                answer = (
                    row.get('Answer') or
                    row.get('answer') or
                    row.get('A') or
                    row.get('a') or
                    ''
                ).strip()

                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })

        except Exception as e:
            logger.error(f"Failed to parse FAQ CSV: {e}")

        return qa_pairs
