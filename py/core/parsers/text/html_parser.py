# type: ignore
import logging
from typing import AsyncGenerator

from bs4 import BeautifulSoup

from core.base.parsers.base_parser import AsyncParser
from core.base.providers import (
    CompletionProvider,
    DatabaseProvider,
    IngestionConfig,
)

logger = logging.getLogger()


class HTMLParser(AsyncParser[str | bytes]):
    """A parser for HTML data using BeautifulSoup."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config

    async def ingest(
        self, data: str | bytes, *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ingest HTML data and yield text."""
        soup = BeautifulSoup(data, "html.parser")
        yield soup.get_text()


class HTMLToMarkdownParser(AsyncParser[str | bytes]):
    """A parser for HTML data that converts to Markdown for better structure preservation."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider=None,  # Accept but ignore ocr_provider for consistency with PDF parsers
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config

    async def ingest(
        self, data: str | bytes, *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ingest HTML data and yield markdown text for better chunking."""
        try:
            from html_to_markdown import convert, ConversionOptions

            logger.info("Starting HTML ingestion using HTMLToMarkdownParser")

            # Convert bytes to string if needed
            if isinstance(data, bytes):
                html_content = data.decode('utf-8')
            else:
                html_content = data

            # Convert HTML to Markdown with clean structure
            options = ConversionOptions(
                heading_style="atx",
                list_indent_width=2,
                bullets="*+-"
            )

            markdown = convert(html_content, options)
            yield markdown

        except ImportError:
            logger.warning("html-to-markdown not installed, falling back to BeautifulSoup")
            soup = BeautifulSoup(data, "html.parser")
            yield soup.get_text()
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {str(e)}")
            # Fallback to BeautifulSoup on error
            soup = BeautifulSoup(data, "html.parser")
            yield soup.get_text()
