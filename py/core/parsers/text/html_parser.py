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

# Default CSS selector to heading mappings for organizations
# Can be overridden via config: html_css_heading_mappings
DEFAULT_CSS_HEADING_MAPPINGS = {
    "div.pv-section": "h2",
    "div.pv-secondary-window-title": "h2",
    "p.pv-header": "h3",
    "div.pv-task-header-anchor": "h3",
    "p.pv-sub-header": "h4",
    "p.pv-task-header": "h5",
}


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
    """
    A parser for HTML data that converts to Markdown for better structure preservation.

    Supports CSS selector-based heading transformation to handle custom HTML structures    """

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

        # Load CSS heading mappings from config or use defaults
        enabled = getattr(config, "enable_html_css_heading_mappings", True)
        if enabled:
            self.css_heading_mappings = getattr(
                config, "html_css_heading_mappings", DEFAULT_CSS_HEADING_MAPPINGS
            )
        else:
            self.css_heading_mappings = {}

    def _preprocess_html_with_css_selectors(self, html_content: str) -> str:
        """
        Preprocess HTML by transforming elements matching CSS selectors into semantic heading tags.

        This allows custom HTML structures (e.g., <div class="pv-section">) to be recognized
        as headings during markdown conversion and subsequent by_title chunking.

        Args:
            html_content: Raw HTML string

        Returns:
            Preprocessed HTML with CSS-based elements transformed to heading tags
        """
        if not self.css_heading_mappings:
            return html_content

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            transformations_made = 0

            # Process each CSS selector mapping
            for selector, target_heading in self.css_heading_mappings.items():
                # Parse selector (simple implementation for tag.class format)
                parts = selector.split(".")
                if len(parts) == 2:
                    tag_name, class_name = parts
                    elements = soup.find_all(tag_name, class_=class_name)
                elif len(parts) == 1:
                    # Just a tag name, no class
                    tag_name = parts[0]
                    elements = soup.find_all(tag_name)
                else:
                    logger.warning(
                        f"Unsupported CSS selector format: {selector}. "
                        "Use 'tag.class' or 'tag' format."
                    )
                    continue

                # Transform matching elements to heading tags
                for elem in elements:
                    elem.name = target_heading
                    # Remove class attribute to avoid clutter in markdown
                    if elem.has_attr("class"):
                        del elem["class"]
                    transformations_made += 1

            if transformations_made > 0:
                logger.info(
                    f"CSS selector preprocessing: transformed {transformations_made} "
                    f"elements to headings using {len(self.css_heading_mappings)} mappings"
                )

            return str(soup)

        except Exception as e:
            logger.warning(
                f"Error during CSS selector preprocessing: {e}. "
                "Continuing with original HTML."
            )
            return html_content

    async def ingest(
        self, data: str | bytes, *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Ingest HTML data and yield markdown text for better chunking.

        Supports CSS selector-based preprocessing for custom HTML structures.
        """
        try:
            from html_to_markdown import convert, ConversionOptions

            # Convert bytes to string if needed
            if isinstance(data, bytes):
                html_content = data.decode("utf-8")
            else:
                html_content = data

            # Preprocess HTML with CSS selector mappings
            preprocessed_html = self._preprocess_html_with_css_selectors(
                html_content
            )

            # Convert HTML to Markdown with clean structure
            options = ConversionOptions(
                heading_style="atx", list_indent_width=2, bullets="*+-"
            )

            markdown = convert(preprocessed_html, options)
            logger.info(
                f"HTML to Markdown conversion: {len(html_content)} chars → "
                f"{len(markdown)} chars"
            )

            yield markdown

        except ImportError as e:
            logger.warning(
                f"html-to-markdown not installed: {e}, "
                "falling back to BeautifulSoup"
            )
            soup = BeautifulSoup(data, "html.parser")
            yield soup.get_text()
        except Exception as e:
            logger.error(
                f"Error converting HTML to Markdown: {str(e)}", exc_info=True
            )
            logger.warning("Falling back to BeautifulSoup due to error")
            soup = BeautifulSoup(data, "html.parser")
            yield soup.get_text()
