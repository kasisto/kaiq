#!/usr/bin/env python3
"""
Quick manual test to verify CSS selector-based HTML chunking works.

Run from kaiq/py/:
    python test_html_css_selectors.py
"""

import asyncio
from core.base.providers import IngestionConfig
from core.parsers.text.html_parser import HTMLToMarkdownParser, DEFAULT_CSS_HEADING_MAPPINGS


class MockDatabaseProvider:
    pass


class MockLLMProvider:
    pass


async def test_html_css_selectors():
    """Test HTML CSS selector transformation."""

    # Sample HTML with custom CSS classes (generic example)
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Product Documentation</title></head>
    <body>
        <h2>User Guide</h2>

        <div class="pv-section">Getting Started</div>
        <p>Welcome to the product documentation.</p>

        <div class="pv-secondary-window-title">Installation</div>
        <p>Follow these steps to install the software.</p>

        <p class="pv-header">System Requirements</p>
        <p>Check your system meets the minimum requirements.</p>

        <div class="pv-task-header-anchor">Prerequisites Checklist</div>
        <ul>
            <li>Operating system compatibility</li>
            <li>Available disk space</li>
            <li>Network connectivity</li>
        </ul>

        <p class="pv-sub-header">Download Instructions</p>
        <p>Download the installer from the official website.</p>

        <p class="pv-task-header">Installation Steps</p>
        <ol>
            <li>Run the installer</li>
            <li>Accept license agreement</li>
            <li>Choose installation directory</li>
            <li>Complete the setup</li>
            <li>Restart your computer</li>
        </ol>

        <div class="pv-section">Configuration</div>
        <p>Configure the application settings.</p>

        <div class="pv-secondary-window-title">Basic Settings</div>
        <p>Essential configuration options.</p>

        <p class="pv-header">User Preferences</p>
        <p>Customize the application to your needs.</p>

        <div class="pv-task-header-anchor">Advanced Options</div>
        <p>Configure advanced features and settings.</p>
    </body>
    </html>
    """

    # Configure parser with CSS mappings
    config = IngestionConfig(
        provider="r2r",
        enable_html_css_heading_mappings=True,
        html_css_heading_mappings=DEFAULT_CSS_HEADING_MAPPINGS
    )

    parser = HTMLToMarkdownParser(
        config=config,
        database_provider=MockDatabaseProvider(),
        llm_provider=MockLLMProvider()
    )

    print("=" * 80)
    print("TESTING HTML CSS SELECTOR TRANSFORMATION")
    print("=" * 80)
    print("\nCSS Mappings:")
    for selector, heading in DEFAULT_CSS_HEADING_MAPPINGS.items():
        print(f"  {selector} → {heading}")

    print("\n" + "=" * 80)
    print("ORIGINAL HTML (excerpt):")
    print("=" * 80)
    print(html_content[:500] + "...")

    # Process HTML
    markdown_chunks = []
    async for chunk in parser.ingest(html_content):
        markdown_chunks.append(chunk)

    markdown = "".join(markdown_chunks)

    print("\n" + "=" * 80)
    print("CONVERTED MARKDOWN:")
    print("=" * 80)
    print(markdown)

    # Verify transformations
    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)

    checks = [
        ("## User Guide", "Standard H2 preserved"),
        ("## Getting Started", "div.pv-section → h2"),
        ("## Installation", "div.pv-secondary-window-title → h2"),
        ("### System Requirements", "p.pv-header → h3"),
        ("### Prerequisites Checklist", "div.pv-task-header-anchor → h3"),
        ("#### Download Instructions", "p.pv-sub-header → h4"),
        ("##### Installation Steps", "p.pv-task-header → h5"),
        ("## Configuration", "div.pv-section → h2"),
        ("## Basic Settings", "div.pv-secondary-window-title → h2"),
        ("### User Preferences", "p.pv-header → h3"),
        ("### Advanced Options", "div.pv-task-header-anchor → h3"),
    ]

    all_passed = True
    for expected, description in checks:
        if expected in markdown:
            print(f"✅ {description}: '{expected}'")
        else:
            print(f"❌ {description}: MISSING '{expected}'")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - CSS selector transformation working!")
    else:
        print("❌ SOME CHECKS FAILED - Review the markdown output above")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_html_css_selectors())
