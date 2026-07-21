"""CTML integration tests for web/trafilatura channel.

Run from app root:
    uv run pytest tests/ -v
"""
import asyncio
from unittest.mock import patch, MagicMock

import pytest

from ghoshell_moss.core.ctml import ctml_shell_test
from ghoshell_moss.core.blueprint.channel_builder import new_channel


HTML = "<html><body><p>Hello world.</p></body></html>"
EXTRACTED = "Hello world."


@pytest.mark.asyncio
async def test_extract_single_url():
    """CTML <apps.web_trafilatura:extract url="..."/> returns Markdown content."""
    chan = new_channel(name="web_trafilatura")

    @chan.build.command(always_observe=True)
    async def extract(url: str, output_format: str = "markdown") -> str:
        loop = asyncio.get_running_loop()
        import trafilatura

        downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
        if downloaded is None:
            return f"Error: could not fetch URL: {url}"
        result = await loop.run_in_executor(
            None, trafilatura.extract, downloaded, output_format
        )
        if result is None:
            return f"Error: could not extract content from: {url}"
        return result

    with (
        patch("trafilatura.fetch_url", return_value=HTML) as mock_fetch,
        patch("trafilatura.extract", return_value=EXTRACTED) as mock_extract,
    ):
        tasks = await ctml_shell_test(
            chan,
            ctml='<apps.web_trafilatura:extract url="https://example.com" />',
        )

    assert len(tasks) == 1
    result = await tasks[0]
    assert result == EXTRACTED
    mock_fetch.assert_called_once_with("https://example.com")
    mock_extract.assert_called_once_with(HTML, "markdown")


@pytest.mark.asyncio
async def test_extract_batch_concurrent():
    """CTML extract_batch fetches multiple URLs concurrently."""
    chan = new_channel(name="web_trafilatura")
    urls = ["https://a.com", "https://b.com"]

    @chan.build.command(always_observe=True)
    async def extract_batch(
        urls: list[str], output_format: str = "markdown"
    ) -> dict[str, str]:
        results = {}

        async def fetch_one(url: str):
            loop = asyncio.get_running_loop()
            import trafilatura

            downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
            if downloaded is None:
                results[url] = f"Error: could not fetch URL"
                return
            text = await loop.run_in_executor(
                None, trafilatura.extract, downloaded, output_format
            )
            results[url] = (
                text if text is not None else f"Error: could not extract content"
            )

        await asyncio.gather(*(fetch_one(u) for u in urls))
        return results

    with (
        patch("trafilatura.fetch_url", return_value=HTML) as mock_fetch,
        patch("trafilatura.extract", return_value=EXTRACTED) as mock_extract,
    ):
        tasks = await ctml_shell_test(
            chan,
            ctml='<apps.web_trafilatura:extract_batch urls=\'["https://a.com", "https://b.com"]\' />',
        )

    assert len(tasks) == 1
    result = await tasks[0]
    assert result == {"https://a.com": EXTRACTED, "https://b.com": EXTRACTED}
    assert mock_fetch.call_count == 2


@pytest.mark.asyncio
async def test_extract_fetch_error():
    """CTML extract returns error message when URL cannot be fetched."""
    chan = new_channel(name="web_trafilatura")

    @chan.build.command(always_observe=True)
    async def extract(url: str, output_format: str = "markdown") -> str:
        loop = asyncio.get_running_loop()
        import trafilatura

        downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
        if downloaded is None:
            return f"Error: could not fetch URL: {url}"
        result = await loop.run_in_executor(
            None, trafilatura.extract, downloaded, output_format
        )
        if result is None:
            return f"Error: could not extract content from: {url}"
        return result

    with patch("trafilatura.fetch_url", return_value=None):
        tasks = await ctml_shell_test(
            chan,
            ctml='<apps.web_trafilatura:extract url="https://down.example.com" />',
        )

    result = await tasks[0]
    assert "Error: could not fetch URL" in result
