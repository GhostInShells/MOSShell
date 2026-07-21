"""Trafilatura web content extraction node.

Channel: web_trafilatura
CTML:     <web_trafilatura:extract url="..." />
"""
import asyncio
import logging

import trafilatura

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

logger = logging.getLogger("WebTrafilatura")


async def main(matrix: Matrix):
    channel = new_channel(
        name="web_trafilatura",
        description="fetch any URL and extract clean Markdown content via trafilatura (local, no API key needed)",
    )

    @channel.build.command(always_observe=True)
    async def extract(url: str, output_format: str = "markdown") -> str:
        """Fetch a URL and extract the main readable content.

        :param url: the web page URL to fetch
        :param output_format: 'markdown' (default), 'txt', 'xml', or 'html'
        """
        loop = asyncio.get_running_loop()
        downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
        if downloaded is None:
            return f"Error: could not fetch URL: {url}"
        result = await loop.run_in_executor(
            None, trafilatura.extract, downloaded, output_format
        )
        if result is None:
            return f"Error: could not extract content from: {url}"
        return result

    @channel.build.command(always_observe=True)
    async def extract_batch(
        urls: list[str], output_format: str = "markdown"
    ) -> dict[str, str]:
        """Fetch multiple URLs concurrently and extract readable content from each.

        :param urls: list of web page URLs to fetch
        :param output_format: 'markdown' (default), 'txt', 'xml', or 'html'
        """
        results = {}

        async def fetch_one(url: str):
            loop = asyncio.get_running_loop()
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

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
