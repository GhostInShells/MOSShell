"""Playwright browser node — module-runtime hub over live browser domains.

Launch:
    moss nodes run nodes/browsers/playwright

Provides a hub channel (`playwright`) that governs module runtimes under
``domains/``. The model opens a domain to get a live runtime and drives it by
writing Python: ``exec`` (blocking) / ``aexec`` (background) / ``history``.
"""
from __future__ import annotations

import pathlib

from ghoshell_moss.channels.module_eval_channel import new_sandbox_hub_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

_NODE_DIR = pathlib.Path(__file__).resolve().parent
_DOMAINS = _NODE_DIR / "domains"


async def main(matrix: Matrix):
    channel = new_sandbox_hub_channel(
        matrix.processes,
        str(_DOMAINS),
        name="playwright",
        description=(
            "Playwright browser — open/close live browser runtimes. "
            "exec/aexec mutate a persistent browser; history is the scrollback."
        ),
    )
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
