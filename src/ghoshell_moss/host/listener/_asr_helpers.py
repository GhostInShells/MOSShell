"""ASR helpers — silence timeout that synthesizes final from last partial."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable

from ghoshell_moss.contracts.asr import ASRResult

__all__ = ["iter_with_silence_timeout"]


async def iter_with_silence_timeout(
    agen: AsyncIterable,
    logger: logging.Logger,
    patience: float = 5.0,
) -> AsyncIterable[ASRResult]:
    """Wrap an async generator with a silence timeout.

    After the first non-empty result, if no subsequent non-empty result
    arrives within *patience* seconds, the iteration stops.  Empty-text
    results do NOT reset the timer.

    If the server never sends ``is_final=True`` before the timeout fires,
    this wrapper synthesizes a final result from the last partial text.
    """
    timeout: float | None = None
    last_result: ASRResult | None = None
    try:
        while True:
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
                else:
                    result = await agen.__anext__()
                if result.error:
                    yield result
                    break
                if result.text:
                    last_result = result
                    timeout = patience
                yield result
            except asyncio.TimeoutError:
                logger.info("ASR silence timeout after %.1fs, finalizing", patience)
                if last_result is not None and not last_result.is_final:
                    logger.info(
                        "Server never sent is_final=True — synthesizing from last partial: %s",
                        last_result.text,
                    )
                    yield ASRResult(text=last_result.text, is_final=True)
                break
            except StopAsyncIteration:
                break
    finally:
        await agen.aclose()
