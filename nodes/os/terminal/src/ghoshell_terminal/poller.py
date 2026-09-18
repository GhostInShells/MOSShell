"""Live output, reconstructed from a window that keeps forgetting.

The subprocess contract has no incremental output API: ``ProcessOutput`` offers
only a bounded in-memory window (``stdout(offset, limit)``) and a whole-file path
that is written block-buffered, so it is invisible to a reader until the process
ends. Neither can be tailed live.

So the node polls the window and works out what is new by matching the tail of
what it has already seen against the head of what it sees now. The window drops
old lines as it fills, which is exactly why ``offset`` cannot be used as a
cursor — an index into the window shifts under you as lines age out.

Two consequences worth knowing (both are limits of the subprocess layer, not of
this module): output is only visible at line granularity — a partial line with no
newline yet never appears — and output faster than ``buffer_lines`` per poll
interval can slip through unseen.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

__all__ = ["stream_output", "PollableProcess"]

_POLL_INTERVAL = 0.35
_KNOWN_LINES = 600


class _Output(Protocol):
    def stdout(self, *, offset: int = 0, limit: int = 0) -> str: ...

    async def wait_drained(self) -> None: ...


class _Inner(Protocol):
    returncode: int | None


class PollableProcess(Protocol):
    @property
    def output(self) -> _Output | None: ...

    @property
    def process(self) -> _Inner: ...


def _window(output: _Output) -> list[str]:
    return output.stdout(limit=0).splitlines(keepends=True)


def _new_lines(known: list[str], window: list[str]) -> list[str]:
    """The lines in ``window`` that ``known`` has not seen.

    The window is ``known``'s suffix (possibly slid forward), so the boundary is
    the longest overlap between the tail of ``known`` and the head of ``window``.
    """
    limit = min(len(known), len(window))
    for k in range(limit, 0, -1):
        if known[-k:] == window[:k]:
            return window[k:]
    return window


async def stream_output(
    managed: PollableProcess,
    on_lines: Callable[[list[str]], Awaitable[None]],
    *,
    interval: float = _POLL_INTERVAL,
) -> None:
    """Push every new output line to ``on_lines`` until the process exits.

    Returns once the process has exited and the streams are fully drained, so
    the caller's buffer is complete when this returns.
    """
    output = managed.output
    if output is None:
        return
    known: list[str] = []
    while True:
        window = _window(output)
        new = _new_lines(known, window)
        if new:
            known = (known + new)[-max(_KNOWN_LINES, len(window)):]
            await on_lines(new)
        if managed.process.returncode is not None:
            break
        await asyncio.sleep(interval)

    await output.wait_drained()
    window = _window(output)
    new = _new_lines(known, window)
    if new:
        await on_lines(new)
