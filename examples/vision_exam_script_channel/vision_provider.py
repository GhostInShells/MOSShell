"""ScriptChannel-based vision provider.

This process exposes an `OpenCVVision` channel to its parent via stdio duplex
protocol, while also running the blocking OpenCV capture loop.

Run via `examples/vision_exam_script_channel/vision_proxy.py`.
"""

import asyncio
import logging
import sys
from queue import Queue
from threading import Thread

from ghoshell_moss import get_container
from ghoshell_moss.transports.script_channel.script_channel import ScriptChannelProvider, StdioProviderConnection
from ghoshell_moss_contrib.channels.opencv_vision import OpenCVVision


async def _build_stdio_connection(logger: logging.Logger | None = None) -> StdioProviderConnection:
    loop = asyncio.get_running_loop()

    reader = asyncio.StreamReader()
    read_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: read_protocol, sys.stdin.buffer)

    write_transport, write_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin,
        sys.stdout.buffer,
    )
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
    return StdioProviderConnection(reader=reader, writer=writer, logger=logger)


def main() -> None:
    container = get_container()
    logger = logging.getLogger(__name__)

    vision = OpenCVVision(container)
    channel = vision.as_channel()

    provider_queue: Queue[ScriptChannelProvider] = Queue(maxsize=1)

    def _provider_thread() -> None:
        async def _run() -> None:
            conn = await _build_stdio_connection(logger)
            provider = ScriptChannelProvider(connection=conn, container=container)
            provider_queue.put(provider)
            async with provider.arun(channel):
                await provider.wait_stop()

        asyncio.run(_run())

    t = Thread(target=_provider_thread, daemon=True)
    t.start()
    provider = provider_queue.get()

    try:
        # This blocks until user quits (press 'q' or close the window).
        vision.run_opencv_loop()
    finally:
        # Best-effort stop provider loop.
        try:
            provider.close()
            provider.wait_closed_sync()
        except Exception:
            logger.exception("failed to stop ScriptChannelProvider")
        vision.close()


if __name__ == "__main__":
    main()
