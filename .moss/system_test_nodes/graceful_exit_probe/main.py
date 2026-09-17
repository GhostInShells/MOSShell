"""Graceful-exit probe — reproduce double-SIGINT during matrix teardown.

Start:  python main.py    # self-fires two SIGINTs; verify exit code + cell log

This node is a regression harness for the ``cell-graceful-exit`` workstream.

The scenario it reproduces: a first SIGINT cancels the run loop; the main
coroutine then drains slowly so ``Matrix.arun`` is still inside its cleanup
``gather`` when a second SIGINT arrives. Before the fix, that second cancel
leaked a ``CancelledError`` out of ``arun`` and was logged as an ``ERROR`` at
``project.py:845``. After the fix, the process exits 0 with no new ERROR.
"""

import asyncio
import os
import signal
import threading
import time

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

# Drain long enough that the second SIGINT (0.5s later) lands while arun is
# still joining the cancelled main coroutine.
DRAIN_SECONDS = 2.0
SECOND_SIGINT_AFTER = 0.5

_ready = threading.Event()


async def main(matrix: Matrix) -> None:
    chan = new_channel(
        name="graceful_exit_probe",
        description="Regression probe: double-SIGINT during teardown must exit gracefully.",
    )

    @chan.build.command(always_observe=True)
    async def ping() -> str:
        """Return pong — proves the channel is live before teardown begins."""
        return "pong"

    _ready.set()
    try:
        await matrix.provide_channel(chan)
    except asyncio.CancelledError:
        await asyncio.sleep(DRAIN_SECONDS)
        raise


def _fire_double_sigint() -> None:
    if not _ready.wait(timeout=30):
        return
    time.sleep(1.0)  # let provide_channel actually suspend
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(SECOND_SIGINT_AFTER)
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == "__main__":
    # Normalize: a background/spawned process may inherit SIGINT as SIG_IGN,
    # which would mask the scenario entirely.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    threading.Thread(target=_fire_double_sigint, daemon=True).start()
    matrix = Matrix.discover()
    matrix.run(main)
    print("[graceful_exit_probe] exited cleanly", flush=True)
