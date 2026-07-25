"""interleaved_probe — semantic instrument panel for interleaved-ctml-thinking.

Every command here is a controlled test case for one cursor-projection
semantic: live progress / empty outcome / observe=True / runtime failure /
critical failure / per-channel FIFO cut points. No real-world side effects —
pure instrumentation, so experience rounds stop abusing bash `sleep` and `ls`
as makeshift probes.

Context: .ai_partners/features/workstreams/2026/07/interleaved-ctml-thinking/FEATURE.md

Start:  moss nodes run .moss/system_test_nodes/interleaved_probe
Debug:  python main.py
"""

import asyncio

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, MutableChannel, new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

probe = new_channel(
    name="probe",
    description=(
        "Interleaved-thinking test instrument. Each command exercises exactly one "
        "cursor-projection semantic; combine them to build multi-channel test tracks."
    ),
)


@probe.build.command()
async def slow(duration: float = 10.0, steps: int = 10) -> str:
    """Run for `duration` seconds, publishing live progress `step k/steps` each tick.

    The progress instrument: observe mid-flight to read the running task's
    progress string; interrupt mid-flight to test cut-point projection.
    """
    interval = duration / max(steps, 1)
    for step in range(1, steps + 1):
        CommandUtil.set_progress(f"step {step}/{steps}")
        await asyncio.sleep(interval)
    return f"finished {steps} steps in {duration:.1f}s"


@probe.build.command(always_observe=True)
async def emit(value: str = "ok") -> str:
    """Return `value` immediately. The happy path: non-empty observed result."""
    return value


@probe.build.command(always_observe=True)
async def silent_observed() -> None:
    """Return nothing, marked observe. The K9 case: an empty outcome must
    project as a placeholder WITH identity — existence never evaporates."""
    return None


@probe.build.command()
async def silent_plain() -> None:
    """Return nothing, no observe mark. Expected: folds into the success
    tally without per-item identity (the token-economy path)."""
    return None


@probe.build.command()
async def fail(msg: str = "boom") -> str:
    """Raise ValueError(msg). Runtime failure: errmsg must surface in the
    cursor map with identity, never silently count as done."""
    raise ValueError(msg)


@probe.build.command()
async def critical(msg: str = "critical boom") -> str:
    """Raise the interpreter's critical observation error. Interrupts the
    whole dispatch — the fail-closed gate trigger."""
    CommandUtil.raise_observe(msg)


def _tick_channel(name: str) -> MutableChannel:
    """A sibling FIFO track. Run probe.a:tick and probe.b:tick in parallel to
    test per-channel cursor positions and per-channel cancel cut anchors."""
    chan = new_channel(
        name=name,
        description=f"parallel FIFO track `{name}` for per-channel cursor / cut-point cases",
    )

    @chan.build.command()
    async def tick(times: int = 5, interval: float = 1.0) -> str:
        """Tick `times` times at `interval`s, publishing progress `tick k/times`."""
        for k in range(1, times + 1):
            CommandUtil.set_progress(f"tick {k}/{times}")
            await asyncio.sleep(interval)
        return f"ticked {times} x {interval:.1f}s"

    return chan


probe.import_channels(_tick_channel("a"), _tick_channel("b"))


async def main(matrix: Matrix):
    # A node without a membrane (channel) does not exist in the model's world.
    await matrix.provide_channel(probe)


if __name__ == "__main__":
    Matrix.discover().run(main)
