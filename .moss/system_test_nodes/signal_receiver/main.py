"""MOSS node cell entry point.

Start:  moss nodes run <path-to-this-dir>    # via CLI (foreground, CLI is owner)
Debug:  python main.py                        # ad-hoc launch (from_proc identity)

Explore:
    moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest
    moss codex blueprint channel_builder
    moss codex blueprint matrix
    moss ctml read
"""

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    import janus
    import asyncio
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.core.blueprint.mindflow import Signal

    chan = new_channel(
        name="signal_receiver",
        description="Receive signals from session bus via janus queue",
    )

    _queue: janus.Queue[Signal] = janus.Queue(maxsize=200)
    _received: list[str] = []

    def _on_signal(signal: Signal):
        _queue.sync_q.put(signal)

    async def _consume():
        while True:
            signal = await _queue.async_q.get()
            desc = signal.description or "(no desc)"
            msgs = [m.to_content_string() for m in (signal.messages or [])]
            body = " ".join(msgs) if msgs else "(no body)"
            line = f"[{desc}] {body}"
            print(line)
            _received.append(line)

    @chan.build.startup
    async def _start():
        matrix.session.on_signal(_on_signal)
        asyncio.create_task(_consume())

    @chan.build.command(always_observe=True)
    async def received(limit: int = 10) -> str:
        """Show recently received signals."""
        if not _received:
            return "[signal_receiver] no signals yet"
        return "\n".join(_received[-limit:])

    await matrix.provide_channel(chan)


if __name__ == "__main__":
    Matrix.discover().run(main)
