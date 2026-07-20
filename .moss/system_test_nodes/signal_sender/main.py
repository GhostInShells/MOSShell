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
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.signals import NotifySignalMeta

    chan = new_channel(
        name="signal_sender",
        description="Send NotifySignal through session bus",
    )

    @chan.build.command(always_observe=True)
    async def send(message: str) -> str:
        """Send a NotifySignal with the given message."""
        meta = NotifySignalMeta()
        signal = meta.to_signal(message, description=f"signal_sender: {message}")
        matrix.session.add_signal(signal)
        return f"[signal_sender] sent: {message}"

    await matrix.provide_channel(chan)


if __name__ == "__main__":
    Matrix.discover().run(main)
