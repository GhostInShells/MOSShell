"""Verify ghost.channel() is registered into shell's channel tree.

Minimal end-to-end: MockGhost + simple channel -> GhostRuntime -> check metas.
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock._meta import MockGhostMeta
from ghoshell_moss.ghosts.mock._runtime import MockGhost
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.channel import Channel


class ChannelGhostMeta(MockGhostMeta):
    """MockGhostMeta that injects a channel into the ghost after factory."""

    def __init__(self, channel: Channel, **kwargs):
        super().__init__(**kwargs)
        self._channel = channel

    def factory(self, container):
        ghost: MockGhost = super().factory(container)
        ghost.set_channel(self._channel)
        return ghost


def build_test_channel() -> Channel:
    ch = new_channel(name="test_ghost", description="Ghost reflexive control channel")

    @ch.build.command(name="ping", doc="Respond pong.")
    def ping() -> str:
        return "pong"

    @ch.build.command(name="echo", doc="Echo back the message.")
    def echo(message: str) -> str:
        return f"echo: {message}"

    return ch


async def main():
    channel = build_test_channel()
    meta = ChannelGhostMeta(
        channel=channel,
        name="channel_test",
        description="Ghost with channel for testing.",
    )

    host = Host()
    gr = host.run_ghost(meta, run_shell=True)

    async with gr:
        shell = gr.moss.shell
        # NOTE: timeout=None waits for full refresh completion.
        # timeout=0 would return immediately before virtual children are scanned.
        await shell.refresh_metas(timeout=None)

        metas = shell.runtime.metas()
        print("=== channel tree metas ===\n")
        for path, m in metas.items():
            print(f"  '{path}'")
            print(f"    description: {m.description}")
            print(f"    commands   : {[c.name for c in m.commands]}")
            print()

        # command lookup
        ping_cmd = shell.runtime.get_command("ghost:ping")
        echo_cmd = shell.runtime.get_command("ghost:echo")
        ghost_child = shell.runtime.get_child_channel("ghost")

        print(f"get_child_channel('ghost')  -> {ghost_child.name() if ghost_child else 'None'}")
        print(f"get_command('ghost:ping')   -> {ping_cmd.name() if ping_cmd else 'None'}")
        print(f"get_command('ghost:echo')   -> {echo_cmd.name() if echo_cmd else 'None'}")

        gr.close()

    ghost_in_metas = any("ghost" in str(p) for p in metas)
    commands_ok = ping_cmd is not None and echo_cmd is not None
    print(f"\n{'PASS' if ghost_in_metas and commands_ok else 'FAIL'}: ghost in metas={ghost_in_metas}, commands={commands_ok}")


if __name__ == "__main__":
    asyncio.run(main())
