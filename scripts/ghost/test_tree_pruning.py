"""Verify stateful ghost channel with sub-channel TREE pruning.

Tree structure:
  ghost (StatefulChannel)
  ├── voice (main_state — always visible)
  │   └── speak
  ├── head  (awake & bored — shared across two states)
  │   └── look_at, nod
  └── body  (awake only)
      └── walk, stand

State pruning:
  - sleeping: voice + wake_up
  - awake:    voice + head + body + all commands
  - bored:    voice + head + play/explore

State switching via command system — same path the model uses via CTML.
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock._meta import MockGhostMeta
from ghoshell_moss.ghosts.mock._runtime import MockGhost
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.channel import Channel


class StatefulGhostMeta(MockGhostMeta):
    def __init__(self, channel: Channel, **kwargs):
        super().__init__(**kwargs)
        self._channel = channel

    def factory(self, container):
        ghost: MockGhost = super().factory(container)
        ghost.set_channel(self._channel)
        return ghost


def build_voice_channel() -> Channel:
    ch = new_channel("voice", "Voice output. Always available.")
    ch.build.command(name="speak", doc="Say something.")(lambda text: f"said: {text}")
    return ch


def build_head_channel() -> Channel:
    ch = new_channel("head", "Head control. Available when awake or bored.")
    ch.build.command(name="look_at", doc="Look at target.")(lambda target: f"looking at {target}")
    ch.build.command(name="nod", doc="Nod head.")(lambda: "nodded")
    return ch


def build_body_channel() -> Channel:
    ch = new_channel("body", "Body control. Only available when awake.")
    ch.build.command(name="walk", doc="Walk to target.")(lambda target: f"walking to {target}")
    ch.build.command(name="stand", doc="Stand still.")(lambda: "standing")
    return ch


def build_ghost_channel(voice, head, body) -> Channel:
    ch = new_prime_channel(name="ghost", description="Ghost reflexive control")
    ch.main_state().import_channels(voice)

    sleeping = ch.new_state("sleeping", "Ghost is sleeping.")

    @sleeping.command(name="wake_up", doc="Wake up.")
    def wake_up() -> str:
        return "woken"

    awake = ch.new_state("awake", "Ghost is fully awake.")
    awake.import_channels(head, body)

    @awake.command(name="sleep", doc="Go to sleep.")
    def sleep() -> str:
        return "sleeping"

    bored = ch.new_state("bored", "Ghost is bored.")
    bored.import_channels(head)

    @bored.command(name="play", doc="Play something.")
    def play(game: str = "idle") -> str:
        return f"playing {game}"

    @bored.command(name="explore", doc="Explore environment.")
    def explore() -> str:
        return "exploring"

    return ch


def _ghost_paths(metas: dict) -> set:
    return {p for p in metas if p.startswith("ghost")}


async def switch_state(shell, state_name: str):
    """Switch ghost state via command system — same path the model uses."""
    await shell.runtime.execute_command("ghost:switch_state", kwargs={"name": state_name})


async def main():
    voice = build_voice_channel()
    head = build_head_channel()
    body = build_body_channel()
    channel = build_ghost_channel(voice, head, body)

    meta = StatefulGhostMeta(channel=channel, name="tree_test", description="Tree pruning test.")
    host = Host()
    gr = host.run_ghost(meta, run_shell=True)

    async with gr:
        shell = gr.moss.shell
        await shell.refresh_metas(timeout=None)
        runtime = shell.runtime

        # -- initial --
        paths = _ghost_paths(runtime.metas())
        assert "ghost" in paths
        assert "ghost.voice" in paths
        assert "ghost.head" not in paths
        assert "ghost.body" not in paths
        print(f"=== 1. Initial: only voice ===\n  {sorted(paths)}\n")

        # -- awake --
        await switch_state(shell, "awake")
        await shell.refresh_metas(timeout=None)
        paths = _ghost_paths(runtime.metas())
        assert "ghost.voice" in paths
        assert "ghost.head" in paths
        assert "ghost.body" in paths
        assert runtime.get_command("ghost:voice:speak") is not None
        assert runtime.get_command("ghost:head:look_at") is not None
        assert runtime.get_command("ghost:body:walk") is not None
        print(f"=== 2. Awake: voice + head + body ===\n  {sorted(paths)}\n")

        # -- sleeping --
        await switch_state(shell, "sleeping")
        await shell.refresh_metas(timeout=None)
        paths = _ghost_paths(runtime.metas())
        assert "ghost.voice" in paths
        assert "ghost.head" not in paths
        assert "ghost.body" not in paths
        print(f"=== 3. Sleeping: head/body pruned ===\n  {sorted(paths)}\n")

        # -- bored --
        await switch_state(shell, "bored")
        await shell.refresh_metas(timeout=None)
        paths = _ghost_paths(runtime.metas())
        assert "ghost.voice" in paths
        assert "ghost.head" in paths
        assert "ghost.body" not in paths
        print(f"=== 4. Bored: voice + head (shared), body pruned ===\n  {sorted(paths)}\n")

        # -- back to awake --
        await switch_state(shell, "awake")
        await shell.refresh_metas(timeout=None)
        paths = _ghost_paths(runtime.metas())
        assert "ghost.voice" in paths
        assert "ghost.head" in paths
        assert "ghost.body" in paths
        print(f"=== 5. Awake: full tree restored ===\n  {sorted(paths)}\n")

        gr.close()

    print("=== PASS: tree pruning via command system ===")


if __name__ == "__main__":
    asyncio.run(main())
