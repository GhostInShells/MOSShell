"""Verify stateful ghost channel: states with different commands per state.

Scenario:
  Ghost has 3 states: sleeping, awake, bored.
  - sleeping: only wake_up
  - awake: head_control, body_control, sleep
  - bored: play, explore

Validates state switching via the command system (same path the model uses via CTML).
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock._meta import MockGhostMeta
from ghoshell_moss.ghosts.mock._runtime import MockGhost
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel
from ghoshell_moss.core.concepts.channel import Channel


class StatefulGhostMeta(MockGhostMeta):
    def __init__(self, channel: Channel, **kwargs):
        super().__init__(**kwargs)
        self._channel = channel

    def factory(self, container):
        ghost: MockGhost = super().factory(container)
        ghost.set_channel(self._channel)
        return ghost


def build_ghost_channel() -> Channel:
    ch = new_prime_channel(name="ghost", description="Ghost reflexive control surface")

    sleeping = ch.new_state("sleeping", "Ghost is sleeping. Only wake_up is available.")

    @sleeping.command(name="wake_up", doc="Wake up from sleep.")
    def wake_up() -> str:
        return "woken up"

    @sleeping.idle
    async def sleep_idle():
        pass

    awake = ch.new_state("awake", "Ghost is fully awake. Head and body control available.")

    @awake.command(name="head_control", doc="Control head: yaw, pitch, roll.")
    def head_control(yaw: float = 0.0, pitch: float = 0.0) -> str:
        return f"head -> yaw={yaw}, pitch={pitch}"

    @awake.command(name="body_control", doc="Control body: posture, gesture.")
    def body_control(posture: str = "stand") -> str:
        return f"body -> {posture}"

    @awake.command(name="sleep", doc="Go to sleep.")
    def sleep() -> str:
        return "going to sleep"

    @awake.idle
    async def awake_idle():
        pass

    bored = ch.new_state("bored", "Ghost is bored. Entertainment commands available.")

    @bored.command(name="play", doc="Play a game or activity.")
    def play(game: str = "idle") -> str:
        return f"playing {game}"

    @bored.command(name="explore", doc="Explore the environment.")
    def explore() -> str:
        return "exploring"

    @bored.idle
    async def bored_idle():
        pass

    return ch


def _ghost_meta(metas: dict):
    for path, m in metas.items():
        if path == "ghost" or path.endswith(".ghost"):
            return m
    return None


async def switch_state(shell, state_name: str):
    """Switch ghost state via command system — same path the model uses."""
    await shell.runtime.execute_command("ghost:switch_state", kwargs={"name": state_name})


async def main():
    channel = build_ghost_channel()
    meta = StatefulGhostMeta(
        channel=channel, name="stateful_test", description="Stateful ghost test.",
    )

    host = Host()
    gr = host.run_ghost(meta, run_shell=True)

    async with gr:
        shell = gr.moss.shell
        await shell.refresh_metas(timeout=None)
        runtime = shell.runtime

        gm = _ghost_meta(runtime.metas())
        assert gm is not None
        assert len(gm.states) == 3
        assert "switch_state" in {c.name for c in gm.commands}
        print("=== 1. Ghost channel in metas, switch_state available ===")
        print(f"  states: {list(gm.states.keys())}")
        print(f"  commands: {[c.name for c in gm.commands]}\n")

        # -- initial: no state-specific commands --
        assert runtime.get_command("ghost:wake_up") is None
        assert runtime.get_command("ghost:head_control") is None
        print("=== 2. Initial: no state commands visible ===\n")

        # -- awake --
        await switch_state(shell, "awake")
        await shell.refresh_metas(timeout=None)
        gm = _ghost_meta(runtime.metas())
        assert gm.current_state == "awake"
        cmds = {c.name for c in gm.commands}
        assert "head_control" in cmds
        assert "body_control" in cmds
        assert "sleep" in cmds
        assert "wake_up" not in cmds
        print("=== 3. Awake: head_control, body_control, sleep ===\n")

        # -- sleeping --
        await switch_state(shell, "sleeping")
        await shell.refresh_metas(timeout=None)
        gm = _ghost_meta(runtime.metas())
        assert gm.current_state == "sleeping"
        cmds = {c.name for c in gm.commands}
        assert "wake_up" in cmds
        assert "head_control" not in cmds
        print("=== 4. Sleeping: only wake_up ===\n")

        # -- bored --
        await switch_state(shell, "bored")
        await shell.refresh_metas(timeout=None)
        gm = _ghost_meta(runtime.metas())
        assert gm.current_state == "bored"
        cmds = {c.name for c in gm.commands}
        assert "play" in cmds
        assert "explore" in cmds
        print("=== 5. Bored: play + explore ===\n")

        # idle hooks
        states = channel.states()
        for name in ("sleeping", "awake", "bored"):
            assert len(states[name]._on_idle_funcs) == 1
        print("=== 6. Idle hooks registered per state ===")

        gr.close()

    print("\n=== PASS: stateful ghost channel via command system ===")


if __name__ == "__main__":
    asyncio.run(main())
