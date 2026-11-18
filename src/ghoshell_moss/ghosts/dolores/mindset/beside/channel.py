"""ghost.beside — the surface, and the factory that decides whether the organ exists at all.

The factory is the whole gate: it reads the ghost's config store, and a disabled beside returns
``None`` — no channel, no runtime, no step-end listener. A channel that exists but can do nothing
would be worse than an absent one: the model would discover a verb that always refuses.
"""

from __future__ import annotations

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, Channel, new_channel

from .config import EgoBesideMindConfig
from .runtime import BesideRuntime

__all__ = ["EgoBesideChannel", "build_beside_channel"]


def _think_doc(max_directions: int) -> str:
    """`think`'s doc — read at the moment of calling, so it is one motion, not a rule list.

    The cap is rendered in rather than written down: it is a config value, and a docstring that
    hardcodes a number drifts from the config the first time either one changes.
    """
    return (
        "Ask yourself in directions — questions, not briefs — so no direction has to wait its\n"
        "turn: a parallel you reads this same context, this command included, and answers from\n"
        "here in later moments.\n"
        "\n"
        f":param text__: one direction per paragraph ({max_directions} at max).\n"
    )


def build_beside_channel(
        runtime: BesideRuntime,
        *,
        name: str = "beside",
        description: str = "",
        debug: Channel | None = None,
) -> MutableChannel:
    """Build the beside channel around a live runtime."""
    if not description:
        description = "Parallel thought issued now and read later — one step behind, off-record."
    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        # Only what the interface cannot say: the timing. The cap is config, not signature, and
        # it rides the command's own docstring — one home, so the number cannot drift.
        return (
            "## Beside (parallel thought)\n"
            "A group dispatched now is read one step behind: it does not move you and does not\n"
            "touch the world."
        )

    @chan.build.command(name="think", doc=lambda: _think_doc(runtime.directions_max()))
    async def think(text__: str) -> str:
        return runtime.think(text__)

    if debug is not None:
        chan.build.add_virtual_channel(debug)

    return chan


class EgoBesideChannel:
    """`ghost.beside` as a factory: IoC → config store → config → channel (or nothing)."""

    @classmethod
    def factory(cls, container: IoCContainer) -> MutableChannel | None:
        """Mount the organ only when config allows and a runtime was built for this boot."""
        from ghoshell_moss.ghosts.dolores.configs import DoloresConfigStore

        store = container.get(DoloresConfigStore)
        if store is None:
            return None
        config = store.get_or_create(EgoBesideMindConfig())
        if not config.enabled:
            return None
        runtime = container.get(BesideRuntime)
        if runtime is None:
            # enabled but not built (a container without the ghost's wiring): no organ rather
            # than a channel whose only verb raises on first use.
            return None

        # The debug child is mounted only when the mechanism it exercises is actually wired:
        # a task runner and a trajectory. A debug verb that can only fail is worse than none.
        from ghoshell_moss.memento import Memento

        from ghoshell_moss.ghosts.dolores.mindset.emanation_task import EmanationTasks
        from .debug import build_debug_channel

        tasks = container.get(EmanationTasks)
        memento = container.get(Memento)
        debug = (
            build_debug_channel(tasks, memento)
            if tasks is not None and memento is not None
            else None
        )
        return build_beside_channel(runtime, debug=debug)
