"""ghost.beside — the surface, and the factory that decides whether the organ exists at all.

The factory is the whole gate: it reads the ghost's config store, and a disabled beside returns
``None`` — no channel, no runtime, no step-end listener. A channel that exists but can do nothing
would be worse than an absent one: the model would discover a verb that always refuses.
"""

from __future__ import annotations

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from .config import EgoBesideMindConfig
from .runtime import BesideRuntime

__all__ = ["EgoBesideChannel", "build_beside_channel"]


def build_beside_channel(
        runtime: BesideRuntime,
        *,
        name: str = "beside",
        description: str = "",
) -> MutableChannel:
    """Build the beside channel around a live runtime."""
    if not description:
        description = "Parallel thought issued now and read later — one step behind, off-record."
    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        return (
            "## Beside (parallel thought)\n"
            "`think` issues a group of directions that runs one step behind you, off-record, and\n"
            "returns as one `<beside-think seq=\"k\">` delivery. It does not move you and does not\n"
            "touch the world."
        )

    @chan.build.command(name="think")
    async def think(text__: str) -> str:
        """Think several directions through in parallel, without holding the conversation.

        Each direction is one independent thought, started at the next step and answered off the
        record on this session's own context. The receipt is immediate; the conclusions arrive
        later as one delivery, shaped `<beside-think seq="k"><direction i="1">…</direction>
        </beside-think>`, with `<error i="n">` where a direction failed.

        Use it when a decision needs several independent angles and you must keep interacting
        meanwhile. Do not use it to think for yourself now — if you can focus, think directly.
        Do not re-think directions you already issued: they come back as one group.

        :param text__: directions separated by `|`; each is a self-contained instruction.
        """
        return runtime.think(text__)

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
        return build_beside_channel(runtime)
