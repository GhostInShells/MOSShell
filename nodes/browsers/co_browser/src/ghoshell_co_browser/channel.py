"""The co_browser channel: exec/aexec become observable frames.

The channel is a thin wrapper over the module_eval runtime: every call is
mirrored to the web surface as a frame, so the human sees the source code
and the result. There is **no per-command approval** — audit is the point,
not consent. Control lives at one place: the store's ``enabled`` flag,
wired into every command's ``available``. Flip it off and the channel
disappears from the model's interface entirely.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.tools.module_eval import ModuleEval

from .frame import Frame, FrameKind, FrameState
from .store import FrameStore

__all__ = ["build_co_browser_channel"]


class _Surface(Protocol):
    async def broadcast(self, frame: dict[str, Any]) -> None: ...


class _NoSurface:
    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None


def build_co_browser_channel(
    module_path: str,
    processes: Subprocesses | None,
    *,
    store: FrameStore,
    surface: _Surface | None = None,
    name: str = "co_browser",
    description: str | None = None,
    history_size: int = 20,
    surface_url: str | Callable[[], str] | None = None,
) -> Channel:
    """Compose the co_browser channel over a store and a ModuleEval runtime."""
    eval = ModuleEval(module_path, subprocesses=processes, history_size=history_size)
    surface = surface or _NoSurface()

    enabled = lambda: store.enabled

    def _url() -> str:
        if surface_url is None:
            return ""
        return surface_url() if callable(surface_url) else surface_url

    async def _head(frame: Frame) -> None:
        await surface.broadcast({"type": "frame.head", "frame": frame.view()})

    async def _full(frame: Frame) -> None:
        await surface.broadcast({"type": "frame.full", "frame": frame.view()})

    async def _run_frame(frame: Frame, timeout: float | None) -> str:
        try:
            result = await eval.exec(frame.source, timeout=timeout if timeout else 600.0)
        except asyncio.CancelledError:
            # ``clear``/``interrupt`` cancelled the channel-layer task. The child
            # keeps running the code — but this await is gone, so the result can
            # never come back through it. Say so on the frame rather than leaving
            # it stuck in ``running`` forever.
            store.append_result(frame.id, "[cleared] the channel task was cleared; "
                                          "the browser kept running this code, but "
                                          "its output is not coming back")
            store.set_state(frame.id, FrameState.ERROR)
            await _full(frame)
            raise
        except Exception as e:
            store.append_result(frame.id, f"[error] {e}")
            store.set_state(frame.id, FrameState.ERROR)
            await _full(frame)
            raise
        store.append_result(frame.id, result)
        store.set_state(frame.id, FrameState.DONE)
        await _full(frame)
        return result

    chan = new_channel(name=name, description=description or (
        "Playwright browser under human observation. Every exec streams onto a "
        "shared web surface so the human can see the source and the result. A "
        "master switch on that surface can disable the whole channel."
    ))

    @chan.build.instruction
    def instruction() -> str:
        return (
            "A live, persistent Python runtime for a headed browser. The human "
            "is watching the exec stream on a shared web surface: every call's "
            "source and result appears there. There is no per-command approval "
            "— the human's single control is a master switch. If these commands "
            "vanish from your interface, they turned the channel off; do "
            "something else.\n\n"
            f"{eval.source}"
        )

    @chan.build.startup
    async def on_startup() -> None:
        await eval.start()

    @chan.build.close
    async def on_close() -> None:
        await eval.shutdown()

    @chan.build.command(name="exec", always_observe=True, available=enabled)
    async def exec_code(text__: str, timeout: float = 30.0) -> str:
        """Execute Python code in the live browser runtime. Blocks until done.

        Wrap the body in CDATA so CTML doesn't chew it:
            <co_browser:exec><![CDATA[
            page.goto("https://example.com")
            print(page.title())
            ]]></co_browser:exec>

        Variables persist across calls. The source and the result show on the
        human's surface.
        """
        frame = store.new_frame(FrameKind.EXEC, text__)
        await _head(frame)
        return await _run_frame(frame, timeout=timeout)

    @chan.build.command(name="aexec", always_observe=False, available=enabled)
    async def exec_async(text__: str) -> str:
        """Fire Python code in the background. Returns a receipt id immediately.

        The frame streams onto the surface as it runs; the result is recorded
        into ``history`` when it finishes.
        """
        frame = store.new_frame(FrameKind.AEXEC, text__)
        await _head(frame)
        asyncio.create_task(_run_frame(frame, timeout=None))
        return f"[{name} #{frame.id}] queued"

    @chan.build.command(name="history", always_observe=True, available=enabled)
    def history(n: int = 10) -> str:
        """Recent executed commands + result summaries from the module runtime.

        :param n: how many entries to show.
        """
        return eval.history(n)

    @chan.build.named_notices
    def notices() -> dict[str, str]:
        running = sum(1 for f in store.frames() if f.state is FrameState.RUNNING)
        out = {name: f"running: {running}"}
        url = _url()
        if url:
            out["url"] = url
        if not store.enabled:
            out[f"{name}_disabled"] = "the human has disabled this channel"
        return out

    return chan
