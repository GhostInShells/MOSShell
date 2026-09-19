"""The model-facing channel: window semantics over one store.

The model owns layout and ordering; the human owns where to look. Every command that
moves the screen lands immediately and returns a receipt; nothing here waits on a
person. The one place time is a first-class argument is the veil — a gesture declares
its duration, and the command resolves when that duration ends, so a gesture and a
spoken clause can share one clock.

Frames go out through an injected ``surface`` (headless = a no-op) rather than through
``CommandUtil``, for the same reason the store is injected: the node wires both to
``Matrix`` and tests wire both to a recorder.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from . import projection as P
from .audio import MockAudioSource
from .mock import demo_items
from .model import ScreenModel

__all__ = ["build_screen_channel"]


class _Surface(Protocol):
    async def broadcast(self, frame: dict[str, Any]) -> None: ...


class _NoSurface:
    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None


def _split_ids(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_screen_channel(
    model: ScreenModel,
    *,
    surface: _Surface | None = None,
    audio: MockAudioSource | None = None,
    surface_url: str | Callable[[], str] | None = None,
    name: str = "screen_manager",
    description: str | None = None,
) -> Channel:
    """Compose the screen channel over a store.

    :param model: window state — the single source of truth, shared with the surface.
    :param surface: the web broadcaster. None = headless (tests).
    :param audio: the audio source the mock commands drive. None = no audio knob.
    :param surface_url: where the human surface lives, surfaced as a warm notice
        fragment so the model can discover it without a fixed port. May be a
        callable (resolved lazily, after the surface binds its ephemeral port).
    """
    surface = surface or _NoSurface()

    def _url() -> str:
        if surface_url is None:
            return ""
        return surface_url() if callable(surface_url) else surface_url

    async def _emit(frame: dict[str, Any]) -> None:
        await surface.broadcast(frame)

    def _require(item_id: str) -> None:
        if model.get(item_id) is None:
            CommandUtil.raise_observe(f"no item {item_id!r} — pool() lists them")

    chan = new_channel(
        name=name,
        description=description
        or (
            "the screen body: windows (items) grouped and laid out on a stage, with a "
            "veil for gestures and a background for MOSS presence"
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "You own how the screen is arranged; the human owns where to look. Items "
            "are windows you materialize into groups; exactly one group is active and "
            "lays its items out on the stage. open() materializes, arrange() sets the "
            "order and the layout family, activate() switches the active group, "
            "fullscreen() makes one item fill the stage. The human can switch groups "
            "and toggle fullscreen too — the notice tells you when that happened. "
            "Veil gestures (mark / arrow / text) are the only time-aware commands: "
            "they resolve when their declared duration ends, so a gesture and a spoken "
            "clause share one clock."
        )

    @chan.build.command(name="open")
    async def open_window(
        id: str, url: str, label: str = "", group: str = ""
    ) -> str:
        """Materialize a window into a group (default: the active group).

        ``id`` is your handle for every later command; ``url`` is the window's
        content (a local node surface or any http page). A fresh ``group`` is created
        on first use.
        """
        try:
            item = model.open(id, url, label=label, group=group)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.open_frame(item))
        if item.group == model.active():
            await _emit(P.arrange_frame(model))
        n = len(model.group_items(item.group))
        return f"[screen] opened #{id} → #{item.group} ({n} item(s))"

    @chan.build.command(name="close")
    async def close_window(id: str) -> str:
        """Remove a window. An emptied group is deleted automatically."""
        was_active = model.group_of(id) == model.active()
        try:
            model.close(id)
        except KeyError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.close_frame(id))
        if was_active:
            await _emit(P.arrange_frame(model))
        return f"[screen] closed #{id}"

    @chan.build.command(name="activate")
    async def activate_group(group: str) -> str:
        """Switch the active group (you may switch too, not only the human)."""
        try:
            model.activate(group)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.activate_frame(model, by_model=True))
        return f"[screen] active #{group} ({len(model.active_items())} item(s))"

    @chan.build.command(name="arrange")
    async def arrange(ids: str, family: str = "grid", dir: str = "lr") -> str:
        """Set the active group's order and layout.

        ``ids`` is a comma-separated list of the active group's items, in display
        order. ``family`` is grid (equal split) or stack (one master + a strip);
        ``dir`` is lr or tb. The shape is derived from the item count — you never give
        sizes, only the order and the family.
        """
        try:
            layout = model.arrange(_split_ids(ids), family=family, dir=dir)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.arrange_frame(model))
        return f"[screen] #{model.active()} → {family} {dir} ({layout.cols}x{layout.rows})"

    @chan.build.command(name="fullscreen")
    async def fullscreen(id: str = "") -> str:
        """Make one item fill the stage; ``id`` empty exits fullscreen."""
        target = id or None
        try:
            model.set_fullscreen(target)
        except KeyError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.fullscreen_frame(target))
        return (
            f"[screen] fullscreen #{target}" if target else "[screen] fullscreen exited"
        )

    @chan.build.command(name="pool", always_observe=True)
    async def pool() -> str:
        """List the whole materialization pool — groups and their items in order."""
        lines = [f"active: #{model.active() or '-'}"]
        for g in model.groups():
            mark = "*" if g == model.active() else " "
            lines.append(f"{mark} #{g}: {', '.join(model.group_items(g))}")
        if model.fullscreen():
            lines.append(f"fullscreen: #{model.fullscreen()}")
        return "\n".join(lines)

    # -- veil gestures (time-aware) ----------------------------------------

    @chan.build.command(name="mark")
    async def mark(item: str, region: str = "full", duration: float = 2.0) -> str:
        """Highlight a region of an item — a veil gesture. Resolves after ``duration``.

        ``region`` is full / left / right / top / bottom.
        """
        _require(item)
        await _emit(P.veil_frame("mark", item=item, region=region, duration=duration))
        await asyncio.sleep(duration)
        return f"[screen] mark #{item}.{region} ({duration}s)"

    @chan.build.command(name="arrow")
    async def arrow(src: str, dst: str, duration: float = 2.0) -> str:
        """Draw an arrow from one item to another — a veil gesture. Resolves after
        ``duration``."""
        _require(src)
        _require(dst)
        await _emit(P.veil_frame("arrow", src=src, dst=dst, duration=duration))
        await asyncio.sleep(duration)
        return f"[screen] arrow #{src} → #{dst} ({duration}s)"

    @chan.build.command(name="text")
    async def text(content: str, duration: float = 2.5) -> str:
        """Show big text over the screen — a veil gesture. Resolves after ``duration``."""
        await _emit(P.veil_frame("text", content=content, duration=duration))
        await asyncio.sleep(duration)
        return f"[screen] text ({duration}s)"

    # -- mock / debug -------------------------------------------------------

    @chan.build.command(name="mock_scene")
    async def mock_scene() -> str:
        """Materialize the demo scene and activate #code. Debug aid — not the product."""
        for spec in demo_items():
            try:
                model.open(
                    spec["id"], spec["url"], label=spec["label"], group=spec["group"]
                )
            except ValueError:
                continue
        if model.active() not in model.groups():
            model.activate(model.groups()[0])
        await _emit(P.snapshot(model))
        return f"[screen] demo scene: {', '.join(model.groups())}"

    @chan.build.command(name="mock_audio")
    async def mock_audio(role: str = "idle") -> str:
        """Set the demo audio feed's mode — idle / ghost / user. Debug aid."""
        if audio is None:
            CommandUtil.raise_observe("no audio source wired")
        try:
            audio.set_mode(role)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        return f"[screen] audio {role}"

    # -- warm state ---------------------------------------------------------

    @chan.build.named_notices
    def notices() -> dict[str, str]:
        out: dict[str, str] = {}
        url = _url()
        if url:
            out["url"] = url
        groups = " ".join(
            f"#{g}({len(model.group_items(g))})" for g in model.groups()
        )
        out["groups"] = groups or "(none)"
        active = model.active()
        if not active:
            out["screen"] = "no active group — activate() one"
        else:
            order = ", ".join(model.active_items())
            fullscreen = f" · fullscreen #{model.fullscreen()}" if model.fullscreen() else ""
            out["screen"] = (
                f"active #{active} [{order}] {model.family()} {model.dir()}{fullscreen}"
            )
        return out

    return chan
