"""The model-facing channel: window semantics over one store.

The model owns arrangement; the human owns where to look. Items materialize into
one pool. Arranging picks a subset of the pool into a named group and shows it;
everything not in a group floats on the desktop. So the model's main move is
``arrange(group, ids)`` — pick and place in one step.

Every command that moves the screen lands immediately and returns a receipt;
nothing here waits on a person. The one place time is a first-class argument is
the veil — a gesture declares its duration, and the command resolves when that
duration ends, so a gesture and a spoken clause can share one clock.

Frames go out through an injected ``surface`` (headless = a no-op) rather than
through ``CommandUtil``, for the same reason the store is injected: the node wires
both to ``Matrix`` and tests wire both to a recorder.
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
from .model import ScreenModel, compute_layout

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
    views_notice: Callable[[], str] | None = None,
    name: str = "webview_screen",
    description: str | None = None,
) -> Channel:
    """Compose the screen channel over a store.

    :param model: window state — the single source of truth, shared with the surface.
    :param surface: the web broadcaster. None = headless (tests).
    :param audio: the audio source the mock commands drive. None = no audio knob.
    :param surface_url: where the human surface lives, surfaced as a warm notice
        fragment so the model can discover it without a fixed port. May be a
        callable (resolved lazily, after the surface binds its ephemeral port).
    :param views_notice: an optional fragment supplier for adopted web views,
        called on each notice refresh (bound to the webview bridge). Returns ''
        to omit the fragment.
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
            "the screen body: windows (items) materialize into a pool, arranged "
            "into groups on a stage; unarranged items float on a desktop. Gestures "
            "ride a veil; MOSS presence rides the background."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "You own how the screen is arranged; the human owns where to look. "
            "open() materializes a window into the pool — by default it floats on "
            "the desktop. arrange(group, ids) is your main move: it pulls the items "
            "you name into a group, in order, with a layout family, and shows it. "
            "Everything you leave out stays floating on the desktop. activate() "
            "switches the view between groups and the desktop; dismiss() sends an "
            "item back to the desktop; fullscreen() makes one item fill the stage. "
            "The human can switch views and toggle fullscreen too — the notice tells "
            "you when that happened. Veil gestures (mark / arrow / text) are the "
            "only time-aware commands: they resolve when their declared duration "
            "ends, so a gesture and a spoken clause share one clock."
        )

    # -- materialization ---------------------------------------------------

    @chan.build.command(name="open")
    async def open_window(
        id: str, url: str, label: str = "", group: str = "", icon: str = ""
    ) -> str:
        """Materialize a window. By default it floats on the desktop.

        ``id`` is your handle for every later command; ``url`` is the window's
        content (a local node surface or any http page). Pass ``group`` to land it
        directly in a group; otherwise it waits on the desktop until you arrange it.
        """
        try:
            item = model.open(
                id, url, label=label, group=group, icon=icon or None
            )
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.open_frame(item))
        await _emit(P.state_frame(model))
        where = f"#{item.group}" if item.group else "desktop"
        return f"[screen] opened #{id} → {where}"

    @chan.build.command(name="arrange")
    async def arrange(ids: str, group: str, family: str = "grid", dir: str = "lr") -> str:
        """Pull items into a group, in order, and show it. Your main move.

        ``ids`` is a comma-separated list of item ids in display order — any subset
        of the pool, including items already in another group (they move here).
        ``group`` names the arrangement (created on first use). ``family`` is grid
        (equal split) or stack (one master + a strip); ``dir`` is lr or tb. The
        shape is derived from the count — you never give sizes.
        """
        try:
            layout = model.arrange(
                group, _split_ids(ids), family=family, dir=dir
            )
        except (ValueError, KeyError) as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.state_frame(model))
        return (
            f"[screen] #{group} ← [{', '.join(_split_ids(ids))}] "
            f"{family} {dir} ({layout.cols}x{layout.rows})"
        )

    @chan.build.command(name="activate")
    async def activate(group: str = "") -> str:
        """Show a group, or the desktop (empty ``group``). You may switch too."""
        try:
            model.activate(group)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.activate_frame(model, by_model=True))
        where = f"#{group}" if group else "desktop"
        return f"[screen] showing {where}"

    @chan.build.command(name="dismiss")
    async def dismiss(id: str) -> str:
        """Send an item back to the desktop — off its group, still in the pool."""
        _require(id)
        item = model.dismiss(id)
        await _emit(P.state_frame(model))
        return f"[screen] #{id} → desktop" if item else f"[screen] #{id} is already floating"

    @chan.build.command(name="destroy")
    async def destroy(id: str) -> str:
        """Remove an item from the pool. An adopted item is tombstoned, not revived."""
        _require(id)
        model.destroy(id)
        await _emit(P.close_frame(id))
        await _emit(P.state_frame(model))
        return f"[screen] removed #{id}"

    @chan.build.command(name="fullscreen")
    async def fullscreen(id: str = "") -> str:
        """Make one active-group item fill the stage; ``id`` empty exits."""
        target = id or None
        try:
            model.set_fullscreen(target)
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.fullscreen_frame(target))
        return (
            f"[screen] fullscreen #{target}" if target else "[screen] fullscreen exited"
        )

    @chan.build.command(name="float")
    async def float_group() -> str:
        """Send the whole active group back to the desktop."""
        freed = model.float_all()
        await _emit(P.state_frame(model))
        return f"[screen] {len(freed)} item(s) → desktop" if freed else "[screen] nothing to float"

    @chan.build.command(name="pool", always_observe=True)
    async def pool() -> str:
        """List the whole pool — the desktop, and every group in order."""
        arena = model.arena() or "desktop"
        lines = [f"showing: #{arena}", f"desktop: {', '.join(model.desktop_items()) or '-'}"]
        for g in model.groups():
            mark = "*" if g == model.arena() else " "
            fs = " [fullscreen]" if g == model.arena() and model.fullscreen() else ""
            lines.append(f"{mark} #{g}: {', '.join(model.group_items(g))}{fs}")
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
        """Materialize the demo scene into groups. Debug aid — not the product."""
        for spec in demo_items():
            if model.get(spec["id"]) is None:
                model.open(
                    spec["id"], spec["url"],
                    label=spec["label"], group=spec["group"],
                )
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
        desktop = model.desktop_items()
        if desktop:
            out["desktop"] = f"{len(desktop)} floating: {', '.join(desktop)}"
        if views_notice is not None:
            fragment = views_notice()
            if fragment:
                out["views"] = fragment
        arena = model.arena()
        if not arena:
            out["screen"] = f"desktop ({len(desktop)} floating)"
        else:
            order = ", ".join(model.arena_items())
            fs = f" · fullscreen #{model.fullscreen()}" if model.fullscreen() else ""
            out["screen"] = (
                f"#{arena} [{order}] {model.family()} {model.dir()}{fs}"
            )
        return out

    return chan
