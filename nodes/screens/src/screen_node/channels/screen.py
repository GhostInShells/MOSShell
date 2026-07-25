"""Screen root channel — window lifecycle + layout switching + context_messages.

Channel tree:

    screen                               # root (StatefulChannel)
      [module] window mgmt: open / close / set_background / switch_layout
      [state]  solo:         focus / front / float / clear
      [state]  split:        focus_left / focus_right / front / float / clear

Events: peek/drain bucket via EventBucket. context_messages shows window
directory, layout snapshot, and recent events.
"""

from __future__ import annotations

import asyncio
import itertools

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.blueprint.states_channel import (
    new_stateful_channel,
    StatefulChannelRuntime,
)
from ghoshell_moss.message import Message

from ..bridge import ScreenBridge
from ..bucket import EventBucket

# Number of recent events to show in context_messages peek.
_PEEK_N = 10

# Short ID counter.
_id_counter = itertools.count(1)


def _next_id(label: str) -> str:
    """Generate a short window ID. Uses label if provided, otherwise #N."""
    if label:
        return f"#{label}"
    n = next(_id_counter)
    return f"#{n:02d}"


def build_screen_channel(bridge: ScreenBridge, bucket: EventBucket):
    """Build and return the complete screen channel tree."""

    # ---- root: StatefulChannel -----------------------------------------------

    screen = new_stateful_channel(
        name="screen",
        description=(
            "screen body compositor. manage windows (any URL) across four slot "
            "layers: background, focus, front, float. windows are URL resources; "
            "this channel owns their lifecycle (open/close/set_background). "
            "layout operations (focus/front/float/clear) are on the current "
            "layout state. badge protocol: web-standard navigator.setAppBadge()."
        ),
    )

    # ---- window management module (always active) ----------------------------

    # Register commands on the main (default) state builder.
    # The main state IS the window management module — always active.

    # open
    @screen.main_state().command(always_observe=True)
    async def open(url: str, label: str = "") -> str:
        """Register a URL as a screen window. Returns the window's short ID.

        Open creates the window resource but does NOT place it in any slot.
        Use layout commands (focus/front/float) to place it. The page's
        icon/title are extracted when the webview first loads; a hidden
        webview is used if the window is registered as float (meta-only).

        :param url: full HTTP URL of the window content
        :param label: short human-readable label, used to generate the window ID
        """
        window_id = _next_id(label)
        f = bridge.submit("open_window", {"id": window_id, "url": url, "label": label})
        await asyncio.wrap_future(f)
        return f"opened {window_id} -> {url}"

    # close
    @screen.main_state().command(always_observe=True)
    async def close(id: str) -> str:
        """Close a window and remove it from all slots.

        :param id: the window's short ID (e.g. #mail, #01_blog)
        """
        f = bridge.submit("close_window", {"id": id})
        await asyncio.wrap_future(f)
        return f"closed {id}"

    # set_background
    @screen.main_state().command(always_observe=True)
    async def set_background(id: str) -> str:
        """Set the global background window (digital avatar / ambient layer).

        The background slot is cross-layout — it persists across layout switches.
        Use an empty string to clear: set_background(id="")

        :param id: window ID, or empty string to clear
        """
        f = bridge.submit("set_background", {"id": id})
        await asyncio.wrap_future(f)
        return f"background = {id}" if id else "background cleared"

    # switch_layout
    @screen.main_state().command(always_observe=True)
    async def switch_layout(name: str) -> str:
        """Switch to a named layout. Occupies screen during transition.

        Available layouts: solo (single focus + front strip + float shelf).
        During the transition, layout commands are blocked.

        :param name: layout name (e.g. 'solo')
        """
        # Update QML first (instant visual switch)
        f = bridge.submit("switch_layout", {"name": name})
        await asyncio.wrap_future(f)
        # Switch channel state for proper command set swap
        runtime = CommandUtil.runtime()
        if isinstance(runtime, StatefulChannelRuntime):
            await runtime.switch_state(name)
        return f"switched to {name} layout"

    # drain
    @screen.main_state().command(always_observe=True)
    async def drain() -> str:
        """Drain accumulated interaction events from the bucket.

        Returns events that the model hasn't seen yet. In MCP mode (no signal
        path), drained events appear in context_messages. In Ghost mode, they
        can be forwarded as signals.
        """
        events = await bucket.drain()
        if not events:
            return "no events to drain"
        lines = [f"{len(events)} event(s) drained:"]
        for ev in events:
            lines.append(f"  [{ev['type']}] {ev.get('window_id', '')} {ev.get('action', ev.get('badge', ''))}")
        return "\n".join(lines)

    # ---- context_messages ----------------------------------------------------

    @screen.build.context_messages
    async def screen_context() -> list[Message]:
        bucket.start()  # idempotent

        snapshot = bridge.snapshot()
        if snapshot is None:
            return [Message.new(tag="screen").with_content("screen not ready")]

        messages: list[Message] = []

        # 1. Window directory
        windows = snapshot.get("windows", {})
        if windows:
            lines = []
            for wid, win in windows.items():
                parts = [wid, win.get("label", ""), win.get("url", "")]
                badge = win.get("badge", 0)
                title = win.get("title", "")
                if badge:
                    parts.append(f"badge={badge}")
                if title:
                    parts.append(f'"{title}"')
                lines.append("  ".join(parts))
            messages.append(
                Message.new(tag="screen", attributes={"section": "windows"}).with_content(
                    "windows:\n" + "\n".join(f"  {l}" for l in lines)
                )
            )
        else:
            messages.append(
                Message.new(tag="screen", attributes={"section": "windows"}).with_content(
                    "windows: (none)"
                )
            )

        # 2. Layout state
        layout = snapshot.get("layout", {})
        bg = layout.get("background", "")
        layout_name = layout.get("name", "solo")
        slots = layout.get("slots", {})

        layout_lines = [f"layout: {layout_name}"]
        if bg:
            layout_lines.append(f"  background: {bg}")
        focus_id = slots.get("focus", "")
        layout_lines.append(f"  focus: {focus_id}" if focus_id else "  focus: —")
        front_ids = slots.get("front", [])
        layout_lines.append(f"  front: {' '.join(front_ids)}" if front_ids else "  front: —")
        float_ids = slots.get("float", [])
        layout_lines.append(f"  float: {' '.join(float_ids)}" if float_ids else "  float: —")

        messages.append(
            Message.new(tag="screen", attributes={"section": "layout"}).with_content(
                "\n".join(layout_lines)
            )
        )

        # 3. Recent events (peek, non-destructive)
        events = bucket.peek(_PEEK_N)
        if events:
            event_lines = []
            for ev in events[-5:]:
                et = ev["type"]
                wid = ev.get("window_id", "")
                if et == "human_clicked":
                    event_lines.append(f"  [click] {wid} {ev.get('action', '')}")
                elif et == "badge_changed":
                    event_lines.append(f"  [badge] {wid} {ev.get('badge', '')}")
                else:
                    event_lines.append(f"  [{et}] {wid}")
            if event_lines:
                messages.append(
                    Message.new(tag="screen", attributes={"section": "events"}).with_content(
                        "recent events (peek):\n" + "\n".join(event_lines)
                    )
                )

        return messages

    # ---- layout states -------------------------------------------------------

    _register_solo_state(screen, bridge)
    _register_split_state(screen, bridge)

    # ---- startup / close -----------------------------------------------------

    @screen.build.startup
    async def _on_startup() -> None:
        bucket.start()
        logger = CommandUtil.logger()
        logger.info("screen channel started")

    @screen.build.close
    async def _on_close() -> None:
        pass

    return screen


# ---- solo layout state -------------------------------------------------------

def _register_solo_state(screen, bridge: ScreenBridge) -> None:
    solo = screen.new_state("solo", "single focus — one main slot + front strip + float shelf")

    @solo.command(always_observe=True)
    async def focus(id: str) -> str:
        """Move a window into the main focus slot.

        :param id: window short ID
        """
        f = bridge.submit("focus_window", {"id": id, "slot": "focus"})
        await asyncio.wrap_future(f)
        return f"focused {id}"

    @solo.command(always_observe=True)
    async def front(id: str, index: int = 0) -> str:
        """Move a window into the expanded front strip at an optional position.

        :param id: window short ID
        :param index: position in front strip (0 = first)
        """
        f = bridge.submit("front_window", {"id": id, "index": index})
        await asyncio.wrap_future(f)
        return f"front {id}"

    @solo.command(always_observe=True)
    async def float(id: str) -> str:
        """Move a window to the float layer — meta icon only, no rendered webview.

        :param id: window short ID
        """
        f = bridge.submit("float_window", {"id": id})
        await asyncio.wrap_future(f)
        return f"floated {id}"

    @solo.command(always_observe=True)
    async def clear(slot: str = "focus") -> str:
        """Clear a slot, returning its window to the float layer.

        :param slot: which slot to clear — 'focus', 'front', or 'float'
        """
        f = bridge.submit("clear_slot", {"slot": slot})
        await asyncio.wrap_future(f)
        return f"cleared {slot}"

    screen.with_state(solo, is_default=True)


# ---- split layout state (placeholder) ----------------------------------------

def _register_split_state(screen, bridge: ScreenBridge) -> None:
    split = screen.new_state("split", "dual focus — two independent focus slots + front + float")

    # Minimal implementation: same commands as solo, just declared separately.
    # The QML handles the slot mapping; the channel interface is identical.
    # Future: left/right-specific commands (focus_left/focus_right).

    @split.command(always_observe=True)
    async def focus(id: str, slot: str = "focus") -> str:
        """Move a window into a focus slot.

        :param id: window short ID
        :param slot: 'left' or 'right' focus slot (default: 'left')
        """
        f = bridge.submit("focus_window", {"id": id, "slot": slot})
        await asyncio.wrap_future(f)
        return f"focused {id} in {slot}"

    @split.command(always_observe=True)
    async def front(id: str, index: int = 0) -> str:
        """Move a window into the expanded front strip.

        :param id: window short ID
        :param index: position in front strip
        """
        f = bridge.submit("front_window", {"id": id, "index": index})
        await asyncio.wrap_future(f)
        return f"front {id}"

    @split.command(always_observe=True)
    async def float(id: str) -> str:
        """Move a window to the float layer — meta icon only.

        :param id: window short ID
        """
        f = bridge.submit("float_window", {"id": id})
        await asyncio.wrap_future(f)
        return f"floated {id}"

    @split.command(always_observe=True)
    async def clear(slot: str = "focus") -> str:
        """Clear a slot, returning its window to the float layer.

        :param slot: 'left', 'right', 'front', or 'float'
        """
        f = bridge.submit("clear_slot", {"slot": slot})
        await asyncio.wrap_future(f)
        return f"cleared {slot}"

    screen.with_state(split)
