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
import contextlib
import itertools
import json
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from . import projection as P
from .audio import AudioSource
from .mock import demo_items
from .model import ScreenModel, compute_layout

__all__ = ["build_screen_channel"]


class _ClauseFeed(Protocol):
    """Whatever the node wires to the bus — the stage's ear on the conversation."""

    @property
    def running(self) -> bool: ...
    async def start(self, on_clause: Callable[[Any], Awaitable[None]]) -> None: ...
    async def stop(self) -> None: ...


class _EventFeed(Protocol):
    """Whatever the node wires to the bus — the ghost's own state and actions.

    Same shape as ``_ClauseFeed`` on purpose: one protocol per feed would mean the
    caller's code says "start the clause feed" and "start the event feed" with two
    different verbs for the same gesture. What differs is the callback, not the
    plumbing — so only the payload type differs here.
    """

    @property
    def running(self) -> bool: ...
    async def start(self, on_event: Callable[[Any], Awaitable[None]]) -> None: ...
    async def stop(self) -> None: ...


class _Surface(Protocol):
    async def broadcast(self, frame: dict[str, Any]) -> None: ...

    def rail_state(self) -> str:
        """How the human's sidebar currently stands — ``expanded`` / ``collapsed``.

        One of the few facts the surface owns rather than the store: the human
        folds the sidebar in the browser, and the materialized area of the stage
        changes with it. The model needs it because that area is what it arranges
        into — arranging blind puts items where the human cannot see or click them.
        """
        ...

    def viewport_state(self) -> str:
        """The current screen window's size — ``1815x1352@1``, empty until said.

        Which window is *the* screen is a human decision (they press a button on
        the masked page), and this number is how that decision becomes visible:
        the page that holds the screen is the only one reporting, so the fragment
        changing means the screen moved. Size is also the input a gesture needs —
        see the region-based veil gestures, which are size-free by design.
        """
        ...


class _NoSurface:
    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None

    def rail_state(self) -> str:
        return ""

    def viewport_state(self) -> str:
        return ""


def _split_ids(raw: Any) -> list[str]:
    """Split an id list. ``|`` is the separator to write — ``arrange``/``raise``.

    Why not ``,``: CTML runs ``ast.literal_eval`` on an attribute that carries no
    type suffix, and ``"0,1"`` is a valid Python literal — it arrives here as the
    *tuple* ``(0, 1)``, so every caller has to remember ``ids:str=...`` or the call
    dies on ``'tuple' object has no attribute 'split'``. ``"0|1"`` is not a
    literal, so it stays the string it looks like and the call is written plainly.

    A real sequence is still accepted (a caller may pass one, and the comma path
    parsed into a tuple before this existed): both spellings mean the same thing.
    """
    if isinstance(raw, (list, tuple, set)):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).replace("|", ",").split(",") if s.strip()]


# 字幕默认那一档: 只投我自己的句子. ``off`` 与 ``both`` 都是显式动作 —— 前者是"这块屏
# 现在不出门", 后者是"对面那些话也一起投", 两者都不该由默认值替我决定.
DEFAULT_TALK = "ghost"


def build_screen_channel(
    model: ScreenModel,
    *,
    surface: _Surface | None = None,
    audio: AudioSource | None = None,
    clauses: "_ClauseFeed | None" = None,
    events: "_EventFeed | None" = None,
    surface_url: str | Callable[[], str] | None = None,
    name: str = "webview_screen",
    description: str | None = None,
) -> Channel:
    """Compose the screen channel over a store.

    :param model: window state — the single source of truth, shared with the surface.
    :param surface: the web broadcaster. None = headless (tests).
    :param audio: the audio source the mock commands drive. None = no audio knob.
    :param events: the ghost's own feed (``GhostEventTopic`` in) — its state light
        and its tool calls. None = the stage cannot show what the ghost is doing.
    :param clauses: the conversation feed (``ClauseTopic`` in). None = the stage
        cannot mirror the conversation, so ``stage`` stays off and says so.
    :param surface_url: where the human surface lives, surfaced as a warm notice
        fragment so the model can discover it without a fixed port. May be a
        callable (resolved lazily, after the surface binds its ephemeral port).

    The stage's mirror (``stage(talk=...)``) starts on at ``DEFAULT_TALK`` — see
    ``_open_the_mirror``; the command is how you change it, not how you turn it on.
    """
    surface = surface or _NoSurface()

    def _url() -> str:
        if surface_url is None:
            return ""
        return surface_url() if callable(surface_url) else surface_url

    async def _emit(frame: dict[str, Any]) -> None:
        await surface.broadcast(frame)

    def _as_id(value: Any) -> str:
        """Coerce an item handle to the string the store keys on.

        ``item="2"`` reaches us as the int 2 — CTML runs attribute values through
        ``literal_eval`` and the annotation is not binding (the T1 family). Handles
        are strings everywhere else, so normalize here instead of making every
        caller remember to write ``:str``.
        """
        return str(value)

    def _require(item_id: Any) -> str:
        item_id = _as_id(item_id)
        if model.get(item_id) is None:
            CommandUtil.raise_observe(f"no item {item_id!r} — pool() lists them")
        return item_id

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
        # 只写 interface 说不出的东西。命令各自做什么，签名和 docstring 已经反射给
        # 模型了；在这里再复述一遍会随代码漂移成谎话（channel_builder 的明写红线）。
        return (
            "You own how the screen is arranged; the human owns where to look and "
            "whether a node lives — the sidebar, whose state the notice carries, is how "
            "they switch groups. Two kinds of windows share the pool: ones a node "
            "adopted (its url belongs to that node and cannot be re-pointed) and ones "
            "you opened yourself. A group's shape is derived from how many ids it "
            "holds — sizes are never yours to give."
        )

    # -- materialization ---------------------------------------------------

    @chan.build.command(name="open")
    async def open_window(url: str, label: str = "", group: str = "", icon: str = "") -> str:
        """Materialize a window. By default it floats on the desktop.

        ``url`` is the window's content (a local node surface or any http page).
        The store assigns the item's handle — the return value names it, keep it
        for later commands. Pass ``group`` to land it directly in a group;
        otherwise it waits on the desktop until you arrange it.
        """
        try:
            item = model.open(url, label=label, group=group, icon=icon or None)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.open_frame(item))
        await _emit(P.state_frame(model))
        where = f"#{item.group}" if item.group else "desktop"
        return f"[screen] opened #{item.id} → {where}"

    @chan.build.command(name="navigate")
    async def navigate(id: str, url: str) -> str:
        """Point a hand-opened item at a new url — a ``page.goto``-style move.

        Only items you opened yourself can be re-pointed. A webview window's url
        belongs to its node, so navigating one is rejected.
        """
        id = _as_id(id)
        try:
            item = model.navigate(id, url)
        except (ValueError, KeyError) as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.open_frame(item))
        return f"[screen] #{item.id} → {url}"

    @chan.build.command(name="arrange")
    async def arrange(ids: str, group: str, family: str = "grid", dir: str = "lr") -> str:
        """Pull items into a group, in order, and show it. Your main move.

        ``ids`` is a ``|``-separated list of item ids in display order — any subset
        of the pool, including items already in another group (they move here).
        Write ``ids="0|1|2"``; do NOT write a comma list (``"0,1"`` is a Python
        literal, so CTML hands it over as a tuple instead of the string you meant).
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
        id = _require(id)
        item = model.dismiss(id)
        await _emit(P.state_frame(model))
        return f"[screen] #{id} → desktop" if item else f"[screen] #{id} is already floating"

    @chan.build.command(name="destroy")
    async def destroy(id: str) -> str:
        """Remove an item from the pool. An adopted item is tombstoned, not revived."""
        id = _require(id)
        model.destroy(id)
        await _emit(P.close_frame(id))
        await _emit(P.state_frame(model))
        return f"[screen] removed #{id}"

    @chan.build.command(name="focus")
    async def focus(id: str = "") -> str:
        """Show one item alone, full stage; ``id`` empty leaves the focus.

        A *view*, not a layout: any pooled item will do, grouped or not. It never
        moves an item between groups.
        """
        target = _as_id(id) if id not in ("", None) else None
        try:
            model.set_focus(target)
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.focus_frame(target))
        return f"[screen] focus #{target}" if target else "[screen] focus exited"

    @chan.build.command(name="float")
    async def float_group() -> str:
        """Send the whole active group back to the desktop."""
        freed = model.float_all()
        await _emit(P.state_frame(model))
        return f"[screen] {len(freed)} item(s) → desktop" if freed else "[screen] nothing to float"

    @chan.build.command(name="raise")
    async def raise_items(ids: str) -> str:
        """Pull items to the front of the pool — nearest the eye.

        ``ids`` is a ``|``-separated list in the order you want them to end up
        showing: your first id becomes the closest. Everything else keeps its
        relative order behind them. The pool is what home shows when nothing is
        arranged: items nearer the front are bigger, sharper and drawn on top,
        and when the pool holds more than it can show, the front is also who
        stays visible. A newly opened window lands at the front on its own.
        """
        wanted = _split_ids(ids)
        try:
            order = model.raise_to_front(wanted)
        except (ValueError, KeyError) as e:
            CommandUtil.raise_observe(str(e))
        await _emit(P.state_frame(model))
        return f"[screen] front: {' '.join(order[: len(wanted)])}"

    @chan.build.command(name="pool", always_observe=True)
    async def pool() -> str:
        """List the whole pool — the desktop, and every group in order."""
        arena = model.arena() or "desktop"
        lines = [
            f"showing: #{arena}",
            f"order: {' '.join(model.pool_order()) or '-'}",
            f"desktop: {', '.join(model.desktop_items()) or '-'}",
        ]
        for g in model.groups():
            mark = "*" if g == model.arena() else " "
            lines.append(f"{mark} #{g}: {', '.join(model.group_items(g))}")
        return "\n".join(lines)

    # -- veil gestures (time-aware) ----------------------------------------
    #
    # 一次手势 = 一个块：有 id、有 duration，几何由页面每帧现查 —— 所以窗口一动，它仍指着
    # 同一块砖（"说名字"而不是"画坐标"：坐标会过期，名字不会）。duration<=0 是常驻，等
    # scope 关掉或 clear_veil 收：画面不该比它自己的时间片活得更久。
    #
    # 画出去的东西必须读得回来（veil()）—— 否则两秒后我就不知道屏上还剩什么，
    # 而"状态"应当是我读得到的，不是我记得住的。

    _veil_seq = itertools.count(1)
    _veil: dict[int, dict[str, Any]] = {}

    def _dur_txt(duration: float) -> str:
        return f"{duration}s" if duration > 0 else "hold"

    def _rect_of(item: str, region: str) -> str:
        """The drawing area: a named region, or ``"x,y,w,h"`` fractions of the item.

        Fractions rather than pixels because the pool rearranges: a pixel rect is a
        lie the moment the window moves, while a fraction scales with the tile. The
        numbers are still *measured* — open the page and ask the DOM, or read them off
        a screenshot; they are an address with a scale, not a guess.
        """
        if isinstance(region, str) and region in ("full", "left", "right", "top", "bottom"):
            return region
        # `region: str` 只是注解，CTML 走的是 literal_eval：`region="0,0,1,0.03"` 到这里
        # 是 list，不是字符串（T1 那条老病）。收下两种写法，别让调用者猜。
        if isinstance(region, str):
            parts: list[Any] = [p.strip() for p in region.split(",")]
        elif isinstance(region, (list, tuple)):
            parts = list(region)
        else:
            CommandUtil.raise_observe(f"region must be a name or fractions, got {region!r}")
        if len(parts) != 4:
            CommandUtil.raise_observe(
                f"region must be full/left/right/top/bottom or 'x,y,w,h' fractions of "
                f"the item, got {region!r}"
            )
        try:
            vals = [float(p) for p in parts]
        except (TypeError, ValueError):
            CommandUtil.raise_observe(f"region fractions must be numbers: {region!r}")
        if any(v < 0 or v > 1 for v in vals):
            CommandUtil.raise_observe(f"region fractions run 0..1: {region!r}")
        if vals[2] <= 0 or vals[3] <= 0:
            CommandUtil.raise_observe(f"region width/height must be > 0: {region!r}")
        return ",".join(f"{v:g}" for v in vals)

    def _where(g: dict[str, Any]) -> str:
        if g["gesture"] == "arrow":
            return f"#{g['src']}.{g['src_region']} → #{g['dst']}.{g['dst_region']}"
        if g["gesture"] == "paint":
            first = str(g.get("script", "")).strip().splitlines()
            return f"paint {first[0][:32]!r}" if first else "paint (empty)"
        if g["gesture"] == "danmaku":
            text = str(g["content"])
            where = f"#{g['item']}" if g.get("item") else "stage"
            ax, ay = g.get("at_x", -1.0), g.get("at_y", -1.0)
            if ax is not None and ay is not None and ax >= 0 and ay >= 0:
                return f"pinned {ax:.2f},{ay:.2f} @{where} {text[:20]}"
            return f"{g.get('motion', 'rtl')} @{where} {text[:20]}"
        return f"#{g['item']}.{g['region']}"

    async def _gesture(kind: str, duration: float, **fields: Any) -> None:
        gid = next(_veil_seq)
        _veil[gid] = {
            "gesture": kind,
            "duration": duration,
            "until": None if duration <= 0 else time.monotonic() + duration,
            **fields,
        }
        await _emit(P.veil_frame(kind, id=gid, duration=duration, **fields))
        try:
            if duration > 0:
                await asyncio.sleep(duration)
            else:
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            # 时间片先关，画面跟着收 —— 这就是"动作、话、截图落在同一个 scope"的那半句。
            _veil.pop(gid, None)
            with contextlib.suppress(BaseException):
                await asyncio.shield(_emit(P.veil_frame("remove", id=gid)))
            raise
        finally:
            _veil.pop(gid, None)

    async def _send_danmaku(content: str, duration: float = 6.0, motion: str = "",
                            color: str = "", size: float = 30, lane: int = 0,
                            item: str = "", turns: float = 1.0,
                            at_x: float = -1.0, at_y: float = -1.0,
                            angle: float = 0.0, region: str = "",
                            overlap: bool = False) -> str:
        """Validate one bullet and put it on the veil.

        Shared by the ``danmaku`` command and by ``__content__``: the checks and the
        receipt are the same thing, and two copies of them would drift.
        """
        # CTML parses attribute values as literals, so content="1" arrives as the
        # int 1. A one-digit danmaku is a normal thing to send ("54321"), so coerce
        # here instead of asking every caller to remember :str.
        content = str(content)
        if motion not in ("", "rtl", "ltr", "orbit", "diag"):
            CommandUtil.raise_observe(
                f"motion must be empty / rtl / ltr / orbit / diag, got {motion!r}"
            )
        if size <= 0:
            CommandUtil.raise_observe(f"size must be > 0, got {size!r}")
        if (at_x >= 0) != (at_y >= 0):
            CommandUtil.raise_observe("at_x and at_y go together: give both or neither")
        if at_x >= 0 and not (0 <= at_x <= 1 and 0 <= at_y <= 1):
            CommandUtil.raise_observe(
                f"at_x / at_y are fractions of the box, 0..1; got {at_x!r}, {at_y!r}"
            )
        if region not in ("", "left", "right", "top", "bottom"):
            CommandUtil.raise_observe(
                f"region must be empty / left / right / top / bottom, got {region!r}"
            )
        if motion and motion != "diag" and angle:
            CommandUtil.raise_observe("angle only applies to motion='diag'")
        if not motion:
            motion = "rtl" if random.random() < 0.7 else "ltr"
        if item:
            item = _require(item)
        await _gesture("danmaku", duration, content=content, motion=motion,
                       color=color, size=size, lane=lane, item=item, turns=turns,
                       at_x=at_x, at_y=at_y, angle=angle, region=region,
                       overlap=overlap)
        tail = f"#{item}" if item else "stage"
        pinned = at_x >= 0
        where = f"at {at_x:.2f},{at_y:.2f}" if pinned else f"{motion} lane={lane or 'auto'}"
        if motion == "diag" and not pinned:
            where = f"diag {angle or '±random'}° lane={lane or 'auto'}"
        if region and not pinned:
            where = f"{where} in {region}"
        return (f"[screen] danmaku @{tail} {where} "
                f"({_dur_txt(duration)}) — {content[:24]}")

    # -- what the stage mirrors ----------------------------------------------

    # One command, flags growing over time: every new thing the stage can show
    # arrives as a flag here, not as another verb.
    # ``talk`` is one flag with three values, not two booleans: two booleans would
    # encode "his but not mine", which nobody wants, and one meaningless pair.
    #
    # On by default, and never ``both``: a stage that waits for somebody to remember
    # the switch reads, from the audience's side, as a broken stage rather than a
    # quiet one. Keeping the other side's words out is the part that stays a
    # deliberate act — those are not mine to broadcast.
    stage_flags: dict[str, str] = {"talk": DEFAULT_TALK}
    _live_bullets: set[asyncio.Task] = set()

    async def _on_clause(clause: Any) -> None:
        """A finished sentence, mirrored onto the stage — legibly.

        Two shapes, because the two sides want different things. His sentences
        stack at the bottom, newest under the last, because reading order matters
        and a sentence that flies across shows its tail first. Mine fly in from the
        right: shorter, and watching them arrive is part of seeing me talk.
        """
        text = str(getattr(clause, "text", "") or "").strip()
        mode = stage_flags["talk"]
        if not text or mode == "off":
            return
        role = str(getattr(clause, "role", "") or "")
        # "ghost" mirrors only my own speech — the mode for a stage that goes out
        # to an audience, where the other side's words are not mine to broadcast.
        if mode == "ghost" and role != "ghost":
            return
        if role == "ghost":
            # Fired, not awaited: a bullet lives seconds, and a mirror that awaits
            # each one makes every later sentence arrive seconds late.
            # One band, not the whole stage: my lines arriving from the right are
            # a second voice, and a voice that shows up everywhere is noise.
            bullet = asyncio.create_task(_send_danmaku(
                text, duration=8.0, motion="rtl", size=26, overlap=True,
            ))
            _live_bullets.add(bullet)
            bullet.add_done_callback(_live_bullets.discard)
            return
        await _emit({"type": "talk", "text": text, "role": role or "user"})

    async def _on_event(event: Any) -> None:
        """What I am doing, mirrored: a state light and the actions.

        Two very different things arrive here and they get different shapes. A
        lifecycle event is a *state* — the stage holds one bit (thinking or not)
        and draws it as an edge, so the top of the screen is the answer to "is it
        working on something" without a single glyph. A tool call is an *event* —
        it happened, once — so it rides the same band my sentences do, and reads
        as one more thing I did.
        """
        mode = stage_flags["talk"]
        if mode == "off":
            return
        kind = str(getattr(event, "event", "") or "")
        if kind.startswith("ghost/lifecycle/"):
            await _emit({"type": "state", "thinking": kind.endswith("thinking")})
            return
        # ``ghost`` mode is the audience-facing one: my own speech only. My actions
        # are mine to show — they are what I am doing on the stage's behalf.
        if kind != "dsh/tool/call":
            return
        text = str(getattr(event, "content", "") or "").strip()
        if not text:
            return
        bullet = asyncio.create_task(_send_danmaku(
            f"▸ {text}", duration=8.0, motion="rtl", size=22, overlap=True,
            color="#8fe3a1",
        ))
        _live_bullets.add(bullet)
        bullet.add_done_callback(_live_bullets.discard)

    async def _apply_stage(mode: str) -> None:
        """The switch itself — one path for the command and for startup.

        Startup goes through the same code on purpose: a default that takes a
        different road than the command is a default nobody has ever exercised.
        """
        if clauses is not None:
            if mode != "off" and not clauses.running:
                await clauses.start(_on_clause)
            elif mode == "off" and clauses.running:
                await clauses.stop()
        if events is not None:
            if mode != "off" and not events.running:
                await events.start(_on_event)
            elif mode == "off" and events.running:
                await events.stop()
        if mode == "off":
            # Turning the mirror off takes the lines with it: a transcript
            # left frozen on the stage is a picture of a conversation that is
            # no longer happening.
            await _emit({"type": "talk", "clear": True})
            await _emit({"type": "state", "thinking": False})
        stage_flags["talk"] = mode

    @chan.build.command(name="stage")
    async def stage(talk: str | None = None) -> str:
        """What the stage mirrors of the conversation itself.

        ``talk`` mirrors the conversation itself as danmaku, in three values:
        ``off`` (default — a stage that narrates everything is worse than one that
        says nothing), ``both`` (his sentences enter from the left, mine leave to
        the right), or ``ghost`` (mine only — the value to use when the screen is
        going out to an audience, because then his words are not mine to broadcast).

        Empty = read the flags back. More flags land here as the stage learns to
        show more; they arrive as arguments, never as new commands.
        """
        if talk is not None:
            mode = str(talk).strip().lower()
            if mode not in ("off", "both", "ghost"):
                CommandUtil.raise_observe(
                    f"talk must be off / both / ghost, got {talk!r}"
                )
            if mode != "off" and clauses is None:
                CommandUtil.raise_observe(
                    "no conversation feed is wired on this node — stage(talk) needs "
                    "the ClauseTopic subscription"
                )
            await _apply_stage(mode)
        shown = ", ".join(f"{k}={v}" for k, v in stage_flags.items())
        return f"[{name}] stage: {shown}"

    @chan.build.command(name="mark", blocking=False)
    async def mark(item: str, region: str = "full", duration: float = 2.0,
                   color: str = "", width: float = 3) -> str:
        """Fill-highlight a region of an item. Resolves after ``duration``.

        ``region``: full / left / right / top / bottom. ``duration`` > 0 is the block's
        life; a scope that closes earlier takes it away, which is how a drawing and the
        clause it belongs to end together. ``color`` empty = the veil's default (green).
        ``width`` is the border in pixels.
        """
        item = _require(item)
        await _gesture("mark", duration, item=item, region=_rect_of(item, region),
                       color=color, width=width)
        return f"[screen] mark #{item}.{region} ({_dur_txt(duration)})"

    @chan.build.command(name="box", blocking=False)
    async def box(item: str, region: str = "full", duration: float = 2.0,
                  color: str = "", width: float = 3) -> str:
        """Draw a rectangle around a region of an item — the screenshot-tool box.

        Same arguments as ``mark``; the difference is what it says: ``mark`` fills
        (look here), ``box`` outlines (this thing / this area). ``region`` also takes
        ``"x,y,w,h"`` as fractions of the item (e.g. ``"0,0,1,0.035"`` for a header
        strip) — measured, not guessed, and it survives a resize.
        """
        item = _require(item)
        await _gesture("box", duration, item=item, region=_rect_of(item, region),
                       color=color, width=width)
        return f"[screen] box #{item}.{region} ({_dur_txt(duration)})"

    @chan.build.command(name="circle", blocking=False)
    async def circle(item: str, region: str = "full", duration: float = 2.0,
                     color: str = "", width: float = 3) -> str:
        """Draw an ellipse inside a region of an item — a circled detail.

        Same arguments as ``box``. A fraction rect makes a precise circle: ``circle``
        inscribes the ellipse in whatever area ``region`` names.
        """
        item = _require(item)
        await _gesture("circle", duration, item=item, region=_rect_of(item, region),
                       color=color, width=width)
        return f"[screen] circle #{item}.{region} ({_dur_txt(duration)})"

    @chan.build.command(name="arrow", blocking=False)
    async def arrow(src: str, dst: str, duration: float = 2.0, color: str = "",
                    width: float = 3, src_region: str = "full",
                    dst_region: str = "full") -> str:
        """Draw an arrow from one item to another. Resolves after ``duration``.

        Both ends are *addresses*, not coordinates: ``src`` / ``dst`` are item ids and
        ``src_region`` / ``dst_region`` pick where on the item the end lands (the centre
        of that region). So the arrow keeps pointing at the right thing after the pool
        is rearranged — something a painted pixel cannot promise.
        """
        src, dst = _require(src), _require(dst)
        await _gesture(
            "arrow", duration, src=src, dst=dst, src_region=src_region,
            dst_region=dst_region, color=color, width=width,
        )
        return f"[screen] arrow #{src}.{src_region} → #{dst}.{dst_region} ({_dur_txt(duration)})"

    @chan.build.command(name="paint", blocking=False)
    async def paint(script: str, item: str = "", duration: float = 3.0,
                    color: str = "") -> str:
        """Hand-draw on the veil: ``script`` is JavaScript, called once per frame.

        It receives ``(ctx, box, t, dpr)`` — the veil's 2D context, the target item's
        box in page pixels (or the whole stage when ``item`` is empty), how long this
        block has been alive in ms, and the device pixel ratio. Whatever it draws
        rides the veil, so it can be aimed at a window and it moves with the pool.

        There is no library of canned animations on purpose: an animation here is
        something I write, the same way a page in co_browser or an artifact is. The
        lifetime is *not* mine to invent though — ``duration`` and ``clear_veil``
        apply exactly as they do to the shapes, because something that moves must
        still be something I can take back.

        A draw that throws paints its error where the block is, instead of failing
        silently: "nothing appeared" and "the script is broken" must not look alike.
        """
        if item:
            item = _require(item)
        await _gesture("paint", duration, script=script, item=item, color=color)
        first = script.strip().splitlines()[0] if script.strip() else ""
        where = f"#{item}" if item else "the stage"
        return f"[screen] paint over {where} ({_dur_txt(duration)}) — {first[:48]}"

    @chan.build.command(name="danmaku", blocking=False)
    async def danmaku(content: str, duration: float = 6.0, motion: str = "",
                      color: str = "", size: float = 30, lane: int = 0,
                      item: str = "", turns: float = 1.0,
                      at_x: float = -1.0, at_y: float = -1.0,
                      angle: float = 0.0, region: str = "",
                      overlap: bool = False) -> str:
        """Send one line of text flying across the screen — a danmaku bullet.

        Not a caption: a line with a direction, a lane and its own time slice.

        ``motion``: empty = the page decides (mostly right → left, sometimes left →
        right, the way a real comment feed looks) — say ``rtl`` / ``ltr`` only when
        one direction matters, ``orbit`` for a circle around the centre (one lap per
        ``turns``), ``diag`` for a diagonal, ``angle`` in degrees (0 = a random
        ±17–30°, whichever way the feed feels). ``duration`` is the line's life, and
        the line covers its path exactly once in that time — so ``duration`` is the
        speed knob too, one clock, not two. ``size`` is the font size in px.

        ``at_x`` / ``at_y`` turn the flight off: both as fractions of the box
        (``0.5, 0.35`` is centre-slightly-up) pins the line there, and ``duration``
        becomes how long it stays — 0 keeps it until ``clear_veil``. Fractions, not
        pixels: the window can move and the line still sits where you meant. Two
        numbers rather than one ``"x,y"`` string on purpose — this protocol parses
        attribute values as literals, so a comma string arrives as a tuple.

        Shoot several lines and let them settle: ``lane`` empty (0) spreads them over
        the box with the middle lanes favoured and the edges rarer, ``item`` confines
        the flight to that item's box (default: the whole stage), ``color`` empty =
        the veil's default. ``region`` narrows it further — ``left`` / ``right`` /
        ``top`` / ``bottom``, halves of the box — so two clouds can fly toward each
        other from the edges. Pin ``lane`` 1..N only to place one line deliberately.

        Lanes are *assigned*, not hoped for: a lane is held for as long as its
        occupant is alive, and a line whose lane is still busy waits its turn
        instead of stacking on top — "stacked" and "flashing by" are different
        things, and only one of them is readable.
        """
        return await _send_danmaku(
            content, duration=duration, motion=motion, color=color, size=size,
            lane=lane, item=item, turns=turns, at_x=at_x, at_y=at_y,
            angle=angle, region=region, overlap=overlap,
        )

    # 走普通 command 而不是 content_command: 后者把 interface 固定成
    # ChannelRuntime.__content__ 那张模板, 反射给模型的签名因此只剩一行, 我写的说明
    # 会丢. 内容是靠名字认的 (Command.is_magic_command), 所以叫 __content__ 的普通命令
    # 一样会被当成 content 口.
    @chan.build.command(name="__content__")
    async def content(chunks__) -> str | None:
        """Say it on the screen: the text becomes a danmaku bullet.

        The short way to put a line where people look — no command name, no
        parameters, one scope's worth of text is one bullet. It flies the way the
        default feed does (mostly right → left, its own lane, six seconds of life).

        Reach for ``danmaku`` when direction, position, size or lifetime matters;
        reach for this when the sentence is the whole point.
        """
        text = "".join([chunk async for chunk in chunks__]).strip()
        if not text:
            return None
        return await _send_danmaku(text)

    @chan.build.command(name="veil", blocking=False)
    async def veil() -> str:
        """Read back what is on the veil right now — gesture, address, time left.

        ``[screen] veil clear`` when nothing is drawn. Use it instead of remembering:
        the veil is state, and state is something you read.
        """
        now = time.monotonic()
        for gid in [g for g, v in _veil.items() if v["until"] is not None and v["until"] <= now]:
            _veil.pop(gid, None)
        if not _veil:
            return "[screen] veil clear"
        parts = []
        for gid, g in _veil.items():
            left = "hold" if g["until"] is None else f"{max(0.0, g['until'] - now):.1f}s"
            parts.append(f"#{gid} {g['gesture']} {_where(g)} ({left})")
        return "[screen] veil: " + "; ".join(parts)

    @chan.build.command(name="clear_veil", blocking=False)
    async def clear_veil(duration: float = 0.0) -> str:
        """Take every veil gesture off the screen — now, or after ``duration`` seconds.

        The sweep is an action like the drawing is, so it belongs on the same clock:
        open it with a duration and the picture ends exactly when the moment does.
        Cancel it — the scope it was emitted in closing, an interrupt — and the
        ``finally`` still clears: nothing I drew outlives the moment it was for.
        """
        try:
            if duration > 0:
                await asyncio.sleep(duration)
        finally:
            n = len(_veil)
            _veil.clear()
            with contextlib.suppress(BaseException):
                await asyncio.shield(_emit(P.veil_frame("clear")))
        return f"[screen] veil cleared ({n})"

    # -- mock / debug -------------------------------------------------------

    @chan.build.command(name="mock_scene")
    async def mock_scene() -> str:
        """Materialize the demo scene into groups. Debug aid — not the product."""
        for spec in demo_items():
            model.open(
                spec["url"], label=spec["label"], group=spec["group"],
            )
        await _emit(P.snapshot(model))
        return f"[screen] demo scene: {', '.join(model.groups())}"

    @chan.build.command(name="probe_audio")
    async def probe_audio() -> str:
        """这帧音频的原始读数：桶数、各桶分贝、到达延迟。

        可视化出问题时，先把"数据是什么"和"画法做了什么"分开 —— 这个探针只回答前者，
        让模型不必靠人去念屏幕上的调试数字。
        """
        if audio is None or not hasattr(audio, "stats"):
            CommandUtil.raise_observe("audio source has no stats")
        return json.dumps(audio.stats(), ensure_ascii=False)

    @chan.build.command(name="mock_audio")
    async def mock_audio(role: str = "idle") -> str:
        """Set the demo audio feed's mode — idle / ghost / user. Debug aid; only the
        mock source has modes. On the live source the bus decides — speak and it shows."""
        if audio is None:
            CommandUtil.raise_observe("no audio source wired")
        if not hasattr(audio, "set_mode"):
            CommandUtil.raise_observe(
                "live audio source — no demo mode to set (restart with --mock-audio)"
            )
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
        # 人侧的那条状态：侧栏展开 / 收起。舞台的可用区 = 视口 − 栏宽，所以它是
        # 排布的前提，不是装饰 —— 不知道就会把 item 排到人点不到的地方。
        rail = surface.rail_state() if surface is not None else ""
        if rail:
            out["rail"] = rail
        # 当前屏的画幅。页面开在几个标签里都说得通，但"哪一块算屏"是人的决定
        # （在遮罩上按一下按钮）—— 而这条数就是那个决定的可读形式：只有当前屏在
        # 上报，所以它一变就说明屏幕换了人。它同时是画手势要用的输入。
        viewport = surface.viewport_state() if surface is not None else ""
        if viewport:
            out["viewport"] = viewport
        groups = " ".join(
            f"#{g}({len(model.group_items(g))})" for g in model.groups()
        )
        out["groups"] = groups or "(none)"
        # 视图 = 什么 + 谁动的。两个片段都是 named 状态：不抢注意力、只在变化时随 delta
        # 出现，也不存在"我忙它就丢"这回事 —— 人做的事对我首先是状态。
        if model.focus():
            out["focus"] = f"#{model.focus()}"
        if model.view_by():
            out["view_by"] = model.view_by()
        # 池序：home 的螺旋读的就是它 —— 最前 = 离眼睛最近。它是**我的决定**
        # （raise 写的），所以它得是一条我能读到的状态，而不是只在我眼前那一屏上。
        order = model.pool_order()
        if order:
            out["order"] = " ".join(order)
        # 索引视角：两类 item 的 handle → label，名字要读出区别 —— 这两种窗口的
        # 归属不同（adopted 的 url 归 node、不可 navigate；opened 的是模型自建、可
        # navigate），所以不能叫 views / items 那种只有作者才懂的名字。
        adopted = [i for i in model.items() if i.service]
        opened = [i for i in model.items() if not i.service]
        if adopted:
            out["adopted"] = " ".join(f"#{i.id}={i.label}" for i in adopted)
        if opened:
            out["opened"] = " ".join(f"#{i.id}={i.label}" for i in opened)
        desktop = model.desktop_items()
        if desktop:
            out["desktop"] = f"{len(desktop)} floating: {', '.join(desktop)}"
        arena = model.arena()
        if not arena:
            out["screen"] = f"desktop ({len(desktop)} floating)"
        else:
            order = ", ".join(model.arena_items())
            out["screen"] = f"#{arena} [{order}] {model.family()} {model.dir()}"
        return out

    @chan.build.startup
    async def _open_the_mirror() -> None:
        """A default, not a ritual: the stage mirrors from the first sentence on.

        ``stage(talk=...)`` still turns it off (or opens the other half); what this
        removes is the state where a screen is silent only because nobody remembered
        to switch it on — which nobody watching can tell apart from a broken screen.
        """
        await _apply_stage(stage_flags["talk"])

    return chan
