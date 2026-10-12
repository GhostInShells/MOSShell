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
import time
from collections.abc import Callable
from typing import Any, Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from . import projection as P
from .audio import AudioSource
from .mock import demo_items
from .model import ScreenModel, compute_layout

__all__ = ["build_screen_channel"]


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


def build_screen_channel(
    model: ScreenModel,
    *,
    surface: _Surface | None = None,
    audio: AudioSource | None = None,
    surface_url: str | Callable[[], str] | None = None,
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

    @chan.build.command(name="focus")
    async def focus(id: str = "") -> str:
        """Show one item alone, full stage; ``id`` empty leaves the focus.

        A *view*, not a layout: any pooled item will do, grouped or not. It never
        moves an item between groups.
        """
        target = id or None
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

    def _where(g: dict[str, Any]) -> str:
        if g["gesture"] == "arrow":
            return f"#{g['src']}.{g['src_region']} → #{g['dst']}.{g['dst_region']}"
        if g["gesture"] == "text":
            text = str(g["content"])
            return text if len(text) <= 24 else text[:23] + "…"
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

    @chan.build.command(name="mark", blocking=False)
    async def mark(item: str, region: str = "full", duration: float = 2.0,
                   color: str = "", width: float = 3) -> str:
        """Fill-highlight a region of an item. Resolves after ``duration``.

        ``region``: full / left / right / top / bottom. ``duration`` > 0 is the block's
        life; a scope that closes earlier takes it away, which is how a drawing and the
        clause it belongs to end together. ``color`` empty = the veil's default (green).
        ``width`` is the border in pixels.
        """
        _require(item)
        await _gesture("mark", duration, item=item, region=region, color=color, width=width)
        return f"[screen] mark #{item}.{region} ({_dur_txt(duration)})"

    @chan.build.command(name="box", blocking=False)
    async def box(item: str, region: str = "full", duration: float = 2.0,
                  color: str = "", width: float = 3) -> str:
        """Draw a rectangle around a region of an item — the screenshot-tool box.

        Same arguments as ``mark``; the difference is what it says: ``mark`` fills
        (look here), ``box`` outlines (this thing / this area).
        """
        _require(item)
        await _gesture("box", duration, item=item, region=region, color=color, width=width)
        return f"[screen] box #{item}.{region} ({_dur_txt(duration)})"

    @chan.build.command(name="circle", blocking=False)
    async def circle(item: str, region: str = "full", duration: float = 2.0,
                     color: str = "", width: float = 3) -> str:
        """Draw an ellipse inside a region of an item — a circled detail.

        Same arguments as ``box``.
        """
        _require(item)
        await _gesture("circle", duration, item=item, region=region, color=color, width=width)
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
        _require(src)
        _require(dst)
        await _gesture(
            "arrow", duration, src=src, dst=dst, src_region=src_region,
            dst_region=dst_region, color=color, width=width,
        )
        return f"[screen] arrow #{src}.{src_region} → #{dst}.{dst_region} ({_dur_txt(duration)})"

    @chan.build.command(name="text", blocking=False)
    async def text(content: str, duration: float = 2.5, color: str = "") -> str:
        """Show big text over the whole screen. Resolves after ``duration``."""
        await _gesture("text", duration, content=content, color=color)
        return f"[screen] text ({_dur_txt(duration)})"

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

    return chan
