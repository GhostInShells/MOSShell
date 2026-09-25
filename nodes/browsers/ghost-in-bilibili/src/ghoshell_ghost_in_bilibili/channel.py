"""The model-facing channel for ghost-in-bilibili.

One channel, pages addressed by ``label`` (p1/p2…). Perception is presence-gated
(the human's main ball), playback/subtitle are per-group gated (satellites). The
store is the shared ``BridgeModel``; outbound commands go through an injected
``Dispatcher`` so the channel stays transport-free — the node injects the WS
sender, tests inject a recorder.

Model contract (Code as Prompt): the docstring of each command IS the prompt.
The instruction must not restate the command list (channel_builder red line) —
it carries only what interface cannot express: the identity model and the
authorization grid.
"""

from __future__ import annotations

from typing import Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from .model import GROUPS, BridgeModel, Page
from .subtitle import SubtitleStore

__all__ = ["build_channel"]


class Dispatcher(Protocol):
    """Outbound transport. ``send_action`` is a blocking round-trip (await the
    extension's result); ``say`` is fire-and-forget (the human replies later, as
    an input signal)."""

    async def send_action(self, label: str, action: str, value: object = None) -> dict: ...
    async def say(self, label: str, text: str) -> None: ...


class _NoopDispatcher:
    async def send_action(self, label, action, value=None):
        return {"ok": True, "result": "(no dispatcher)"}

    async def say(self, label, text):
        return None


def build_channel(
    model: BridgeModel,
    subtitles: SubtitleStore,
    *,
    dispatch: Dispatcher | None = None,
    name: str = "ghost_in_bilibili",
) -> Channel:
    dispatch = dispatch or _NoopDispatcher()

    chan = new_channel(
        name=name,
        description="bilibili 共享观看:ghost 陪人类一起看视频,读字幕获得时间戳认知。",
    )

    # -- cold ------------------------------------------------------------

    @chan.build.instruction
    def instruction() -> str:
        return (
            "bilibili 共享观看 body。``page`` 是页面 label(p1/p2…):它绑的是浏览器窗口,"
            "不是视频 —— B 站会自动播放,label 不变但 bvid 会换。\n"
            "授权是 (页面 × 能力组) 两维,人类在页面上点球授予:主球=presence(你能感知这个"
            "页面),卫星=sense(实时状态)/control(播放控制)/subtitle(全文字幕)/interact"
            "(弹幕评论)。\n"
            "你只能对已授权的组发命令,未授权会报错 —— 那时用 say 在该页面的 panel 里请求"
            "人类点对应卫星,别反复重试。"
        )

    # -- warm ------------------------------------------------------------

    @chan.build.notice
    def notice() -> str:
        pages = model.open_pages()
        if not pages:
            return "(none) — 还没有人类授权感知的页面"
        return f"{len(pages)} 页在线: " + " ".join(pages)

    @chan.build.named_notices
    def named_notices() -> dict[str, str | None]:
        return {label: _fragment(p) for label, p in model.open_pages().items()}

    # -- hot -------------------------------------------------------------

    @chan.build.context_messages
    def context_messages() -> list[str]:
        out = []
        for p in model.open_pages().values():
            if p.reachable and p.grants["sense"]:
                out.append(_context(p, subtitles))
        return out

    # -- command helpers -------------------------------------------------

    def _require_page(label: str) -> Page:
        page = model.page_by_label(label)
        if page is None:
            CommandUtil.raise_observe(f"没有 {label!r} 这个页面 —— 见 notice 里在线页面")
        if not page.presence:
            CommandUtil.raise_observe(f"{label!r} 未授权感知(主球灰)—— 请人类点亮主球")
        if not page.reachable:
            CommandUtil.raise_observe(f"{label!r} 离线 —— 浏览器连接不在")
        return page

    def _require_group(label: str, group: str) -> Page:
        page = _require_page(label)
        if not page.grants.get(group):
            CommandUtil.raise_observe(f"{label!r} 的 {group} 未授权 —— 请人类点对应卫星")
        return page

    async def _run(label: str, action: str, value=None) -> str:
        _require_group(label, "control")
        result = await dispatch.send_action(label, action, value)
        if result.get("ok"):
            return f"[ghost] {action} ✓ {result.get('result', 'done')}"
        return f"[ghost] {action} ✗ {result.get('error', 'failed')}"

    # -- commands --------------------------------------------------------

    @chan.build.command(name="play")
    async def play(page: str) -> str:
        """让某个页面的视频开始播放。page 是页面 label(p1/p2,见 notice)。需 control 授权。"""
        return await _run(page, "play")

    @chan.build.command(name="pause")
    async def pause(page: str) -> str:
        """暂停某个页面的视频。"""
        return await _run(page, "pause")

    @chan.build.command(name="seek")
    async def seek(page: str, seconds: float) -> str:
        """把某个页面的视频进度跳到 seconds 秒。"""
        return await _run(page, "seek", seconds)

    @chan.build.command(name="speed")
    async def speed(page: str, rate: float) -> str:
        """设置某个页面视频的播放倍速(如 2.0)。"""
        return await _run(page, "speed", rate)

    @chan.build.command(name="subtitle")
    async def subtitle(page: str, start: float, end: float) -> str:
        """查询页面当前视频在 [start, end] 秒区间内的字幕,逐句返回时间区间与文本。需 subtitle 授权。"""
        p = _require_group(page, "subtitle")
        if not p.bvid:
            CommandUtil.raise_observe(f"{page!r} 还没有 bvid(页面刚打开,稍等)")
        lines = subtitles.query(p.bvid, start, end)
        if not lines:
            return f"[ghost] {page} {start:.0f}~{end:.0f}s 无字幕(或字幕未加载)"
        return "\n".join(f"[{l['from']:.1f}-{l['to']:.1f}] {l['content']}" for l in lines)

    @chan.build.command(name="say")
    async def say(page: str, text: str) -> str:
        """往某个页面的 panel 回一句话,给正在看该页面的人看。不等待回复。"""
        _require_page(page)
        await dispatch.say(page, text)
        return f"[ghost] → {page}: {text}"

    return chan


def _fragment(p: Page) -> str:
    granted = [g for g in GROUPS if p.grants[g]]
    g = ",".join(granted) if granted else "-"
    reach = "离线" if not p.reachable else "在线"
    return f"{p.bvid or '?'} {p.title} | 授权[{g}] | {reach}"


def _context(p: Page, subtitles: SubtitleStore) -> str:
    playing = "暂停" if p.paused else f"{p.rate:.1f}x"
    head = f"[{p.label}] {p.t:.0f}s/{p.duration:.0f}s {playing}"
    if p.bvid and subtitles.available(p.bvid):
        lines = subtitles.window(p.bvid, p.t, before=15.0)
        if lines:
            body = " · ".join(l["content"] for l in lines[-3:])
            return f"{head} · 「{body}」"
    return head
