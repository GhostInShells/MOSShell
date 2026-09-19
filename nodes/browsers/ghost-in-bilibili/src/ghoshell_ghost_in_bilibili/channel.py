"""The model-facing channel for ghost-in-bilibili.

One channel, pages addressed by ``label`` (p1/p2/...). Perception is consent-gated
(the human toggles the green ball), playback commands are pre-authorized, and raw JS
goes through the human's accept/deny on the page. The store is the shared
``BridgeModel`` — the HTTP handler mutates it, this channel reads it and dispatches.
"""

from __future__ import annotations

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from .model import BridgeModel

__all__ = ["build_channel"]


def build_channel(
    model: BridgeModel,
    *,
    name: str = "ghost_in_bilibili",
    description: str | None = None,
) -> Channel:
    chan = new_channel(
        name=name,
        description=description
        or (
            "bilibili 共享观看 body:人类点绿球授权感知,ghost 感知页面、拉字幕、"
            "在授权后控制播放和下发 JS"
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "bilibili 共享观看 body。页面对应一个 label(p1/p2/...),notice 列在线页面,"
            "named_notices 给每页离散状态(播放/暂停·倍速·字幕就绪),context 是热数据"
            "(当前播放秒 + 当前字幕句)。人类在视频页点绿球 = 授权感知这个页面;播放组"
            "命令(play/pause/seek/speed)预设授权,但要人类在页面上点 accept 才执行,一次一条。"
            "完整字幕/弹幕的返回值是文件路径,内容在文件里,读文件取内容,别指望返回值带全文。"
        )

    # -- commands ----------------------------------------------------------

    def _require_page(page: str) -> None:
        if page not in model.pages:
            CommandUtil.raise_observe(f"no page {page!r} — see notice")

    def _dispatch(page: str, action: str, value=None) -> str:
        _require_page(page)
        cmd = model.dispatch_action(page, action, value)
        return f"[ghost] {action} #{cmd['id']} → {page} (pending human accept/deny)"

    @chan.build.command(name="play")
    async def play(page: str) -> str:
        """让某个页面的视频开始播放。``page`` 是 label(p1/p2,见 notice)。人类 accept 后执行。"""
        return _dispatch(page, "play")

    @chan.build.command(name="pause")
    async def pause(page: str) -> str:
        """暂停某个页面的视频。"""
        return _dispatch(page, "pause")

    @chan.build.command(name="seek")
    async def seek(page: str, seconds: float) -> str:
        """把某个页面的视频进度跳到 seconds 秒。"""
        return _dispatch(page, "seek", seconds)

    @chan.build.command(name="speed")
    async def speed(page: str, rate: float) -> str:
        """设置某个页面视频的播放倍速。"""
        return _dispatch(page, "speed", rate)

    @chan.build.command(name="pages", always_observe=True)
    async def pages() -> str:
        """列出所有页面:label、bvid、标题、状态。"""
        if not model.pages:
            return "[ghost] no pages"
        lines = [
            f"{l} {p['bvid']} {p['title']} [{p['state']}]" for l, p in model.pages.items()
        ]
        return "[ghost]\n" + "\n".join(lines)

    # -- warm state --------------------------------------------------------

    @chan.build.notice
    def notice() -> str:
        open_ = model.open_pages()
        if not open_:
            return "no page online"
        return f"{len(open_)} page(s): " + " ".join(open_)

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        out: dict[str, str] = {}
        for label, p in model.open_pages().items():
            state = "on" if p["state"] == "on" else "off"
            out[label] = f"{p['bvid']} {p['title']} ({state})"
        return out

    # -- hot state ---------------------------------------------------------

    @chan.build.context_messages
    def context_messages() -> list[str]:
        # 占位:state 推(每页 currentTime + 当前字幕句)尚未接线,等扩展侧 state 流补上。
        # 届时每个 open page 产出一条 "[p1] 120s · 当前句"。
        return []

    return chan
