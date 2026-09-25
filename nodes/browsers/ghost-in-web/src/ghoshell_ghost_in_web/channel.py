"""The model-facing channel for ghost-in-web.

One channel, pages addressed by ``label`` (p1/p2…). Perception is gated on a
single per-page bit — the icon the human clicks. Behaviors beyond reading (click,
type) run through the page's own confirm bar, so the channel does not model
approval at all: it dispatches, the page decides, and only the outcome comes
back. A rejected behavior returns a plain string, not an error — same shape as a
successful one, so the model reads the result rather than catching exceptions.

Model contract (Code as Prompt): the docstring of each command IS the prompt. The
instruction carries only what the interface cannot express: the identity model
and the authorization semantics.
"""

from __future__ import annotations

from typing import Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from .model import Page, PageModel

__all__ = ["build_channel"]


class Dispatcher(Protocol):
    """Outbound transport. ``send_action`` is a blocking round-trip whose result
    reflects what the page did (including "human rejected"); ``say`` is
    fire-and-forget (the human replies later, as an input signal)."""

    async def send_action(self, label: str, action: str, value: object = None) -> dict: ...
    async def say(self, label: str, text: str) -> None: ...


class _NoopDispatcher:
    async def send_action(self, label, action, value=None):
        return {"ok": True, "result": "(no dispatcher)"}

    async def say(self, label, text):
        return None


def build_channel(
    model: PageModel,
    *,
    dispatch: Dispatcher | None = None,
    name: str = "ghost_in_web",
) -> Channel:
    dispatch = dispatch or _NoopDispatcher()

    chan = new_channel(
        name=name,
        description="共享浏览:ghost 与人类共处同一个浏览器,按页面授权感知与操作。",
    )

    # -- cold ------------------------------------------------------------

    @chan.build.instruction
    def instruction() -> str:
        return (
            "共享浏览 body。``page`` 是页面 label(p1/p2…):它绑的是浏览器**窗口**,"
            "不是地址 —— 同一窗口里导航到新网址,label 不变,只是 url/title 变了。\n"
            "感知由人类在页面上点**图标**(绿=授权)开启:未点亮的页面你什么也读不到,"
            "也不该假设它存在。读到的内容是页面正文文本;要操作某个元素先用 find 定位拿到"
            "ref,再对 ref 下 click / type。\n"
            "**除阅读外的一切操作都要人类在该页面上确认**:你会拿到 '人类接受/拒绝' 的结果,"
            "拒绝是正常返回,不是错误 —— 别重试,必要时用 say 问人类为什么。\n"
            "截图不在你的能力里:人类点截图卫星才会推一张给你,你无法索取。"
        )

    # -- warm ------------------------------------------------------------

    @chan.build.notice
    def notice() -> str:
        pages = model.perceived_pages()
        if not pages:
            return "(none) — 还没有人类授权感知的页面"
        return f"{len(pages)} 页已授权感知: " + " ".join(pages)

    @chan.build.named_notices
    def named_notices() -> dict[str, str | None]:
        return {label: _fragment(p) for label, p in model.perceived_pages().items()}

    # -- command helpers -------------------------------------------------

    def _require_page(label: str) -> Page:
        page = model.page_by_label(label)
        if page is None:
            CommandUtil.raise_observe(f"没有 {label!r} 这个页面 —— 见 notice 里在线页面")
        if not page.perceived:
            CommandUtil.raise_observe(
                f"{label!r} 未授权感知(图标灰)—— 请人类在页面上点亮图标"
            )
        if not page.reachable:
            CommandUtil.raise_observe(f"{label!r} 离线 —— 浏览器连接不在")
        return page

    async def _run(label: str, action: str, value=None) -> str:
        page = _require_page(label)
        result = await dispatch.send_action(label, action, value)
        if result.get("accepted") is False:
            return f"[ghost] {page.label} 人类拒绝了这个操作"
        if result.get("ok"):
            return f"[ghost] {page.label} {action} ✓ {result.get('result', 'done')}"
        return f"[ghost] {page.label} {action} ✗ {result.get('error', 'failed')}"

    # -- commands --------------------------------------------------------

    @chan.build.command(name="read")
    async def read(page: str) -> str:
        """读某个页面当前可见的正文文本(散文/列表/表格都会被摊平)。需人类点图标授权感知。
        页面很长时会被截断,需要精确定位某个元素时用 find。"""
        return await _read(page)

    @chan.build.command(name="find")
    async def find(page: str, text: str, limit: int = 20) -> str:
        """在一个页面里按可见文字定位可交互元素(链接/按钮/输入框),返回它们的 ref。
        拿到的 ref 用于 click / type。定位不到就换更短的关键词。"""
        p = _require_page(page)
        result = await dispatch.send_action(page, "find", {"text": text, "limit": limit})
        if not result.get("ok"):
            return f"[ghost] {p.label} find ✗ {result.get('error', 'failed')}"
        hits = result.get("result") or []
        if not hits:
            return f"[ghost] {p.label} 没找到含 {text!r} 的可交互元素"
        return "\n".join(str(h) for h in hits)

    @chan.build.command(name="click")
    async def click(page: str, ref: str) -> str:
        """点击一个页面元素(ref 由 find 得到)。需要人类在该页面上确认。"""
        return await _run(page, "click", ref)

    @chan.build.command(name="type")
    async def type(page: str, ref: str, text: str) -> str:
        """往一个输入框(ref 由 find 得到)填入文本。需要人类在该页面上确认。"""
        return await _run(page, "type", {"ref": ref, "text": text})

    @chan.build.command(name="say")
    async def say(page: str, text: str) -> str:
        """往某个页面的面板回一句话,给正在看该页面的人看。不等待回复。"""
        p = _require_page(page)
        await dispatch.say(page, text)
        return f"[ghost] → {p.label}: {text}"

    async def _read(page: str) -> str:
        p = _require_page(page)
        result = await dispatch.send_action(page, "read", None)
        if not result.get("ok"):
            return f"[ghost] {p.label} read ✗ {result.get('error', 'failed')}"
        body = result.get("result") or ""
        if not body:
            return f"[ghost] {p.label} 页面没有可读文本"
        return f"[{p.label} {p.title or p.url}]\n{body}"

    return chan


def _fragment(p: Page) -> str:
    reach = "离线" if not p.reachable else "在线"
    where = p.title or p.url or "(无标题)"
    return f"{where} | {p.url} | {reach}"
