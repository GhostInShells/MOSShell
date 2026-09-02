"""DoloresRun — Dolores thinking 交易的 run 对象 (async with 交易边界 + events() 事件流).

async with 是交易边界 (aenter 开 / aexit 关), events() 是被动拉的原始 session event 流:

    async with ego.run_thinking(thinking) as run:
        async for event in run.events():
            ...  # 消费方在 turn/end / wait_next_moment (yield) 处 break 收线

生命周期契约:
  aenter: 先绑 session catch-all 监听 (避免丢 enter 广播) → 再建 enter task (async).
  aexit : cancel enter task → 解绑监听 → 补发 thinking/exit (enter 未通过也发, 带超时
          fail-safe) → 异常时 thinking.abort(reason).
  events(): 队列消费; 毒丸只承载 enter 异常 (正常路径由消费方 turn/end break 收线).

yield tool (moss_wait_next_moment): 消费方认出 tool/call = moss_wait_next_moment → break 收线并置
run.yielded → exit 时 plugin 不 cancel (tool 留 pending, 下一轮 enter 用 moment 解锁).
moment 的生产归 mindflow 正常 loop, 不归 run.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from typing_extensions import Self
from ghoshell_moss.core.blueprint.mindflow import Thinking, Articulator
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, ToolCallEvent, AssistantChunk

from ._tools import FetchNextMomentToolCall, WaitNextMomentToolCall, AppendCtmlToolCall, ToolCallResult

if TYPE_CHECKING:
    from ._ego import DoloresEgo
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["DoloresRun"]

# 毒丸 sentinel: enter task 异常时入队, 消费方读到即抛 _enter_error 终止.
# 正常路径不塞 — 收线由 logos() 内部遇 turn/end 自止, 毒丸只管 enter 异常.
_POISON = object()


def _get_text_chunk(event: SessionEvent) -> str | None:
    if assistant_chunk := AssistantChunk.from_session_event(event):
        if assistant_chunk.chunk.type == 'text-delta':
            return assistant_chunk.chunk.text
    return None


_LogosDelta = str


class _CtmlParser:

    def __init__(self, articulator: Articulator, ctml_quoter: str = '<|CTML|>') -> None:
        self._articulator = articulator
        self._ctml_chars = [c for c in ctml_quoter]
        self._ctml_mark_length = len(ctml_quoter)
        self._ctml_buffer = ''
        self._is_in_ctml = False

    async def add(self, text: str) -> _LogosDelta:
        if self._ctml_mark_length == 0:
            if text:
                await self._articulator.send(text)
            return text
        logos = ''
        for char in text:
            delta = self._add_char(char)
            if delta is not None:
                logos += delta
        if logos:
            await self._articulator.send(logos)
        return logos

    def _add_char(self, char: str) -> _LogosDelta | None:
        # 没有缓存过命中 ctml 标记的信息.
        if self._ctml_buffer == '':
            # 命中了第一个字符.
            if char == self._ctml_chars[0]:
                # 等下一个字符.
                self._ctml_buffer += char
                return None
            else:
                # 既没有 ctml mark, 又没有命中需观测的字符, 则直接判定返回.
                if self._is_in_ctml:
                    # in ctml 这种情况是返回 logos delta.
                    return char
                else:
                    # 否则返回非 logos.
                    return None
        # 已经缓存过, 则需要判断 ctml 标记是否命中.
        else:
            index = len(self._ctml_buffer)
            # 命中了 ctml 标记的情况.
            if index < self._ctml_mark_length and char == self._ctml_chars[index]:
                # 增加新的 buffer.
                self._ctml_buffer += char
                # 增加完后, 可能正好满足了 ctml 标记.
                if (index + 1) == self._ctml_mark_length:
                    # 满足标记, 立刻反转.
                    self._is_in_ctml = not self._is_in_ctml
                    self._ctml_buffer = ''
                return None
            else:
                # 没有命中 ctml 标记的情况, 则应该准备发送.
                # 已经存在的 ctml buffer 都可以清理掉.
                if self._is_in_ctml:
                    buffer = self._ctml_buffer
                    buffer += char
                    self._ctml_buffer = ''
                    return buffer
                else:
                    self._ctml_buffer = ''
                    return None

    async def __aenter__(self) -> Self:
        await self._articulator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._is_in_ctml and self._ctml_buffer:
            await self._articulator.send(self._ctml_buffer)
            self._ctml_buffer = ''

        await self._articulator.wait_action_done()
        await self._articulator.__aexit__(exc_type, exc_val, exc_tb)
        return None


class DoloresRun:
    """Dolores thinking 交易 run 对象 — async with 交易边界 + events() 事件流.

    生命周期契约见模块 docstring. 构造: ego (DoloresEgo, 供 session/enter/exit 窄桥)
    + thinking + thinking_event (ego 持有的"交易进行中" Event — run aenter/aexit set/clear,
    供 ego self-wake gate 读取). 依赖经公有接口.
    """

    def __init__(
            self,
            ego: "DoloresEgo",
            thinking: "Thinking",
            thinking_event: asyncio.Event,
            facade: "MShellContextFacade",
            ctml_quoter: str = '<|CTML|>',
    ) -> None:
        self._ego = ego
        self._thinking = thinking
        self._facade = facade
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._dispose_listener: "Callable[[], None] | None" = None
        self._enter_task: "asyncio.Task[None] | None" = None
        self._enter_error: Exception | None = None
        self._thinking_event: asyncio.Event = thinking_event
        self._ctml_quoter = ctml_quoter
        # yield 收线标记: 消费方认出 tool/call == wait_next_moment 并 break 时置 True,
        # __aexit__ 经 exit_thinking(yielded=...) 传给 plugin — yield 时绝不 cancel.
        self.yielded = False
        # logos() 单次消费 guard — 一个 run 只能有一个 logos 流 (多个会分食 _queue).
        self._logos_started = False

    # ── transaction 边界 ──────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """开 transaction. 先绑监听 (避免丢 enter 广播), 再建 enter task."""
        self._thinking_event.set()
        self._dispose_listener = self._ego.session.on_session_event("*", self._on_event)
        self._enter_task = asyncio.create_task(self._drive_enter())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """关 transaction. cancel enter task → 解绑 → 补发 exit → abort (异常时)."""
        self._thinking_event.clear()
        task = self._enter_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._dispose_listener is not None:
            self._dispose_listener()
        # 补发 exit — enter 未通过也要发 (清理 plugin 侧状态), 阻塞到确认 (带超时 fail-safe).
        # yielded 标记: 本次 break 是否 yield 收线, plugin 据此决定是否 cancel.
        await self._ego.exit_thinking(yielded=self.yielded)
        if isinstance(exc_val, asyncio.CancelledError):
            return None
        reason = exc_val if exc_val is not None else self._enter_error
        if reason is not None:
            self._thinking.abort(reason)
        return None

    # ── 事件流 ─────────────────────────────────────────────────────

    async def _events(self) -> "AsyncIterator[SessionEvent]":
        """从队列拉原始 session event. 毒丸抛 _enter_error; 正常由 logos() aclose 终止."""
        while True:
            item = await self._queue.get()
            if item is _POISON:
                if self._enter_error is not None:
                    raise self._enter_error
                return
            yield item

    async def _handle_tool_use_event(self, event: ToolCallEvent) -> None:
        """tool/call 分派 — 按函数名判别 typed tool 并路由.

        fetch_next_moment / append_ctml → run_tool 产 ToolCallResult, 经 tool-result RPC 回.
        wait_next_moment (yield) → 置 self.yielded (logos() 据此 break 收线), 不走 tool-result.
        """
        result = await FetchNextMomentToolCall.run_tool(event, self._handle_fetch_next_moment)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await AppendCtmlToolCall.run_tool(event, self._handle_append_ctml)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        if WaitNextMomentToolCall.from_tool_call(event) is not None:
            self.yielded = True

    async def _handle_fetch_next_moment(self, call: FetchNextMomentToolCall) -> ToolCallResult:
        """fetch_next_moment handler — 产 moment, 回 {moment_ref} 结构化 result, 携带 moment."""
        moment = self._thinking.observe()
        moment_ref = f"{self._thinking.observer.epoch.index}-{moment.index}"
        return ToolCallResult(
            call=call.tool_call_event,
            result={"moment_ref": moment_ref},
            moment=moment,
        )

    async def _handle_append_ctml(self, call: AppendCtmlToolCall) -> str:
        """append_ctml handler — 追加 ctml 到执行, 思维超前于行为 (interleaved).

        refresh_meta: 执行前刷新 shell meta (经 facade.shell.refresh_metas).
        wait_done: true → wait_action_done (等执行完), false → wait_compiled (思维超前).
        """
        if call.refresh_meta:
            await self._facade.shell.refresh_metas(timeout=5.0)
        async with self._thinking.articulator(replan=False, wait_action_done=call.wait_done) as articulator:
            await articulator.send(call.ctml)
            if call.wait_done:
                await articulator.wait_action_done()
            else:
                await articulator.wait_compiled()
        return "ok"

    async def _dispatch_tool_result(self, result: ToolCallResult) -> None:
        """把 ToolCallResult 经 tool-result RPC 回给 plugin: result 解锁 tool, moment 注入.

        moment_id 从 result["moment_ref"] 显式取 (fetch_next_moment 的结构化 result), 不反推/重拼.
        """
        moment_parts = None
        if result.moment is not None and isinstance(result.result, dict):
            moment_ref = result.result.get("moment_ref")
            if moment_ref is not None:
                moment_parts = self._ego.moment_context_parts(result.moment, moment_ref)
        await self._ego.rpc_tool_result(result.call.callId, result.result, moment_parts)

    async def logos(self) -> "AsyncIterator[str]":
        """消费事件流, 提取 logos delta (经 <|CTML|> 模式分隔), 遇 turn/end 自止.

        单次消费: 一个 run 只能有一个 logos 流. tool 及其它非 text 事件留空 (见
        _handle_tool_use_event). 收线 (turn/end) 内化在此, 消费方无需 break.
        """
        if self._logos_started:
            raise RuntimeError("DoloresRun.logos() can only be consumed once")
        self._logos_started = True
        events = self._events()
        try:
            while True:
                event = await anext(events)
                text = _get_text_chunk(event)
                if text is not None:
                    parser = _CtmlParser(
                        self._thinking.articulator(replan=False, wait_action_done=True),
                        ctml_quoter=self._ctml_quoter,
                    )
                    async with parser:
                        while text is not None:
                            delta = await parser.add(text)
                            if delta:
                                yield delta
                            event = await anext(events)
                            text = _get_text_chunk(event)
                # 非 text 事件: tool 留空, turn/end 收线.
                if event.meta.type == "turn/end":
                    return
                if tool := ToolCallEvent.from_session_event(event):
                    await self._handle_tool_use_event(tool)
                    if self.yielded:
                        return
        finally:
            await events.aclose()

    # ── 内部 ──────────────────────────────────────────────────────

    async def _on_event(self, event: "SessionEvent") -> None:
        """catch-all 回调: 入队 (async 包装 — on_session_event 消费方 await)."""
        self._queue.put_nowait(event)

    async def _drive_enter(self) -> None:
        """enter task: thinking/enter RPC. 只在异常时塞毒丸.

        正常路径不塞毒丸 — turn 由 dsh 自行 run, live moment 的事件经 catch-all 监听
        流式入队, 消费方在 turn/end 处 break 收线. 毒丸只在 enter 异常时钉下收尾标志,
        让 events() 能借 _enter_error 向消费方抛出. 若在正常路径也塞毒丸, 会抢在
        模型产出 logos 之前终止事件流 — enter RPC 返回的时刻模型尚未生成任何帧.
        """
        try:
            await self._ego.enter_thinking(self._thinking)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._enter_error = error
            self._queue.put_nowait(_POISON)
