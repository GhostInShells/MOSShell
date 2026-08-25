"""
Action 基础结构实现.

重构自历史文件 ./base_attention.py
"""

import contextlib
from typing import AsyncIterator, AsyncGenerator, Awaitable, Callable
from typing_extensions import Self

from ghoshell_moss.core.blueprint.mindflow import (
    Attention, Action, ActionExitedException, Articulator, AttentionExitedException, StatementExitedException,
    ActionGate
)
from ghoshell_moss.core.blueprint.moment import Moments, Moment
from ghoshell_moss.core.helpers import ThreadSafeEvent
import asyncio
import janus
import logging

__all__ = ['BaseAction', 'BaseArticulator', 'BaseActionGate']

ApproveCallback = Callable[[str], Awaitable[tuple[bool, str]]]


class BaseArticulator(Articulator):

    def __init__(
            self,
            *,
            moment: Moment,
            logos_queue: janus.Queue[str | None],
            compiled_event: ThreadSafeEvent,
            action_stop_event: ThreadSafeEvent,
            gate: 'BaseActionGate | None' = None,
            action: 'BaseAction | None' = None,
            put_action: Callable[[Action], None] | None = None,
            logger: logging.Logger | None = None,
    ):
        self._logos_queue = logos_queue
        self._moment = moment
        self._compiled_event = compiled_event
        self._gate = gate
        self._action = action
        self._put_action = put_action
        self._action_stop_event = action_stop_event
        self._committed = False
        self._buffered_logos = ''
        self._logger = logger or logging.getLogger('moss.Articulator')
        self._started = False

    async def wait_compiled(self) -> None:
        await self._commit()
        await self._compiled_event.wait()

    async def wait_action_done(self) -> None:
        if not self._started:
            raise RuntimeError('Articulator must be started before wait_action_done()')
        await self._commit()
        await self._action_stop_event.wait()

    def send_nowait(self, logos_delta: str) -> None:
        try:
            if self._committed:
                raise RuntimeError('Articulator has already been committed')
            elif self._action_stop_event.is_set():
                return
            else:
                if self._gate is not None:
                    self._buffered_logos += logos_delta
                else:
                    self._logos_queue.sync_q.put_nowait(logos_delta)
                self._moment.logos += logos_delta
        except asyncio.QueueFull:
            raise
        except janus.AsyncQueueShutDown:
            raise ActionExitedException()

    async def send(self, logos_delta: str) -> None:
        try:
            if self._committed:
                raise RuntimeError('Articulator has already been committed')
            elif self._action_stop_event.is_set():
                return
            else:
                if self._gate is not None:
                    self._buffered_logos += logos_delta
                else:
                    await self._logos_queue.async_q.put(logos_delta)
                self._moment.logos += logos_delta
        except asyncio.QueueFull:
            raise
        except janus.AsyncQueueShutDown:
            raise ActionExitedException()

    async def _commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        if self._gate is not None:
            # gated: await approve 裁决完整 logos (commit 锁). approved 才投递 action,
            # rejected/cancelled 直接 abort action (进而 abort attention), 不重新 loop.
            approved, message = await self._gate.approve(self._buffered_logos)
            if approved:
                if message:
                    self._action.moments.add_result([message], need_observe=False)
                self._logos_queue.sync_q.put_nowait(self._buffered_logos)
                self._logos_queue.sync_q.put_nowait(None)
                self._put_action(self._action)
            else:
                self._action.abort(message)
        else:
            self._logos_queue.sync_q.put_nowait(None)

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 仅正常退出 (无异常) 时才 commit + 等编译; 异常退出 (含 CancelledError) 不 commit/
        # approve 也不等 compiled_event, 避免 action 不会 set_compiled 时的 articulate 互锁.
        if exc_val is None:
            await self._commit()
            if not self._action_stop_event.is_set():
                await self._compiled_event.wait()
        if exc_val:
            if isinstance(exc_val, ActionExitedException):
                return True
            elif not isinstance(exc_val, StatementExitedException):
                self._action_stop_event.set()
        return None


class BaseAction(Action):

    def __init__(
            self,
            *,
            attention: Attention,
            moments: Moments,
            replaned: bool,
            logos_queue: janus.Queue[str | None],
            compiled_event: ThreadSafeEvent,
            action_stop_event: ThreadSafeEvent,
            mindflow_stop_event: ThreadSafeEvent,
            thinking_stop_event: ThreadSafeEvent | None = None,
            logger: logging.Logger | None = None,
    ):
        self._attention = attention
        self._moments = moments
        self._compiled_event = compiled_event
        self._logger = logger or logging.getLogger(__name__)
        self._action_stop_event = action_stop_event
        self._mindflow_stop_event = mindflow_stop_event
        self._replaned = replaned
        self._logos_queue = logos_queue
        self._has_meaningful_logos = ThreadSafeEvent()
        self._waiting_futures: list[asyncio.Future] = []
        self._lifecycle_task: asyncio.Task | None = None
        self._prefetched_delta = ''
        self._terminated = False
        self._thinking_stop_event = thinking_stop_event
        self._started = False
        self._stopped = False

    @property
    def logos_queue(self) -> janus.Queue[str | None]:
        """仅限内部测试使用. """
        return self._logos_queue

    @property
    def replaned(self) -> bool:
        return self._replaned

    @property
    def attention(self) -> Attention:
        return self._attention

    async def wait_ready(self) -> None:
        if not self._started:
            raise RuntimeError('Action must be started before wait_ready()')
        # 等待 abort 或第一个有语义的帧. 预取首个有意义块, 顺序对齐 ghost_runtime:
        # wait_ready() 先返回, _logos() 再被消费 (预取内容由 _logos() 起手 drain).
        self._prefetched_delta = ''
        while not self.is_aborted():
            try:
                delta = await asyncio.wait_for(self._logos_queue.async_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except janus.AsyncQueueShutDown:
                return
            if delta is None:
                # 空流哨兵: 没有任何有意义帧, 记标志位让 _logos() 零 yield 立即返回.
                self._terminated = True
                return
            self._prefetched_delta += delta
            # 提权运行.
            self._attention.escalate()
            if delta.strip():
                self._has_meaningful_logos.set()
                return
        # abort 事件触发: 调用方借 is_aborted() 判别并干净退出, 不抛异常走通用 except.

    async def wait_abort(self) -> None:
        await self._action_stop_event.wait()

    def is_aborted(self) -> bool:
        return (
                self._action_stop_event.is_set()
                or self._attention.is_aborted()
                or self._mindflow_stop_event.is_set()
                or (self._thinking_stop_event is not None and self._thinking_stop_event.is_set())
        )

    async def wait_until_done(self, *futures: asyncio.Future) -> None:
        ensured = []
        for future in futures:
            fut = asyncio.ensure_future(future)
            ensured.append(fut)
            if self.is_aborted():
                fut.cancel()

        self._waiting_futures.extend(ensured)
        # return_exceptions: 被取消 / abort 的子任务不应中止 wait_until_done 自身,
        # 与 stop() 里取消 _waiting_futures 的语义一致.
        await asyncio.gather(*ensured, return_exceptions=True)

    def abort(self, reason: str | Exception | None) -> None:
        self._attention.abort(reason)
        self._action_stop_event.set()
        if not self._logos_queue.sync_q.closed:
            self._logos_queue.sync_q.shutdown()

    def abort_reason(self) -> str:
        return self._attention.abort_reason()

    def set_compiled(self):
        self._compiled_event.set()

    def logos(self) -> AsyncIterator[str]:
        return self._deliver_logos()

    def abort_thinking(self) -> None:
        if self._thinking_stop_event and not self._thinking_stop_event.is_set():
            self._thinking_stop_event.set()

    async def _deliver_logos(self) -> AsyncGenerator[str, None]:
        async for delta in self._logos():
            # 确认 executed logos 被添加了.
            self.moments.add_executed_logos(delta)
            self._attention.escalate()
            yield delta

    async def _logos(self) -> AsyncGenerator[str, None]:
        # 空流: wait_ready 已观察到 None 哨兵, 零 yield 立即返回.
        if self._terminated:
            return
        # drain wait_ready 预取的有意义块.
        if self._prefetched_delta:
            chunk = self._prefetched_delta
            self._prefetched_delta = ''
            self._has_meaningful_logos.set()
            yield chunk
        while self.is_running():
            try:
                if self.is_aborted():
                    # 方便跳出调用栈.
                    raise ActionExitedException()
                delta = await asyncio.wait_for(self._logos_queue.async_q.get(), timeout=0.5)
                if delta is None:
                    break

                if self._has_meaningful_logos.is_set():
                    yield delta
                    continue
                self._prefetched_delta += delta
                if delta.strip():
                    self._has_meaningful_logos.set()
                    chunk = self._prefetched_delta
                    self._prefetched_delta = ''
                    # 确认 executed logos 被添加了.
                    yield chunk
            except asyncio.TimeoutError:
                continue
            except janus.AsyncQueueShutDown:
                raise ActionExitedException()

    @property
    def moments(self) -> Moments:
        return self._moments

    async def stop(self) -> None:
        if self._action_stop_event.is_set():
            return
        self._action_stop_event.set()
        if not self._logos_queue.sync_q.closed:
            self._logos_queue.sync_q.shutdown()
        # 关闭控制的生命周期.
        if len(self._waiting_futures) > 0:
            for t in self._waiting_futures:
                t.cancel()
            _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

    def is_running(self) -> bool:
        return (
                self._started and not self._stopped
                and not self._mindflow_stop_event.is_set()
                and not self._attention.is_aborted()
                and not (self._thinking_stop_event is not None and self._thinking_stop_event.is_set())
        )

    async def _lifecycle_aborted_monitor(self) -> None:
        await asyncio.wait(
            [
                asyncio.create_task(self._attention.wait_abort()),
                asyncio.create_task(self._action_stop_event.wait()),
                asyncio.create_task(self._mindflow_stop_event.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        await self.stop()

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        self._lifecycle_task = asyncio.create_task(self._lifecycle_aborted_monitor())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._stopped:
            return None
        self._stopped = True
        try:
            await self.stop()
            if self._lifecycle_task and not self._lifecycle_task.done():
                self._lifecycle_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._lifecycle_task
                self._lifecycle_task = None
            if len(self._waiting_futures) > 0:
                for t in self._waiting_futures:
                    if not t.done():
                        t.cancel()
                _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

            if exc_val:
                # 内部用来中断循环的异常.
                if isinstance(exc_val, ActionExitedException):
                    return True
                # 其它未控制的异常, 需要 abort attention.
                self._attention.abort(exc_val)
                if isinstance(exc_val, AttentionExitedException):
                    return True
            return None
        finally:
            self._logos_queue.shutdown(immediate=True)
            self._has_meaningful_logos.set()
            self._action_stop_event.set()
            self._compiled_event.set()


class BaseActionGate(ActionGate):
    """thinking 级别的 logos 审批闸门 — 单一 approve 回调容器.

    articulator 在 commit 时调 ``approve(logos)``: 注册了回调就 await 它裁决,
    没注册 (非 gated) 直接 ``(True, '')`` 放行。
    """

    def __init__(self):
        self._approve: ApproveCallback | None = None

    def register(self, approve: ApproveCallback) -> None:
        """注册单一 approve 回调。重复注册覆盖旧值。"""
        self._approve = approve

    def has_approve(self) -> bool:
        return self._approve is not None

    async def approve(self, logos: str) -> tuple[bool, str]:
        if self._approve is None:
            return (True, '')
        return await self._approve(logos)
