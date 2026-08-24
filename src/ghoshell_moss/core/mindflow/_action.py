"""
Action 基础结构实现.

重构自历史文件 ./base_attention.py
"""

import contextlib
from typing import AsyncIterator, AsyncGenerator, Callable
from typing_extensions import Self

from ghoshell_moss.core.blueprint.mindflow import (
    Attention, Action, ActionExitedException, Articulator, AttentionExitedException, StatementExitedException,
    ActionGate, LogosRequest
)
from ghoshell_moss.core.blueprint.moment import Moments, Moment
from ghoshell_moss.core.helpers import ThreadSafeEvent
from collections import deque
import asyncio
import janus
import logging

__all__ = ['BaseAction', 'BaseArticulator', 'ActionLogosRequest', 'BaseActionGate']


class ActionLogosRequest(LogosRequest):

    def __init__(
            self,
            logos: str,
            action: 'BaseAction',
            put_action: Callable[[Action], None],
    ):
        self._logos = logos
        self._action = action
        self._put_action = put_action
        self._committed_event = ThreadSafeEvent()
        self._result: bool | None = None

    async def wait_commited(self):
        await self._committed_event.wait()

    @property
    def logos(self) -> str:
        return self._logos

    @property
    def action(self) -> 'BaseAction':
        return self._action

    def add_logos(self, logos_delta: str) -> None:
        self._logos += logos_delta

    def commit(self) -> None:
        self._committed_event.set()

    def is_done(self) -> bool:
        return self._action.is_aborted() or self._committed_event.is_set()

    async def approve(self, message: str = '') -> None:
        if self._result is not None:
            raise RuntimeError('Logos Request has already been approved')
        if self._action.is_aborted():
            return
        self._result = True
        if message:
            self._action.moments.add_result([message], need_observe=False)
        self._action.logos_queue.sync_q.put_nowait(self._logos)
        self._action.logos_queue.sync_q.put_nowait(None)
        # action 回调.
        self._put_action(self._action)

    async def reject(self, reason: str = '') -> str:
        if self._result is not None:
            raise RuntimeError('Logos Request has already been rejected')
        if self._action.is_aborted():
            return "action is aborted before rejected"
        self._result = False
        self._action.abort(reason)
        return "abort the running attention"

    def approved(self) -> bool | None:
        return self._result


class BaseArticulator(Articulator):

    def __init__(
            self,
            *,
            moment: Moment,
            logos_queue: janus.Queue[str | None],
            compiled_event: ThreadSafeEvent,
            action_stop_event: ThreadSafeEvent,
            logos_request: ActionLogosRequest | None = None,
            logger: logging.Logger | None = None,
    ):
        self._logos_queue = logos_queue
        self._moment = moment
        self._compiled_event = compiled_event
        self._logos_request: ActionLogosRequest | None = logos_request
        self._action_stop_event = action_stop_event
        self._committed = False
        self._logger = logger or logging.getLogger('moss.Articulator')
        self._started = False

    async def wait_compiled(self) -> None:
        self._commit()
        await self._compiled_event.wait()

    async def wait_action_done(self) -> None:
        if not self._started:
            raise RuntimeError('Articulator must be started before wait_action_done()')
        self._commit()
        await self._action_stop_event.wait()

    def send_nowait(self, logos_delta: str) -> None:
        try:
            if self._committed:
                raise RuntimeError('Articulator has already been committed')
            elif self._action_stop_event.is_set():
                return
            else:
                if self._logos_request:
                    self._logos_request.add_logos(logos_delta)
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
                if self._logos_request:
                    self._logos_request.add_logos(logos_delta)
                else:
                    await self._logos_queue.async_q.put(logos_delta)
                self._moment.logos += logos_delta
        except asyncio.QueueFull:
            raise
        except janus.AsyncQueueShutDown:
            raise ActionExitedException()

    def _commit(self):
        if self._committed:
            return
        self._committed = True
        if self._logos_request:
            self._logos_request.commit()
        else:
            self._logos_queue.sync_q.put_nowait(None)

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._commit()
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
            if delta.strip():
                self._has_meaningful_logos.set()
                return
        # abort 事件触发: 调用方借 is_aborted() 判别并干净退出, 不抛异常走通用 except.

    async def wait_abort(self) -> None:
        await self._action_stop_event.wait()

    def is_aborted(self) -> bool:
        return self._action_stop_event.is_set() or self._attention.is_aborted() or self._mindflow_stop_event.is_set()

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
        return self._logos()

    async def _logos(self) -> AsyncGenerator[str, None]:
        # 空流: wait_ready 已观察到 None 哨兵, 零 yield 立即返回.
        if self._terminated:
            return
        # drain wait_ready 预取的有意义块.
        if self._prefetched_delta:
            chunk = self._prefetched_delta
            self._prefetched_delta = ''
            self._has_meaningful_logos.set()
            self.moments.add_executed_logos(chunk)
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
                    # 确认 executed logos 被添加了.
                    self.moments.add_executed_logos(delta)
                    yield delta
                    continue
                self._prefetched_delta += delta
                if delta.strip():
                    self._has_meaningful_logos.set()
                    chunk = self._prefetched_delta
                    self._prefetched_delta = ''
                    # 确认 executed logos 被添加了.
                    self.moments.add_executed_logos(chunk)
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
                and not self._mindflow_stop_event.is_set() and not self._attention.is_aborted()
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

    def __init__(self, stop_event: ThreadSafeEvent):
        self._has_new_logos_request = ThreadSafeEvent()
        self._logos_requests: deque[ActionLogosRequest] = deque()
        self._stop_event = stop_event

    def add_request(self, request: ActionLogosRequest):
        self._has_new_logos_request.set()
        self._logos_requests.append(request)

    async def wait_request(self) -> LogosRequest | None:
        if self._stop_event.is_set():
            return None
        wait_stop = asyncio.create_task(self._stop_event.wait())
        get_request = asyncio.create_task(self._wait_request())
        done, pending = await asyncio.wait([wait_stop, get_request], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            return await get_request
        # stop 事件先完成: 返回 None, 表示 gate 已终止.
        return None

    async def _wait_request(self) -> LogosRequest | None:
        try:
            while True:
                if self._stop_event.is_set():
                    return None
                while len(self._logos_requests) > 0:
                    r = self._logos_requests.popleft()
                    if r is None or r.is_done():
                        continue
                    return r
                # 队列空: 清掉唤醒标志后重查一次, 避免 add_request 在 clear 与 append 之间
                # 的 set() 被 clear 吞掉而丢唤醒.
                self._has_new_logos_request.clear()
                if self._logos_requests:
                    continue
                await self._has_new_logos_request.wait()
        except asyncio.CancelledError:
            return None
