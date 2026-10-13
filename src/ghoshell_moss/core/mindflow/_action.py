"""
Action 基础结构实现.

重构自历史文件 ./base_attention.py
"""

import contextlib
from typing import AsyncIterator, AsyncGenerator, Awaitable, Callable
from typing_extensions import Self

from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.message import Message
from ghoshell_moss.core.blueprint.mindflow import (
    Attention, Action, ActionExitedException, Articulator, AttentionExitedException, StatementExitedException,
)
from ghoshell_moss.core.blueprint.moment import Moments, Moment
from ghoshell_moss.core.helpers import ThreadSafeEvent
import asyncio
import janus
import logging

__all__ = ['BaseAction', 'BaseArticulator']

ApproveCallback = Callable[[str], Awaitable[tuple[bool, str]]]


class BaseArticulator(Articulator):

    def __init__(
            self,
            *,
            moment: Moment,
            logos_queue: janus.Queue[str | None],
            compiled_event: ThreadSafeEvent,
            observed_event: ThreadSafeEvent,
            action_stop_event: ThreadSafeEvent,
            warrant: ApproveCallback | None = None,
            action: 'BaseAction',
            put_action: Callable[[Action], None] | None = None,
            logger: logging.Logger | None = None,
    ):
        self._logos_queue = logos_queue
        self._moment = moment
        self._compiled_event = compiled_event
        self._observed_event = observed_event
        self._warrant = warrant
        self._action = action
        self._put_action = put_action
        self._action_stop_event = action_stop_event
        self._committed = False
        self._buffered_logos = ''
        self._logger = logger or logging.getLogger('moss.Articulator')
        self._started = False
        # gated 模式下被持有的审批 task, 供 wait 动作 await / __aexit__ 退出时 cancel.
        self._approve_task: asyncio.Task | None = None

    async def wait_compiled(self, raise_interpret_error: bool = False) -> None:
        await self._commit()
        await self._await_approve()
        await self._compiled_event.wait()
        if raise_interpret_error:
            error = self._action.interpret_error
            if error is not None:
                raise InterpretError.from_error(error)

    async def wait_observed(self, raise_interpret_error: bool = False) -> None:
        await self.wait_compiled(raise_interpret_error)
        await self._observed_event.wait()

    async def wait_action_done(self) -> None:
        if not self._started:
            raise RuntimeError('Articulator must be started before wait_action_done()')
        await self._commit()
        await self._await_approve()
        await self._action_stop_event.wait()

    def send_nowait(self, logos_delta: str) -> None:
        try:
            if self._committed:
                raise RuntimeError('Articulator has already been committed')
            elif self._action_stop_event.is_set():
                return
            else:
                if self._warrant is not None:
                    self._buffered_logos += logos_delta
                else:
                    self._logos_queue.sync_q.put_nowait(logos_delta)
                self._moment.logos += logos_delta
        except asyncio.QueueFull:
            raise
        except (janus.ShutDown, janus.QueueShutDown):
            # sync put_nowait raises janus.ShutDown; async put raises janus.QueueShutDown.
            # The two are unrelated classes — both must be caught.
            raise ActionExitedException()

    async def send(self, logos_delta: str) -> None:
        try:
            if self._committed:
                raise RuntimeError('Articulator has already been committed')
            elif self._action_stop_event.is_set():
                return
            else:
                if self._warrant is not None:
                    self._buffered_logos += logos_delta
                else:
                    await self._logos_queue.async_q.put(logos_delta)
                self._moment.logos += logos_delta
        except asyncio.QueueFull:
            raise
        except (janus.ShutDown, janus.QueueShutDown):
            raise ActionExitedException()

    async def _commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        if self._warrant is not None:
            # gated: 把 warrant 包装成被持有的 task, 供 wait 动作 await / 退出时 cancel.
            self._approve_task = asyncio.create_task(self._approve_and_dispatch())
        else:
            self._logos_queue.sync_q.put_nowait(None)

    async def _approve_and_dispatch(self) -> None:
        """被持有的阻塞 task — await warrant 裁决完整 logos, 据此投递 action 或 abort.

        approved 才投递 action (approve-note 由 warrant 自己落到 mindflow.moments,
        此处只认 approved/abort); rejected 直接 abort action (进而 abort attention).
        被 cancel (articulator 退出) 时转成 StatementExitedException, 不裸抛 CancelledError.
        """
        try:
            approved, message = await self._warrant(self._buffered_logos)
        except asyncio.CancelledError:
            raise StatementExitedException('articulator exited during gated approval')
        if approved:
            self._logos_queue.sync_q.put_nowait(self._buffered_logos)
            self._logos_queue.sync_q.put_nowait(None)
            self._put_action(self._action)
        else:
            self._action.abort(message)

    async def _await_approve(self) -> None:
        """await 被持有的 approve task 结束; 结束后清引用 (退出兜底 cancel 见 __aexit__)."""
        if self._approve_task is None:
            return
        await self._approve_task
        self._approve_task = None

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 仅正常退出 (无异常) 时才 commit + 等编译; 异常退出 (含 CancelledError) 既不 commit /
        # approve, 也不等 compiled_event —— 这个 region 不会再被喂, 编译/观测位不会有人生产,
        # 等它们的一方会被永远挂在这里 (articulate 互锁). 中断退出因此要把三个端点直接落地.
        try:
            if exc_val is None:
                await self._commit()
                await self._await_approve()
                if not self._action_stop_event.is_set():
                    await self._compiled_event.wait()
            else:
                self._action.close_lifecycle()
                if isinstance(exc_val, ActionExitedException):
                    return True
            return None
        finally:
            # 退出时若审批 task 仍未裁决, cancel 掉, 避免泄漏阻塞下一轮.
            if self._approve_task is not None:
                if not self._approve_task.done():
                    self._approve_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, StatementExitedException):
                        await self._approve_task
                self._approve_task = None


class BaseAction(Action):

    def __init__(
            self,
            *,
            attention: Attention,
            moments: Moments,
            replaned: bool,
            logos_queue: janus.Queue[str | None],
            compiled_event: ThreadSafeEvent,
            observed_event: ThreadSafeEvent,
            action_stop_event: ThreadSafeEvent,
            mindflow_stop_event: ThreadSafeEvent,
            thinking_stop_event: ThreadSafeEvent | None = None,
            notify_thinking_stop: Callable[[], None] | None = None,
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
        # 走 thinking 的 _notify_stop (而非裸置 _thinking_stop_event): 这样 abort_thinking
        # 也会同帧向本 thinking 的其它 action/articulator 级联 close_lifecycle, 不再漏窗口.
        self._notify_thinking_stop = notify_thinking_stop
        self._interpret_error: Exception | None = None
        self._observed_done_event: ThreadSafeEvent = observed_event
        self._started = False
        self._stopped = False
        # stop() 的"清理只做一次"标记 (停止位与清理分开, 见 stop()).
        self._cleaned = False
        # thinking 停止回调的撤销函数 —— 由 BaseThinking.articulator() 在**构造时**挂牌
        # (见 bind_to_thinking_stop); None 表示这一帧没挂上, 退出时无需撤牌.
        self._dispose_thinking_stop: Callable[[], None] | None = None
        # attention 的 abort 回调撤销函数 —— 契约: Action 持有 attention, attention 退出
        # 时它也应该退出. 构造即从属, 不等 enter (与 thinking 同形), 保证直接构造 action
        # (无签发 thinking) 的场景下 wait_abort 仍能被 attention.abort() 释放.
        self._dispose_attention_abort: Callable[[], None] | None = self._attention.on_abort(
            self.close_lifecycle
        )

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

    def set_observed_done(self):
        self._observed_done_event.set()

    async def wait_observed_done(self):
        """Wait for the observed node to settle.

        The node settles in one of three ways: the observable tasks finished
        (:meth:`set_observed_done`), the action was aborted (preempted / stopped), or the
        logos failed to compile. The last raises :class:`InterpretError`; the first two
        both release the waiter, and the caller tells them apart with :meth:`is_aborted`.
        """
        await self._observed_done_event.wait()
        if self._interpret_error:
            raise InterpretError.from_error(self._interpret_error)

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
        # 终结 compiled / observed 两个生命周期节点: abort 之后不会再有 compile 或 observe,
        # 释放等待方 (与 stop() 一致), 由 is_aborted() 判别干净退出还是异常.
        self._compiled_event.set()
        self._observed_done_event.set()
        if not self._logos_queue.sync_q.closed:
            self._logos_queue.sync_q.shutdown()

    def abort_reason(self) -> str | None:
        return self._attention.abort_reason()

    def set_compiled(self):
        self._compiled_event.set()

    async def wait_compiled(self, raise_interpret_error: bool = False):
        await self._compiled_event.wait()
        if self._interpret_error and raise_interpret_error:
            raise InterpretError.from_error(self._interpret_error)

    def set_interpret_error(self, error: Exception) -> None:
        self._compiled_event.set()
        self._interpret_error = error
        # 终结 observed 节点: 编译失败的 action 不会再产生 observe, 释放 wait_observed_done()
        # 让它在读 _interpret_error 时抛 InterpretError, 而不是挂到 stop().
        self._observed_done_event.set()

    @property
    def interpret_error(self) -> Exception | None:
        """The interpreter's parse error, if this action's logos failed to compile."""
        return self._interpret_error

    def logos(self) -> AsyncIterator[str]:
        return self._deliver_logos()

    def abort_thinking(self, reason: Message | None = None, need_observe: bool = False) -> None:
        """停当前 thinking, 并 (可选) 由本次 abort 显式驱动下一帧自愈.

        置位 thinking 的 stop event → 当前帧思考退场 (attention 不释放, 仍会产生下一帧).
        ``need_observe`` 为 True 时提交一个 need_observe 回声: 下一帧 thinking 的签发由
        这个显式事件驱动, 不依赖解释器 close 落盘的时序 —— 这是 interpret error 自愈的机制。
        """
        # 优先走 thinking 的回调: 同帧向其它 action/articulator 级联 close_lifecycle,
        # 不止是置自己的停止位 (裸事件会漏掉"已构造未 enter"的子对象).
        if self._notify_thinking_stop is not None:
            self._notify_thinking_stop()
        elif self._thinking_stop_event and not self._thinking_stop_event.is_set():
            self._thinking_stop_event.set()
        echo_result = []
        if reason and isinstance(reason, Message):
            echo_result.append(reason)
        self._moments.add_echoes(echo_result, need_observe=need_observe)

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
        # 停止位与清理分开 (同 BaseThinking.stop): 停止位可能已被 abort() 或 abort 回调
        # **同步**置位, 但等待中的 future 仍然要有人取消一次, 所以不能因为 event 已置位就早退.
        self._action_stop_event.set()
        self._compiled_event.set()
        self._observed_done_event.set()
        if not self._logos_queue.sync_q.closed:
            self._logos_queue.sync_q.shutdown()
        if self._cleaned:
            return
        self._cleaned = True
        # 关闭控制的生命周期.
        if len(self._waiting_futures) > 0:
            for t in self._waiting_futures:
                t.cancel()
            _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

    def is_running(self) -> bool:
        return (
                self._started and not self._stopped
                and not self._action_stop_event.is_set()
                and not self._mindflow_stop_event.is_set()
                and not self._attention.is_aborted()
                and not (self._thinking_stop_event is not None and self._thinking_stop_event.is_set())
        )

    async def _lifecycle_aborted_monitor(self) -> None:
        """等"该收线"的两路来源, 到了就把这一帧收干净.

        不再等 attention: 本帧的终止由 thinking 的停止回调**同步**置位 ``_action_stop_event``
        (见 ``close_lifecycle``), 这里只是在同一事件上补做异步清理.
        """
        stop_waiter = asyncio.create_task(self._action_stop_event.wait(), name="action-stop-event")
        mindflow_waiter = asyncio.create_task(self._mindflow_stop_event.wait(), name="mindflow-stop-event")
        try:
            await asyncio.wait(
                {stop_waiter, mindflow_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # asyncio.wait 不管被取消时的遗留: 取消 monitor 不能把这两个 waiter 漏在循环里.
            for waiter in (stop_waiter, mindflow_waiter):
                if not waiter.done():
                    waiter.cancel()
        await self.stop()

    def bind_to_thinking_stop(self, register: Callable[[Callable[[], None]], Callable[[], None]]) -> None:
        """把本 action 的生命周期终结点挂到**签发它的 thinking** 的停止事件上.

        由 ``BaseThinking.articulator()`` 在构造时调用, 不等 ``__aenter__``: articulator()
        在构造的同一刻就把 action 放进了动作队列, 于是"已构造、还没被 enter"是一段真实存在的
        窗口 —— 窗口里的 action 没有生命周期任务, 若也没挂上这张牌, 就没有任何人会置它的
        compiled/observed, 任何在它身上等的人会被永远挂住 (2026-10-13 的卡死).

        从属层次按契约: attention 级联到 thinking, thinking 再级联到它签发的 action / articulator
        (Go ctx 的语义: 派生关系在构造那一刻成立, 不等 goroutine 被调度).
        """
        self._dispose_thinking_stop = register(self.close_lifecycle)

    def close_lifecycle(self) -> None:
        """本帧终止 → 三个生命周期端点全部落地. 同步, 跑在停止方的调用栈上.

        只置停止位不够: 没被 enter 过的 action 没有生命周期任务, 这一步是它唯一的终点.
        形状与 :meth:`abort` 一致 —— 释放等待方, 退出性质由 ``is_aborted()`` /
        ``abort_reason()`` 判别, 不靠事件值区分.
        """
        self._action_stop_event.set()
        self._compiled_event.set()
        self._observed_done_event.set()

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        # 牌已在构造时挂好 (构造即从属); 这里只起异步收尾的监视任务.
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
            # 撤牌: 这一帧已收线, 之后的停止不该再回调到它. 两张都要撤 (thinking 和 attention).
            if self._dispose_thinking_stop is not None:
                self._dispose_thinking_stop()
                self._dispose_thinking_stop = None
            if self._dispose_attention_abort is not None:
                self._dispose_attention_abort()
                self._dispose_attention_abort = None
