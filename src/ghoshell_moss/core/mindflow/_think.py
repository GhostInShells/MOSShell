"""
Thinking 实现.

重构自历史文件 ./base_attention.py
"""

import asyncio
import contextlib
from typing import Callable
from typing_extensions import Self

import janus

from ghoshell_moss.core.blueprint.mindflow import (
    Action, ActionExitedException, Articulator, Attention, AttentionExitedException,
    Thinking, ThinkExitedException, ThinkingEffort, ActionGate
)
from ghoshell_moss.core.blueprint.moment import Moment, Moments, Observer
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ._action import ActionLogosRequest, BaseAction, BaseActionGate, BaseArticulator

__all__ = ['BaseThinking']


class BaseThinking(Thinking):
    """Thinking 的基类实现.

    职责: 持有 attention / moment / observer, 把 thinking 与 action 通过
    logos_queue + events 装线在一起. gate 惰性创建, per-thinking 一个;
    articulator() 依据 gate 是否已实例化来区分 gated / non-gated 接入策略.

    从 attention 能拿到的都走 attention: effort 由 attention.draw_from() 的
    发起 impulse 提供; 生命周期/中断由 attention.wait_abort() 级联下来.
    """

    def __init__(
            self,
            *,
            attention: Attention,
            observer: Observer,
            put_action: Callable[[Action], None],
            mindflow_stop_event: ThreadSafeEvent,
            moment: Moment | None = None,
            logger: LoggerItf | None = None,
    ):
        self._attention = attention
        self._observer = observer
        self._put_action = put_action
        self._mindflow_stop_event = mindflow_stop_event
        # 初始帧来自 mindflow 预观测+折帧; 为 None 时首次访问 moment 才懒观测一帧.
        self._moment = moment
        self._logger = logger or get_moss_logger()
        self._log_prefix = f"<Thinking attention={attention.id}>"

        self._gate: BaseActionGate | None = None
        self._stop_event = ThreadSafeEvent()
        self._started = False
        self._stopped = False
        self._lifecycle_task: asyncio.Task | None = None
        self._waiting_futures: list[asyncio.Future] = []

    def __repr__(self) -> str:
        return self._log_prefix

    # ── Thinking ABC ──────────────────────────────

    @property
    def attention(self) -> Attention:
        return self._attention

    @property
    def observer(self) -> Observer:
        return self._observer

    @property
    def moment(self) -> Moment:
        if self._moment is None:
            self._moment = self._observer.observe()
        return self._moment

    def observe(self) -> Moment:
        self._moment = self._observer.observe()
        return self._moment

    def effort(self) -> ThinkingEffort:
        return self._attention.draw_from().thinking_effort

    def gate(self) -> ActionGate:
        if self._gate is None:
            self._gate = BaseActionGate(stop_event=self._stop_event)
        return self._gate

    def articulator(self, replan: bool = False, wait_action_done: bool = False) -> Articulator:
        """
        创建一个可以发布 logos 的 articulator, 与一个新的 BaseAction 成对.

        gate 已实例化 (= gated 模式) 时走 ActionLogosRequest buffer, 等审批通过才
        投递 action; 否则立即 put_action, logos 直接进 queue.
        """
        logos_queue: janus.Queue[str | None] = janus.Queue()
        compiled_event = ThreadSafeEvent()
        action_stop_event = ThreadSafeEvent()

        action = BaseAction(
            attention=self._attention,
            moments=self._observer,
            replaned=replan,
            logos_queue=logos_queue,
            compiled_event=compiled_event,
            action_stop_event=action_stop_event,
            mindflow_stop_event=self._mindflow_stop_event,
            logger=self._logger,
        )

        gated = self._gate is not None
        logos_request: ActionLogosRequest | None = None
        if gated:
            logos_request = ActionLogosRequest(
                logos='',
                action=action,
                put_action=self._put_action,
            )
            self._gate.add_request(logos_request)
        else:
            self._put_action(action)

        return BaseArticulator(
            moment=self.moment,
            logos_queue=logos_queue,
            compiled_event=compiled_event,
            action_stop_event=action_stop_event,
            logos_request=logos_request,
            logger=self._logger,
        )

    async def wait_until_done(self, *futures: asyncio.Future) -> None:
        ensured = []
        for future in futures:
            fut = asyncio.ensure_future(future)
            ensured.append(fut)
            if self.is_aborted():
                fut.cancel()
        self._waiting_futures.extend(ensured)
        await asyncio.gather(*ensured, return_exceptions=True)

    # ── MindflowStatement 生命周期 ────────────────

    def is_running(self) -> bool:
        return (
                self._started and not self._stopped
                and not self._mindflow_stop_event.is_set()
                and not self._attention.is_aborted()
        )

    def is_aborted(self) -> bool:
        return (
                self._stop_event.is_set()
                or self._attention.is_aborted()
                or self._mindflow_stop_event.is_set()
        )

    def abort(self, reason: str | Exception | None) -> None:
        self._attention.abort(reason)
        self._stop_event.set()

    def abort_reason(self) -> str:
        return self._attention.abort_reason()

    async def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._waiting_futures:
            for t in self._waiting_futures:
                t.cancel()
            _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

    async def wait_abort(self) -> None:
        await self._stop_event.wait()

    async def _lifecycle_aborted_monitor(self) -> None:
        await asyncio.wait(
            [
                asyncio.create_task(self._attention.wait_abort()),
                asyncio.create_task(self._stop_event.wait()),
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

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
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
            if self._waiting_futures:
                for t in self._waiting_futures:
                    if not t.done():
                        t.cancel()
                _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

            if exc_val:
                if isinstance(exc_val, (ActionExitedException, AttentionExitedException, ThinkExitedException)):
                    return True
                self._attention.abort(exc_val)
            return None
        finally:
            self._stop_event.set()
