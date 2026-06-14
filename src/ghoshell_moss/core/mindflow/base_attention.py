from abc import abstractmethod, ABC
from typing import Coroutine, Callable, AsyncIterator, AsyncGenerator, Iterable
from typing_extensions import Self

from ghoshell_moss.core.blueprint import ThinkingEffort
from ghoshell_moss.message import Message
from ghoshell_moss.core.blueprint.mindflow import (
    Attention, Impulse, Flag, Priority, Moment,
    AttentionAbortedError, Action, Articulator, Logos, Reaction, ObserveError,
    ArticulateAbortedError, ActionAbortedError, ImpulseAbsorbed
)
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from collections import deque
import time
import threading
import asyncio
import janus

__all__ = [
    'AbsAttention',
    'BaseAttention',
    'AttentionContext', 'BaseAction', 'BaseArticulator',
]


class AttentionContext:

    def __init__(
            self,
            *,
            attention_id: str,
            moment: Moment,
            aborted_event: ThreadSafeEvent,
            flags: dict[str, ThreadSafeEvent],
            logger: LoggerItf | None = None,
            max_size: int = 8000,
    ):
        self.logos_queue: janus.Queue[str | None] = janus.Queue(maxsize=max_size)
        self._max_size = max_size
        self.attention_id = attention_id
        self.moment = moment
        self.logger = logger or get_moss_logger()
        self.logger_prefix = f"<AttentionContext id={attention_id} observation={moment.id}>"

        self._flags: dict[str, ThreadSafeEvent] = flags
        self._flag_lock = threading.Lock()

        self._aborted_event = aborted_event
        self._exception: BaseException | None = None
        self._stop_reason: str | None = None
        self._logos: str = ''
        self._outcome_messages: list[Message] = []
        # observe 可能是多方会触发的.
        self._observe_messages: list[Message] | None = None
        self._observe_lock = threading.Lock()

    def __repr__(self):
        return self.logger_prefix

    def buffer_executed_logos(self, delta: str) -> None:
        self._logos += delta

    def is_aborted(self) -> bool:
        return self._aborted_event.is_set()

    def abort(self, error: str | BaseException | None) -> None:
        """线程共享的, 关闭 Attention 的信号. """
        if self._aborted_event.is_set():
            # 处理过了就 skip.
            return None
        if self._aborted_event.is_set():
            return None
        if isinstance(error, str):
            self._stop_reason = error
        elif isinstance(error, BaseException):
            self._stop_reason = f"aborted on: {error}"
            self._exception = error
        for flag in list(self._flags.values()):
            flag.clear()
        self.logos_queue.sync_q.put_nowait(None)
        self._aborted_event.set()
        return None

    async def wait_aborted(self) -> None:
        await self._aborted_event.wait()

    def get_observe_messages(self) -> list[Message] | None:
        """通常只有 Attention 所在位置会调用. """
        return self._observe_messages

    def exception(self) -> Exception | None:
        return self._exception

    def stop_at_outcome(self) -> Reaction:
        """生成新对象, 只有 Attention 调用, 应该是线程安全的. """
        last = self.moment.new_reaction()
        last.executed_logos = self._logos
        if self._outcome_messages:
            last.messages.extend(self._outcome_messages)
        if self._observe_messages:
            last.messages.extend(self._observe_messages)
        if self._stop_reason:
            last.stop_reason = self._stop_reason
        return last

    def new_moment(self) -> Moment:
        last = self.stop_at_outcome()
        return last.new_moment()

    def next_frame(self) -> Self:
        """继承创建下一个 Ctx. """
        moment = self.new_moment()
        return AttentionContext(
            attention_id=self.attention_id,
            moment=moment,
            aborted_event=self._aborted_event,
            flags=self._flags,
            logger=self.logger,
            max_size=self._max_size,
        )

    def observe(self, message: str) -> None:
        """两边线程可能都会调度的 observe 方法. """
        with self._observe_lock:
            if self._observe_messages is None:
                self._observe_messages = []
            if message:
                self._observe_messages.append(Message.new().with_content(message))
            # observe 不直接关闭什么.
            return None

    def outcome(self, *messages: Message, observe: bool) -> None:
        """outcome 目前只有 actions 侧使用. """
        self._outcome_messages.extend(messages)
        if observe:
            self.observe('')

    def capture_error(self, error: BaseException) -> bool | None:
        """共享的异常处理逻辑. 主要协助 __aexit__ 处理拦截异常. """
        if isinstance(error, asyncio.CancelledError):
            return None
        elif isinstance(error, asyncio.TimeoutError):
            return True
        elif isinstance(error, ActionAbortedError):
            # 正常的关闭讯号.
            return True
        elif isinstance(error, ArticulateAbortedError):
            # 正常的关闭讯号.
            return True
        elif isinstance(error, ObserveError):
            with self._observe_lock:
                if not self._observe_messages:
                    self._observe_messages = []
                self._observe_messages.extend(error.as_messages())
            return True
        elif isinstance(error, AttentionAbortedError):
            self.abort(error)
            return True
        else:
            self.logger.error("%s capture exception: %s", self.logger_prefix, error)
            self.abort(error)
            return False

    def flag(self, name: str) -> ThreadSafeEvent:
        """调用的频率应该非常低. """
        with self._flag_lock:
            if name not in self._flags:
                self._flags[name] = ThreadSafeEvent()
            return self._flags[name]


class BaseArticulator(Articulator):

    def __init__(
            self,
            *,
            ctx: AttentionContext,
            exited_event: ThreadSafeEvent,
            thinking_effort: ThinkingEffort,
    ):
        self._ctx = ctx
        self._task_group = BaseTaskGroup()
        self._exited_event = exited_event
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._thinking_effort = thinking_effort
        self._started = False
        self._closing = False

    @property
    def moment(self) -> Moment:
        self._check_running()
        return self._ctx.moment

    def thinking_effort(self) -> ThinkingEffort:
        return self._thinking_effort

    def _check_running(self):
        if not self._started:
            raise RuntimeError("Articulate is not entered")
        elif self._exited_event.is_set():
            raise ArticulateAbortedError("Articulate is already exited")

    async def _wait_aborted_and_cancel(self) -> None:
        await self._ctx.wait_aborted()
        raise AttentionAbortedError("aborted")

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("Articulate is already entered")
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        # 启动一个检查, 确保 Attention 退出时可以影响到这里.
        self._task_group.add_task(self._event_loop.create_task(self._wait_aborted_and_cancel()))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._closing:
            return None
        self._closing = True
        try:
            self._ctx.logos_queue.sync_q.put_nowait(None)
            await self._task_group.aclose()
            if exc_val is not None:
                return self._ctx.capture_error(exc_val)
            return None
        finally:
            # 通知运行结束.
            self._exited_event.set()

    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        self._ctx.abort(error)
        self._task_group.close()

    async def send_logos(self, logos: Logos) -> None:
        self._check_running()
        async for delta in logos:
            self.send_nowait(delta)

    def create_task(self, cor: Coroutine) -> asyncio.Future:
        self._check_running()
        task = self._event_loop.create_task(cor)
        self._task_group.add_task(task)
        return task

    def flag(self, name: str) -> Flag:
        return self._ctx.flag(name)

    def send_nowait(self, logos_delta: str) -> None:
        if self._ctx.is_aborted() or self._exited_event.is_set():
            self._ctx.logger.debug("%r articulate drop delta %s after aborted", self._ctx, logos_delta)
            # 中断循环及其外部逻辑.
            raise AttentionAbortedError("Attention is already aborted")
        try:
            self._ctx.logos_queue.sync_q.put_nowait(logos_delta)
        except janus.SyncQueueShutDown:
            raise AttentionAbortedError("Attention is already aborted")


class BaseTaskGroup:

    def __init__(self):
        self.tasks: set[asyncio.Task] = set()
        self._closed = False

    def add_task(self, task: asyncio.Task) -> None:
        if self._closed:
            task.cancel('closed')
            return
        self.tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        if self._closed:
            return
        self.tasks.discard(task)
        if task.cancelled():
            return
        elif task.exception():
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self.tasks)
        for t in tasks:
            if not t.done():
                t.cancel()

    async def aclose(self) -> None:
        self.close()
        tasks = list(self.tasks)
        wait_all = []
        for t in tasks:
            if not t.done():
                t.cancel()
                wait_all.append(t)
        if len(wait_all) > 0:
            await asyncio.gather(*wait_all, return_exceptions=True)


class BaseAction(Action):

    def __init__(
            self,
            *,
            ctx: AttentionContext,
            exited_event: ThreadSafeEvent,
    ):
        self._ctx = ctx
        self._task_group = BaseTaskGroup()
        self._exited_event = exited_event
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._closing = False
        self._prefetched_delta: str | None = None

    async def wait_ready(self) -> None:
        """Wait until first logos delta arrives or attention is aborted.
        消费第一个 delta 并缓存，received_logos() 从缓存开始消费。
        """
        if self._ctx.is_aborted():
            return
        try:
            while not self._ctx.is_aborted():
                try:
                    item = await asyncio.wait_for(
                        self._ctx.logos_queue.async_q.get(),
                        timeout=0.05,
                    )
                    self._prefetched_delta = item
                    return
                except asyncio.TimeoutError:
                    continue
        except janus.AsyncQueueShutDown:
            pass

    def received_logos(self) -> Logos:
        return self._logos()

    async def _logos(self) -> AsyncGenerator[str, None]:
        # drain prefetched delta first
        if self._prefetched_delta is not None:
            item = self._prefetched_delta
            self._prefetched_delta = None
            if item is None:
                return
            self._ctx.buffer_executed_logos(item)
            yield item
        try:
            while not self._ctx.is_aborted() and not self._exited_event.is_set():
                try:
                    item = await asyncio.wait_for(self._ctx.logos_queue.async_q.get(), 1)
                except asyncio.TimeoutError:
                    continue
                except janus.AsyncQueueShutDown:
                    return

                if item is None:
                    break
                self._ctx.buffer_executed_logos(item)
                yield item
        except janus.SyncQueueShutDown:
            return

    def outcome(self, *messages: Message | str, observe: bool = False) -> None:
        saving = []
        for message in messages:
            if isinstance(message, Message):
                saving.append(message)
            else:
                saving.append(Message.new().with_content(message))
        # 这里会记录 observe, 但是不会中断什么.
        # 如果希望触发 observe 就立刻中断, 还是应该外部 Action 的逻辑里处理.
        self._ctx.outcome(*saving, observe=observe)

    def _check_running(self):
        if not self._started:
            raise RuntimeError("Action is not entered")
        elif self._exited_event.is_set():
            raise ActionAbortedError("Action is already exited")

    async def _wait_aborted_and_cancel(self) -> None:
        # 创建到 task group 里保证 aborted 的时候会自动退出.
        await self._ctx.wait_aborted()
        raise AttentionAbortedError("aborted")

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("Action is already entered")
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        self._task_group.add_task(self._event_loop.create_task(self._wait_aborted_and_cancel()))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._closing:
            return None
        self._closing = True
        try:
            # 阻塞等待到运行结束.
            await self._task_group.aclose()
            if exc_val is not None:
                return self._ctx.capture_error(exc_val)
            return None
        finally:
            # 通知运行结束.
            self._exited_event.set()

    def is_aborted(self) -> bool:
        return self._ctx.is_aborted()

    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        self._ctx.abort(error)
        self._task_group.close()

    def create_task(self, cor: Coroutine) -> asyncio.Future:
        self._check_running()
        task = self._event_loop.create_task(cor)
        self._task_group.add_task(task)
        return task

    def flag(self, name: str) -> Flag:
        return self._ctx.flag(name)


class AbsAttention(Attention, ABC):
    """
    Attention 的抽象基类. 管理全部生命周期机械: loop, observe 循环, context func,
    abort, Articulator/Action 创建, 强度状态存储.

    challenge() 和 current_strength() 留给子类实现仲裁策略.
    """

    def __init__(
            self,
            *,
            previous: Reaction,
            impulse: Impulse,
            logger: LoggerItf | None = None,
            system_floor_strength: float = 0.0,  # 当 current_strength() <= 此值时自行 fade out
    ):
        self._init_impulse: Impulse = impulse
        self._has_any_completed_impulse = ThreadSafeEvent()

        self._logger = logger or get_moss_logger()

        # 关键的 flags.
        self._aborted_event = ThreadSafeEvent()
        self._flags: dict[str, ThreadSafeEvent] = {}
        # 继承的回合.
        self._previous_reaction: Reaction = previous
        # 发送 observation 时的回调.
        self._on_moment_callbacks: list[Callable[[Moment], None]] = []
        self._perspectives_funcs: dict[str, Callable[[], list[Message]]] = {}
        self._thinking_effort = impulse.thinking_effort

        # 运行时.
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._inner_arbiter_task: asyncio.Task | None = None

        # 这三个值通过 update impulse 更新，子类只读.
        self._strength_start_value: float = 0.0
        self._strength_refreshed_at: float = 0.0
        self._strength_decay_time: float = 0.0

        # 强度 floor — _inner_attention_lifecycle 用它判断是否自行 fade out.
        self._system_floor_strength: float = system_floor_strength

        self._started: bool = False
        self._closing: bool = False
        self._closed_event = ThreadSafeEvent()
        # update the impulse
        self._log_prefix = f"<Attention id={self._init_impulse.id}>"

        self._articulate_stop_event = ThreadSafeEvent()
        self._action_stop_event = ThreadSafeEvent()
        self._articulate_stop_event.set()
        self._action_stop_event.set()
        self._percepts_funcs: dict[str, Callable[[], list[Message]]] = {}
        self._buffered_impulses: deque[Impulse] = deque()
        self._buffered_impulse_ids: set[str] = set()
        self._buffer_impulse(impulse)

        # ctx 会持续存在.
        self._ctx = AttentionContext(
            attention_id=self._init_impulse.id,
            # 当前的 moment 帧. 每一帧在生产时都会更新.
            moment=self._previous_reaction.new_moment(),
            aborted_event=self._aborted_event,
            logger=self._logger,
            flags=self._flags,
            max_size=8000,
        )

    def __repr__(self):
        return self._log_prefix

    def _buffer_impulse(self, impulse: Impulse) -> None:
        """更新当前持有的 impulse. """
        if impulse.id in self._buffered_impulse_ids:
            # 说明这个 impulse 已经被 buffer 过了.
            # 可能有某个 nucleus 的机制不生效, 造成了重复请求.
            return
        # 起始强度, 用于计算当前强度.
        self._strength_start_value = impulse.strength
        # 强度最后更新时间.
        self._strength_refreshed_at = time.monotonic()
        # 保护期所在时间点.
        self._protected_until = self._strength_refreshed_at + impulse.protection_time
        self._strength_decay_time = self._init_impulse.strength_decay_seconds
        if self._strength_decay_time <= 0:
            # 不要让它为0.
            self._strength_decay_time = 1
        if impulse.complete:
            # 只有 complete 类型的 impulse 才会进入 buffer, 其它的只是占据注意力.
            self._buffered_impulses.append(impulse)
            self._buffered_impulse_ids.add(impulse.id)
            self._thinking_effort = impulse.thinking_effort
            self._has_any_completed_impulse.set()
        else:
            self._has_any_completed_impulse.clear()

    def _pop_buffered_impulses(self) -> Iterable[Impulse]:
        """退出最后一个 impulse. """
        while len(self._buffered_impulses) > 0:
            impulse = self._buffered_impulses.popleft()
            yield impulse

    @property
    def strength_refreshed_at(self) -> float:
        """测试专用参数, 避免取私有值. """
        return self._strength_refreshed_at

    @property
    def thinking_effort(self) -> ThinkingEffort:
        """当前 attention 持有的 thinking_effort, 由最后一个 complete impulse 决定.
        与 yielded Articulator.thinking_effort() 同源, 用于测试与反身性观察."""
        return self._thinking_effort

    @property
    def strength_start_value(self) -> float:
        """强度衰减曲线起点. 由最新进入 buffer 的 impulse 刷新."""
        return self._strength_start_value

    @property
    def strength_decay_time(self) -> float:
        """attention 衰减总时长, 钉在 init impulse 的 strength_decay_seconds (clamp 到 >=1)."""
        return self._strength_decay_time

    @property
    def protected_until(self) -> float:
        """同优先级保护期截止 monotonic 时间."""
        return self._protected_until

    @property
    def buffered_impulses(self) -> list[Impulse]:
        """快照尚未被 drain 的 buffered impulses (不消费). 用于测试与反身性观察."""
        return list(self._buffered_impulses)

    def draw_from(self) -> Impulse:
        return self._init_impulse

    def is_aborted(self) -> bool:
        return self._aborted_event.is_set()

    async def wait_any_completed_impulse(self) -> None:
        # 阻塞等待第一个 complete event.
        await self._has_any_completed_impulse.wait()

    def flag(self, name: str) -> Flag:
        # 让 ctx 的状态对齐到一起.
        return self._ctx.flag(name)

    def on_moment(self, callback: Callable[[Moment], None]) -> None:
        """register observation callback"""
        self._on_moment_callbacks.append(callback)

    def with_perspective_func(self, perspective_key: str, perspective_func: Callable[[], list[Message]]) -> Self:
        """注册获取动态上下文的方式. """
        # 直接覆盖存在的 context func. Attention 应该在创建时, 至少包含 Mindflow 的
        self._perspectives_funcs[perspective_key] = perspective_func

    def with_percepts_func(self, percepts_key: str, percepts_func: Callable[[], list[Message]]) -> Self:
        # 注册上下文函数, 在运行前丰满上下文.
        self._percepts_funcs[percepts_key] = percepts_func

    async def wait_aborted(self) -> None:
        # 单纯阻塞到失效.
        await self._aborted_event.wait()

    def is_started(self) -> bool:
        return self._started

    def last_outcome(self) -> Reaction:
        # 返回最后一个 ctx 帧的 outcome 记录.
        if self.is_started():
            return self._ctx.stop_at_outcome()
        return self._previous_reaction

    async def wait_closed(self) -> None:
        await self._aborted_event.wait()

    def _escalation_on_active(self) -> None:
        # 先简单用时间刷新来做提权. 方便 AI 大神未来帮我改.
        self._strength_refreshed_at = time.monotonic()

    @abstractmethod
    def current_strength(self) -> int:
        """基于剩余生存权重的线性衰减模型."""
        pass

    def loop(self) -> AsyncIterator[tuple[Articulator, Action]]:
        return self._loop()

    def _prepare_moment(self, moment: Moment) -> None:
        if len(self._perspectives_funcs) > 0:
            # 从缓存中获取数据. 速度应该是很快的.
            for key, func in self._perspectives_funcs.items():
                try:
                    messages = func()
                    moment.perspectives[key] = messages
                except Exception as e:
                    self._logger.error(
                        "%s failed to prepare perspective messages of %s: %s",
                        self._log_prefix, key, e,
                    )
        if len(self._percepts_funcs) > 0:
            for key, func in self._percepts_funcs.items():
                try:
                    messages = func()
                    moment.percepts.extend(messages)
                except Exception as e:
                    self._logger.error(
                        "%s failed to prepare percepts messages of %s: %s",
                        self._log_prefix, key, e,
                    )
        for impulse in self._pop_buffered_impulses():
            # 更新哩个 moment.
            impulse.update_moment(moment)

    def _callback_moment(self, moment: Moment) -> None:
        if len(self._on_moment_callbacks) > 0:
            for func in self._on_moment_callbacks:
                try:
                    func(moment)
                except Exception as e:
                    self._logger.error(
                        "%s failed to callback observation to %s: %s",
                        self._log_prefix, func, e,
                    )

    async def _loop(self) -> AsyncGenerator[tuple[Articulator, Action], None]:
        # 等待第一个完整的信号. 本质是一个抢占式注意力锁, 比如 ASR 首包打断时
        # 已经抢占了注意力, 但要等待一个完整的逻辑包才采取行动.
        await self.wait_any_completed_impulse()
        while not self.is_aborted():
            # 每次刷新时会更新权重.
            self._escalation_on_active()
            current_moment = self._ctx.moment
            # 1. 准备本轮的 moment. 要更新所有的运行时消息.
            self._prepare_moment(current_moment)
            # 回调 moment.
            self._callback_moment(current_moment)

            # 2. 创建双工流 (8000 是个缓冲区大小，可以自定)
            # 3. 准备退出同步信号
            self._action_stop_event.clear()
            self._articulate_stop_event.clear()

            articulate = BaseArticulator(
                ctx=self._ctx,
                exited_event=self._articulate_stop_event,
                thinking_effort=self._thinking_effort,
            )
            action = BaseAction(ctx=self._ctx, exited_event=self._action_stop_event)

            # 4. 交给外部执行线程/任务
            yield articulate, action

            # 5. 等待双子星运行结束. 顺序不重要.
            # 注意, attention 即便 aborted 了, 这里也需要等待运行结束.
            # 主要是确保 Articulate / Action 的运行周期正式结束. 所有回收逻辑完成.
            await self._articulate_stop_event.wait()
            await self._action_stop_event.wait()

            # 6. 核心：检查是否需要继续观察
            # 看看 Action 是否调用了 outcome(observe=True) 或者触发了 ObserveError
            if self._ctx.get_observe_messages() is None:
                # 没有任何一方要求继续看，注意力自然结束
                # 当前的 ctx 就是最后一帧了.
                break

            # 7. 如果要继续, 要更新 ctx 准备下一轮.
            # 更新后, moment 也是新的.
            self._ctx = self._ctx.next_frame()

    def challenge(self, challenger: Impulse) -> bool | None:
        """
        True: 抢占成功, 当前 attention 被 abort, 新 impulse 接管.
        False: 压制, 挑战失败. 挑战者被 suppress.
        None: 吸收, 挑战被内部消化 (如同源更新/buffer/DEBUG), 不影响当前 attention.
        """
        if challenger.is_stale():
            return False
        if challenger.id == self._init_impulse.id:
            # 相同 id 的永远可以 buffer. 但只 buffer 一次.
            self._buffer_impulse(challenger)
            return ImpulseAbsorbed
        elif challenger.priority == Priority.FATAL or challenger.priority > self._init_impulse.priority:
            return True
        elif challenger.priority < self._init_impulse.priority:
            return False
        # 相同优先级保护期.
        elif self._protected_until > time.monotonic():
            return False
        return self.arbit_challenge_by_strength(challenger)

    @abstractmethod
    def arbit_challenge_by_strength(self, challenger: Impulse) -> bool | None:
        """
        True: 抢占成功, 当前 attention 被 abort, 新 impulse 接管.
        False: 压制, 挑战失败. 挑战者被 suppress.
        None: 吸收, 挑战被内部消化 (如同源更新/buffer/DEBUG), 不影响当前 attention.
        """
        pass

    def is_closed(self) -> bool:
        return self._aborted_event.is_set()

    def abort(self, error: str | Exception | None) -> None:
        self._ctx.abort(error)

    def _check_running(self) -> None:
        if not self._started or self._aborted_event.is_set() or self._event_loop is None:
            raise asyncio.CancelledError("Attention is not running")

    async def _inner_attention_lifecycle(self) -> None:
        """
        在自己内部做自己是否应该结束的仲裁.
        收到挑战, 第一时间返回属于条件反射.
        实际上仍然可以有一个周期去内省.
        """
        try:
            ttl = self._strength_decay_time
            wait_task = asyncio.create_task(asyncio.sleep(ttl))
            wait_done_task = asyncio.create_task(self._ctx.wait_aborted())
            done, pending = await asyncio.wait(
                [wait_task, wait_done_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            # 如果 abort 先触发，直接退出
            if self._aborted_event.is_set():
                return None
            # 做一个低阶的自省, 防止另外两个循环卡死.
            while not self._aborted_event.is_set():
                if self.current_strength() <= self._system_floor_strength:
                    # 自主结束.
                    self.abort(asyncio.TimeoutError("attention fade out"))
                    break
                try:
                    await asyncio.wait_for(self._aborted_event.wait(), 0.5)
                except asyncio.TimeoutError:
                    continue
            return None
        finally:
            # 这个任务退出时, 一种情况是 aborted, 另一种情况是 aexit, 两种情况都去清理所有可能阻塞的锁.
            self._has_any_completed_impulse.set()
            self._action_stop_event.set()
            self._articulate_stop_event.set()

    async def __aenter__(self):
        if self._started:
            raise RuntimeError("Attention is already entered")
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        # 启动自身的超时检查.
        self._inner_arbiter_task = self._event_loop.create_task(self._inner_attention_lifecycle())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关键是哪些异常是需要对外抛出的.
        """
        if self._closing:
            return None
        self._closing = True
        try:
            # 取消 inner task.
            self._ctx.abort(exc_val)
            if self._inner_arbiter_task is not None and not self._inner_arbiter_task.done():
                self._inner_arbiter_task.cancel()
                try:
                    await self._inner_arbiter_task
                except asyncio.CancelledError:
                    pass
            self._inner_arbiter_task = None
            # 再执行一次好了.
            self._event_loop = None
            if exc_val is not None:
                # 判断是否要拦截.
                return self._ctx.capture_error(exc_val)
            await self._articulate_stop_event.wait()
            await self._action_stop_event.wait()
        finally:
            # 清除一些容易互相持有的逻辑.
            self._perspectives_funcs.clear()
            self._on_moment_callbacks.clear()
            # 两个确保能够退出的标记.
            self._aborted_event.set()
            self._closed_event.set()


class BaseAttention(AbsAttention):
    """
    Beta 版本的强度衰减仲裁实现.
    保持原有构造签名和行为不变, 用于 BaseMindflow._create_attention_from_impulse() 向后兼容.
    """

    def __init__(
            self,
            *,
            previous: Reaction,
            impulse: Impulse,
            logger: LoggerItf | None = None,
            system_floor_strength: float = 0.0,
            source_escalation: float = 1.1,
            max_protection_time: float = 3.0,
            protection_duration_ratio: float = 0.2,
    ):
        super().__init__(
            previous=previous,
            impulse=impulse,
            logger=logger,
            system_floor_strength=system_floor_strength,
        )
        self._source_escalation: float = source_escalation
        self._max_protection_time: float = max_protection_time
        self._protection_duration_ratio: float = min(max(protection_duration_ratio, 0.0), 1.0)

    def arbit_challenge_by_strength(self, challenger: Impulse) -> bool | None:
        challenger_strength = challenger.strength
        if challenger.source == self._init_impulse.source:
            challenger_strength = int(challenger_strength * self._source_escalation)
        current_strength = self.current_strength()
        return current_strength < challenger_strength

    def current_strength(self) -> int:
        """基于剩余生存权重的线性衰减模型. 强度在不提权情况下, 默认要衰减到 0"""
        now = time.monotonic()
        elapsed = now - self._strength_refreshed_at

        protection_time = min(
            self._strength_decay_time * self._protection_duration_ratio,
            self._max_protection_time,
        )
        if elapsed < protection_time:
            return int(self._strength_start_value * self._source_escalation)

        decay_elapsed = elapsed - protection_time
        decay_duration = self._strength_decay_time - protection_time
        progress = min(decay_elapsed / decay_duration, 1.0)
        decay_factor = 1.0 if self._init_impulse.complete else 1.5
        current = self._strength_start_value * (1.0 - (progress * decay_factor))
        return int(max(current, 0))
