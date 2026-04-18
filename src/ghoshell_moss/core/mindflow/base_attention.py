from typing import Coroutine, Callable, Self, AsyncIterator, AsyncGenerator
from ghoshell_moss import Message
from ghoshell_moss.core.concepts.mindflow import (
    Attention, Impulse, Flag, Priority, Observation,
    AttentionAbortedError, Action, Articulate, Logos, Outcome, ObserveError,
    ArticulateAbortedError, ActionAbortedError,
)
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from collections import deque
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from anyio import ClosedResourceError, BrokenResourceError, create_memory_object_stream, create_task_group
import time
import math
import threading
import asyncio

__all__ = [
    'BaseAttention',
    'AttentionContext', 'BaseAction', 'BaseArticulate',
]


class AttentionContext:

    def __init__(
            self,
            *,
            attention_id: str,
            observation: Observation,
            aborted_event: ThreadSafeEvent,
            flags: dict[str, ThreadSafeEvent],
            logger: LoggerItf | None = None,
    ):
        self.attention_id = attention_id
        self.observation = observation
        self.logger = logger or get_moss_logger()
        self.logger_prefix = f"<AttentionContext id={attention_id} observation={observation.id}>"

        self._flags: dict[str, ThreadSafeEvent] = flags
        self._flag_lock = threading.Lock()

        self._aborted_event = aborted_event
        self._aborted_lock = threading.Lock()
        self._exception: Exception | None = None
        self._stop_reason: str | None = None
        self._logos: str = ''
        self._outcome_messages: list[Message] = []
        # observe 可能是多方会触发的.
        self._observe_messages: list[Message] | None = None
        self._observe_lock = threading.Lock()

    def __repr__(self):
        return self.logger_prefix

    def clear(self) -> None:
        """
        清理 ctx 里的阻塞状态.
        clear 应该是 attention 调用的.
        """
        for flag in list(self._flags.values()):
            flag.clear()

    def add_logos(self, delta: str) -> None:
        self._logos += delta

    def is_aborted(self) -> bool:
        return self._aborted_event.is_set()

    def abort(self, error: str | Exception | None) -> None:
        """线程共享的, 关闭 Attention 的信号. """
        if self._aborted_event.is_set():
            # 处理过了就 skip.
            return None
        with self._aborted_lock:
            if self._aborted_event.is_set():
                return None
            if isinstance(error, str):
                self._stop_reason = error
            elif isinstance(error, Exception):
                self._stop_reason = f"aborted on: {error}"
                self._exception = error
            self._aborted_event.set()
            return None

    async def wait_aborted(self) -> None:
        await self._aborted_event.wait()

    def get_observe_messages(self) -> list[Message] | None:
        """通常只有 Attention 所在位置会调用. """
        return self._observe_messages

    def exception(self) -> Exception | None:
        return self._exception

    def stop_at_outcome(self) -> Outcome:
        """生成新对象, 只有 Attention 调用, 应该是线程安全的. """
        last = self.observation.new_outcome()
        last.logos = self._logos
        if self._outcome_messages:
            last.messages.extend(self._outcome_messages)
        if self._observe_messages:
            last.messages.extend(self._observe_messages)
        if self._stop_reason:
            last.stop_reason = self._stop_reason
        return last

    def to_new_observation(self) -> Observation:
        last = self.stop_at_outcome()
        return last.new_observation()

    def next_frame(self) -> Self:
        """继承创建下一个 Ctx. """
        observation = self.to_new_observation()
        return AttentionContext(
            attention_id=self.attention_id,
            observation=observation,
            aborted_event=self._aborted_event,
            flags=self._flags,
            logger=self.logger,
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

    def capture_error(self, error: Exception) -> bool | None:
        """共享的异常处理逻辑. 主要协助 __aexit__ 处理拦截异常. """
        if isinstance(error, asyncio.CancelledError):
            return True
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


class BaseArticulate(Articulate):

    def __init__(
            self,
            *,
            ctx: AttentionContext,
            sender: MemoryObjectSendStream[str],
            exited_event: ThreadSafeEvent,
    ):
        self._ctx = ctx
        self._sender = sender
        self._task_group: TaskGroup | None = None
        self._exited_event = exited_event
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    @property
    def observation(self) -> Observation:
        self._check_running()
        return self._ctx.observation

    def _check_running(self):
        if not self._started:
            raise RuntimeError("Articulate is not entered")
        elif self._exited_event.is_set():
            raise RuntimeError("Articulate is already exited")

    async def _wait_aborted_and_cancel(self) -> None:
        await self._ctx.wait_aborted()
        raise AttentionAbortedError("aborted")

    async def __aenter__(self) -> Self:
        if self._started:
            return
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        self._task_group = create_task_group()
        await self._task_group.__aenter__()
        # 启动一个检查, 确保 Attention 退出时可以影响到这里.
        self._task_group.start_soon(self._wait_aborted_and_cancel)
        # 实际上底层是空的.
        await self._sender.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exited_event.is_set():
            return None
        try:
            await self._sender.__aexit__(exc_type, exc_val, exc_tb)
            if self._task_group:
                self._task_group.cancel_scope.cancel("exited")
                # 别抛出异常了.
                try:
                    await self._task_group.__aexit__(None, None, None)
                except Exception as e:
                    self._ctx.logger.info("%r task group canceled on error: %s", self, e)
            if exc_val is not None:
                return self._ctx.capture_error(exc_val)
            return None
        finally:
            # 通知运行结束.
            self._exited_event.set()

    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        self._ctx.abort(error)
        if self._task_group:
            self._task_group.cancel_scope.cancel("aborted")

    async def send_logos(self, logos: Logos) -> None:
        self._check_running()
        try:
            async for delta in logos:
                if self._ctx.is_aborted():
                    self._ctx.logger.debug("%r articulate drop delta %s after aborted", self._ctx, delta)
                    # 中断循环极其外部逻辑.
                    raise AttentionAbortedError("Attention is already aborted")
                await self._sender.send(delta)
        except ClosedResourceError:
            self._ctx.logger.error("%r articulate receive delta after closed", self._ctx)
            return None
        except BrokenResourceError:
            self._ctx.logger.debug("%r articulate drop delta after rejected", self._ctx)
            return None

    def create_task(self, cor: Coroutine) -> asyncio.Future:
        self._check_running()
        task = self._event_loop.create_task(cor)

        async def _wait_task():
            try:
                nonlocal task
                await task
            except Exception as e:
                if not self._ctx.capture_error(e):
                    raise e

        self._task_group.start_soon(_wait_task)
        return task

    def flag(self, name: str) -> Flag:
        return self._ctx.flag(name)

    async def send(self, delta: str) -> None:
        if self._ctx.is_aborted():
            self._ctx.logger.debug("%r articulate drop delta %s after aborted", self._ctx, delta)
            # 中断循环及其外部逻辑.
            raise AttentionAbortedError("Attention is already aborted")
        try:
            await self._sender.send(delta)
        except ClosedResourceError:
            # 当资源被关闭时, 说明 articulate 需要被结束了.
            # 需要一个关键讯号中断运行逻辑.
            self._ctx.logger.error("%r articulate receive delta %s after closed", self._ctx, delta)
            raise ArticulateAbortedError("Articulate shall close")
        except BrokenResourceError:
            # 接收者先退出.
            self._ctx.logger.debug("%r articulate drop delta %s after rejected", self._ctx, delta)
            raise ArticulateAbortedError("Articulate shall close")


class BaseAction(Action):

    def __init__(
            self,
            *,
            ctx: AttentionContext,
            receiver: MemoryObjectReceiveStream[str],
            exited_event: ThreadSafeEvent,
    ):
        self._ctx = ctx
        self._receiver = receiver
        self._task_group: TaskGroup | None = None
        self._exited_event = exited_event
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    def logos(self) -> Logos:
        return self._logos()

    async def _logos(self) -> AsyncGenerator[str, None]:
        try:
            async for delta in self._receiver:
                # 实际上被消费的 logo delta 才会被记录.
                self._ctx.add_logos(delta)
                yield delta
        except ClosedResourceError:
            # 直接退出即可,
            return
        except BrokenResourceError:
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
            raise RuntimeError("Articulate is not entered")
        elif self._exited_event.is_set():
            raise RuntimeError("Articulate is already exited")

    async def _wait_aborted_and_cancel(self) -> None:
        # 创建到 task group 里保证 aborted 的时候会自动退出.
        await self._ctx.wait_aborted()
        raise AttentionAbortedError("aborted")

    async def __aenter__(self) -> Self:
        if self._started:
            return
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        self._task_group = create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._wait_aborted_and_cancel)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exited_event.is_set():
            return None
        try:
            self._receiver.close()
            if self._task_group:
                self._task_group.cancel_scope.cancel("exited")
                # 别抛出异常了.
                try:
                    await self._task_group.__aexit__(None, None, None)
                except Exception as e:
                    self._ctx.capture_error(e)
            if exc_val is not None:
                return self._ctx.capture_error(exc_val)
            return None
        finally:
            # 通知运行结束.
            self._exited_event.set()

    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        self._ctx.abort(error)
        if self._task_group:
            self._task_group.cancel_scope.cancel("aborted")

    def create_task(self, cor: Coroutine) -> asyncio.Future:
        self._check_running()
        task = self._event_loop.create_task(cor)

        async def _wait_task():
            try:
                nonlocal task
                await task
            except Exception as e:
                # 加一个异常检查, 避免一个 task 抛出了低级的未处理异常, 系统仍然被 cancel 了.
                if self._ctx.capture_error(e):
                    raise e

        self._task_group.start_soon(_wait_task)
        return task

    def flag(self, name: str) -> Flag:
        return self._ctx.flag(name)


class BaseAttention(Attention):
    """
    基础的 Attention 机制实现.
    只要这个机制通过了单元测试, 就能够把系统的复杂度都屏蔽到这套实现的内侧.
    """

    def __init__(
            self,
            *,
            last_outcome: Outcome,
            impulse: Impulse,
            logger: LoggerItf | None = None,
            system_floor_strength: float = 0.0,  # 决定强度衰减到合适中断.
            source_escalation: float = 1.1,  # 决定同源 impulse 提权比例.
            max_protection_time: float = 3.0,  # 决定最大的保护时间.
            protection_duration_ratio: float = 0.2,  # 决定保护时间在总时间的比例.
    ):
        self._init_impulse: Impulse = impulse
        self._init_impulse_is_complete_event = ThreadSafeEvent()

        # 一个可以接受新消息的 buffer.
        self._info_impulse_buffer: deque[Impulse] = deque()

        self._logger = logger or get_moss_logger()

        # 关键的 flags.
        self._aborted_event = ThreadSafeEvent()
        self._flags: dict[str, ThreadSafeEvent] = {}
        # 继承的回合.
        self._inherit_outcome: Outcome = last_outcome
        # 发送 observation 时的回调.
        self._observation_callbacks: list[Callable[[Observation], None]] = []
        self._context_funcs: dict[str, Callable[[], list[Message]]] = {}

        # 运行时.
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._inner_arbiter_task: asyncio.Task | None = None

        # 这三个值通过 update impulse 更新.
        self._initial_strength: float = 0.0
        self._strength_refreshed_at: float = 0.0
        self._strength_decay_time: float = 0.0

        # 强度计算的相关参数.
        self._system_floor_strength: float = system_floor_strength
        # 当前 impulse 默认的提权效果.
        self._source_escalation: float = source_escalation
        self._max_protection_time: float = max_protection_time
        self._protection_duration_ratio: float = min(max(protection_duration_ratio, 0.0), 1.0)

        self._started: bool = False
        self._closed_event = ThreadSafeEvent()
        # update the impulse
        self._log_prefix = "?? 别忘记了."
        self._update_current_impulse(impulse)

        self._articulate_stop_event = ThreadSafeEvent()
        self._action_stop_event = ThreadSafeEvent()

        # ctx 会持续存在.
        self._ctx = AttentionContext(
            attention_id=self._init_impulse.id,
            observation=self._inherit_outcome.new_observation(),
            aborted_event=self._aborted_event,
            logger=self._logger,
            flags=self._flags,
        )

    def _update_current_impulse(self, impulse: Impulse) -> None:
        """更新当前持有的 impulse. """
        self._init_impulse = impulse
        self._initial_strength = impulse.strength
        self._strength_refreshed_at = time.monotonic()
        self._strength_decay_time = self._init_impulse.strength_decay_seconds
        if self._strength_decay_time <= 0:
            # 不要让它为0.
            self._strength_decay_time = 1
        if impulse.complete:
            # 最后才设置.
            self._init_impulse_is_complete_event.set()
        else:
            self._init_impulse_is_complete_event.clear()

    @property
    def strength_refreshed_at(self) -> float:
        return self._strength_refreshed_at

    def peek(self) -> Impulse:
        return self._init_impulse

    def is_aborted(self) -> bool:
        return self._aborted_event.is_set()

    async def wait_impulse(self) -> Impulse:
        # 阻塞等待第一个 complete event.
        await self._init_impulse_is_complete_event.wait()
        # 等待到了可能是别的原因. aborted 了.
        if self._aborted_event.is_set():
            raise AttentionAbortedError("Attention is aborted")
        return self._init_impulse

    def flag(self, name: str) -> Flag:
        # 让 ctx 的状态对齐到一起.
        return self._ctx.flag(name)

    def wait_complete_impulse(self) -> asyncio.Future[Impulse]:
        self._check_running()
        if self._init_impulse.complete:
            future = self._event_loop.create_future()
            future.set_result(self._init_impulse)
            return future
        # 直接创建 task 即可. 因为 attention 退出时, 会清理一遍锁.
        task = self._event_loop.create_task(self.wait_impulse())
        return task

    def on_observation(self, callback: Callable[[Observation], None]) -> None:
        """register observation callback"""
        self._observation_callbacks.append(callback)

    def with_context_func(self, context_name: str, context_func: Callable[[], list[Message]]) -> Self:
        """注册获取动态上下文的方式. """
        # 直接覆盖存在的 context func. Attention 应该在创建时, 至少包含 Mindflow 的
        self._context_funcs[context_name] = context_func

    async def wait_aborted(self) -> None:
        # 单纯阻塞到失效.
        await self._aborted_event.wait()

    def is_started(self) -> bool:
        return self._started

    def last_outcome(self) -> Outcome:
        # 返回最后一个 ctx 帧的 outcome 记录.
        if self.is_started():
            return self._ctx.stop_at_outcome()
        return self._inherit_outcome

    async def wait_closed(self) -> None:
        await self._aborted_event.wait()

    def _escalation_on_active(self) -> None:
        # 先简单用时间刷新来做提权. 方便 AI 大神未来帮我改.
        self._strength_refreshed_at = time.monotonic()

    def current_strength(self) -> int:
        """
        Beta 版本实现：基于剩余生存权重的线性衰减模型。
        """
        now = time.monotonic()
        elapsed = now - self._strength_refreshed_at

        # 1. 启动保护区 (Protection Buffer)
        # 逻辑：在前 20% 的时间里，Strength 保持 100% 且不会衰减，
        # 确保 Attention 建立初期不会被微小的抖动打断。
        # 由于 ttl 可能会设置很长, 所以也设置一个阈值.
        protection_time = min(self._strength_decay_time * self._protection_duration_ratio, self._max_protection_time)
        if elapsed < protection_time:
            return int(self._initial_strength * self._source_escalation)

        # 2. 运行者提权 (Escalation Gain)
        # 逻辑：我们引入一个 'active_boost'，如果系统在运行，
        # 我们认为它的“惯性”更高。
        # 只有当 elapsed 超过保护区后，才开始衰减。
        decay_elapsed = elapsed - protection_time
        decay_duration = self._strength_decay_time - protection_time

        # 归一化衰减进度 (0.0 -> 1.0)
        progress = min(decay_elapsed / decay_duration, 1.0)

        # 3. 线性衰减 + 提权惯性
        # 核心设计：如果 impulse.complete 为 True (运行中)，
        # 我们让衰减斜率减半（即：运行中的任务比待办任务更难被打断）。
        decay_factor = 1.0 if self._init_impulse.complete else 1.5

        # 计算最终强度
        # 初始强度 * (1 - 进度 * 衰减斜率)
        current = self._initial_strength * (1.0 - (progress * decay_factor))

        return int(max(current, 0))

    def loop(self) -> AsyncIterator[tuple[Articulate, Action]]:
        return self._loop()

    def _prepare_observation(self, observation: Observation) -> None:
        if len(self._context_funcs) > 0:
            # 从缓存中获取数据. 速度应该是很快的.
            for key, func in self._context_funcs.items():
                try:
                    messages = func()
                    observation.context[key] = messages
                except Exception as e:
                    self._logger.error(
                        "%s failed to prepare context messages of %s: %s",
                        self._log_prefix, key, e,
                    )

    def _callback_observation(self, observation: Observation) -> None:
        if len(self._observation_callbacks) > 0:
            for func in self._observation_callbacks:
                try:
                    func(observation)
                except Exception as e:
                    self._logger.error(
                        "%s failed to callback observation to %s: %s",
                        self._log_prefix, func, e,
                    )

    async def _loop(self) -> AsyncGenerator[tuple[Articulate, Action], None]:
        # 等待第一个完整的信号. 本质是一个抢占式注意力锁, 比如 ASR 首包打断时
        # 已经抢占了注意力, 但要等待一个完整的逻辑包才采取行动.
        await self._init_impulse_is_complete_event.wait()
        impulse = self._init_impulse
        # 完成第一轮输入的赋值. 其中 mindflow context 应该是通过 context func 更新的.
        observation = self._ctx.observation
        observation.inputs = impulse.messages
        observation.prompt = impulse.prompt
        while not self.is_aborted():
            # 每次刷新时会更新权重.
            self._escalation_on_active()
            current_observation = self._ctx.observation
            while len(self._info_impulse_buffer) > 0:
                impulse_buffer = self._info_impulse_buffer.popleft()
                # buffer messages.
                current_observation.inputs.extend(impulse_buffer.messages)
                current_observation.prompt = impulse_buffer.prompt

            # 1. 准备本轮的 Observation
            # 这里的逻辑要把 context_funcs 执行一遍，塞进 self._ctx.observation
            self._prepare_observation(current_observation)

            # 2. 创建双工流 (8000 是个缓冲区大小，可以自定)
            tx, rx = create_memory_object_stream(8000)

            # 3. 准备退出同步信号
            self._action_stop_event.clear()
            self._articulate_stop_event.clear()

            articulate = BaseArticulate(ctx=self._ctx, sender=tx, exited_event=self._articulate_stop_event)
            action = BaseAction(ctx=self._ctx, receiver=rx, exited_event=self._action_stop_event)

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
            self._ctx = self._ctx.next_frame()

    def on_challenge(self, challenger: Impulse) -> bool | None:
        """
        计算逻辑本身考虑线程安全. 重写这个函数, 可以实现不同的机制.
        """
        if challenger.is_stale():
            return False
        # challenge 要有序调用, Mindflow 需要对它进行原子操作.
        # 自己就不加锁了, 如果外层没有原子操作, 加锁也只会卡死.
        if challenger.priority == Priority.DEBUG:
            # mindflow 会 pop impulse 并丢弃.
            # debug 类型不应该走到这一步.
            self._ctx.logger.warning(
                "%s receive debug level impulse: %s",
                self._log_prefix, challenger
            )
            return None
        if challenger.id == self._init_impulse.id:
            # 来自自身的消息.
            self._update_current_impulse(challenger)
            return None
        elif challenger.source == self._init_impulse.source and challenger.priority == Priority.INFO:
            self._info_impulse_buffer.append(challenger)
            return None
        # priority is superior
        if challenger.priority == Priority.FATAL or challenger.priority > self._init_impulse.priority:
            return True
        elif challenger.priority < self._init_impulse.priority:
            return False
        challenger_strength = challenger.strength
        if challenger.source == self._init_impulse.source:
            # 同源数据提权.
            challenger_strength = int(challenger_strength * self._source_escalation)

        current_strength = self.current_strength()
        return current_strength < challenger_strength

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
            await asyncio.sleep(ttl)

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
            self._init_impulse_is_complete_event.set()
            # 清除 ctx 的锁状态. 释放所有的阻塞.
            self._ctx.clear()
            self._action_stop_event.set()
            self._articulate_stop_event.set()

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        # 启动自身的超时检查.
        self._inner_arbiter_task = self._event_loop.create_task(self._inner_attention_lifecycle())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关键是哪些异常是需要对外抛出的.
        """
        if self._closed_event.is_set():
            return None
        try:
            # 取消 inner task.
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
        finally:
            self._ctx.clear()
            # 清除一些容易互相持有的逻辑.
            self._context_funcs.clear()
            self._observation_callbacks.clear()
            # 两个确保能够退出的标记.
            self._aborted_event.set()
            self._closed_event.set()
