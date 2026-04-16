import threading
from collections import defaultdict
from typing import Coroutine, Callable, Self, AsyncIterator

from click import Abort

from ghoshell_moss import Message
from ghoshell_moss.core.concepts.mindflow import (
    Attention, Impulse, Flag, Priority, Observation,
    AbortAttentionError, Actions, Observations, Logos,
)
from ghoshell_moss.core.concepts.errors import FatalError
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
import asyncio
import anyio
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
import time
import math
import janus
import threading

__all__ = [
    'BaseAttention',
]


class BaseLogosWriter(LogosWriter):
    def __init__(
            self,
            attention_id: str,
            *,
            attention_aborted: ThreadSafeEvent,
            stream_callback: Callable[[MemoryObjectReceiveStream], None],
            record_logos_callback: Callable[[str], None],
            # 第一个 logos writer 会拿到 on_start_logs.
            on_logos_start: str = '',
            logger: LoggerItf | None = None,
    ) -> None:
        self._attention_id = attention_id
        self._stream_callback = stream_callback
        self._logger = logger or get_moss_logger()
        self._record_logos = record_logos_callback
        self._on_logos_start = on_logos_start
        self._logos_buffer: str = ''
        self._has_contents: bool = False
        self._attention_aborted = attention_aborted
        self._started = False
        self._closed = False
        self._stream_sender: MemoryObjectSendStream[str] | None = None
        self._log_prefix = f"<LogosWriter attention={attention_id}>"

    def _check_running(self) -> None:
        if not self._started:
            raise RuntimeError("Logos shall run in with statement")
        elif self._closed:
            raise RuntimeError("Logos already exit")
        elif self._attention_aborted.is_set():
            # attention 已经被取消了. 扔出一个可忽略的 Cancel Error.
            # raise cancel error?
            raise asyncio.CancelledError("Attention already aborted")

    def send_nowait(self, delta: str) -> None:
        # 任何高级异常都会
        self._check_running()
        # 先检查第一个有内容的消息块, 决定是否发布.
        if self._stream_sender is None:
            # 第一个有语义元素才算发布.
            is_empty_delta = len(delta.strip()) == 0
            if not is_empty_delta:
                self._has_contents = True
                # 第一次发布所有的 buffer.
                sender, receiver = anyio.create_memory_object_stream[str]()
                self._stream_sender = sender
                # 在这里启动.
                sender.__enter__()
                # 发送所有的 buffer.
                sender.send_nowait(self._logos_buffer)
                # 记录 logos 的回调.
                self._record_logos(self._logos_buffer)
                # 回调 receiver. 预计要对 Attention 做一次提权.
                self._stream_callback(receiver)
        # buffer
        if self._stream_sender is not None:
            self._stream_sender.send_nowait(delta)
            self._record_logos(delta)
        else:
            self._logos_buffer += delta

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        # 需要有启动.
        if self._on_logos_start:
            # 直接将 on logos start 发送, 预期可以自动创建 sender.
            self.send_nowait(self._on_logos_start)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """通过 Logos writer 来捕获, 处理异常, 方便向上抛出. """
        if self._stream_sender is not None:
            # 确认 stream sender 被关闭了.
            self._stream_sender.__exit__(exc_type, exc_val, exc_tb)
        self._closed = True
        self._stream_sender = None
        self._stream_callback = None
        self._record_logos = None
        if exc_val is None:
            # 没事了, 直接结束.
            return None
        # cancel 交给外层处理.
        try:
            if isinstance(exc_type, asyncio.CancelledError):
                return None
            elif isinstance(exc_type, AbortAttentionError):
                self._logger.info("%s abort the attention on AbortAttentionError: %s", self._log_prefix, exc_val)
                # raise it
                return False
            else:
                self._logger.error("%s exit on unexpected error %s, raise abort", self._log_prefix, exc_val)
                # raise an abort error.
                raise AbortAttentionError(f"Logos exit on exception")
        finally:
            self._logger.info("%s finally close the logos writer", self._log_prefix)


class BaseActions(Actions):

    def __init__(
            self,
            *,
            attention: "BaseAttention"
    ):
        self._attention = attention
        self._iterated = False

    def __aiter__(self) -> AsyncIterator[Logos]:
        """实际上抽象为了屏蔽有并发问题的函数, 本质上还是用 attention 做统一的状态管理. """
        if self._iterated:
            raise RuntimeError("Actions already iterated")
        self._iterated = True
        return self

    async def __anext__(self) -> Logos:
        logos = await self._attention.wait_next_logos()
        if logos is None:
            raise StopAsyncIteration
        return logos

    def outcome(self, message: Message, observe: bool = False) -> None:
        self._attention.outcome(message, observe=observe)

    def fail(self, error: Exception) -> None:
        self._attention.abort(error)

    def flag(self, name: str) -> Flag:
        return self._attention.flag(name)


class BaseObservations(Observations):

    def __init__(
            self,
            *,
            attention: "BaseAttention",
            wait_next_observation: Callable[[], Coroutine[None, None, Observation]],
    ):
        self._attention = attention
        self._wait_next_observation = wait_next_observation
        self._iterated = False

    def __aiter__(self) -> AsyncIterator[Observation]:
        """抽象只是为了屏蔽有并发隐患的逻辑, 实际上仍然走同一个对象做状态管理. """
        if self._iterated:
            raise RuntimeError("Observations already iterated")
        self._iterated = True
        return self

    async def __anext__(self) -> Observation:
        observation = await self._wait_next_observation()
        if observation is None:
            raise StopAsyncIteration
        return observation

    async def send_logos(self, logos: Logos) -> None:
        async for delta in logos:
            await self._attention.send_logos_delta(delta)

    def send_nowait(self, delta: str) -> None:
        self._attention.send_logos_delta_nowait(delta)

    def observe(self, message: str) -> None:
        self._attention.outcome(Message.new().with_content(message), observe=True)

    def flag(self, name: str) -> Flag:
        return self._attention.flag(name)


class BaseAttention(Attention):
    """
    基础的 Attention 机制实现.
    只要这个机制通过了单元测试, 就能够把系统的复杂度都屏蔽到这套实现的内侧.
    """

    def __init__(
            self,
            *,
            impulse: Impulse,
            logger: LoggerItf | None = None,
            last_observation: Observation = None,
    ):
        self._impulse = impulse
        self._impulse_is_complete_event = ThreadSafeEvent()
        self._logger = logger or get_moss_logger()

        # 关键的 flags.
        self._aborted_event = ThreadSafeEvent()
        self._flags: dict[str, ThreadSafeEvent] = {}
        self._flag_lock = threading.Lock()
        sender, receiver = anyio.create_memory_object_stream[str]()
        self._logos_writer = sender
        self._logos_stream = receiver
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._exception: Exception | None = None
        self._task_groups: set[asyncio.Task] = set()

        # 当前 impulse 默认的提权效果.
        self._escalation: float = 1.2

        # 这三个值通过 update impulse 更新.
        self._initial_strength: float = 0.0
        self._strength_refreshed_at: float = 0.0
        self._strength_decay_time: float = 0.0

        # 防重入的 flag.
        self._is_iterating_observation: bool = False
        # observation
        self._observation_buffer: Observation = Observation(
            # 只有第一个 observation 才有资格使用.
            logos=self._impulse.on_logos_start
        )
        # inherit last observation buffer.
        if last_observation is not None:
            self._observation_buffer.parent_id = last_observation.id
            self._observation_buffer.logos = last_observation.logos
            self._observation_buffer.outcomes = last_observation.outcomes
            self._observation_buffer.stop_reason = last_observation.stop_reason

        self._has_new_observation_event = ThreadSafeEvent()
        self._set_new_observation_lock = threading.Lock()
        if self._impulse.complete:
            # 启动时 observation.
            self._has_new_observation_event.set()
        self._waiting_observation = False
        self._popped_any_observation = False
        self._logos_sender: MemoryObjectSendStream | None = None
        self._logos_receiver: MemoryObjectSendStream | None = None
        self._waiting_logos = False
        # 先约定一个致命的最大轮次, 实际运行看情况. 否则必须要做背压了.
        # 理论上 articulate 不可能消费比 interpreter 快.
        self._logos_receiver_queue: janus.Queue[MemoryObjectReceiveStream[str] | None] = janus.Queue(maxsize=10)
        self._observation_callbacks: list[Callable[[Observation], None]] = []
        self._context_funcs: dict[str, Callable[[], list[Message]]] = {}

        self._started: bool = False
        self._closed_event = ThreadSafeEvent()
        # update the impulse
        self._log_prefix = "?? 别忘记了."
        self._update_current_impulse(impulse)
        # attention 初始化逻辑预计应该到不了毫秒.

    def _update_current_impulse(self, impulse: Impulse) -> None:
        """更新当前持有的 impulse. """
        self._impulse = impulse
        self._initial_strength = impulse.strength * self._escalation
        self._strength_refreshed_at = time.time()
        self._strength_decay_time = self._impulse.strength_decay_seconds
        if self._strength_decay_time <= 0:
            # 不要让它为0.
            self._strength_decay_time = 1
        if impulse.complete:
            # 最后才设置.
            self._impulse_is_complete_event.set()

    async def _run_articulate_func(self, func: Callable[[Observations], Coroutine[None, None, None]]) -> None:
        observations = BaseObservations(
            attention=self,
            wait_next_observation=self._wait_next_observation,
        )
        try:
            await func(observations)
        except asyncio.CancelledError:
            raise
        except AbortAttentionError as e:
            self.abort(e)
        except Exception as e:
            self._logger.exception("%s run articulate func failed: %s", self._log_prefix, e)
            self.abort(e)
        finally:
            self._logger.info("%s run articulate func finished", self._log_prefix)

    async def _wait_next_logos(self) -> Logos | None:
        """屏蔽在抽象下, 不直接对外暴露的函数."""
        if self._aborted_event.is_set():
            return None
        # 返回值可能是 None.
        try:
            if self._logos_receiver_queue.async_q.empty() and self._waiting_observation:
                return None
            self._waiting_logos = True
            item = await self._logos_receiver_queue.async_q.get()
            # 返回值可能是 None.
            return item
        finally:
            self._waiting_logos = False

    async def _wait_next_observation(self) -> Observation | None:
        """屏蔽在抽象下, 不直接对外暴露的函数. """
        try:
            if self._aborted_event.is_set():
                return None
            if self._popped_any_observation and self._logos_sender is not None:
                # 提前完成流结束.
                self._logos_sender.close()

            # 确保在 abort 后这个事件一定会 set
            if not self._impulse.complete:
                # 第一个 impulse. 除非 on_impulse 支持新的未完成事件阻塞.
                await self._impulse_is_complete_event.wait()
                return self._pop_observation()
            elif self._has_new_observation_event.is_set():
                return self._pop_observation()
            elif self._waiting_logos:
                # 让外部退出. 也就是不会再有 observation 发送.
                self._logos_receiver_queue.sync_q.put_nowait(None)
                return None
            else:
                self._waiting_observation = True
                await self._has_new_observation_event.wait()
                return self._pop_observation()
        finally:
            self._waiting_observation = False

    def _pop_observation(self) -> Observation | None:
        # 在这里统一检查, 评估只有一个地方用了这个函数.
        if self._aborted_event.is_set():
            # 没有的话, 返回 None.
            return None
        pop = self._observation_buffer
        # 替换容器.
        with self._set_new_observation_lock:
            self._observation_buffer = Observation()
            self._has_new_observation_event.clear()

        # 只有 pop 时才添加.
        try:
            # 永远结合上下文返回.
            for name, context_func in list(self._context_funcs.items()):
                # 如果出现异常, 这里做个兜底.
                context_messages = context_func()
                pop.context[name] = context_messages
            # 初始化新的 logos stream 做准备.
            # 虽然设置了 max_buffer_size, 但实际上是并行消费的. 不太可能触发. 先保留一个值, 不处理异常, 调试时看是否会有异常.
            # 基本原理是, ctml interpreter 如果在运行时选择 append 类型, 在 compiled 完后会直接退出.
            # 使用它的 on_task_compiled() 回调可以注册独立的通讯, 让 task 运行时通知到 outcome, 而不需要依赖 interpreter 生命周期.
            sender, receiver = anyio.create_memory_object_stream[str](max_buffer_size=1000)
            if self._logos_sender is not None:
                # 旧的 sender 记得删除.
                self._logos_sender.close()

            self._logos_sender: MemoryObjectSendStream = sender
            # 其实不用这一行. 不过怕未来有变化.
            sender.__enter__()
            # 发送要输出的 logos. 只有 pop 新的 observation 才会配套发送.
            self._logos_receiver_queue.sync_q.put_nowait(receiver)
            # 发送基本讯息.
            if pop.logos.strip():
                sender.send_nowait(pop.logos)
            # 任何一个正常 pop 的都会标记 True.
            self._popped_any_observation = True
            if len(self._observation_callbacks) > 0:
                for callback in self._observation_callbacks:
                    callback(pop)
            return pop
        except Exception as e:
            # 暂时不考虑容错. 先跑起来看看会有什么异常.
            self._logger.exception("%s failed to create attention messages: %s", self._log_prefix, e)
            raise e

    def peek(self) -> Impulse:
        return self._impulse

    def is_aborted(self) -> bool:
        return self._aborted_event.is_set()

    async def wait_impulse(self) -> Impulse:
        await self._impulse_is_complete_event.wait()
        if self._aborted_event.is_set():
            raise AbortAttentionError("Attention is aborted")
        return self._impulse

    def flag(self, name: str) -> Flag:
        flag = self._flags.get(name)
        if flag is not None:
            return flag
        # 这里做个线程锁, 速度应该非常快. 实际上 asyncio 也会是同步调用. 用锁是为了解决未来多线程调度三循环的问题.
        with self._flag_lock:
            if name in self._flags:
                return self._flags[name]
            event = ThreadSafeEvent()
            self._flags[name] = event
            return event

    async def send_logos_delta(self, delta: str) -> None:
        # 做一个快速校验.
        if self.is_aborted():
            raise AbortAttentionError("Attention is aborted")
        # 添加给当前的 observation.
        self._observation_buffer.logos += delta
        if self._logos_sender is not None:
            # 发送物料.
            await self._logos_sender.send(delta)
        # 有活跃的信号输入.
        self._escalation_on_active()

    def send_logos_delta_nowait(self, delta: str) -> None:
        # 做一个快速校验.
        if self.is_aborted():
            raise AbortAttentionError("Attention is aborted")
        # 添加给当前的 observation.
        self._observation_buffer.logos += delta
        if self._logos_sender is not None:
            # 发送物料.
            self._logos_sender.send_nowait(delta)
        # 有活跃的信号输入.
        self._escalation_on_active()

    def wait_complete_impulse(self) -> asyncio.Future[Impulse]:
        self._check_running()
        if self._impulse.complete:
            future = self._event_loop.create_future()
            future.set_result(self._impulse)
            return future
        # add task for sake of close after
        task = self._event_loop.create_task(self.wait_impulse())
        self._add_task(task)
        return task

    def on_observation(self, callback: Callable[[Observation], None]) -> None:
        """register observation callback"""
        self._observation_callbacks.append(callback)

    def with_context_func(self, context_name: str, context_func: Callable[[], list[Message]]) -> Self:
        # 直接覆盖存在的 context func.
        self._context_funcs[context_name] = context_func

    def outcome(self, *outcomes: Message, observe: bool = False) -> None:
        self._observation_buffer.outcomes.extend(outcomes)
        if observe:
            with self._set_new_observation_lock:
                self._has_new_observation_event.set()
            # 不会在 observe 设置时, 清空当前的 logos.
            # 因为 logos 只有连续, 没有语法错误时才是合法的.
            # 当 observe 发生时, ctml interpreter 会直接退出.
            # 同时下一个 interpreter 启动是, incomplete_tasks 会继承给它.
            # 所以在新的观察启动的时候, 旧的运行还不会停止. 要打断旧的运行, 需要显式调用 interrupt.
        return None

    async def wait_aborted(self) -> None:
        # 单纯阻塞到失效.
        await self._aborted_event.wait()

    def is_started(self) -> bool:
        return self._started

    def stop_at(self) -> Observation:
        return self._observation_buffer

    async def wait_closed(self) -> None:
        await self._aborted_event.wait()

    def _escalation_on_active(self) -> None:
        # 先简单用时间刷新来做提权. 方便 AI 大神未来帮我改.
        self._strength_refreshed_at = time.time()

    def _current_strength(self) -> int:
        # by gemini 3.0
        now = time.time()
        elapsed = now - self._strength_refreshed_at

        # 基础衰减因子：未完成的脉冲衰减更快（急迫感）
        decay_rate = 1.0 if self._impulse.complete else 2.5

        # 使用指数衰减模拟生物神经突触信号
        remaining_ratio = math.exp(- (elapsed / self._strength_decay_time) * decay_rate)

        current = self._initial_strength * remaining_ratio
        return int(max(current, 0))

    def on_challenge(self, challenger: Impulse) -> bool:
        if challenger.id == self._impulse.id:
            self._update_current_impulse(challenger)
            return False
        # priority is superior
        if challenger.priority == Priority.FATAL or challenger.priority > self._impulse.priority:
            return True
        elif challenger.priority < self._impulse.priority:
            return False
        challenger_strength = challenger.strength
        if challenger.source == self._impulse.source:
            challenger_strength = challenger_strength * self._escalation
        current_strength = self._current_strength()
        return current_strength < challenger_strength

    def create_task(self, cor: Coroutine) -> asyncio.Future:
        self._check_running()
        task = self._event_loop.create_task(cor)
        self._add_task(task)
        return task

    def _add_task(self, task: asyncio.Task) -> None:
        if self._aborted_event.is_set():
            task.cancel("aborted")
        else:
            self._task_groups.add(task)
            task.add_done_callback(self._on_task_done_callback)

    def _on_task_done_callback(self, task: asyncio.Task) -> None:
        if not task.done():
            return
        self._task_groups.discard(task)
        if self._aborted_event.is_set():
            return
        if task.cancelled():
            return
        exception = task.exception()
        # 任何异常都会导致全体退出.
        if exception is None:
            return
        elif isinstance(exception, BaseException):
            self.abort(str(exception))
            return
        elif isinstance(exception, Exception):
            self.abort(exception)
            return

    def is_closed(self) -> bool:
        return self._aborted_event.is_set()

    def exception(self) -> Exception | None:
        return self._exception

    def abort(self, error: str | Exception | None) -> None:
        if self._aborted_event.is_set():
            return None
        if isinstance(error, str):
            cancel_error = asyncio.CancelledError(error)
        elif isinstance(error, Exception):
            cancel_error = asyncio.CancelledError(f"aborted on: {error}")
        else:
            cancel_error = None
        self._exception = cancel_error
        # stop all the tasks immediately
        tasks = self._task_groups.copy()
        for task in tasks:
            task.cancel("attention aborted")
        self._aborted_event.set()
        if cancel_error:
            self._observation_buffer.stop_reason = str(cancel_error)
        # 可能阻塞的事件都调用一次.
        self._impulse_is_complete_event.set()
        self._logos_receiver_queue.sync_q.put_nowait(None)
        return None

    def _check_running(self) -> None:
        if not self._started or self._aborted_event.is_set() or self._event_loop is None:
            raise asyncio.CancelledError("Attention is not running")

    async def _inner_arbiter(self) -> None:
        """
        在自己内部做自己是否应该结束的仲裁.
        收到挑战, 第一时间返回属于条件反射.
        实际上仍然可以有一个周期去内省.
        """
        ttl = self._strength_decay_time
        await asyncio.sleep(ttl)
        if self._current_strength() <= 0:
            self.abort(asyncio.TimeoutError("ttl timed out"))
            return None
        while not self._aborted_event.is_set():
            if self._current_strength() <= 0:
                self.abort(asyncio.TimeoutError("ttl timed out"))
                return None
            # tick every 1 second
            await asyncio.sleep(1)
        return None

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        # 启动自身的超时检查.
        _ = self.create_task(self._inner_arbiter())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关键是哪些异常是需要对外抛出的.
        """
        if self._closed_event.is_set():
            return None
        try:
            intercept = False
            self._aborted_event.set()
            self._event_loop = None
            # clear all the tasks
            tasks = self._task_groups.copy()
            self._task_groups.clear()
            wait_all = []
            for task in tasks:
                if not task.done():
                    task.cancel("aborted")
                    wait_all.append(task)
            if len(wait_all) > 0:
                r = await asyncio.gather(*wait_all, return_exceptions=True)
                for e in r:
                    if not isinstance(e, Exception):
                        continue
                    elif isinstance(e, asyncio.CancelledError):
                        continue
                    elif isinstance(e, asyncio.TimeoutError):
                        continue
                    else:
                        self._logger.error("attention cancel task failed on exception %s", e)

            if exc_val is not None:
                if isinstance(exc_val, BaseException) or isinstance(exc_val, FatalError):
                    self._logger.error("attention aborted on fatal exception %s", exc_val)
                    return None
                elif self._exception is exc_val:
                    return True
                elif isinstance(exc_val, asyncio.TimeoutError):
                    # box stop here
                    return True
                elif isinstance(exc_val, asyncio.CancelledError):
                    # always raise cancel error
                    return None
                else:
                    self._logger.error("attention aborted on unexpected exception %s", exc_val)
                    # intercept any exception
                    return True
            return intercept
        finally:
            # 清除一些容易互相持有的逻辑.
            self._context_funcs.clear()
            self._observation_callbacks.clear()
            # 两个确保能够退出的标记.
            self._aborted_event.set()
            self._closed_event.set()
