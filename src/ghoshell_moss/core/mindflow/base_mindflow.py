from abc import abstractmethod, ABC
from typing import AsyncGenerator, AsyncIterator, Callable
from typing_extensions import Self

import janus

from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, Attention, Impulse, Nucleus, Signal, SignalName, Priority, ImpulseAbsorbed,
    Reaction, ChallengeVerdict, MindflowHook,
    ChallengeMode,
)
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, Channel
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.message import Message
from .base_attention import BaseAttention
import asyncio
import contextlib
import threading

_SignalName = str
_NucleusName = str

__all__ = [
    'BaseMindflow',
    'AbsMindflow',
    'new_default_mindflow',
]


class _DirectImpulseNucleus(Nucleus):
    """内置 cache nucleus, 服务 ``AbsMindflow.add_impulse`` 公开调试入口.

    把"直接注入的 impulse"包装成标准 nucleus 的 peek/pop 行为, 让 rank/challenge 协议
    在 nucleus-path 和 direct-path 上共用同一套流程, 不引入旁路.

    职责保持极简:
    - 不监听 signal
    - 不维护队列, 只保留最后一次 ``set_impulse`` 的值 (last-impulse cache)
    - suppress / pop 直接清空 (调用方语义已决定它的命运)
    """

    NAME = "_direct"

    def __init__(self):
        self._impulse: Impulse | None = None
        self._running = False
        self._notify: Callable[[Impulse], None] | None = None

    def set_impulse(self, impulse: Impulse) -> None:
        """缓存一个 impulse 并通知 mindflow. impulse.source 被锚定为 NAME."""
        impulse.source = self.NAME
        self._impulse = impulse
        if self._notify is not None:
            self._notify(impulse)

    def name(self) -> str:
        return self.NAME

    def description(self) -> str:
        return "direct add_impulse cache (internal)"

    def status(self) -> str:
        return ""

    def signals(self) -> list[SignalName]:
        return []

    def clear(self) -> None:
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        return None

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            impulse_notify: Callable[[Impulse], None],
    ) -> None:
        self._notify = impulse_notify

    def suppress(self, suppress_by: Impulse) -> None:
        self._impulse = None

    def pop_impulse(self, impulse: Impulse) -> None:
        if self._impulse is impulse:
            self._impulse = None

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse is None:
            return None
        if no_stale and self._impulse.is_stale():
            self._impulse = None
            return None
        return self._impulse

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> "_DirectImpulseNucleus":
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        return None


class MindflowHookGroup(MindflowHook):

    def __init__(self, logger: LoggerItf | None = None):
        self._hooks: dict[str, MindflowHook] = {}
        self._has_any: bool = False
        self._logger = logger or get_moss_logger()
        self._hook_lock = threading.Lock()

    def name(self) -> str:
        return 'MindflowHookGroup'

    def add_hook(self, hook: MindflowHook):
        with self._hook_lock:
            self._hooks[hook.name()] = hook
        self._has_any = True

    def remove_hook(self, hook: str):
        with self._hook_lock:
            if hook in self._hooks:
                del self._hooks[hook]

    def description(self) -> str:
        return 'group of mindflow hooks'

    def on_impulse_challenged(
            self,
            challenger: Impulse,  # challenger — 发起挑战的 Impulse
            defender: Impulse | None,  # defender   — 当前占据注意力的 Impulse，None 表示无当前 attention
            verdict: ChallengeVerdict,  # verdict    — 仲裁结果
    ) -> None:
        if not self._has_any:
            return
        # todo: 考虑用 functools.wrap 方式包装子 hook.
        for name, hook in self._hooks.items():
            try:
                hook.on_impulse_challenged(challenger, defender, verdict)
            except Exception as e:
                self._logger.error(
                    "MindflowHook %s failed on on_impulse_challenged with exception %r",
                    name, e
                )

    def on_error(self, error: Exception) -> None:
        if not self._has_any:
            return
        for name, hook in self._hooks.items():
            try:
                hook.on_error(error)
            except Exception as e:
                self._logger.error(
                    "MindflowHook %s failed on on_impulse_challenged with exception %r",
                    name, e
                )


class AbsMindflow(Mindflow, ABC):
    """
    Mindflow 抽象基类: 信号路由, impulse 排队, attention 调度.

    _build_attention() 留给子类实现仲裁策略.
    """

    def __init__(
            self,
            *nuclei: Nucleus,
            logger: LoggerItf | None = None,
            raise_nucleus_start_error: bool = True,
    ):
        # Nucleus 可能只是一个接口. 内部有别的技术实现.
        self._faculties: dict[_NucleusName, Nucleus] = {}
        self._input_signal_name_routes: dict[_SignalName, dict[_NucleusName, Nucleus]] = {}
        self._logger = logger or get_moss_logger()
        self._log_prefix = "<MindflowBus>"
        self._current_attention: Attention | None = None
        # 这是内部循环使用的队列.
        self._pop_new_attention_queue: janus.Queue[Attention | None] = janus.Queue(maxsize=1)
        self._starting = False
        self._started_event = ThreadSafeEvent()
        self._closed = False
        self._paused = False
        self._unpaused_event = ThreadSafeEvent()
        self._unpaused_event.set()
        self._looping_attention = False
        # 设置线程安全的优先级队列, 用来卸载信号量到本地循环, 避免线程安全上的震荡.
        self._signal_low_queue: janus.PriorityQueue[tuple[int, int, Signal]] = self._new_signal_queue()
        self._signal_high_queue: janus.PriorityQueue[tuple[int, int, Signal]] = self._new_signal_queue()
        self._signal_count: int = 0
        self._has_impulse_event = ThreadSafeEvent()
        self._set_impulse_lock = asyncio.Lock()
        self._buffered_messages = []
        self._signal_priority_bar = Priority.BACKGROUND
        self._impulse_priority_bar = Priority.BACKGROUND

        # 内部循环检测是否有新的 impulse.
        self._consuming_signal_task: asyncio.Task | None = None
        self._consuming_impulse_task: asyncio.Task | None = None
        # 是否对启动异常容错.
        self._raise_nucleus_start_error = raise_nucleus_start_error
        # 内置 direct nucleus, 服务 public add_impulse 入口. 必须先于用户 nuclei 注册.
        self._direct_nucleus = _DirectImpulseNucleus()
        self.with_nucleus(self._direct_nucleus)
        for nucleus in nuclei:
            self.with_nucleus(nucleus)
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._hooks_group: MindflowHookGroup = MindflowHookGroup(self._logger)
        self._buffered_messages_lock = threading.Lock()

    def get_buffered(self, pop: bool = True) -> list[Message]:
        if pop:
            return self._pop_buffer()
        with self._buffered_messages_lock:
            messages = self._buffered_messages.copy()
        return messages

    def buffer(self, messages: list[Message]) -> None:
        with self._buffered_messages_lock:
            self._buffered_messages.extend(messages)

    @staticmethod
    def _new_signal_queue() -> janus.PriorityQueue[tuple[int, int, Signal]]:
        return janus.PriorityQueue(maxsize=100)

    def is_running(self) -> bool:
        return self._started_event.is_set() and not self._closed

    def faculties(self) -> dict[str, Nucleus]:
        return self._faculties

    def with_hook(self, hook: MindflowHook) -> Self:
        self._hooks_group.add_hook(hook)
        return self

    def remove_hook(self, hook: str | MindflowHook) -> None:
        if isinstance(hook, MindflowHook):
            hook = hook.name
        self._hooks_group.remove_hook(hook)

    async def wait_started(self) -> None:
        await self._started_event.wait()

    def wait_started_sync(self, timeout: float | None = None) -> bool:
        return self._started_event.wait_sync(timeout)

    def with_nucleus(self, nucleus: Nucleus, override: bool = False) -> None:
        self._bind_nucleus_with_bus(nucleus, override)

        channel = self.as_channel()
        # 注册 nucleus 的 channel.
        if channel and isinstance(channel, MutableChannel):
            if sub_channel := nucleus.as_channel():
                channel.import_channels(sub_channel)

    def _bind_nucleus_with_bus(self, nucleus: Nucleus, override: bool = False) -> None:
        if self._started_event.is_set():
            raise RuntimeError(f"Mindflow only with nucleus before started, use add_nucleus instead")

        # 注册运行总线. 只能在启动前用.
        _name = nucleus.name()
        if not override and _name in self._faculties:
            raise NameError(f"nucleus {_name} already exists")

        nucleus.with_bus(self.add_signal, self._nucleus_has_impulse)
        self._faculties[_name] = nucleus

    def _register_nucleus_to_signal_routes(self, nucleus: Nucleus) -> None:
        for listening in nucleus.signals():
            if listening not in self._input_signal_name_routes:
                self._input_signal_name_routes[listening] = {}
            # 使用 dict 注册防止重复.
            # always override
            self._input_signal_name_routes[listening][nucleus.name()] = nucleus

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Mindflow is not running.")

    async def add_nucleus(self, nucleus: Nucleus, override: bool = False) -> Self:
        self._check_running()
        if not override and self._has_nucleus(nucleus.name()):
            raise NameError(f"nucleus {nucleus.name()} already exists")
        # 启动 nucleus 并且加入.
        if not nucleus.is_running():
            await nucleus.__aenter__()
        if channel := self.as_channel():
            if isinstance(channel, PrimeChannel):
                if sub_channel := nucleus.as_channel():
                    #  添加channel.
                    channel.add_virtual_channel(sub_channel)

        self._bind_nucleus_with_bus(nucleus, override=override)
        self._register_nucleus_to_signal_routes(nucleus)

    def _has_nucleus(self, name: str) -> bool:
        return name in self._faculties

    def add_signal(self, signal: Signal) -> None:
        """接受signal"""
        # 这个函数很可能是接受跨线程的回调, 比如 zenoh session 的回调.
        # 所以它的核心目标是卸载 signal 到当前线程 (loop).
        if not self.is_running():
            self._logger.error("%s on signal but not running: %r", self._log_prefix, signal)
            signal.__state__ = 'ignored'
            return None
        elif self._paused:
            self._logger.warning("%s ignore signal cause paused: %r", self._log_prefix, signal)
            signal.__state__ = 'ignored'
            return None
        elif signal.is_stale():
            self._logger.debug("%s ignore stale signal: %s", self._log_prefix, signal.id)
            signal.__state__ = 'ignored'
            return None
        elif signal.priority < self._signal_priority_bar:
            self._logger.debug(
                "%s ignore signal lower than priority %d: %s",
                self._log_prefix, self._signal_priority_bar, signal.id,
            )
            return None
        signal.max_hop -= 1
        if signal.max_hop < 0:
            self._logger.error("%s ignore signal max_hop negative: %r", self._log_prefix, signal)
            signal.__state__ = 'ignored'
            return None

        self._signal_count += 1
        priority_count = signal.priority_strength()
        try:
            if self._signal_low_queue.sync_q.full() and signal.priority >= Priority.CRITICAL:
                # 特殊的信号, 丢到高优队列. 不抛弃不放弃.
                self._signal_high_queue.sync_q.put_nowait((-priority_count, self._signal_count, signal))
            else:
                self._signal_low_queue.sync_q.put_nowait((-priority_count, self._signal_count, signal))
            signal.__state__ = 'pending'
        except janus.SyncQueueFull:
            # 直接 ignore 掉. 反应不过来了.
            self._logger.debug("%s ignore signal queue full: %r", self._log_prefix, signal)
            return None
        except janus.SyncQueueShutDown:
            self._logger.debug("%s ignore signal queue shutdown: %r", self._log_prefix, signal)

    async def _on_signal_consuming_loop(self):
        """信号消费队列, 将 signal 卸载到当前循环中. """
        while self.is_running():
            # 队列是单一消费者, 所以可以检查 empty.
            try:
                if not self._signal_high_queue.async_q.empty():
                    p, count, item = self._signal_high_queue.async_q.get_nowait()
                else:
                    # 如果高优队列不为空, 一定是低优队列满了. 所以低优队列阻塞时永远不会阻塞高优队列.
                    p, count, item = await self._signal_low_queue.async_q.get()
                # 丢弃过期对象.
                if self._paused or item.is_stale():
                    # 丢弃过期的信号量. 这个日志要不要记录呢?
                    self._logger.debug("%s ignore stale signal: %s", self._log_prefix, item.id)
                    item.__state__ = 'ignored'
                    continue
                await self._dispatch_signal(item)
            except janus.AsyncQueueShutDown:
                continue

    def set_signal_priority_bar(self, priority: Priority) -> None:
        self._signal_priority_bar = priority

    def set_impulse_priority_bar(self, priority: Priority) -> None:
        self._impulse_priority_bar = priority

    async def _dispatch_signal(self, signal: Signal) -> None:
        try:
            if signal.priority < self._signal_priority_bar:
                self._logger.debug(
                    "%s ignore signal %s priority %d lower than bar %d",
                    self._log_prefix, signal.id, signal.priority, self._signal_priority_bar,
                )
            name = signal.name
            broadcasted = 0
            if len(self._faculties) == 0:
                signal.__state__ = 'ignored'
                return None
            if name not in self._input_signal_name_routes:
                # 丢弃不监听的 signal.
                signal.__state__ = 'ignored'
                return None
            dispatched = False
            for n in self._input_signal_name_routes[name].values():
                # 触发分配.
                n.add_signal(signal)
                dispatched = True
            signal.__state__ = 'dispatched' if dispatched else 'ignored'
            self._logger.debug("%s receive signal and send to %d nuclei", self._log_prefix, broadcasted)
            return None
        except asyncio.CancelledError:
            # 只有 cancel 才 raise.
            raise
        except Exception as e:
            # 拦截所有的异常, 不要影响外部循环.
            self._logger.error("%s dispatch signal error on %r: %s", self._log_prefix, signal, e)

    def add_impulse(self, impulse: Impulse) -> None:
        """公开调试入口, 直接注入一个 impulse 走标准 rank/challenge 流程.

        实现上路由到内置 ``_DirectImpulseNucleus`` 缓存, rank 时与 nuclei impulse
        混合排序, 仲裁路径与 nucleus 产出完全一致. impulse.source 会被锚定为
        ``_DirectImpulseNucleus.NAME``.

        典型用法: 协议级集成测试, 绕开 nucleus 信号链直接验证 ImpulsePrimitive
        组合的仲裁行为.
        """
        if not self.is_running() or self._paused:
            return None
        if impulse.is_stale():
            return None
        self._direct_nucleus.set_impulse(impulse)
        return None

    def _nucleus_has_impulse(self, impulse: Impulse) -> None:
        """Bus 回调, nuclei 产出 impulse 后调用. 仅标记 ``_has_impulse_event``,
        真正的 impulse 由 rank 循环通过 ``nucleus.peek()`` 拉取.

        Note: 注意 ``on_signal / _nucleus_has_impulse`` 作为总线提供给 Nucleus 时,
        要防止信号成环无限传播. 暂无系统机制百分之百预防.
        """
        if self._paused:
            self._logger.info("%s drop impulse cause paused: %r", self._log_prefix, impulse)
            return None
        elif not self.is_running():
            self._logger.error("%s drop impulse cause not running: %r", self._log_prefix, impulse)
            return None
        # 仅仅标记一个信号.
        self._has_impulse_event.set()
        return None

    async def _on_impulse_consuming_loop(self):
        while self.is_running():
            if self._paused:
                # 阻塞等到 unpause.
                await self._unpaused_event.wait()
            try:
                # 创建一个搏动的循环, 用来做impulse 检查.
                await asyncio.wait_for(self._has_impulse_event.wait(), 0.5)
            except asyncio.TimeoutError:
                continue
            self._has_impulse_event.clear()
            # 进行一次排队.
            try:
                impulse = self._rank_nuclei()
                # 使用 await, 方便感知 cancel?
                if impulse is None:
                    # 以 rank 的瞬间为准. 如果出现极端情况, rank完的瞬间又有新的 impulse, 那也只能等下一轮.
                    continue
                else:
                    await self._challenge_attention(impulse)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._logger.error("%s impulse consuming loop error: %s", self._log_prefix, e)

    def _suppress_impulse(self, impulse: Impulse, by: Impulse) -> None:
        """supress 指定的 impulse"""
        nucleus = self._faculties.get(impulse.source, None)
        if nucleus is not None:
            nucleus.suppress(by)

    def _pop_impulse(self, impulse: Impulse) -> None:
        """通知 nucleus 被 pop 了. """
        nucleus = self._faculties.get(impulse.source, None)
        if nucleus is not None:
            # 应该要将 impulse 给踢掉.
            if impulse is nucleus.peek():
                nucleus.pop_impulse(impulse)

    async def _challenge_attention(self, impulse: Impulse) -> None:
        """原子操作."""
        try:
            if impulse.is_stale():
                self._pop_impulse(impulse)
                return None
            # attention 或者.
            if self._current_attention and not self._current_attention.is_aborted():
                defender = self._current_attention.draw_from()
                # Fatal always prevails
                if impulse.priority == Priority.FATAL.value:
                    done = True
                elif impulse.priority == Priority.BACKGROUND.value:
                    done = False
                else:
                    done = self._current_attention.challenge(impulse)

                # 如果被吸收了.
                if done is ImpulseAbsorbed:
                    # 同 ID 更新 complete, 不抢占.
                    self._pop_impulse(impulse)
                    self._fire_challenge(impulse, defender, 'absorbed')
                    return None
                elif done:
                    if impulse.mode == ChallengeMode.silent.value:
                        self._pop_impulse(impulse)
                        self.buffer(impulse.messages)
                        self._fire_challenge(impulse, defender, 'buffered')
                        return None

                    # 抢占成功, 创建新 Attention.
                    await self._create_attention_from_impulse(impulse)
                    self._fire_challenge(impulse, defender, 'preempted')
                else:
                    # notify 挑战失败, 将 notify 的消息入队.
                    if impulse.mode == ChallengeMode.notify.value:
                        self._pop_impulse(impulse)
                        self.buffer(impulse.messages)
                        self._fire_challenge(impulse, defender, 'buffered')
                        return None
                    # 被压制.
                    self._suppress_impulse(impulse, defender)
                    self._fire_challenge(impulse, defender, 'suppressed')
                return None
            else:
                if impulse.mode == ChallengeMode.silent.value:
                    # silent 模式不创建注意力, 只做 buffer.
                    self._pop_impulse(impulse)
                    self.buffer(impulse.messages)
                    self._fire_challenge(impulse, None, 'buffered')
                    return None

                # 创建一个新的 impulse.
                await self._create_attention_from_impulse(impulse)
                self._fire_challenge(impulse, None, 'initial')
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 只记录异常, 不要抛出终止. 保证循环运行.
            self._logger.exception(
                "%s failed to challenge attention with impulse %r: %s",
                self._log_prefix, impulse, e,
            )

    def _fire_challenge(
            self,
            challenger: Impulse,
            defender: Impulse | None,
            verdict: ChallengeVerdict,
    ) -> None:
        self._hooks_group.on_impulse_challenged(challenger, defender, verdict)

    def attention(self) -> Attention | None:
        if self._current_attention is None:
            return None
        elif self._current_attention.is_aborted():
            return None
        return self._current_attention

    def is_quiet(self) -> bool:
        """有时候要检查一下"""
        if not self.is_running():
            return True
        elif self._current_attention is not None and not self._current_attention.is_aborted():
            return False
        for nucleus in self._faculties.values():
            impulse = nucleus.peek()
            if impulse is not None:
                return False
        return True

    def set_impulse(self, impulse: Impulse) -> None:
        if impulse.is_stale():
            return None
        if not self.is_running():
            return None
        self._event_loop.create_task(self._create_attention_from_impulse(impulse))
        return None

    async def _create_attention_from_impulse(self, impulse: Impulse) -> None:
        """直接用 impulse 创建 attention"""
        self._pop_impulse(impulse)
        async with self._set_impulse_lock:
            if impulse.is_stale():
                # 仍然做一次校验.
                return None
            if self._current_attention is not None:
                if not self._current_attention.is_aborted():
                    # 在这里 abort.
                    self._current_attention.abort("interrupted")
                # 在 last outcome 里做了判断, 如果没有 started 过, 则会返回原始的对象.
                inherit_outcome = self._current_attention.last_outcome()
            else:
                inherit_outcome = Reaction()
            attention = self._build_attention(impulse, inherit_outcome)
            self._set_attention(attention)
            return None

    def _pop_buffer(self) -> list[Message]:
        with self._buffered_messages_lock:
            messages = self._buffered_messages.copy()
            self._buffered_messages.clear()
        return messages

    @abstractmethod
    def _build_attention(self, impulse: Impulse, inherit_outcome: Reaction) -> Attention:
        """子类实现: 用指定的仲裁策略构建 Attention 实例."""
        pass

    @abstractmethod
    def as_channel(self) -> Channel | None:
        """强调子类要重新实现 channel 逻辑. """
        pass

    def _set_attention(self, attention: Attention) -> None:
        # 这个函数只在 set impulse 处可以被调用.
        # 考虑到未来 set attention 可能不止一个地方调用 (比如命令行的行为), 所以加一个 set.
        if not self.is_running():
            self._logger.warning("%s set attention but not running: %r", self._log_prefix, attention)
            attention.abort("not running")
            return None
        elif self._paused:
            # paused 仍然可以设置. 这是系统指令.
            pass
        # 系统指令, 立刻生效.
        if self._current_attention is not None and not self._current_attention.is_aborted():
            # 多做一次 abort 检查, 用来做容错.
            self._current_attention.abort("interrupted")
        self._current_attention = attention
        # 注册 mindflow 自身的 context message 函数.
        self._current_attention.with_percepts_func("MindflowBuffer", self._pop_buffer)
        self._current_attention.with_perspective_func("Mindflow", self.perspective)
        # 这个队列里的其实都是上一个 current attention.
        try:
            while not self._pop_new_attention_queue.sync_q.empty():
                # maxsize 为 1 的队列.
                attention = self._pop_new_attention_queue.sync_q.get_nowait()
            self._pop_new_attention_queue.sync_q.put_nowait(self._current_attention)

        except janus.AsyncQueueShutDown:
            return None
        # 新 attention 入队.
        self._logger.info("%s set attention %r", self._log_prefix, attention)
        return None

    def _rank_nuclei(self, best_impulse: Impulse = None) -> Impulse | None:
        best_impulse = best_impulse
        best_n = None
        best_p = 0 if best_impulse is None else best_impulse.priority_strength()
        losers: list[Nucleus] = []
        for nucleus in self._faculties.values():
            impulse = nucleus.peek()
            # 是否 impulse 也要做一个过期?
            if impulse is None:
                continue
            elif impulse.is_stale():
                # pop stale impulse for the nucleus
                nucleus.pop_impulse(impulse)
                self._logger.info("%s pop stale impulse %r", self._log_prefix, impulse)
                continue
            elif impulse.priority < self._impulse_priority_bar:
                # 低于优先级的直接忽视.
                continue
            # 加一行代码防蠢.
            impulse.source = nucleus.name()
            impulse_priority_strength = impulse.priority_strength()
            if best_impulse is None:
                best_impulse = impulse
                best_n = nucleus
                best_p = impulse_priority_strength
                continue
            elif best_n and impulse_priority_strength > best_p:
                best_impulse = impulse
                losers.append(best_n)
                best_n = nucleus
                best_p = impulse_priority_strength
                continue
            else:
                losers.append(nucleus)
                continue
        if best_impulse and len(losers) > 0:
            for nucleus in losers:
                # 在这里通知完 suppress.
                nucleus.suppress(best_impulse)
        return best_impulse

    def pause(self, toggle: bool) -> None:
        if not self.is_running():
            return
        self._paused = toggle
        if toggle:
            if self._current_attention is not None:
                # 通过这种方式 stop the attention.
                self._current_attention.abort('paused')
            self._unpaused_event.clear()
            self._clear()
        else:
            self._unpaused_event.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unpaused_event.set()
        self._clear()
        # 用来通知退出.
        if not self._pop_new_attention_queue.sync_q.closed:
            self._pop_new_attention_queue.shutdown(immediate=True)

    def clear(self) -> None:
        if not self.is_running():
            return
        self._clear()

    def _clear(self) -> None:
        # 其实这两个通常是同一个. 不排除在队列中.
        if self._current_attention is not None and not self._current_attention.is_aborted():
            self._current_attention.abort('closed')

        _signal_low_queue = self._signal_low_queue
        _signal_low_queue.shutdown(immediate=True)
        self._signal_low_queue = self._new_signal_queue()
        _signal_high_queue = self._signal_high_queue
        _signal_high_queue.shutdown(immediate=True)
        self._signal_high_queue = self._new_signal_queue()
        for nucleus in self._faculties.values():
            # 清空所有的状态.
            nucleus.clear()
        self._has_impulse_event.clear()
        while not self._pop_new_attention_queue.sync_q.empty():
            self._pop_new_attention_queue.sync_q.get_nowait()

    def perspective(self) -> list[Message]:
        """
        返回 Mindflow 的瞬时状态图谱。
        通过简单的列表描述，让模型快速评估当前各 Nucleus 的优先级与待处理任务压力。
        """
        context_lines = []
        for name, nucleus in self._faculties.items():
            if not nucleus.is_running():
                continue

            try:
                status = nucleus.status()
                description = nucleus.description()

                # 只有当 nucleus 有明确状态告知时才加入，保持上下文纯净
                if status:
                    # 格式化建议："[Name] (Desc): Status"
                    # 这种格式在 Prompt 中极易被模型 parse 出来
                    line = f"- [{name}] {description}: {status}"
                    context_lines.append(line)
            except Exception as e:
                self._logger.error("%s get context message from nucleus %s failed: %s", self._log_prefix, name, e)
                continue

        if not context_lines:
            return []

        # 简单清晰的描述块，不引入复杂 XML，直接用纯文本提示组件当前焦点
        content_str = "Current Mindflow State:\n" + "\n".join(context_lines)

        return [Message.new(tag="mindflow").with_content(content_str)]

    def loop(self) -> AsyncIterator[Attention]:
        return self._loop_attention()

    async def _loop_attention(self) -> AsyncGenerator[Attention, None]:
        """需要实现一个特别稳定的流程."""
        if self._looping_attention:
            raise RuntimeError('looping attention already running')
        self._looping_attention = True
        try:
            last_popped_attention = None
            while self.is_running():
                self._looping_attention = True
                try:
                    if last_popped_attention is not None and not last_popped_attention.is_aborted():
                        # 阻塞等到下一帧运行结束.
                        await last_popped_attention.wait_closed()
                        # 不要再次进入这里.
                        last_popped_attention = None
                    # 如果进入等待的瞬间没有任何 attention, 最常见的就是一大堆的 Impulse 被压抑住了.
                    # 而被压抑住的 attention 结束时, 反而没有新的 impulse 进入.
                    if self._current_attention is None or self._current_attention.is_aborted():
                        if impulse := self._rank_nuclei():
                            # 提醒一下有事件.
                            self._has_impulse_event.set()
                    # 尝试尽快拿到最新的.
                    try:
                        _attention = await asyncio.wait_for(self._pop_new_attention_queue.async_q.get(), 1)
                    except asyncio.TimeoutError:
                        continue
                    except janus.AsyncQueueShutDown:
                        return

                    if _attention is None:
                        # 拿到毒丸, 退出循环.
                        # 当 mindflow 显式关闭时, 一定要发送毒丸.
                        return
                    if _attention.is_aborted():
                        # 拿到的一瞬间已经关闭了.
                        continue
                    last_popped_attention = _attention
                    yield _attention
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self._logger.error(
                        "%s loop attention failed on exception: %r", self._log_prefix, e
                    )
                    self._hooks_group.on_error(e)
        finally:
            self._looping_attention = False

    @contextlib.asynccontextmanager
    async def _make_sure_attention_cleared(self):
        """确保在线的 attention 都被退出了. """
        try:
            yield
        finally:
            current_attention = None
            if self._current_attention is not None and not self._current_attention.is_aborted():
                self._current_attention.abort('mindflow closed')
                # 稍稍等待一下退出.
                current_attention = self._current_attention
            if current_attention is not None:
                await current_attention.wait_closed()
            if not self._pop_new_attention_queue.sync_q.closed:
                self._pop_new_attention_queue.shutdown(immediate=True)

    @contextlib.asynccontextmanager
    async def _signal_consuming_task_ctx_manager(self):
        try:
            self._consuming_signal_task = asyncio.create_task(self._on_signal_consuming_loop())
            yield
        finally:
            if self._consuming_signal_task and not self._consuming_signal_task.done():
                self._consuming_signal_task.cancel()
                try:
                    await self._consuming_signal_task
                except asyncio.CancelledError:
                    pass
                self._consuming_signal_task = None

    @contextlib.asynccontextmanager
    async def _impulse_consuming_task_ctx_manager(self):
        try:
            self._consuming_impulse_task = asyncio.create_task(self._on_impulse_consuming_loop())
            yield
        finally:
            if self._consuming_impulse_task and not self._consuming_impulse_task.done():
                self._consuming_impulse_task.cancel()
                try:
                    await self._consuming_impulse_task
                except asyncio.CancelledError:
                    pass
                self._consuming_impulse_task = None

    @contextlib.asynccontextmanager
    async def _faculties_lifecycle_ctx_manager(self):
        nuclei = list(self._faculties.values())
        result = await asyncio.gather(*[n.__aenter__() for n in nuclei if not n.is_running()], return_exceptions=True)
        idx = 0
        for r in result:
            nucleus = nuclei[idx]
            if isinstance(r, Exception):
                self._logger.error("%s failed to start nucleus %r: %s", self._log_prefix, nucleus, r)
                if self._raise_nucleus_start_error:
                    # 严格模式下启动不做任何容错. 仅仅作为一个保留开发点. 默认是抛出异常.
                    raise r
            else:
                # 正式注册监听.
                self._register_nucleus_to_signal_routes(nucleus)

            idx += 1
        try:
            yield
        finally:
            faculties = list(self._faculties.values())
            self._faculties.clear()
            close_all = []
            for nucleus in faculties:
                close_all.append(nucleus.__aexit__(None, None, None))
            result = await asyncio.gather(*close_all, return_exceptions=True)
            idx = 0
            for r in result:
                if isinstance(r, Exception):
                    self._logger.error(
                        "%s failed to stop nucleus %r: %s", self._log_prefix, faculties[idx], r)
                idx += 1

    async def __aenter__(self):
        if self._starting:
            raise RuntimeError("Mindflow is already entered")
        self._starting = True
        self._event_loop = asyncio.get_running_loop()
        await self._async_exit_stack.__aenter__()
        # 退出顺序很重要:
        # 开关 faculties
        await self._async_exit_stack.enter_async_context(self._faculties_lifecycle_ctx_manager())
        # attention 最后退出.
        await self._async_exit_stack.enter_async_context(self._make_sure_attention_cleared())
        # impulse 消费停止.
        await self._async_exit_stack.enter_async_context(self._impulse_consuming_task_ctx_manager())
        # 先停止 signal.
        await self._async_exit_stack.enter_async_context(self._signal_consuming_task_ctx_manager())
        self._started_event.set()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        self._started_event.clear()
        self._starting = False
        # 走到这一步时, 就不会有信号输入了.
        self._clear()
        await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        # 简单处理下异常. 未来再考虑 error handler
        if isinstance(exc_val, Exception):
            expecting = [asyncio.CancelledError, asyncio.TimeoutError, SystemExit, KeyboardInterrupt]
            for e in expecting:
                if isinstance(exc_val, e):
                    return None
            self._logger.exception(
                "%s mindflow stopped on unexpected exception: %s",
                self._log_prefix, exc_val,
            )
        # do not block any exception
        return None


class BaseMindflow(AbsMindflow):

    def __init__(
            self,
            *nuclei: Nucleus,
            logger: LoggerItf | None = None,
            raise_nucleus_start_error: bool = True,
            system_floor_strength: float = 0.0,
            source_escalation: float = 1.1,
            max_protection_time: float = 3.0,
    ):
        super().__init__(*nuclei, logger=logger, raise_nucleus_start_error=raise_nucleus_start_error)
        self._system_floor_strength = system_floor_strength
        self._max_protection_time = max_protection_time
        self._source_escalation = source_escalation

    def as_channel(self) -> Channel | None:
        return None

    def _build_attention(self, impulse: Impulse, inherit_outcome: Reaction) -> Attention:
        return BaseAttention(
            previous=inherit_outcome,
            impulse=impulse,
            logger=self._logger,
            system_floor_strength=self._system_floor_strength,
            source_escalation=self._source_escalation,
            max_protection_time=self._max_protection_time,
        )


def new_default_mindflow(
        *nuclei: Nucleus,
        logger: LoggerItf | None = None,
) -> BaseMindflow:
    from ghoshell_moss.core.mindflow.input_signal_nucleus import InputSignalNucleus
    return BaseMindflow(
        InputSignalNucleus(),
        *nuclei,
        logger=logger,
    )
