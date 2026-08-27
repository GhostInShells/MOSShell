"""
Mindflow 调度逻辑实现.

重构自历史文件 ./base_mindflow.py
"""

import inspect
from abc import abstractmethod, ABC
from typing import AsyncGenerator, AsyncIterator, Callable, Awaitable, Iterable
from typing_extensions import Self

import janus

from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, Attention, Impulse, Nucleus, Signal, SignalName, Priority,
    ChallengeVerdict, MindflowHook,
    ChallengeMode, Action, Thinking, ActionExitedException
)
from ghoshell_moss.core.blueprint.moment import BaseMomentsObserver, Moments, Moment
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, Channel
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ._channel import build_mindflow_channel
from ._attention import BaseAttention
from ._think import BaseThinking
import asyncio
import contextlib
import threading
import logging

_SignalName = str
_NucleusName = str

__all__ = [
    'BaseMindflow',
    'AbsMindflow',
    'new_default_mindflow',
    'DirectImpulseNucleus',
]


class DirectImpulseNucleus(Nucleus):
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

    def attended(self, impulse: Impulse) -> None:
        if self._impulse is impulse:
            self._impulse = None

    def ignored(self, impulse: Impulse) -> None:
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

    async def __aenter__(self) -> "DirectImpulseNucleus":
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
            description: str = '',
            logger: logging.Logger | None = None,
            raise_nucleus_start_error: bool = True,
            max_moments_size: int = 100,
    ):
        # Nucleus 可能只是一个接口. 内部有别的技术实现.
        self._description = description
        self._nuclei: dict[_NucleusName, Nucleus] = {}
        self._input_signal_name_routes: dict[_SignalName, dict[_NucleusName, Nucleus]] = {}
        self._logger = logger or get_moss_logger()
        self._log_prefix = "<MindflowBus>"
        self._current_attention: Attention | None = None
        # 这是内部循环使用的队列.
        self._pop_new_attention_queue: janus.Queue[Attention | None] = janus.Queue(maxsize=1)
        self._starting = False
        self._started_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
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
        self._signal_priority_bar = Priority.BACKGROUND
        self._impulse_priority_bar = Priority.BACKGROUND

        # 内部循环检测是否有新的 impulse.
        self._consuming_signal_task: asyncio.Task | None = None
        self._consuming_impulse_task: asyncio.Task | None = None
        # 是否对启动异常容错.
        self._raise_nucleus_start_error = raise_nucleus_start_error
        # 内置 direct nucleus, 服务 public add_impulse 入口. 必须先于用户 nuclei 注册.
        self._direct_nucleus = DirectImpulseNucleus()
        self.with_nucleus(self._direct_nucleus)
        for nucleus in nuclei:
            self.with_nucleus(nucleus)
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._hooks_group: MindflowHookGroup = MindflowHookGroup(self._logger)
        self._moments_observer = BaseMomentsObserver(max_size=max_moments_size, logger=self._logger)
        self._idle_event = ThreadSafeEvent()
        self._inner_tasks: set[asyncio.Task] | None = None
        self._on_idle_callbacks: set[Callable[[Moments], None] | Callable[[Moments], Awaitable[None]]] = set()
        self._idle_callback_tasks: set[asyncio.Future] = set()
        self._mindflow_channel: Channel | None = None

        # 供外部使用的队列.
        self._thinking_loop_queue: janus.Queue[Thinking] = janus.Queue(maxsize=10)
        self._is_looping_thinking = False

        self._attention_created_callbacks: set[Callable[[Attention], None]] = set()

        # 测试专用逻辑.
        self._action_loop_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
        self._is_looping_action = False

        # 观测轨迹: 帧中到达的 impulse (absorb 续包) 暂存于此, 下一帧生成时折进 moment.
        # 以及上一轮 attention 的 abort reason, 织进下一帧 moment.previous.stop_reason.
        self._pending_frame_impulses: list[Impulse] = []
        self._last_abort_reason: str = ''

    def nuclei(self) -> dict[_NucleusName, Nucleus]:
        return self._nuclei

    # --- idle 治理 --- #

    def is_idle(self) -> bool:
        return self._idle_event.is_set()

    def description(self) -> str:
        return self._description

    def when_idle(self, callback: Callable[[Moments], None] | Callable[[Moments], Awaitable[None]]) -> Callable[
        [], None]:
        self._on_idle_callbacks.add(callback)

        def _disposer():
            if callback in self._on_idle_callbacks:
                self._on_idle_callbacks.discard(callback)

        return _disposer

    def _set_idle(self) -> None:
        try:
            if self._idle_event.is_set():
                return
            if len(self._on_idle_callbacks) == 0:
                return
            for callback in self._on_idle_callbacks:
                fut = self._event_loop.create_task(self._run_idle_callback_future(callback))
                self._add_idle_callback_tasks(fut)
        finally:
            self._idle_event.set()

    async def _stop_idling(self) -> None:
        try:
            if len(self._idle_callback_tasks) == 0:
                return
            futures = self._idle_callback_tasks.copy()
            self._idle_callback_tasks.clear()
            for fut in futures:
                fut.cancel()
            # 取消所有的 idling.
            await asyncio.gather(*futures, return_exceptions=True)
        finally:
            self._idle_event.clear()

    async def _run_idle_callback_future(
            self,
            callback: Callable[[Moments], Awaitable[None]] | Callable[[Moments], None]
    ) -> None:
        try:
            if inspect.iscoroutinefunction(callback):
                await callback(self._moments_observer)
            else:
                fut = self._event_loop.run_in_executor(None, callback, self._moments_observer)
                await fut
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.exception("% run idle callback error: %s", self._log_prefix, e)
            self._hooks_group.on_error(e)

    def _add_idle_callback_tasks(self, callback_task: asyncio.Future) -> None:
        self._idle_callback_tasks.add(callback_task)

        def _done(_task: asyncio.Task) -> None:
            if _task in self._idle_callback_tasks:
                self._idle_callback_tasks.discard(_task)

        callback_task.add_done_callback(_done)

    @property
    def moments(self) -> Moments:
        return self._moments_observer

    @staticmethod
    def _new_signal_queue() -> janus.PriorityQueue[tuple[int, int, Signal]]:
        return janus.PriorityQueue(maxsize=100)

    def is_running(self) -> bool:
        return self._started_event.is_set() and not self._closed_event.is_set()

    def with_hook(self, hook: MindflowHook) -> Self:
        self._hooks_group.add_hook(hook)
        return self

    def remove_hook(self, hook: str | MindflowHook) -> None:
        if isinstance(hook, MindflowHook):
            hook = hook.name()
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
        if not override and _name in self._nuclei:
            raise NameError(f"nucleus {_name} already exists")

        nucleus.with_bus(self.add_signal, self._nucleus_has_impulse)
        self._nuclei[_name] = nucleus

    def _register_nucleus_to_signal_routes(self, nucleus: Nucleus) -> None:
        for listening in nucleus.signals():
            if listening not in self._input_signal_name_routes:
                self._input_signal_name_routes[listening] = {}
            # 使用 dict 注册防止重复.
            # always override
            self._input_signal_name_routes[listening][nucleus.name()] = nucleus
        nucleus.with_moments(self._moments_observer)

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
        self._bind_nucleus_with_bus(nucleus, override=override)
        self._register_nucleus_to_signal_routes(nucleus)

    def _has_nucleus(self, name: str) -> bool:
        return name in self._nuclei

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
            if len(self._nuclei) == 0:
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
            self._hooks_group.on_error(e)

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
                impulse = self._rank_best_impulse_from_nuclei()
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
                self._hooks_group.on_error(e)

    def _suppress_impulse(self, impulse: Impulse, by: Impulse) -> None:
        """supress 指定的 impulse"""
        nucleus = self._nuclei.get(impulse.source, None)
        if nucleus is not None:
            nucleus.suppress(by)

    def _notify_impulse_attended(self, impulse: Impulse) -> None:
        """通知 nucleus 被 pop 了. """
        nucleus = self._nuclei.get(impulse.source, None)
        if nucleus is not None:
            # 应该要将 impulse 给踢掉.
            nucleus.attended(impulse)

    def _notify_impulse_ignored(self, impulse: Impulse) -> None:
        nucleus = self._nuclei.get(impulse.source, None)
        if nucleus is not None:
            nucleus.ignored(impulse)

    async def _challenge_attention(self, impulse: Impulse) -> None:
        """impulse 与当前 attention 的仲裁入口. 原子操作.

        三 mode (default/silent/notify) 沿"抢占成功 vs 失败" 双轴对称分布,
        见 ``ChallengeMode`` 注释的对称表. 本函数把对称表展开成实际分支:
        - 抢占成功 + silent → buffer messages (silent 偏离侧)
        - 抢占成功 + 其他   → 创建新 attention (default)
        - 抢占失败 + notify → buffer messages (notify 偏离侧)
        - 抢占失败 + 其他   → suppress nucleus (default)

        quiet 系统 (无 defender) 走单独分支: silent 同样 buffer 不创建 attention,
        其他模式直接创建初始 attention.

        FATAL/BACKGROUND 在进入 challenge() 之前先短路 — 这是协议级承诺,
        防止子类重写 challenge() 把绝对性退化掉.

        strength=0 在 stale 之前先短路 — Impulse "绝不竞争" 的协议承诺
        (Zen 静默心智模型预留): 不进任何 mode 分支, 不 buffer 不 suppress,
        从 nucleus pop 后 fire 'yielded' verdict.
        """
        try:
            # strength=0 协议承诺: 绝不竞争 — 比 stale 更先短路.
            # 不分 defender/quiet, 一律礼让: 不打任何 mode 分支, 不建 attention,
            # 从 nucleus pop 后 fire 'yielded', 由 nucleus 自然清理缓存.
            if impulse.strength == 0:
                defender = None
                if self._current_attention and not self._current_attention.is_aborted():
                    defender = self._current_attention.draw_from()
                await self._fire_challenge(impulse, defender, 'yielded')
                return None

            if impulse.is_stale():
                # 通知已经被丢弃.
                self._notify_impulse_ignored(impulse)
                return None

            # 实装所有强约定规则, 在 Attention 自行实现的机制外, 保证可预见性结果.
            verdict: ChallengeVerdict = 'suppressed'
            if self._current_attention and not self._current_attention.is_aborted():
                defender = self._current_attention.draw_from()
                # Fatal always prevails (silent mode 抢占成功但只 buffer 不创建 attention)
                if impulse.priority == Priority.FATAL.value:
                    verdict = 'buffered' if impulse.mode == ChallengeMode.silent.value else 'preempted'
                    await self._fire_challenge(impulse, defender, verdict)
                    return None
                elif impulse.priority == Priority.BACKGROUND.value:
                    # BACKGROUND 永不抢占; notify 偏离侧: 失败时 buffer 而非 suppress.
                    verdict = 'buffered' if impulse.mode == ChallengeMode.notify.value else 'suppressed'
                    await self._fire_challenge(impulse, defender, verdict)
                    return None
                if self._current_attention.is_protected():
                    # 保护期, 同/低优先级失败. notify 偏离侧: 失败时 buffer 而非 suppress.
                    verdict = 'buffered' if impulse.mode == ChallengeMode.notify.value else 'suppressed'
                    await self._fire_challenge(impulse, defender, verdict)
                    return None

                # 执行挑战逻辑.
                result = await self._current_attention.challenge(impulse)
                # 如果被吸收了.
                if result == 'win':
                    # 同 ID 更新 complete, 不抢占.
                    verdict = 'preempted'
                    if impulse.mode == ChallengeMode.silent.value:
                        verdict = 'buffered'
                    await self._fire_challenge(impulse, defender, verdict)
                    return None
                elif result == 'lose':
                    if impulse.mode == ChallengeMode.notify.value:
                        verdict = 'buffered'
                    else:
                        verdict = 'suppressed'
                    await self._fire_challenge(impulse, defender, verdict)
                    return None
                elif result == 'absorb':
                    # notify 挑战失败, 将 notify 的消息入队.
                    await self._fire_challenge(impulse, defender, 'absorbed')
                return None
            else:
                verdict = 'initial'
                if impulse.mode == ChallengeMode.silent.value:
                    # silent 模式不创建注意力, 只做 buffer.
                    verdict = 'buffered'
                # 创建一个新的 impulse.
                await self._fire_challenge(impulse, None, verdict)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 只记录异常, 不要抛出终止. 保证循环运行.
            self._logger.exception(
                "%s failed to challenge attention with impulse %r: %s",
                self._log_prefix, impulse, e,
            )
            self._hooks_group.on_error(e)

    async def _fire_challenge(
            self,
            challenger: Impulse,
            defender: Impulse | None,
            verdict: ChallengeVerdict,
    ) -> None:
        if verdict == 'suppressed':
            self._suppress_impulse(challenger, defender)
        elif verdict == 'preempted':
            # 创建一个新的 impulse.
            self._notify_impulse_attended(challenger)
            await self._create_attention_from_impulse(challenger)
        elif verdict == 'buffered':
            self._notify_impulse_attended(challenger)
            self._moments_observer.inject_percepts(*challenger.messages)
        elif verdict == 'absorbed':
            self._notify_impulse_attended(challenger)
            # absorbed 也是一种 attended 动作: 让持有 attention 更新仲裁状态 (优先级/强度/保护期).
            # absorb_impulse 返回 Impulse 表示载荷仍需交由 mindflow 折进观测 (暂存到下一帧);
            # 返回 None 表示 attention 内部已消化, 无需 mindflow 继续暂存.
            if self._current_attention is not None:
                pending = self._current_attention.absorb_impulse(challenger)
                if pending is not None:
                    self._pending_frame_impulses.append(pending)
        elif verdict == 'initial':
            self._notify_impulse_attended(challenger)
            await self._create_attention_from_impulse(challenger)
        elif verdict == 'yielded':
            # 绝不竞争: 不经手任何 mode 分支, 直接通知 nucleus 自然清理缓存.
            self._notify_impulse_attended(challenger)
        self._hooks_group.on_impulse_challenged(challenger, defender, verdict)

    def attention(self) -> Attention | None:
        if self._current_attention is None:
            return None
        elif self._current_attention.is_aborted():
            return None
        return self._current_attention

    def set_impulse(self, impulse: Impulse) -> None:
        if impulse.is_stale():
            return None
        if not self.is_running():
            return None
        self._event_loop.create_task(self._create_attention_from_impulse(impulse))
        return None

    async def _create_attention_from_impulse(self, impulse: Impulse) -> None:
        """直接用 impulse 创建 attention"""
        self._notify_impulse_attended(impulse)
        async with self._set_impulse_lock:
            if impulse.is_stale():
                # 仍然做一次校验.
                return None
            if self._current_attention is not None:
                if not self._current_attention.is_aborted():
                    # 在这里 abort.
                    self._current_attention.abort("interrupted")
                # 在 last outcome 里做了判断, 如果没有 started 过, 则会返回原始的对象.
            attention = None
            # 创建 attention
            if nucleus := self._nuclei.get(impulse.source):
                # 允许 impulse 自行构建 attention.
                attention = nucleus.create_attention(observer=self._moments_observer, impulse=impulse)
            if attention is None:
                attention = self._build_attention(impulse)
            self._set_attention(attention)
            return None

    @abstractmethod
    def _build_attention(self, impulse: Impulse) -> Attention:
        """子类实现: 用指定的仲裁策略构建 Attention 实例."""
        pass

    @abstractmethod
    def as_channel(self) -> Channel | None:
        """强调子类要重新实现 channel 逻辑. """
        if self._mindflow_channel is None:
            self._mindflow_channel = build_mindflow_channel(self)
        return self._mindflow_channel

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

        try:
            # 这个队列里的其实都是上一个 current attention.
            # 要考虑 attention 在还没启动时就结束.
            while not self._pop_new_attention_queue.sync_q.empty():
                # maxsize 为 1 的队列.
                attention = self._pop_new_attention_queue.sync_q.get_nowait()
            self._pop_new_attention_queue.sync_q.put_nowait(self._current_attention)

        except janus.AsyncQueueShutDown:
            return None
        # 新 attention 入队.
        self._logger.info("%s set attention %r", self._log_prefix, attention)
        return None

    def peek_impulses(self) -> Iterable[tuple[Nucleus, Impulse]]:
        for nucleus in self._nuclei.values():
            if not nucleus.is_running():
                continue
            impulse = nucleus.peek()
            # 是否 impulse 也要做一个过期?
            if impulse is None:
                continue
            elif impulse.is_stale():
                # pop stale impulse for the nucleus
                nucleus.ignored(impulse)
                self._logger.info("%s pop stale impulse %r", self._log_prefix, impulse)
                continue
            else:
                impulse.source = nucleus.name()
                yield nucleus, impulse

    def _rank_best_impulse_from_nuclei(self, best_impulse: Impulse = None) -> Impulse | None:
        """从所有的 nuclei 中获取最重要的 impulse. """
        best_impulse = best_impulse
        best_n = None
        best_p = 0 if best_impulse is None else best_impulse.priority_strength()
        losers: list[Nucleus] = []
        for nucleus, impulse in self.peek_impulses():
            if impulse.priority < self._impulse_priority_bar:
                # 低于优先级的直接忽视.
                continue
            # 加一行代码防蠢.
            # 基于最基础的信号优先级返回 impulse.
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

    def when_attention_created(self, callback: Callable[[Attention], None]) -> Callable[[], None]:
        self._attention_created_callbacks.add(callback)

        def _disposer():
            self._attention_created_callbacks.discard(callback)

        return _disposer

    async def attention_loop(self) -> AsyncGenerator[Attention, None]:
        """ 测试专用的接口. 可重入. 不可在下游治理 attention 的声明周期 (已经被处理). """
        q = janus.Queue[Attention]()
        disposer: Callable[[], None] | None = None
        try:
            disposer = self.when_attention_created(q.sync_q.put_nowait)
            while self.is_running():
                try:
                    attention = await q.async_q.get()
                    yield attention
                except janus.AsyncQueueShutDown:
                    break
        finally:
            if disposer is not None:
                disposer()

    def _on_attention_created(self, attention: Attention) -> None:
        if len(self._attention_created_callbacks) > 0:
            for callback in self._attention_created_callbacks:
                try:
                    callback(attention)
                except Exception as e:
                    self._logger.exception(
                        "%s _on_attention_created callback %r failed: %s", self._log_prefix, callback, e,
                    )

    async def _loop_attention(self) -> AsyncGenerator[Attention, None]:
        """需要实现一个特别稳定的流程."""
        if self._looping_attention:
            raise RuntimeError('looping attention already running')
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
                        if impulse := self._rank_best_impulse_from_nuclei():
                            # 提醒一下有事件.
                            self._has_impulse_event.set()
                        elif not self.is_idle():
                            # 开启 idle.
                            self._set_idle()
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
                    # 停止 idle.
                    await self._stop_idling()
                    self._on_attention_created(_attention)
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

    def _make_thinking(self, attention: Attention, impulse: Impulse | None, moment: Moment) -> Thinking:
        def _put_action(action: Action) -> None:
            try:
                self._action_loop_queue.sync_q.put_nowait(action)
            except janus.SyncQueueShutDown:
                # janus 关停语义不应泄漏到 action 协议层, 统一转成 statement exit.
                raise ActionExitedException()

        return BaseThinking(
            attention=attention,
            observer=self._moments_observer,
            put_action=_put_action,
            mindflow_stop_event=self._closed_event,
            moment=moment,
            logger=self._logger,
        )

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
        if self._closed_event.is_set():
            return
        self._closed_event.set()
        self._unpaused_event.set()
        self._clear()
        # 用来通知退出.
        if not self._pop_new_attention_queue.sync_q.closed:
            self._pop_new_attention_queue.shutdown(immediate=True)
        if not self._thinking_loop_queue.sync_q.closed:
            self._thinking_loop_queue.shutdown(immediate=True)
        if not self._action_loop_queue.sync_q.closed:
            self._action_loop_queue.shutdown(immediate=True)

    async def wait_close(self) -> None:
        await self._closed_event.wait()

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
        for nucleus in self._nuclei.values():
            # 清空所有的状态.
            nucleus.clear()
        # clear moments observer also
        self._moments_observer.clear()
        self._has_impulse_event.clear()
        # clear the task groups
        while not self._pop_new_attention_queue.sync_q.empty():
            self._pop_new_attention_queue.sync_q.get_nowait()
        # 清空观测轨迹的暂存.
        self._pending_frame_impulses = []
        self._last_abort_reason = ''

    @staticmethod
    def _is_useful_frame(impulse: Impulse | None) -> bool:
        """是否携带可折进观测帧的载荷 (有用帧). """
        if impulse is None:
            return False
        return bool(impulse.messages or impulse.dynamic_messages or impulse.hint or impulse.logos)

    def _fold_frame_impulses(self, moment: Moment, impulse: Impulse | None) -> None:
        """把本帧携带的 impulse 载荷织进 moment (用 Impulse.update_moment, 而非新增 observer 接口). """
        if self._is_useful_frame(impulse):
            impulse.update_moment(moment)
        if self._pending_frame_impulses:
            incoming = self._pending_frame_impulses
            self._pending_frame_impulses = []
            for absorbed in incoming:
                if self._is_useful_frame(absorbed):
                    absorbed.update_moment(moment)

    async def _generate_thinking(self) -> None:
        """基于 attention 循环, 生产 thinking 循环. """
        while self.is_running():
            async for attention in self._loop_attention():
                try:
                    async with attention:
                        impulse = await attention.wait_ready()
                        while not attention.is_aborted():
                            moment = self._moments_observer.observe()
                            # 把上一轮(或上一 attention)的 abort reason 织进这一帧的接缝.
                            if self._last_abort_reason and moment.previous is not None:
                                moment.previous.stop_reason = self._last_abort_reason
                                self._last_abort_reason = ''
                            # 本帧的有用 impulse (创建帧的 impulse + 帧中到达的 absorb 续包) 折进 moment.
                            self._fold_frame_impulses(moment, impulse)
                            think = self._make_thinking(attention, impulse, moment)
                            impulse = None
                            await self._thinking_loop_queue.async_q.put(think)
                            await think.wait_abort()
                            if attention.is_aborted():
                                self._last_abort_reason = attention.abort_reason()
                                break
                            if not self._moments_observer.need_observe():
                                break
                except asyncio.CancelledError:
                    raise
                except janus.AsyncQueueShutDown:
                    # 退出时才会关闭循环.
                    break
                except Exception as e:
                    self._logger.exception("%s generate_thinking failed: %s", self._log_prefix, e)
                    self._hooks_group.on_error(e)

    def thinking_loop(self) -> AsyncIterator[Thinking]:
        return self._thinking_loop()

    async def _thinking_loop(self) -> AsyncGenerator[Thinking, None]:
        if self._is_looping_thinking:
            raise RuntimeError(f"looping thinking once at a time")
        self._is_looping_thinking = True
        try:
            while self.is_running():
                try:
                    think = await self._thinking_loop_queue.async_q.get()
                    if think.is_aborted():
                        continue
                    yield think
                except asyncio.CancelledError:
                    raise
                except janus.AsyncQueueShutDown:
                    break
                except Exception as e:
                    self._logger.exception("%s thinking_loop failed: %s", self._log_prefix, e)
                    self._hooks_group.on_error(e)
                finally:
                    pass
        finally:
            self._is_looping_thinking = False

    def action_loop(self) -> AsyncIterator[Action]:
        return self._action_loop()

    async def _action_loop(self) -> AsyncGenerator[Action, None]:
        if self._is_looping_action:
            raise RuntimeError(f"looping action once at a time")
        self._is_looping_action = True
        try:
            while self.is_running():
                try:
                    action = await self._action_loop_queue.async_q.get()
                    if action.is_aborted():
                        continue
                    yield action
                except asyncio.CancelledError:
                    raise
                except janus.AsyncQueueShutDown:
                    break
                except Exception as e:
                    self._logger.exception("%s action_loop failed: %s", self._log_prefix, e)
                    self._hooks_group.on_error(e)
                finally:
                    pass
        finally:
            self._is_action_looping = False

    @contextlib.asynccontextmanager
    async def _attention_generation_task_ctx(self):
        task = None
        try:
            task = self._event_loop.create_task(self._generate_thinking())
            yield
        finally:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # mindflow 退出时清理仍阻塞的 idle 回调, 不把它们放养到事件循环关闭.
            await self._stop_idling()

            current_attention = None
            if self._current_attention is not None and not self._current_attention.is_aborted():
                self._current_attention.abort('mindflow closed')
                # 稍稍等待一下退出.
                current_attention = self._current_attention
            if current_attention is not None:
                await current_attention.wait_abort()
            if not self._pop_new_attention_queue.sync_q.closed:
                self._pop_new_attention_queue.shutdown(immediate=True)
            if not self._thinking_loop_queue.sync_q.closed:
                self._thinking_loop_queue.shutdown(immediate=True)
            if not self._action_loop_queue.sync_q.closed:
                self._action_loop_queue.shutdown(immediate=True)

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
    async def _nuclei_lifecycle_ctx_manager(self):
        nuclei = list(self._nuclei.values())
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
                    self._hooks_group.on_error(r)
            else:
                # 正式注册监听.
                self._register_nucleus_to_signal_routes(nucleus)

            idx += 1
        try:
            yield
        finally:
            faculties = list(self._nuclei.values())
            self._nuclei.clear()
            close_all = []
            for nucleus in faculties:
                close_all.append(nucleus.__aexit__(None, None, None))
            result = await asyncio.gather(*close_all, return_exceptions=True)
            idx = 0
            for r in result:
                if isinstance(r, Exception):
                    self._logger.error(
                        "%s failed to stop nucleus %r: %s", self._log_prefix, faculties[idx], r)
                    self._hooks_group.on_error(r)
                idx += 1

    async def __aenter__(self):
        if self._starting:
            raise RuntimeError("Mindflow is already entered")
        self._starting = True
        self._event_loop = asyncio.get_running_loop()
        await self._async_exit_stack.__aenter__()
        # 退出顺序很重要:
        # 开关 faculties
        await self._async_exit_stack.enter_async_context(self._nuclei_lifecycle_ctx_manager())
        # attention 最后退出.
        await self._async_exit_stack.enter_async_context(self._attention_generation_task_ctx())
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
            self._hooks_group.on_error(exc_val)
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

    def _build_attention(self, impulse: Impulse) -> Attention:
        return BaseAttention(
            impulse=impulse,
            logger=self._logger,
            system_floor_strength=self._system_floor_strength,
            source_escalation=self._source_escalation,
            max_protection_time=self._max_protection_time,
        )


def new_default_mindflow(
        *nuclei: Nucleus,
        logger: logging.Logger | None = None,
) -> BaseMindflow:
    from ghoshell_moss.core.mindflow.input_signal_nucleus import InputSignalNucleus
    return BaseMindflow(
        InputSignalNucleus(),
        *nuclei,
        logger=logger,
    )
