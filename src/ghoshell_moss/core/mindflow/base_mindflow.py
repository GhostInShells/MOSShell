from typing import Self, Iterable, AsyncGenerator, AsyncIterator

import janus

from ghoshell_moss.core.concepts.mindflow import (
    Mindflow, Attention, Impulse, Nucleus, Signal, Priority, BufferImpulse,
    Outcome,
)
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.message import Message
from .base_attention import BaseAttention
import asyncio
import threading
import contextlib


class BaseMindflow(Mindflow):
    """
    基础 Mindflow 的实现.
    """

    def __init__(
            self,
            *nuclei: Nucleus,
            logger: LoggerItf | None = None,
            strict: bool = True,
    ):
        # Nucleus 可能只是一个接口. 内部有别的技术实现.
        self._faculties: dict[str, Nucleus] = {}
        self._faculties_count: int = 0
        self._signal_name_routes: dict[str, list[Nucleus]] = {}
        self._logger = logger or get_moss_logger()
        self._log_prefix = "<MindflowBus>"
        self._current_attention: Attention | None = None
        # 这是内部循环使用的队列.
        self._pop_new_attention_queue: janus.Queue[Attention | None] = janus.Queue(maxsize=1)
        self._last_popped_attention: Attention | None = None
        self._starting = False
        self._started = False
        self._closed = False
        self._paused = False
        self._unpaused_event = ThreadSafeEvent()
        self._unpaused_event.set()
        self._looping_attention = False
        self._set_attention_lock = threading.Lock()
        # 设置线程安全的优先级队列, 用来卸载信号量到本地循环, 避免线程安全上的震荡.
        self._signal_low_queue: janus.PriorityQueue[tuple[int, Signal]] = janus.PriorityQueue(maxsize=100)
        self._signal_high_queue: janus.PriorityQueue[tuple[int, Signal]] = janus.PriorityQueue(maxsize=100)

        # 内部循环检测是否有新的 impulse.
        self._has_impulse_event = ThreadSafeEvent()
        self._consuming_signal_task: asyncio.Task | None = None
        self._consuming_impulse_task: asyncio.Task | None = None
        self._strict = strict
        for nucleus in nuclei:
            self.with_nucleus(nucleus)
        self._async_exit_stack = contextlib.AsyncExitStack()

    def is_running(self) -> bool:
        return self._started and not self._closed

    def faculties(self) -> Iterable[Nucleus]:
        return self._faculties.values()

    def with_nucleus(self, nucleus: Nucleus) -> None:
        if self._started:
            raise RuntimeError(f"Mindflow only with nucleus before started, use add_nucleus instead")
        # 注册运行总线. 只能在启动前用.
        nucleus.with_bus(self.on_signal, self.on_impulse)
        self._faculties[nucleus.name()] = nucleus
        for listening in nucleus.signals():
            if listening not in self._signal_name_routes:
                self._signal_name_routes[listening] = []
            self._signal_name_routes[listening].append(nucleus)
        self._faculties_count = len(self._faculties)

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Mindflow is not running.")

    async def add_nucleus(self, nucleus: Nucleus) -> Self:
        self._check_running()
        # 启动 nucleus 并且加入.
        await nucleus.__aenter__()
        self.with_nucleus(nucleus)

    def on_signal(self, signal: Signal) -> None:
        """接受signal"""
        # 这个函数很可能是接受跨线程的回调, 比如 zenoh session 的回调.
        # 所以它的核心目标是卸载 signal 到当前线程 (loop).
        if not self.is_running():
            self._logger.error("%s on signal but not running: %r", self._log_prefix, signal)
            return None
        elif self._paused:
            self._logger.warning("%s ignore signal cause paused: %r", self._log_prefix, signal)
            return None
        elif signal.is_stale():
            self._logger.debug("%s ignore stale signal: %s", self._log_prefix, signal.id)
            return None
        signal.max_hop -= 1
        if signal.max_hop < 0:
            self._logger.error("%s ignore signal max_hop negative: %r", self._log_prefix, signal)
            return None

        priority_count = signal.priority_strength()
        try:
            if self._signal_low_queue.sync_q.full() and signal.priority >= Priority.CRITICAL:
                # 特殊的信号, 丢到高优队列. 不抛弃不放弃.
                self._signal_high_queue.sync_q.put_nowait((priority_count, signal))
            else:
                self._signal_low_queue.sync_q.put_nowait((priority_count, signal))
        except janus.SyncQueueFull:
            # 直接 ignore 掉. 反应不过来了.
            self._logger.debug("%s ignore signal queue full: %r", self._log_prefix, signal)
            return None

    async def _on_signal_consuming_loop(self):
        """信号消费队列, 将 signal 卸载到当前循环中. """
        while self.is_running():
            # 队列是单一消费者, 所以可以检查 empty.
            if not self._signal_high_queue.async_q.empty():
                p, item = self._signal_high_queue.async_q.get_nowait()
            else:
                # 如果高优队列不为空, 一定是低优队列满了. 所以低优队列阻塞时永远不会阻塞高优队列.
                p, item = await self._signal_low_queue.async_q.get()
            # 丢弃过期对象.
            if self._paused or item.is_stale():
                # 丢弃过期的信号量. 这个日志要不要记录呢?
                self._logger.debug("%s ignore stale signal: %s", self._log_prefix, item.id)
                continue
            await self._dispatch_signal(item)

    async def _dispatch_signal(self, signal: Signal) -> None:
        try:
            name = signal.name
            broadcasted = 0
            if len(self._faculties) == 0:
                return None
            if name not in self._signal_name_routes:
                # 丢弃不监听的 signal.
                return None
            for n in self._signal_name_routes[name]:
                # 触发分配.
                n.on_signal(signal)
            self._logger.debug("%s receive signal and send to %d nuclei", self._log_prefix, broadcasted)
            return None
        except asyncio.CancelledError:
            # 只有 cancel 才 raise.
            raise
        except Exception as e:
            # 拦截所有的异常, 不要影响外部循环.
            self._logger.error("%s dispatch signal error on %s: %s", self._log_prefix, signal, e)

    def on_impulse(self, impulse: Impulse) -> None:
        """
        接受新的 impulse 并且进行排队.
        """
        # impulse 本身可能也是跨线程的, 有几种情况:
        # 1. Nucleus 自身不是从 on_signal 进行决策的, 动作不是在同一个 loop 里触发.
        # 2. Mindflow 接受进程级别的 Impulse 通讯, 不是从持有的 Nucleus 回调的.
        if self._paused:
            self._logger.info("%s drop impulse cause paused: %s", self._log_prefix, impulse)
            return None
        elif not self.is_running():
            self._logger.error("%s drop impulse cause not running: %s", self._log_prefix, impulse)
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
            impulse = self._rank_nuclei()
            # 使用 await, 方便感知 cancel?
            if impulse is None:
                # 以 rank 的瞬间为准. 如果出现极端情况, rank完的瞬间又有新的 impulse, 那也只能等下一轮.
                continue
            else:
                await self._challenge_attention(impulse)

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
            nucleus.pop_impulse(impulse)

    async def _challenge_attention(self, impulse: Impulse) -> None:
        try:
            if impulse.is_stale():
                self._pop_impulse(impulse)
                return None
            # attention 或者.
            if self._current_attention and not self._current_attention.is_aborted():
                # 校验出现结果.
                done = self._current_attention.on_challenge(impulse)
                if done is BufferImpulse:
                    # 挑战通过, 已经被 buffer 了. 通知一下.
                    self._pop_impulse(impulse)
                elif done:
                    # 挑战成功. 先完成通知, 然后再替换 attention.
                    self._pop_impulse(impulse)
                    # set impulse 时会终止原来的. 并继承对应参数.
                    self.set_impulse(impulse)
                else:
                    # 通知 suppress.
                    self._suppress_impulse(impulse, self._current_attention.peek())
                return None
            # 不需要排序, 因为消费过程中, 本身拿到的就是优先级队列里的 impulse.
            self.set_impulse(impulse)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 只记录异常, 不要抛出终止. 保证循环运行.
            self._logger.exception(
                "%s failed to challenge attention with impulse %r: %s",
                self._log_prefix, impulse, e,
            )

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
        """直接用 impulse 创建 attention"""
        if impulse.is_stale():
            # 仍然做一次校验.
            return None
        with self._set_attention_lock:
            if self._current_attention is not None:
                if not self._current_attention.is_aborted():
                    # 在这里 abort.
                    self._current_attention.abort("interrupted")
                # 在 last outcome 里做了判断, 如果没有 started 过, 则会返回原始的对象.
                inherit_outcome = self._current_attention.last_outcome()
            else:
                inherit_outcome = Outcome()
            attention = BaseAttention(
                impulse=impulse,
                last_outcome=inherit_outcome,
                logger=self._logger,
                system_floor_strength=0.0,  # 决定强度衰减到合适中断.
                source_escalation=1.1,  # 决定同源 impulse 提权比例.
                max_protection_time=3.0,  # 决定最大的保护时间.
                protection_duration_ratio=0.2,  # 决定保护时间在总时间的比例.
            )
            self._set_attention(attention)
            return None

    def _set_attention(self, attention: Attention) -> None:
        # 这个函数只在 set impulse 处可以被调用.
        # 考虑到未来 set attention 可能不止一个地方调用 (比如命令行的行为), 所以加一个 set.
        if not self.is_running():
            self._logger.error("%s set attention but not running: %r", self._log_prefix, attention)
            attention.abort("not running")
            # 保持 attention 上下文的连续性.
            self._current_attention = attention
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
        self._current_attention.with_context_func("mindflow", self.context_messages)
        # 这个队列里的其实都是上一个 current attention.
        while not self._pop_new_attention_queue.sync_q.empty():
            # maxsize 为 1 的队列.
            attention = self._pop_new_attention_queue.sync_q.get_nowait()

        self._pop_new_attention_queue.sync_q.put_nowait(self._current_attention)
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
        self._pop_new_attention_queue.sync_q.put_nowait(None)

    def clear(self) -> None:
        if not self.is_running():
            return
        self._clear()

    def _clear(self) -> None:
        # 其实这两个通常是同一个. 不排除在队列中.
        if self._current_attention is not None and not self._current_attention.is_aborted():
            self._current_attention.abort('closed')
        if self._last_popped_attention is not None and not self._last_popped_attention.is_aborted():
            self._last_popped_attention.abort("interrupted")

        while not self._signal_low_queue.sync_q.empty():
            _ = self._signal_low_queue.sync_q.get_nowait()
        for nucleus in self._faculties.values():
            # 清空所有的状态.
            nucleus.clear()
        self._has_impulse_event.clear()
        while not self._pop_new_attention_queue.sync_q.empty():
            self._pop_new_attention_queue.sync_q.get_nowait()

    def context_messages(self) -> list[Message]:
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
            while self.is_running():
                self._looping_attention = True
                try:
                    # 获取 attention 不去关心 pause. 因为 pause 了 仍然可以 set impulse.

                    # 理论上 last popped attention 永远是被处理完, 才可能吐出一个 attention.
                    # 一个 mindflow 只能吐出一个 attention. 用来做单一状态管理.
                    # 不过仍然做一层冗余, 好像没有什么代价, 但会更安心.
                    if self._last_popped_attention is not None and not self._last_popped_attention.is_aborted():
                        # 等待到上一帧 attention 执行完毕.
                        # 这种情况只有一种, 就是 attention 被发送给别的队列了, 导致这个阻塞点立刻重入.
                        await self._last_popped_attention.wait_closed()
                        # 做一次 running 的检查.
                        continue
                    self._last_popped_attention = None

                    # 如果进入等待的瞬间没有任何 attention, 最常见的就是一大堆的 Impulse 被压抑住了.
                    # 而被压抑住的 attention 结束时, 反而没有新的 impulse 进入.
                    if self._current_attention is None:
                        if impulse := self._rank_nuclei():
                            # 强行设置 Impulse, 不再进行排序.
                            self.set_impulse(impulse)
                    _attention = await self._pop_new_attention_queue.async_q.get()
                    if _attention is None:
                        # 拿到毒丸, 退出循环.
                        # 当 mindflow 显式关闭时, 一定要发送毒丸.
                        return
                    if _attention.is_aborted():
                        continue
                    self._last_popped_attention = _attention
                    yield _attention
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    continue
        finally:
            self._looping_attention = False

    @contextlib.asynccontextmanager
    async def _make_sure_attention_cleared(self):
        """确保在线的 attention 都被退出了. """
        try:
            yield
        finally:
            if self._current_attention is not None and not self._current_attention.is_aborted():
                self._current_attention.abort('mindflow closed')
                # 稍稍等待一下退出.
                await self._current_attention.wait_closed()
                self._current_attention = None
            if self._last_popped_attention is not None and not self._last_popped_attention.is_aborted():
                self._last_popped_attention.abort("mindflow closed")
                await self._current_attention.wait_closed()
                self._last_popped_attention = None
            while not self._pop_new_attention_queue.sync_q.empty():
                self._pop_new_attention_queue.sync_q.get_nowait()
            self._pop_new_attention_queue.sync_q.put_nowait(None)

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
        # 从头开始启动.
        self._faculties.clear()
        result = await asyncio.gather(*[n.__aenter__() for n in nuclei], return_exceptions=True)
        idx = 0
        for r in result:
            nucleus = nuclei[idx]
            if isinstance(r, Exception):
                self._logger.error("%s failed to start nucleus %r: %s", self._log_prefix, nucleus, r)
                if self._strict:
                    # 严格模式下启动不做任何容错. 仅仅作为一个保留开发点. 默认是抛出异常.
                    raise r
            else:
                self.with_nucleus(nucleus)
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
            return
        self._starting = True
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
        self._started = True

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        self._started = False
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
