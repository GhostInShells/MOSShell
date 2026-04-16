from typing import Self, Iterable

from ghoshell_moss.core.concepts.mindflow import (
    Mindflow, Attention, Impulse, Nucleus, Signal,
)
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.helpers import Timeleft
import asyncio
import threading


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
        self._faculties: dict[str, Nucleus] = {}
        self._logger = logger or get_moss_logger()
        self._log_prefix = "<MindflowBus>"
        for nucleus in nuclei:
            self._faculties[nucleus.name()] = nucleus
        self._current_attention: Attention | None = None
        self._pop_new_attention_queue = asyncio.Queue(maxsize=1)
        self._last_popped_attention: Attention | None = None
        self._starting = False
        self._started = False
        self._closed = False
        self._paused = False
        self._set_attention_lock = threading.Lock()
        # 定义一个简单的开关可以选择启动时的容错性.
        self._strict = strict
        self._unpaused_event = ThreadSafeEvent()
        self._unpaused_event.set()

    def _create_attention_from_impulse(self, impulse: Impulse) -> Attention:
        pass

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

    async def add_nucleus(self, nucleus: Nucleus) -> Self:
        if self.is_running():
            await nucleus.__aenter__()
        self.with_nucleus(nucleus)

    def on_impulse(self, impulse: Impulse) -> None:
        """
        接受新的 impulse 并且进行排队.
        """
        if self._paused:
            self._logger.info("%s drop impulse cause paused: %s", self._log_prefix, impulse)
            return None
        if not self.is_running():
            self._logger.error("%s drop impulse cause not running: %s", self._log_prefix, impulse)
            return None

        if self._current_attention and not self._current_attention.is_aborted():
            # 校验出现结果.
            if self._current_attention.on_challenge(impulse):
                attention = self._create_attention_from_impulse(impulse)
                self.set_attention(attention)
            else:
                suppressing = self._faculties.get(impulse.source, None)
                if suppressing is not None:
                    suppressing.suppress(self._current_attention.peek())
            return None

        self._current_attention = None
        # 排序获取最优先的 impulse.
        best_impulse = self._rank_nuclei()
        if best_impulse is not None:
            best_impulse = best_impulse or impulse
        if best_impulse:
            if impulse := self._pop_impulse(best_impulse.source):
                attention = self._create_attention_from_impulse(impulse)
                self.set_attention(attention, 'new impulse emerge')
        return None

    def _pop_impulse(self, source: str) -> Impulse | None:
        nucleus = self._faculties.get(source, None)
        if nucleus is not None:
            return nucleus.pop_impulse()
        return None

    def attention(self) -> Attention | None:
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

    def on_signal(self, signal: Signal) -> None:
        if not self.is_running():
            self._logger.error("%s on signal but not running: %r", self._log_prefix, signal)
            return None
        if self._paused:
            self._logger.warning("%s ignore signal cause paused: %r", self._log_prefix, signal)
            return None
        name = signal.name
        # 这里不做异常治理了, 先假设实现都合乎理性.
        # todo: 未来好像可以考虑频率治理. 用 janus_queue 做有 maxsize 的优先级队列来限频?
        if signal.is_stale():
            return None
        broadcasted = 0
        for nucleus in self._faculties.values():
            if name in nucleus.signals():
                nucleus.on_signal(signal)
                broadcasted += 1
        self._logger.debug("%s receive signal and send to %d nuclei", self._log_prefix, broadcasted)
        return None

    def set_attention(self, attention: Attention, reason: str = 'set new attention') -> None:
        # 加一个线程锁. 从逻辑上看, 这里本身都是同步逻辑, 加不加无所谓.
        # 考虑到未来 set attention 可能不止一个地方调用, 所以加一个 set.
        with self._set_attention_lock:
            if not self.is_running():
                self._logger.error("%s set attention but not running: %r", self._log_prefix, attention)
                return None
            elif self._paused:
                # paused 仍然可以设置. 这是系统指令.
                pass
            # 系统指令, 立刻生效.
            if self._current_attention is not None and not self._current_attention.is_aborted():
                # 多做一次 abort 检查, 用来做容错.
                self._current_attention.abort(reason)
            self._current_attention = attention
            while not self._pop_new_attention_queue.empty():
                attention = self._pop_new_attention_queue.get_nowait()
                # 通常不全部都 aborted 了.
                if not attention.is_aborted():
                    attention.abort(reason)
            self._pop_new_attention_queue.put_nowait(self._current_attention)
            self._logger.info("%s set attention %r", self._log_prefix, attention)
            return None

    def set_impulse(self, impulse: Impulse) -> None:
        """直接用 impulse 创建 attention"""
        if impulse.is_stale():
            # 仍然做一次校验.
            return None
        attention = self._create_attention_from_impulse(impulse)
        self.set_attention(attention, 'set new impulse')
        return None

    def _rank_nuclei(self, best_impulse: Impulse = None) -> Impulse | None:
        best_impulse = best_impulse
        for nucleus in self._faculties.values():
            impulse = nucleus.peek()
            # 加一行代码防蠢.
            impulse.source = nucleus.name()
            # 是否 impulse 也要做一个过期?
            if impulse is None:
                continue
            elif best_impulse is None:
                best_impulse = impulse
                continue
            elif impulse.priority > best_impulse.priority:
                best_impulse = impulse
                continue
            elif impulse.priority == best_impulse.priority and impulse.strength > best_impulse.strength:
                best_impulse = impulse
            else:
                continue
        return best_impulse

    async def _wait_pop_attention(self, timeout: float = 1.0) -> Attention:
        """等待下一帧的 attention 关键帧. """
        if timeout <= 0:
            timeout = 1.0
        timeleft = Timeleft(timeout)
        while self.is_running() and timeleft.alive():
            attention = await asyncio.wait_for(self._pop_new_attention_queue.get(), timeout=timeleft.left())
            if attention.is_aborted():
                continue
            return attention
        raise asyncio.TimeoutError()

    def pause(self, toggle: bool) -> None:
        if not self.is_running():
            return
        self._paused = toggle
        if toggle:
            if self._current_attention is not None:
                # 通过这种方式 stop the attention.
                self._current_attention.abort('paused')

            self._unpaused_event.clear()
        else:
            self._unpaused_event.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unpaused_event.set()

    async def __anext__(self) -> Attention:
        """需要实现一个特别稳定的流程."""
        while self.is_running():
            try:
                # 如果是 paused, 阻塞等到释放.
                # 关闭时也会释放.
                if self._paused:
                    await self._unpaused_event.wait()
                # 理论上 last popped attention 永远是被处理完, 才可能吐出一个 attention.
                # 一个 mindflow 只能吐出一个 attention. 用来做单一状态管理.
                # 不过仍然做一层冗余, 好像没有什么代价, 但会更安心.
                if self._last_popped_attention is not None:
                    if not self._last_popped_attention.is_aborted():
                        await self._last_popped_attention.wait_closed()
                self._last_popped_attention = None

                # 如果进入等待的瞬间没有任何 attention, 最常见的就是一大堆的 Impulse 被压抑住了.
                # 而被压抑住的 attention 结束时, 反而没有新的 impulse 进入.
                if self._current_attention is None:
                    if impulse := self._rank_nuclei():
                        # 强行设置 Impulse, 不再进行排序.
                        self.set_impulse(impulse)
                attention = await self._wait_pop_attention(1.0)
                if attention.is_aborted():
                    continue
                self._last_popped_attention = attention
                return attention
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                continue
        raise StopAsyncIteration

    async def __aenter__(self):
        if self._starting:
            return
        self._starting = True
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
        self._started = True

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        self._started = False
        self._starting = False
        self._unpaused_event.set()
        if self._current_attention is not None:
            self._current_attention.abort('mindflow stopped')
            self._current_attention = None
        while not self._pop_new_attention_queue.empty():
            attention = self._pop_new_attention_queue.get_nowait()
            if not attention.is_aborted():
                attention.wait_closed("mindflow stopped")
        faculties = list(self._faculties.values())
        self._faculties.clear()
        close_all = []
        for nucleus in faculties:
            close_all.append(nucleus.__aexit__(exc_type, exc_val, exc_tb))
        result = await asyncio.gather(*close_all, return_exceptions=True)
        idx = 0
        for r in result:
            if isinstance(r, Exception):
                self._logger.error("%s failed to stop nucleus %r: %s", self._log_prefix, faculties[idx], r)
            idx += 1
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
