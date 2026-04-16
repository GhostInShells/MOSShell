import asyncio
import datetime
from typing import Callable, Dict, List, Optional

from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.core.concepts.mindflow import Impulse, Signal, MindPulse, Mindflow, InputSignal
from ghoshell_container import BootstrapProvider, Provider, IoCContainer

__all__ = [
    "Mindflow", 'MindPulse', 'Signal', 'InputSignal', 'Impulse',
    "MindflowBus",
    'PriorityMindPulse',
    'MindflowBusProvider', 'PriorityMindPulseProvider',
    'default_mindflow',
]


class PriorityMindPulse(MindPulse):

    def __init__(
            self,
            pulse_name: str,
            description: str,
            signals: List[str],
            instruction: str = "",
            max_size: int = 10,
            logger: LoggerItf | None = None,
    ):
        self._name = pulse_name
        self._description = description
        self._receiving = signals
        self._instruction = instruction
        self._max_size = max_size
        self._buffer: List[Signal] = []
        self._notify_cb: Optional[Callable[[Impulse], None]] = None
        self._bus_cb: Optional[Callable[[Signal], None]] = None
        self._logger = logger or get_moss_logger()
        self._lock = asyncio.Lock()
        self._started = False
        self._cached_impulse: Impulse | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def receiving(self) -> List[str]:
        return self._receiving

    def with_bus(self, impulse_notify: Callable[[Impulse], None], signal_bus: Callable[[Signal], None]) -> None:
        self._notify_cb = impulse_notify
        self._bus_cb = signal_bus

    def on_signal(self, signal: Signal):
        """
        接收信号，异常处理机制确保不中断总线。
        """
        try:
            if signal.is_stale():
                self._logger.debug(f"信号 {signal.name} 已过期，丢弃。")
                return
            if self._event_loop is None:
                return

            # 异步处理入队，防止阻塞信号分发
            self._event_loop.create_task(self._enqueue(signal))
        except Exception as e:
            self._logger.error(f"处理信号时发生异常: {e}")

    async def _enqueue(self, signal: Signal):
        if signal.is_stale():
            return
        self._buffer.append(signal)
        # 按优先级从大到小排序
        self._buffer.sort(key=lambda s: s.priority, reverse=True)
        # 超过容量则丢弃低优信号
        if len(self._buffer) > self._max_size:
            self._buffer.pop()

        self._logger.debug(f"信号入队成功。当前 Buffer 大小: {len(self._buffer)}")

        # 通知 Mindflow 产生了新脉冲（peek 模式）
        if self._buffer[0] is signal:
            stale_timeout = 0
            if signal.stale_timeout > 0:
                now = datetime.datetime.now()
                stale_timeout = signal.stale_timeout - (now.timestamp() - signal.created_at.timestamp())
            self._cached_impulse = Impulse(
                belongs_to=self._name,
                trace=signal.trace,
                priority=signal.priority,
                description=signal.description,
                messages=signal.messages,
                instruction=self._instruction,
                stale_timeout=stale_timeout
            )
            if self._notify_cb:
                self._notify_cb(self._cached_impulse)
        return

    def peek(self) -> Optional[Impulse]:
        if self._cached_impulse is not None:
            if not self._cached_impulse.is_stale():
                return self._cached_impulse
        if not self._buffer: return None

        # 清理掉 buffer 顶部的过期信号
        while self._buffer and self._buffer[0].is_stale():
            self._logger.info(f"清理过期信号: {self._buffer[0].name}")
            self._buffer.pop(0)

        if not self._buffer: return None
        top = self._buffer[0]
        return Impulse(
            belongs_to=self._name,
            trace=top.trace,
            priority=top.priority,
            description=top.description,
            messages=top.messages,
            instruction=self._instruction
        )

    def pop_impulse(self) -> Optional[Impulse]:
        if not self._buffer: return None
        top_signal = self._buffer.pop(0)
        return Impulse(
            belongs_to=self._name,
            priority=top_signal.priority,
            messages=top_signal.messages,
            instruction=self._instruction
        )

    def supress(self, other: Impulse):
        self._logger.info(f"被 {other.belongs_to} (优先级:{other.priority}) 压制。")
        # 此处可以根据业务逻辑实现衰减（Decay）或重新调度

    async def start(self):
        if self._started:
            return
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        self._logger.info(f"脉冲节点 {self._name} 启动。")

    async def stop(self):
        self._logger.info(f"脉冲节点 {self._name} 正在关闭...")


# --- 具体实现：Mindflow 总线 ---

class MindflowBus(Mindflow):
    def __init__(
            self,
            *mind_pulses: MindPulse,
            logger: LoggerItf | None = None,
    ):
        self._pulses: Dict[str, MindPulse] = {}
        self._logger = logger or get_moss_logger()
        self._impulse_event = asyncio.Event()
        self._prior_impulse: Impulse | None = None
        self._wait_impulse_futures: dict[asyncio.Future[Impulse], int] = {}
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._listening_pulse_map: dict[str, set[MindPulse]] = {}
        # 完成初始化注册.
        for mind_pulse in mind_pulses:
            self.with_pulse(mind_pulse)
        self._log_prefix = "<MindflowBus>"

    def with_pulse(self, pulse: MindPulse):
        pulse.with_bus(self._on_inner_impulse, self.on_signal)
        self._pulses[pulse.name()] = pulse
        for signal_name in pulse.receiving():
            if signal_name not in self._listening_pulse_map:
                self._listening_pulse_map[signal_name] = set()
            self._listening_pulse_map[signal_name].add(pulse)
        return self

    def pulses(self) -> Dict[str, MindPulse]:
        return self._pulses

    def context(self) -> str:
        """
        面向大模型的上下文格式化。
        """
        lines = []
        for p in self._pulses.values():
            imp = p.peek()
            if imp and imp.description:
                lines.append(f'  <pulse name="{p.name()}" priority="{imp.priority}">')
                lines.append(f'    <desc>{p.description()}</desc>')
                lines.append(f'    <impulse>{imp.description}</impulse>')
                if imp.instruction:
                    lines.append(f'    <instruction>{imp.instruction}</instruction>')
                lines.append(f'  </pulse>')

        if not lines:
            return "<mindflow_context status=\"clear\" />"

        return "<mindflow_context>\n" + "\n".join(lines) + "\n</mindflow_context>"

    def on_signal(self, signal: Signal):
        """
        信号分发路由。
        """
        if signal.name not in self._listening_pulse_map:
            self._logger.warning(f"发现未路由信号: {signal.name}")
            return
        for p in self._listening_pulse_map[signal.name]:
            p.on_signal(signal)

    def set_impulse(self, impulse: Impulse):
        """
        MindPulse 回调，唤醒正在等待的 wait_impulse。
        """
        # todo: 日志都加上前缀, 然后改成英文.
        self._logger.info(f"探测到新脉冲: {impulse.belongs_to} (优先级:{impulse.priority})")
        self._prior_impulse = impulse
        # 直接设置所有的 wait future done.
        for future in self._wait_impulse_futures.keys():
            future.set_result(impulse)

    def _on_inner_impulse(self, impulse: Impulse):
        # 提取 items 到 list，避免遍历时字典尺寸发生变化
        for future, priority in list(self._wait_impulse_futures.items()):
            # 等于 priority 也有打断效果.
            if future.done():
                continue
            if impulse.priority >= priority:
                future.set_result(impulse)

    def _check_running(self):
        if self._event_loop is None or not self._event_loop.is_running():
            raise RuntimeError(f"{self._log_prefix} MindflowBus is not running.")

    def wait_impulse(self, *, priority: int = -1, wait_new: bool = False) -> asyncio.Future[Impulse]:
        self._check_running()

        # --- 关键检查：如果当前已有更高优脉冲，直接返回 ---
        if not wait_new:
            best_imp, best_p = self._peek_best_impulse(priority)  # 封装一下你 pop 里的寻找逻辑
            if best_imp:
                fut = self._event_loop.create_future()
                fut.set_result(best_imp)
                return fut
        # ---------------------------------------------

        future = self._event_loop.create_future()
        self._wait_impulse_futures[future] = priority
        future.add_done_callback(self._remove_done_wait_impulse_future)
        return future

    def _remove_done_wait_impulse_future(self, future: asyncio.Future[Impulse]):
        # 似乎不要加锁.
        if future in self._wait_impulse_futures:
            del self._wait_impulse_futures[future]

    def pop_impulse(self, pulse_name: str | None = None) -> Optional[Impulse]:
        if pulse_name:
            return self._pulses[pulse_name].pop_impulse() if pulse_name in self._pulses else None
        # 通过 on impulse
        if self._prior_impulse is not None:
            impulse = self._prior_impulse
            self._prior_impulse = None
            return self._suppress_all(impulse)

        # 找全局最优 pop
        best_imp, best_p = self._peek_best_impulse()
        return self._suppress_all(best_p.pop_impulse()) if best_p else None

    def _peek_best_impulse(self, priority: int = -1) -> tuple[Impulse | None, MindPulse | None]:
        best_impulse = None
        best_p = None
        for p in self._pulses.values():
            imp = p.peek()
            if not imp:
                continue
            # 优先级更高才有入场券.
            elif imp.priority > priority:
                best_impulse = imp
                best_p = p
                priority = imp.priority
                continue
            # 如果是两个 impulse 做比较, 则可采取时序比较.
            elif best_impulse and imp > best_impulse:
                best_impulse = imp
                best_p = p
                priority = imp.priority
        return best_impulse, best_p

    def _suppress_all(self, impulse: Impulse) -> Impulse | None:
        if impulse is None:
            return None
        for p in self._pulses.values():
            if impulse.belongs_to == p.name():
                # 不包括自己.
                continue
            imp = p.peek()
            if imp is not None:
                # 告知被 supress 了.
                p.supress(imp)
        return impulse

    async def __aenter__(self):
        self._event_loop = asyncio.get_running_loop()
        self._logger.info("Mindflow 核心启动中...")
        for p in self._pulses.values():
            try:
                await p.start()
            except Exception as e:
                self._logger.error(f"启动节点 {p.name()} 失败: {e}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        futures = list(self._wait_impulse_futures.keys())
        self._wait_impulse_futures.clear()
        for future in futures:
            future.cancel("closed")

        self._logger.info("Mindflow 核心关闭中...")
        for p in self._pulses.values():
            try:
                await p.stop()
            except Exception as e:
                self._logger.error(f"关闭节点 {p.name()} 时发生异常: {e}")


class MindflowBusProvider(Provider[Mindflow]):

    def __init__(
            self,
            *mind_pulses: MindPulse,
    ):
        self._pulses = list(mind_pulses)

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> Mindflow:
        logger = con.get(LoggerItf)
        mindflow = MindflowBus(
            *self._pulses,
            logger=logger,
        )
        return mindflow


class PriorityMindPulseProvider(BootstrapProvider):
    """
    方便通过 manifest 对 Mindflow 进行注册.
    """

    def __init__(
            self,
            *pulses: PriorityMindPulse,
    ):
        self._pulses = list(pulses)

    def singleton(self) -> bool:
        return True

    def contract(self):
        # 返回自身, 保证全局唯一注册, 可被覆盖.
        return type(self)

    def factory(self, con: IoCContainer):
        return self

    def bootstrap(self, container: IoCContainer) -> None:
        # container 启动的时候, 对 mindflow 进行注册.
        mindflow = container.force_fetch(Mindflow)
        for pulse in self._pulses:
            mindflow.with_pulse(pulse)


def default_mindflow(container: IoCContainer) -> Mindflow:
    logger = container.get(LoggerItf)
    return MindflowBus(
        PriorityMindPulse(
            pulse_name=InputSignal.signal_name(),
            signals=[InputSignal.signal_name()],
            logger=logger,
            description='',
            instruction='',
        ),
        logger=logger,
    )
