"""InterleavedThinkingToolset — 服务于 interleaved thinking 场景的 shell 观察器.

订阅 shell 的 Tracer Protocol 钩子, 收集 task 完成 / interpreter 停止事件, 对外提供
四原语: buffered / drain / status / wait_interpreter_done.

生命周期:
    async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
        events = toolset.drain()
        status = toolset.status()
        await toolset.wait_interpreter_done()

线程模型: shell 的 fire 是同步直调, 可能来自 channel 线程或 asyncio 线程. Toolset
用 threading.Lock 保护 buffer / waiters, 用 ThreadSafeEvent 唤醒 asyncio waiter.

线程安全纪律:
- 锁临界区极小, 只做 dict/list 原子操作.
- 两把锁独立获取释放, 绝不嵌套.
- event.set() 和 shell 递归调用一律在锁外.
- 事件投影 (as_message) 延迟到 drain/buffered 之后, 不在 shell 线程做.
"""

import threading
import datetime
from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import Self
from pydantic import BaseModel, Field

from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.concepts.command import CommandTask, CommandTaskResult
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.message import Message

__all__ = [
    'ShellEvent', 'TaskDone', 'InterpreterStopped',
    'InterpreterStatus',
    'InterleavedThinkingToolset',
    'project_events',
]


# --- events --- #


class ShellEvent(ABC):
    """一次 shell 观察事件. 通过 as_message() 投影为大模型可读的消息.

    扩展点: 加字段就改子类 __init__, 加事件类型就加新子类; ShellEvent + as_message() 接口稳定.
    """

    @abstractmethod
    def as_message(self) -> list[Message]:
        ...


class TaskDone(ShellEvent):
    """一个 command task 完成事件. 最小状态: 只持 CommandTaskResult 指针 + task 侧的成/败/取消判据.

    K9 作用域收窄 (对齐 Interpretation.on_done_task):
    - 有 payload → 返回 payload.
    - 空 payload + observe=True (含 is_critical 自动 observe) → 给身份占位, 存在性不蒸发.
    - 空 payload + observe=False → 返回 [], 由投影层聚合成计数. "连身份都不必给" (K8).
    """

    def __init__(
            self,
            result: CommandTaskResult,
            *,
            success: bool,
            cancelled: bool,
    ):
        self.result = result
        self.success = success
        self.cancelled = cancelled

    @property
    def failed(self) -> bool:
        return not self.success and not self.cancelled

    @property
    def observe(self) -> bool:
        return self.result.observe

    def as_message(self) -> list[Message]:
        msgs = self.result.as_messages(name=self.result.caller)
        if len(msgs) > 0:
            return msgs
        # 空 payload 分支
        if self.result.observe:
            # observe=True (含 is_critical 自动 observe): K9 占位, 存在性不蒸发
            attrs = {'command': self.result.caller} if self.result.caller else None
            return [Message.new(tag='result', attributes=attrs).with_content('(no output)')]
        # observe=False + 空: 交给投影层聚合成计数
        return []


class InterpreterStopped(ShellEvent):
    """一个 interpreter close 事件. 只在有 parsing_exception 时进 buffer —
    清洁停止在 wait_interpreter_done 语义里表现, 不需要生成 event.
    """

    def __init__(self, exception: str):
        self.exception = exception

    def as_message(self) -> list[Message]:
        return [Message.new(tag='interpret_error').with_content(self.exception)]


# --- status snapshot --- #


class InterpreterStatus(BaseModel):
    """当下 shell 里 interpreter 的活指针快照 (同步读一次)."""

    running: bool = Field(description="是否有 interpreter 处于 running 状态")
    parsing_exception: Optional[str] = Field(default=None, description="当前 interpreter 的解析异常")
    ongoing_callers: list[str] = Field(
        default_factory=list,
        description="managing_tasks 中未 done 的 command caller names",
    )

    def as_message(self) -> list[Message]:
        lines = [f"running: {self.running}"]
        if self.parsing_exception:
            lines.append(f"parsing_exception: {self.parsing_exception}")
        if self.ongoing_callers:
            lines.append(f"ongoing: {', '.join(self.ongoing_callers)}")
        return [
            Message.new(
                tag='shell_status',
                attributes={'at': _now_short()},
            ).with_content('\n'.join(lines))
        ]


def _now_short() -> str:
    """shell 观察节点用的简化时间戳 HH:MM:SS (本地时区).

    时间是第一公民, 但 shell 层只需要给 "这次观察的相对秩序" 打点; 完整 datetime
    是 command 层的责任 (message.attributes.at 已带). 8 字符 tally-friendly.
    """
    return datetime.datetime.now().strftime('%H:%M:%S')


def project_events(events: list[ShellEvent], status: InterpreterStatus) -> list[Message]:
    """把 drain 出的事件列表 + 当下 status 投影成 list[Message].

    分桶规则 (对齐 K8/K9 payload 分层, 参考 Interpretation.on_done_task):
    - 有 payload 的 TaskDone (含 failed 带 errmsg / observe=True 占位): 逐条 emit, 保留身份.
    - 空 payload 的 TaskDone: 按 success / cancelled / failed 计数聚合成一条 <shell_tally>.
      "连身份都不必给" (K8) — 25年小红帽实测: 一轮 logos 可含数百 task, 批量 cancel
      时 caller name 是纯 token 开销. 是否需要 debug 身份由 is_critical/observe=True
      flag 精准闸口, 不走计数分桶.
    - InterpreterStopped / 其他 ShellEvent: 走各自的 as_message().
    """
    messages: list[Message] = []
    success_count = 0
    cancelled_count = 0
    failed_count = 0

    for ev in events:
        projected = ev.as_message()
        if projected:
            messages.extend(projected)
            continue
        # 空 payload 的 TaskDone → 计数聚合
        if isinstance(ev, TaskDone):
            if ev.success:
                success_count += 1
            elif ev.cancelled:
                cancelled_count += 1
            else:
                failed_count += 1
        # 非 TaskDone 的空 projected 忽略 (InterpreterStopped 无异常时不进 buffer,
        # 到不了这里; 其他 ShellEvent 子类若刻意返回空, 视作显式静默)

    if success_count or cancelled_count or failed_count:
        tally_lines = []
        if success_count:
            tally_lines.append(f"success: {success_count}")
        if cancelled_count:
            tally_lines.append(f"cancelled: {cancelled_count}")
        if failed_count:
            tally_lines.append(f"failed: {failed_count}")
        messages.append(
            Message.new(
                tag='shell_tally',
                attributes={'at': _now_short()},
            ).with_content('\n'.join(tally_lines))
        )

    messages.extend(status.as_message())
    return messages


# --- toolset --- #


class InterleavedThinkingToolset:
    """跨-interpreter 的 shell 观察器, 服务于 interleaved thinking 场景.

    满足 Tracer Protocol (duck-type). 通过 ``new_from_shell(shell)`` 工厂注册到 shell.
    """

    def __init__(self, shell: MOSShell):
        self._shell = shell
        self._buffer: list[ShellEvent] = []
        self._buffer_lock = threading.Lock()
        self._waiters: set[ThreadSafeEvent] = set()
        self._waiters_lock = threading.Lock()
        self._running = True
        self._closed = False

    @classmethod
    def new_from_shell(cls, shell: MOSShell) -> Self:
        """构造 toolset 并注册到 shell.add_tracer.

        推荐用法: ``async with InterleavedThinkingToolset.new_from_shell(shell) as ts:``
        """
        toolset = cls(shell)
        shell.add_tracer(toolset)
        return toolset

    # --- Tracer Protocol impl (shell 线程调用, 极短) --- #

    def is_running(self) -> bool:
        return self._running

    def is_closed(self) -> bool:
        return self._closed

    def on_task_pushed(self, task: CommandTask) -> None:
        # 无副作用: status() 从 shell.interpreting() 读活指针, tracer 不重复维护映射.
        pass

    def on_task_done(self, task: CommandTask) -> None:
        result = task.task_result()
        if result is None:
            return
        # 分桶判据在 shell 线程一次读定, 避免投影时再回访 task 引发跨线程读
        event = TaskDone(
            result,
            success=task.success(),
            cancelled=task.cancelled(),
        )
        with self._buffer_lock:
            self._buffer.append(event)
        # 不 wake waiter: wait_interpreter_done 语义是等 interpreter stop, 不是等 task done.
        # 未来若需要 wait-any-event (全双工推送), 用独立的 waiter 集合, 别复用这个.

    def on_interpreter_stopped(self, interpreter: Interpreter) -> None:
        exc = interpreter.exception()
        if exc is not None:
            with self._buffer_lock:
                self._buffer.append(InterpreterStopped(str(exc)))
        # 无论有无 exception 都唤醒 wait — 清洁停止也是 "interpreter 到 idle" 信号.
        self._wake_waiters()

    def _wake_waiters(self) -> None:
        # 锁内只 snapshot, 锁外 fire; ThreadSafeEvent.set() 内部 call_soon_threadsafe, 非阻塞.
        with self._waiters_lock:
            waiters = list(self._waiters)
        for ev in waiters:
            ev.set()

    # --- 四原语 (调用方线程) --- #

    def buffered(self) -> list[ShellEvent]:
        """返回缓冲区所有事件的快照, 不 drain. 用于 debug / UI."""
        with self._buffer_lock:
            return list(self._buffer)

    def drain(self) -> list[ShellEvent]:
        """取空缓冲区并返回所有事件. 语义为一次性消费."""
        with self._buffer_lock:
            events = self._buffer
            self._buffer = []
        return events

    def status(self) -> InterpreterStatus:
        """读一次 shell 当下的 interpreter 活指针状态. 同步方法."""
        interp = self._shell.interpreting()
        if interp is None:
            return InterpreterStatus(running=False)
        # 防御: managing_tasks 可能在读的瞬间被清 (close 竞态), copy 一次
        try:
            tasks = list(interp.managing_tasks().values())
        except Exception:
            tasks = []
        ongoing = [t.caller_name() for t in tasks if not t.done()]
        exc = interp.exception()
        return InterpreterStatus(
            running=interp.is_running(),
            parsing_exception=str(exc) if exc is not None else None,
            ongoing_callers=ongoing,
        )

    async def wait_interpreter_done(self) -> InterpreterStatus:
        """等待 "调用时刻正在运行的" interpreter (若有) 停止, 然后返回 status.

        语义:
        - 调用时若无 interpreter running, 立即返回 status.
        - 若有, 等到下一次 on_interpreter_stopped fire, 返回 status.
        - toolset close 时所有 pending waiter 会被立即唤醒 (返回当下 status).
        """
        interp = self._shell.interpreting()
        if interp is None or not interp.is_running():
            return self.status()

        ev = ThreadSafeEvent()
        with self._waiters_lock:
            self._waiters.add(ev)
        try:
            await ev.wait()
        finally:
            with self._waiters_lock:
                self._waiters.discard(ev)
        return self.status()

    # --- lifecycle --- #

    async def close(self) -> None:
        """标记 toolset 为 closed. shell 下次 fire 时会跳过. 唤醒所有 pending waiter."""
        if self._closed:
            return
        self._closed = True
        self._running = False
        with self._waiters_lock:
            waiters = list(self._waiters)
            self._waiters.clear()
        for ev in waiters:
            ev.set()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
