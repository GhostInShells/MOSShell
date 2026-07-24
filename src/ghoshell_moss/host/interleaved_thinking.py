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
    """一个 command task 完成事件. 最小状态: 只持 CommandTaskResult 指针.

    K9 兜底: 空 outcome (result=None + messages=[]) 会被 CommandTaskResult.as_messages()
    过滤成空列表, 存在性蒸发. TaskDone.as_message 在这种情况下给出一个合法非空包裹,
    让 "这条命令跑过了" 的存在性不丢失.
    """

    def __init__(self, result: CommandTaskResult):
        self.result = result

    def as_message(self) -> list[Message]:
        msgs = self.result.as_messages(name=self.result.caller)
        if len(msgs) > 0:
            return msgs
        # K9 兜底: 存在性 vs payload 丰度解耦
        attrs = {'command': self.result.caller} if self.result.caller else None
        return [Message.new(tag='result', attributes=attrs).with_content('(no output)')]


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
        return [Message.new(tag='shell_status').with_content('\n'.join(lines))]


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
        with self._buffer_lock:
            self._buffer.append(TaskDone(result))
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
