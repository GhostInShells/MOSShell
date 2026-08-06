"""ShellContext — Shell 上下文组装面的抽象契约.

Shell 初始化并治理生命周期, 为不同消费者 (mcp / ghost runtime / articulate)
提供统一的上下文片段:

  第一类 — 静态只读透传: meta_instruction / static_messages / dynamic_messages
    保留在 MOSShell, 不进入此 ABC.

  第二类 — channel 游标 (warm+hot 治理): snapshot / ack
    帧级 compare-and-emit, 对比基准为最后一次 ACK 的投影.

  第三类 — interpreter 游标 (执行事件): drain / status / wait_interpreter_done
    订阅 Tracer Protocol, 收集 task 完成 / interpreter 停止事件.

三类共享生命周期 (async context manager) 与线程安全, 各自数据路径独立.
"""

from __future__ import annotations

import datetime
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.core.concepts.channel import ChannelMeta
from ghoshell_moss.core.concepts.command import CommandTask, CommandTaskResult
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.message import Message

__all__ = [
    # Tracer (从 shell.py 迁移)
    "Tracer",
    # warm/hot 数据契约
    "WarmUnit",
    "WarmDelta",
    "ContextSnapshot",
    # 执行事件契约
    "ShellEvent",
    "TaskDone",
    "InterpreterStopped",
    "InterpreterStatus",
    # 投影
    "project_events",
    # ABC
    "ShellContext",
]


# ── Tracer Protocol (从 shell.py 提取, shell.py 重导出保持向前兼容) ──


class Tracer(Protocol):
    """对 shell 运行时的观察模块. shell 关键生命周期节点回调它.

    fire and forget: shell 遍历 tracers 时, is_closed() 或 not is_running() 都会跳过,
    异常被 shell 捕获并记 log, 不影响主流程.

    实现要点:
    - 所有 on_xxx 方法必须线程安全 (可能被 shell 线程 / channel 线程调用).
    - 方法体保持轻量, 不阻塞 shell 主流程.
    - is_closed()=True 是终态, 表示 tracer 已终结; shell 不再 fire.
    """

    def is_running(self) -> bool:
        """是否处于活跃接收状态. False 时 shell 跳过本次 fire (可用于暂停)."""
        ...

    def is_closed(self) -> bool:
        """是否已关闭. True 时 shell 永久跳过, 未来可能被 GC."""
        ...

    def on_task_pushed(self, task: CommandTask) -> None:
        """一个 command task 被 push 到 shell 时回调."""
        ...

    def on_task_done(self, task: CommandTask) -> None:
        """一个 command task 完成时回调 (成功 / 失败 / 取消 都算 done)."""
        ...

    def on_interpreter_stopped(self, interpreter: Interpreter) -> None:
        """一个 interpreter close 完成时回调.

        可从 ``interpreter.exception()`` 拿到编译期异常 (INTERPRET_ERROR),
        从 ``interpreter.interpretation()`` 拿到最终 Interpretation 快照.
        """
        ...


# ── warm/hot 数据契约 ──────────────────────────────────────


class WarmUnit(str, Enum):
    """warm 数据单元 — 字段组粒度. 每个单元独立 hash/delta/降级."""

    DESC_INSTRUCTION = "desc_instruction"
    STATES = "states"
    INTERFACE = "interface"


@dataclass(frozen=True)
class WarmDelta:
    """一次 warm 变更事件.

    kind: add=整 channel 出现 / update=某单元变更 / remove=channel 移除 (tombstone).
    block 是 channel 包裹的渲染文本. 历史里同 path 后块字段级覆盖前块.
    """

    kind: Literal["add", "update", "remove"]
    path: str
    unit: WarmUnit | None
    block: str

    def to_messages(self) -> list[Message]:
        return [Message.new(tag="", timestamp=False).with_content(self.block)]


@dataclass(frozen=True)
class ContextSnapshot:
    """一次 context monitor 快照 — 冻帧, ack 的显式令牌.

    调用方用 warm_deltas 拼入 durable 段, hot_messages 拼入 ephemeral 尾部,
    落库后把同一快照交给 ``monitor.ack(snapshot)`` 推进基线.

    warm_deltas 已按 path 排序, 可直接使用.
    """

    warm_deltas: tuple[WarmDelta, ...]
    hot_messages: tuple[Message, ...]

    # ack 验证与落基线所需的内态.
    _ack_id: int
    _frame_projection: dict[str, dict[WarmUnit, str]]


# ── 执行事件契约 ────────────────────────────────────────────


class ShellEvent(ABC):
    """一次 shell 观察事件. 通过 as_message() 投影为大模型可读的消息."""

    @abstractmethod
    def as_message(self) -> list[Message]:
        ...


class TaskDone(ShellEvent):
    """一个 command task 完成事件. 最小状态: 只持 CommandTaskResult 指针 + task 侧的成/败/取消判据."""

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
        if self.result.observe:
            attrs = {'command': self.result.caller} if self.result.caller else None
            return [Message.new(tag='result', attributes=attrs).with_content('(no output)')]
        return []


class InterpreterStopped(ShellEvent):
    """一个 interpreter close 事件. 只在有 parsing_exception 时进 buffer."""

    def __init__(self, exception: str):
        self.exception = exception

    def as_message(self) -> list[Message]:
        return [Message.new(tag='interpret_error').with_content(self.exception)]


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


# ── 投影 ────────────────────────────────────────────────────


def _now_short() -> str:
    return datetime.datetime.now().strftime('%H:%M:%S')


def project_events(events: list[ShellEvent], status: InterpreterStatus) -> list[Message]:
    """把 drain 出的事件列表 + 当下 status 投影成 list[Message].

    分桶规则:
    - 有 payload 的 TaskDone (含 failed 带 errmsg / observe=True 占位): 逐条 emit, 保留身份.
    - 空 payload 的 TaskDone: 按 success / cancelled / failed 计数聚合成一条 <shell_tally>.
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
        if isinstance(ev, TaskDone):
            if ev.success:
                success_count += 1
            elif ev.cancelled:
                cancelled_count += 1
            else:
                failed_count += 1

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


# ── ShellContext ABC ────────────────────────────────────────


class ShellContext(ABC):
    """Shell 上下文组装面.

    Shell 通过 ``new_from_shell(shell)`` 初始化, 注册为 shell 的 Tracer,
    治理生命周期. 消费者通过 channel cursor (snapshot/ack) 和 interpreter
    cursor (drain/status/wait_interpreter_done) 获取统一的上下文片段.
    """

    def __init__(self, shell: "MOSShell") -> None:
        self._shell = shell

    # ── factory ───────────────────────────────────────────

    @classmethod
    def new_from_shell(cls, shell: "MOSShell") -> Self:
        ctx = cls(shell)
        shell.add_tracer(ctx)
        return ctx

    # ── lifecycle ─────────────────────────────────────────

    @abstractmethod
    async def close(self) -> None:
        """标记为 closed, 唤醒所有 pending waiter."""

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    # ── channel cursor (warm+hot) ─────────────────────────

    @abstractmethod
    def snapshot(self, metas: dict[str, ChannelMeta]) -> ContextSnapshot:
        """对当前 metas 帧做 compare-and-emit, 产出一帧冻快照.

        对比基准是最后一次 ACK 的投影. ACK 前多次 snapshot 会重算同一批 delta.
        """

    @abstractmethod
    def ack(self, snapshot: ContextSnapshot) -> None:
        """确认快照的 warm delta 已进入历史, 推进基线.

        快照的 _ack_id 不大于上次已 ack 帧号时 no-op (已过时或重复 ack).
        """

    # ── interpreter cursor (执行事件) ──────────────────────

    @abstractmethod
    def drain(self) -> list[ShellEvent]:
        """取空事件缓冲区并返回所有事件. 语义为一次性消费."""

    @abstractmethod
    def status(self) -> InterpreterStatus:
        """读一次 shell 当下的 interpreter 活指针状态. 同步方法."""

    @abstractmethod
    async def wait_interpreter_done(self) -> InterpreterStatus:
        """等待当前正在运行的 interpreter 停止, 然后返回 status.

        调用时若无 interpreter running, 立即返回 status.
        """
