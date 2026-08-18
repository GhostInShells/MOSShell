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

import datetime
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias
from pydantic import BaseModel, Field
from typing_extensions import Self
import logging
from dateutil import tz

from ghoshell_moss.core.concepts.channel import ChannelMeta, ChannelFullPath
from ghoshell_moss.core.concepts.shell import MOSShell, Tracer
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.concepts.command import CommandTask
from ghoshell_moss.core.ctml.v1_0.prompts import ChannelMetaPrompter
from ghoshell_moss.message import Message, format_timestamp

__all__ = [
    'Tracer',
    'MShellContextFacade', 'MShellTrajectory',
    'TrajectoryFrame',
    'MShellEventTracer', 'MShellEvent', 'InterpreterStoppedEvent', 'ShellTaskDoneEvent',
    # Tracer (从 shell.py 迁移)
    # warm/hot 数据契约
    # "WarmUnit",
    # "WarmDelta",
    # "ContextSnapshot",
    # # 执行事件契约
    # "ShellEvent",
    # "TaskDone",
    # "InterpreterStopped",
    # "InterpreterStatus",
    # # 投影
    # "project_events",
    # # ABC
    # "ShellContext",
]

# # ── warm/hot 数据契约 ──────────────────────────────────────
#
#
# class WarmUnit(str, Enum):
#     """warm 数据单元 — 字段组粒度. 每个单元独立 hash/delta/降级."""
#
#     DESC_INSTRUCTION = "desc_instruction"
#     STATES = "states"
#     INTERFACE = "interface"
#
#
# @dataclass(frozen=True)
# class WarmDelta:
#     """一次 warm 变更事件.
#
#     kind: add=整 channel 出现 / update=某单元变更 / remove=channel 移除 (tombstone).
#     block 是 channel 包裹的渲染文本. 历史里同 path 后块字段级覆盖前块.
#     """
#
#     kind: Literal["add", "update", "remove"]
#     path: str
#     unit: WarmUnit | None
#     block: str
#
#     def to_messages(self) -> list[Message]:
#         return [Message.new(tag="", timestamp=False).with_content(self.block)]
#
#
# @dataclass(frozen=True)
# class ContextSnapshot:
#     """一次 context monitor 快照 — 冻帧, ack 的显式令牌.
#
#     调用方用 warm_deltas 拼入 durable 段, hot_messages 拼入 ephemeral 尾部,
#     落库后把同一快照交给 ``monitor.ack(snapshot)`` 推进基线.
#
#     warm_deltas 已按 path 排序, 可直接使用.
#     """
#
#     warm_deltas: tuple[WarmDelta, ...]
#     hot_messages: tuple[Message, ...]
#
#     # ack 验证与落基线所需的内态.
#     _ack_id: int
#     _frame_projection: dict[str, dict[WarmUnit, str]]
#
#
# # ── 执行事件契约 ────────────────────────────────────────────
#
#
# class ShellEvent(ABC):
#     """一次 shell 观察事件. 通过 as_message() 投影为大模型可读的消息."""
#
#     @abstractmethod
#     def as_message(self) -> list[Message]:
#         ...
#
#
# class TaskDone(ShellEvent):
#     """一个 command task 完成事件. 最小状态: 只持 CommandTaskResult 指针 + task 侧的成/败/取消判据."""
#
#     def __init__(
#             self,
#             result: CommandTaskResult,
#             *,
#             success: bool,
#             cancelled: bool,
#     ):
#         self.result = result
#         self.success = success
#         self.cancelled = cancelled
#
#     @property
#     def failed(self) -> bool:
#         return not self.success and not self.cancelled
#
#     @property
#     def observe(self) -> bool:
#         return self.result.observe
#
#     def as_message(self) -> list[Message]:
#         msgs = self.result.as_messages(name=self.result.caller)
#         if len(msgs) > 0:
#             return msgs
#         if self.result.observe:
#             attrs = {'command': self.result.caller} if self.result.caller else None
#             return [Message.new(tag='result', attributes=attrs).with_content('(no output)')]
#         return []
#
#
# class InterpreterStopped(ShellEvent):
#     """一个 interpreter close 事件. 只在有 parsing_exception 时进 buffer."""
#
#     def __init__(self, exception: str):
#         self.exception = exception
#
#     def as_message(self) -> list[Message]:
#         return [Message.new(tag='interpret_error').with_content(self.exception)]
#
#
# class InterpreterStatus(BaseModel):
#     """当下 shell 里 interpreter 的活指针快照 (同步读一次)."""
#
#     running: bool = Field(description="是否有 interpreter 处于 running 状态")
#     parsing_exception: Optional[str] = Field(default=None, description="当前 interpreter 的解析异常")
#     ongoing_callers: list[str] = Field(
#         default_factory=list,
#         description="managing_tasks 中未 done 的 command caller names",
#     )
#
#     def as_message(self) -> list[Message]:
#         lines = [f"running: {self.running}"]
#         if self.parsing_exception:
#             lines.append(f"parsing_exception: {self.parsing_exception}")
#         if self.ongoing_callers:
#             lines.append(f"ongoing: {', '.join(self.ongoing_callers)}")
#         return [
#             Message.new(
#                 tag='shell_status',
#                 attributes={'at': _now_short()},
#             ).with_content('\n'.join(lines))
#         ]
#
#
# # ── 投影 ────────────────────────────────────────────────────
#
#
# def _now_short() -> str:
#     return datetime.datetime.now().strftime('%H:%M:%S')
#
#
# def project_events(events: list[ShellEvent], status: InterpreterStatus) -> list[Message]:
#     """把 drain 出的事件列表 + 当下 status 投影成 list[Message].
#
#     分桶规则:
#     - 有 payload 的 TaskDone (含 failed 带 errmsg / observe=True 占位): 逐条 emit, 保留身份.
#     - 空 payload 的 TaskDone: 按 success / cancelled / failed 计数聚合成一条 <shell_tally>.
#     - InterpreterStopped / 其他 ShellEvent: 走各自的 as_message().
#     """
#     messages: list[Message] = []
#     success_count = 0
#     cancelled_count = 0
#     failed_count = 0
#
#     for ev in events:
#         projected = ev.as_message()
#         if projected:
#             messages.extend(projected)
#             continue
#         if isinstance(ev, TaskDone):
#             if ev.success:
#                 success_count += 1
#             elif ev.cancelled:
#                 cancelled_count += 1
#             else:
#                 failed_count += 1
#
#     if success_count or cancelled_count or failed_count:
#         tally_lines = []
#         if success_count:
#             tally_lines.append(f"success: {success_count}")
#         if cancelled_count:
#             tally_lines.append(f"cancelled: {cancelled_count}")
#         if failed_count:
#             tally_lines.append(f"failed: {failed_count}")
#         messages.append(
#             Message.new(
#                 tag='shell_tally',
#                 attributes={'at': _now_short()},
#             ).with_content('\n'.join(tally_lines))
#         )
#
#     messages.extend(status.as_message())
#     return messages
#
#
# # ── ShellContext ABC ────────────────────────────────────────
#
#
# class ShellContext(ABC):
#     """Shell 上下文
#
#     Shell 通过 ``new_from_shell(shell)`` 初始化, 注册为 shell 的 Tracer,
#     治理生命周期. 消费者通过 channel cursor (snapshot/ack) 和 interpreter
#     cursor (drain/status/wait_interpreter_done) 获取统一的上下文片段.
#     """
#
#     def __init__(self, shell: "MOSShell") -> None:
#         self._shell = shell
#
#     # ── factory ───────────────────────────────────────────
#
#     @classmethod
#     def new_from_shell(cls, shell: "MOSShell") -> Self:
#         ctx = cls(shell)
#         shell.add_tracer(ctx)
#         return ctx
#
#     # ── lifecycle ─────────────────────────────────────────
#
#     @abstractmethod
#     async def close(self) -> None:
#         """标记为 closed, 唤醒所有 pending waiter."""
#
#     @abstractmethod
#     async def __aenter__(self) -> Self:
#         ...
#
#     @abstractmethod
#     async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
#         ...
#
#     # ── channel cursor (warm+hot) ─────────────────────────
#
#     @abstractmethod
#     def snapshot(self, metas: dict[str, ChannelMeta]) -> ContextSnapshot:
#         """对当前 metas 帧做 compare-and-emit, 产出一帧冻快照.
#
#         对比基准是最后一次 ACK 的投影. ACK 前多次 snapshot 会重算同一批 delta.
#         """
#
#     @abstractmethod
#     def ack(self, snapshot: ContextSnapshot) -> None:
#         """确认快照的 warm delta 已进入历史, 推进基线.
#
#         快照的 _ack_id 不大于上次已 ack 帧号时 no-op (已过时或重复 ack).
#         """
#
#     # ── interpreter cursor (执行事件) ──────────────────────
#
#     @abstractmethod
#     def drain(self) -> list[ShellEvent]:
#         """取空事件缓冲区并返回所有事件. 语义为一次性消费."""
#
#     @abstractmethod
#     def status(self) -> InterpreterStatus:
#         """读一次 shell 当下的 interpreter 活指针状态. 同步方法."""
#
#     @abstractmethod
#     async def wait_interpreter_done(self) -> InterpreterStatus:
#         """等待当前正在运行的 interpreter 停止, 然后返回 status.
#
#         调用时若无 interpreter running, 立即返回 status.
#         """

MShellState: TypeAlias = Literal[
    'not_running',
    'paused',
    'running',
    'idle',
]


class MShellStatus(BaseModel):
    """shell 的运行状态"""
    state: MShellState = Field(description="state")
    commands_count: int = Field(
        default=0,
        description="command count",
    )
    executing: list[str] = Field(
        default_factory=list,
        description="executing command task callers",
    )
    need_observe: int = Field(
        default=0,
        description="need observe"
    )
    progresses: dict[str, str] = Field(
        default_factory=dict,
        description="command task with progress",
    )
    pending: int = Field(
        default=0,
        description="pending task count",
    )
    cancelled: int = Field(
        default=0,
        description="cancel task count",
    )
    completed: int = Field(
        default=0,
        description="completed task count",
    )
    failed: int = Field(
        default=0,
        description="failed task count",
    )

    @classmethod
    def new(cls, shell: MOSShell, interpreter: Interpreter | None = None) -> 'MShellStatus':
        """返回 shell 的运行时状态快照 """
        interpreter = interpreter or shell.interpreting()
        if not shell.is_running():
            state = 'not_running'
        elif shell.is_paused():
            state = 'paused'
        elif shell.is_idle():
            state = 'idle'
        else:
            state = 'running'
        if interpreter is None:
            return MShellStatus(state=state)

        completed = 0
        cancelled = 0
        failed = 0
        observe = 0
        progresses = {}
        pending = 0
        executing = []
        commands_count = 0
        for task in interpreter.managing_tasks().values():
            commands_count += 1
            if task.observe():
                observe += 1
            if task.cancelled():
                cancelled += 1
            elif e := task.exception():
                failed += 1
            elif task.done():
                completed += 1
            elif task.state == "executing":
                executing.append(task.caller_name())
                if task.progress:
                    progresses[task.caller_name()] = task.progress
            else:
                pending += 1

        return MShellStatus(
            state=state,
            need_observe=observe,
            completed=completed,
            cancelled=cancelled,
            failed=failed,
            pending=pending,
            executing=executing,
            progresses=progresses,
        )

    def description(self, tag: str = "status") -> str:
        if self.state != 'running':
            return f"<{tag} {self.state}/>"
        body_lines = ["Commands: %d" % self.commands_count]
        if self.cancelled:
            body_lines.append('cancelled: %d' % self.cancelled)
        if self.failed:
            body_lines.append('failed: %d' % self.failed)
        if self.completed:
            body_lines.append('completed: %d' % self.completed)
        if self.need_observe:
            body_lines.append('need_observe: %d' % self.need_observe)
        if self.executing:
            body_lines.append('executing: %d' % len(self.executing))
            if len(self.progresses) > 0:
                for caller, progress in self.progresses.items():
                    progress = progress.replace("\n", ' ')
                    body_lines.append(f" {caller}: {progress}")
            body_lines.append(f'last executing: {self.executing[-1]}')
        if self.pending:
            body_lines.append('pending: %d' % self.pending)
        body = "\n".join(body_lines)
        return f"<{tag} {self.state}>\n{body}\n</{tag}>"


class MShellContextFacade:
    """
    Shell 上下文获取函数的 Facade
    """

    def __init__(self, shell: "MOSShell", *selected_channels: ChannelFullPath):
        self.shell = shell
        _selected_channels = []
        _selected_channel_wildcards = []
        for channel_path_match_pattern in selected_channels:
            # 允许后缀通配符匹配.
            if channel_path_match_pattern.endswith('*'):
                _selected_channel_wildcards.append(channel_path_match_pattern[:-1])
            else:
                _selected_channels.append(channel_path_match_pattern)

        self._selected_channels = _selected_channels
        self._selected_channel_wildcards = _selected_channel_wildcards
        self._accept_all = len(self._selected_channels) == 0 and len(self._selected_channel_wildcards) == 0
        self._cached_channel_metas: dict[ChannelFullPath, ChannelMeta] = {}
        # 更新 channels.
        self._on_channel_metas_generation(shell.channel_metas())
        # 注册更新监听, 持有 discard 句柄.
        self._discard = shell.on_channel_metas_generation(self._on_channel_metas_generation)

    def discard(self) -> None:
        """注销 channel metas 重建监听. 幂等."""
        if self._discard is not None:
            self._discard()
            self._discard = None

    def _on_channel_metas_generation(self, metas: dict[ChannelFullPath, ChannelMeta]) -> None:
        """监听 channel_metas 被重新构建完. """
        if self._accept_all:
            self._cached_channel_metas = dict(sorted(metas.items(), key=lambda item: item[0]))
            return
        result = {}
        for path in self._selected_channels:
            if path in metas:
                result[path] = metas[path]
        if len(self._selected_channel_wildcards) > 0:
            for path in metas.keys():
                if path in result:
                    continue
                for wildcard in self._selected_channel_wildcards:
                    if path.startswith(wildcard):
                        result[path] = metas[path]
                        break
        # 确保排序符合预期.
        metas = dict(sorted(result.items(), key=lambda item: item[0]))
        self._cached_channel_metas = metas

    def meta_instruction(self) -> str:
        """返回 shell 自己的 meta instruction, 通常主要是 ctml (logos) 语法"""
        return self.shell.meta_instruction()

    def channel_metas(self, available_only: bool = True) -> dict[ChannelFullPath, ChannelMeta]:
        """有序排列 metas. """
        if available_only:
            return {key: meta for key, meta in self._cached_channel_metas.items() if meta.available}
        return self._cached_channel_metas

    def get_channel_meta(self, path: ChannelFullPath) -> ChannelMeta | None:
        """获取指定的 channel meta"""
        return self._cached_channel_metas.get(path)

    def full_facade(
            self,
            available_only: bool = True,
    ) -> str:
        """shell 当前状态的完整操作表面. 包含 instruction 和运行时信息."""
        channel_metas = self.channel_metas(available_only=available_only)
        lines = []
        for path, channel_meta in channel_metas.items():
            prompter = ChannelMetaPrompter(path, channel_meta)
            lines.append(prompter.full_facade())
        return "\n".join(lines)

    def get_channel_full_facade(self, path: ChannelFullPath) -> str:
        """指定channel 的操作表面, 包含除 context messages 之外的全量信息. """
        meta = self.get_channel_meta(path)
        if meta:
            return ChannelMetaPrompter(path, meta).full_facade()
        return ''

    def dynamic_context(self) -> dict[ChannelFullPath, list[Message]]:
        """获取动态上下文"""
        metas = self.channel_metas(
            available_only=True,
        )
        result = {}
        for path, channel_meta in metas.items():
            prompter = ChannelMetaPrompter(path, channel_meta)
            result[path] = prompter.dynamic_context_messages()
        return result

    def channels_description(self) -> str:
        """全量 channel 的 name + description 描述信息. 通常用于筛选. """
        from ghoshell_common.helpers import yaml_pretty_dump
        metas = self.channel_metas(available_only=True)
        kv = {path: meta.description or "(no desc)" for path, meta in metas.items()}
        return yaml_pretty_dump(kv)

    def status(self) -> MShellStatus:
        """返回当前的状态. """
        return MShellStatus.new(self.shell)


class MShellEvent(ABC):
    """shell 的流式运行时事件. """
    index: int
    created: float

    @abstractmethod
    def as_messages(self) -> list[Message]:
        ...


@dataclass
class ShellTaskDoneEvent(MShellEvent):
    """一个 command task 完成了. """
    index: int
    created: float
    caller: str
    task_id: str
    messages: list[Message]

    @classmethod
    def from_command_task(cls, task: CommandTask) -> Self | None:
        if not task.done():
            return None
        return ShellTaskDoneEvent(
            index=0,
            task_id=task.cid,
            caller=task.caller_name(),
            messages=task.task_result().as_messages(),
            created=time.time(),
        )

    def as_messages(self) -> list[Message]:
        return self.messages


@dataclass
class InterpreterStoppedEvent(MShellEvent):
    index: int
    created: float
    state: str  # done / interrupted / error, 来自 Interpretation.state()
    error: str = ""  # exception 文本, 无异常时为空
    completed: int = 0
    cancelled: int = 0
    failed: int = 0

    @classmethod
    def from_interpreter(cls, interpreter: Interpreter) -> Self:
        interpretation = interpreter.interpretation()
        return InterpreterStoppedEvent(
            index=0,
            created=time.time(),
            state=interpretation.state(),
            error=interpretation.exception,
            completed=len(interpretation.success_tasks),
            cancelled=len(interpretation.cancelled_tasks),
            failed=len(interpretation.failed_tasks),
        )

    def as_messages(self) -> list[Message]:
        body_lines = []
        if self.completed:
            body_lines.append(f"completed: {self.completed}")
        if self.cancelled:
            body_lines.append(f"cancelled: {self.cancelled}")
        if self.failed:
            body_lines.append(f"failed: {self.failed}")
        if self.error:
            body_lines.append(f"error: {self.error}")
        message = Message.new(
            tag='interpreter',
            attributes={'state': self.state},
            timestamp=True,
        )
        if body_lines:
            message.with_content('\n'.join(body_lines))
        return [message]


@dataclass
class TrajectoryFrame:
    """ Shell 运行时的关键帧数据, 它记录了 shell 的瞬间状态. """

    epoch_index: int  # 在一个 shell trajectory 中的第几个 epoch.
    index: int  # 在一个 trajectory epoch 中的位置.
    events: list[MShellEvent]  # 这一帧 会要提取出来的 shell events
    status: MShellStatus  # 生产 Frame 瞬间的状态.
    previous_metas: dict[ChannelFullPath, ChannelMeta]  # 上一帧持有的关键帧 shell metas.
    metas: dict[ChannelFullPath, ChannelMeta]  # 当前帧获取时的 shell 状态.
    created: float  # 创建的 timestamp
    committed: bool = False  # 是否已经完成了确认. 如果没有完成确认, 缓冲不会更新.

    def facade_delta(self) -> str:
        """比较 metas 后返回 shell 变更后的数据表面.

        上一帧有而当前帧没有的 channel, emit 墓碑标记, 让模型知道它已下线.
        """
        lines = []
        for path in sorted(set(self.previous_metas) | set(self.metas)):
            meta = self.metas.get(path)
            if meta is None:
                lines.append(f'<channel path="{path}" removed/>')
                continue
            previous_meta = self.previous_metas.get(path)
            if previous_meta is None:
                facade = ChannelMetaPrompter(path, meta).full_facade()
            else:
                facade = ChannelMetaPrompter(path, previous_meta).diff_facade(meta)
            if facade:
                lines.append(facade)
        return "\n".join(lines)

    def drained_event_messages(self) -> list[Message]:
        """从历史中抽取的 MOSS event 消息列表"""
        result = []
        # 返回 drain 的事件.
        for event in self.events:
            result.extend(event.as_messages())
        if len(result) > 0:
            result = [
                Message.new().with_content("<events>"),
                *result,
                Message.new().with_content("</events>"),
            ]
        return result

    def dynamic_context_messages(self) -> list[Message]:
        result = []
        for path, meta in self.metas.items():
            if meta.available and meta.context:
                messages = ChannelMetaPrompter(path, meta).dynamic_context_messages()
                result.extend(messages)
        return result

    def project(self, *, now: float | None = None) -> list[Message]:
        """投影本帧为消息列表.

        :param now: 发送时刻, 作帧级时间锚. 请求重试时应重新传入当前时间,
            避免模型把上次发送时刻误认为 now. 缺省用帧的 created.
        """
        result = []
        # 返回 drain 的事件.
        if drained_event_messages := self.drained_event_messages():
            result.extend(drained_event_messages)
        # 返回 shell status 数据.
        result.append(Message.new().with_content(self.status.description()))
        # 返回 facade delta
        if delta := self.facade_delta():
            result.append(Message.new(tag="facade").with_content(delta))

        at_ts = now if now is not None else self.created
        at = format_timestamp(datetime.datetime.fromtimestamp(at_ts, tz.gettz()))
        result = [
            Message.new().with_content(f'<moss at="{at}">'),
            *result,
            Message.new().with_content("</moss>"),
        ]
        return result


class MShellEventTracer(Tracer):
    """
    自动注册到 Shell, 观测 shell 的运行状态. 
    """

    def __init__(
            self,
            shell: MOSShell,
            index: int = 0,
            max_shell_events: int = 1000,
            logger: logging.Logger = None,
    ):
        self.shell = shell
        self._last_event_index: int = index
        self._closed = False
        self._shell_events: list[MShellEvent] = []
        self._shell_events_lock = threading.Lock()
        self._max_shell_events = max_shell_events
        self.shell.add_tracer(self)
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def index(self) -> int:
        return self._last_event_index

    def close(self):
        self._closed = True

    def peek(self) -> tuple[list[MShellEvent], int]:
        with self._shell_events_lock:
            return self._shell_events.copy(), self._last_event_index

    def drain(self, index: int) -> list[MShellEvent]:
        with self._shell_events_lock:
            events = self._shell_events.copy()
            result = []
            for event in events:
                if event.index > index:
                    break
                result.append(event)
            self._shell_events = self._shell_events[len(result):]
        return result

    def is_running(self) -> bool:
        return not self._closed

    def is_closed(self) -> bool:
        return self._closed

    def on_task_pushed(self, task: CommandTask) -> None:
        return None

    def on_task_done(self, task: CommandTask) -> None:
        if self._closed:
            return
        self._append_event(ShellTaskDoneEvent.from_command_task(task))

    def on_interpreter_stopped(self, interpreter: Interpreter) -> None:
        if self._closed:
            return
        self._append_event(InterpreterStoppedEvent.from_interpreter(interpreter))

    def _append_event(self, event: MShellEvent | None) -> None:
        if event is None:
            return
        with self._shell_events_lock:
            self._last_event_index += 1
            event.index = self._last_event_index
            self._shell_events.append(event)
            while len(self._shell_events) > self._max_shell_events:
                self._shell_events.pop(0)


class MShellTrajectory:
    """有状态的观测轨迹."""

    def __init__(
            self,
            shell: "MOSShell",
            selected_channels: list[ChannelFullPath] | None = None,
    ):
        """
        :param shell: 持有一个 MOSShell 
        :param selected_channels: 约束 Trajectory 的追踪面, 可以用来做注意力. 
        """
        self.facade = MShellContextFacade(shell, *(selected_channels or []))
        self.shell = shell
        self.logger = shell.container.get(logging.Logger) or logging.getLogger(__name__)
        self._selected_channels = selected_channels or []
        self._epoch_index = 0
        self._tracer: MShellEventTracer | None = None
        self._last_frame: TrajectoryFrame = TrajectoryFrame(
            epoch_index=0,
            index=0,
            events=[],
            previous_metas={},
            status=MShellStatus.new(shell),
            metas={},
            created=time.time(),
        )
        self._started = False
        self._stopped = False

    def is_running(self) -> bool:
        return self._started and not self._stopped and self._tracer is not None

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError(f"ShellTrajectory is not running.")

    def new_epoch(self) -> None:
        """从头开始观测, 清空观测结果, 进入新的 epoch. """
        if self._tracer:
            self._tracer.close()
            self._tracer = None
        self._epoch_index += 1
        # create new tracer.
        self._tracer = MShellEventTracer(self.shell)
        self._last_frame: TrajectoryFrame = TrajectoryFrame(
            index=self._tracer.index,
            events=[],
            previous_metas={},
            status=self.facade.status(),
            metas=self.facade.channel_metas(available_only=True),
            created=time.time(),
            epoch_index=self._epoch_index,
        )

    def epoch_start_point(self, refresh: bool = True) -> str:
        if refresh:
            self.new_epoch()
        return self.facade.full_facade(available_only=True)

    def peek(self) -> TrajectoryFrame:
        """生成一个当前帧的快照. """
        self._check_running()
        events, index = self._tracer.peek()
        return TrajectoryFrame(
            index=index,
            epoch_index=self._epoch_index,
            events=events,
            previous_metas=self._last_frame.metas,
            status=self.facade.status(),
            metas=self.facade.channel_metas(available_only=True),
            created=time.time(),
        )

    def commit(self, frame: TrajectoryFrame) -> bool:
        """ack 一个 snapshot"""
        self._check_running()
        if frame.index <= self._last_frame.index:
            return False
        _ = self._tracer.drain(frame.index)
        frame.committed = True
        self._last_frame = frame
        return True

    def pop_frame(self) -> TrajectoryFrame:
        """"""
        snapshot = self.peek()
        self.commit(snapshot)
        return snapshot

    async def __aenter__(self) -> 'MShellTrajectory':
        if self._stopped:
            raise RuntimeError(f"ShellTrajectory is stopped, create a new one.")
        if not self.shell.is_running():
            raise RuntimeError(f"Shell is not running for ShellTrajectory.")
        if self._started:
            return self
        self._started = True
        self.new_epoch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.facade.discard()
        if self._tracer:
            self._tracer.close()
            self._tracer = None
        self._stopped = True
