"""ShellTrajectory — pull 型观测轨迹, 取代旧观测面 (ContextMonitor / ShellContext / InterleavedThinking).

以帧 (events + status + context + facade delta) 承载 shell 运行时的观测:
- MShellTrajectory: 有状态轨迹, peek/commit 推进基线, pop_frame 拉取每帧 delta.
- MShellContextFacade: channel 操作表面 facade (full_facade / per-channel / delta).
- MShellEventTracer: 订阅 Shell Tracer 收集 task-done / interpreter-stopped 事件.
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
]

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
        elif interpreter is not None and interpreter.is_running():
            state = 'running'
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


def _today_anchor() -> str:
    """绝对日期锚, 形如 2026-08-20+8. 帧里的 D 短时间戳由它补全年月."""
    now = datetime.datetime.now(tz.gettz())
    offset = now.utcoffset()
    tz_str = ""
    if offset is not None:
        total = int(offset.total_seconds())
        sign = '+' if total >= 0 else '-'
        total = abs(total)
        hours, minutes = divmod(total, 3600)
        tz_str = f"{sign}{hours}" if minutes == 0 else f"{sign}{hours}:{minutes:02d}"
    return now.strftime('%Y-%m-%d') + tz_str


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
        lines = [f'<today>{_today_anchor()}</today>']
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
        at = format_timestamp(datetime.datetime.fromtimestamp(self.created, tz.gettz()))
        message = Message.new(
            tag='interpreter',
            attributes={'state': self.state, 'at': at},
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
        """从历史中抽取的命令事件消息, 恒包 <events> 容器 (空则自闭合)."""
        result = []
        for event in self.events:
            # InterpreterStoppedEvent 本帧不投影: status/interpreter 语义未定, 先 skip.
            if isinstance(event, ShellTaskDoneEvent):
                result.extend(event.as_messages())
        if result:
            return [
                Message.new().with_content("<events>"),
                *result,
                Message.new().with_content("</events>"),
            ]
        return [Message.new().with_content("<events/>")]

    def dynamic_context_messages(self) -> list[Message]:
        result = []
        for path, meta in self.metas.items():
            if meta.available and meta.context:
                messages = ChannelMetaPrompter(path, meta).dynamic_context_messages()
                result.extend(messages)
        return result

    def project(self, *, now: float | None = None, with_dynamic: bool = True) -> list[Message]:
        """投影本帧为消息列表.

        :param now: 发送时刻, 作帧级时间锚. 请求重试时应重新传入当前时间,
            避免模型把上次发送时刻误认为 now. 缺省用帧的 created.
        :param with_dynamic: 是否携带 channel 的动态讯息 (每轮都可能不一样, 属于 hot 数据).
        """
        result = []
        # 返回 drain 的事件.
        if drained_event_messages := self.drained_event_messages():
            result.extend(drained_event_messages)
        # 返回 shell status 数据.
        result.append(Message.new().with_content(self.status.description()))
        # 返回当前 context messages (channel 运行时数据, 有则发).
        if with_dynamic:
            if context := self.dynamic_context_messages():
                result.append(Message.new().with_content("<context>"))
                result.extend(context)
                result.append(Message.new().with_content("</context>"))
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
        self._selected_channels = selected_channels or []
        self.facade = MShellContextFacade(shell, *self._selected_channels)
        self.shell = shell
        self.logger = shell.container.get(logging.Logger) or logging.getLogger(__name__)
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
