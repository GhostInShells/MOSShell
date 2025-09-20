from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from ghoshell_moss.concepts.command import (
    CommandTask, Command, CommandToken, CommandTokenType, BaseCommandTask, CommandDeltaType,
    CancelAfterOthersTask,
)
from ghoshell_moss.concepts.interpreter import CommandTaskElement, CommandTaskCallback, CommandTaskParseError
from ghoshell_moss.concepts.shell import MOSSShell, OutputStream, Output
from ghoshell_moss.helpers.stream import create_thread_safe_stream
from .token_parser import CMTLElement
from logging import Logger, getLogger
from threading import Event


class RootCommandTaskElement(CommandTaskElement):

    def __init__(
            self,
            *,
            commands: Dict[str, Command],
            output: Output,
            callback: Optional[CommandTaskCallback] = None,
            logger: Optional[Logger] = None,
    ) -> None:
        self._logger = logger or getLogger("MOSShell")
        self._output = output
        self._commands = commands
        self.children = {}
        self._unclose_child: Optional[CommandTaskElement] = None
        self._callback = callback
        self._end = False

    def with_callback(self, callback: CommandTaskCallback) -> None:
        self._callback = callback

    def on_token(self, token: CommandToken) -> bool:
        pass

    def is_end(self) -> bool:
        pass

    def destroy(self) -> None:
        """
        destroy manually in case of memory leaking
        """
        del self._commands
        del self._output
        del self.children
        del self._unclose_child
        del self._callback
        del self._logger


class BaseCommandTaskElement(CommandTaskElement):
    """
    标准的 command task 节点.
    """

    def __init__(
            self,
            cid: str,
            current_task: Optional[CommandTask],
            *,
            commands: Dict[str, Command],
            output: Output,
            callback: Optional[CommandTaskCallback] = None,
            logger: Optional[Logger] = None,
            stop_event: Optional[Event] = None,
    ) -> None:
        self.cid = cid
        self._current_task: Optional[CommandTask] = current_task
        """当前的 task"""
        self._current_task_sent = False
        """当前 task 是否已经发送了. """
        self._stop_event = stop_event or Event()

        self._logger = logger or getLogger("MOSShell")
        self._output = output
        """output interface 用来发送 speak"""

        self._commands = commands
        """上下文中所有可以使用的 commands """

        self.children = {}
        """所有的子节点"""

        self._unclose_child: Optional[CommandTaskElement] = None
        """没有结束的子节点"""

        self._callback = callback
        """the command task callback method"""

        self._end = False
        """这个 element 是否已经结束了"""

        self._tokens = current_task.tokens
        """这个 command task 完整的输出 tokens """

        self._current_stream: Optional[OutputStream] = None
        """当前正在发送的 output stream"""

        self._children_tasks: List[CommandTask] = []

        self._on_self_start()

    def with_callback(self, callback: CommandTaskCallback) -> None:
        """设置变更 callback """
        self._callback = callback

    def on_token(self, token: CommandToken) -> None:
        if self._stop_event.is_set():
            # 避免并发操作中存在的乱续.
            return None

        if self._end:
            # 当前 element 已经运行结束了, 却拿到了新的 token.
            # todo: 优化异常.
            raise CommandTaskParseError(f"receive end token after the command is stopped")

        # 如果有子节点状态已经变更, 但没有被更新, 临时更新一下. 容错.
        if self._unclose_child is not None and self._unclose_child.is_end():
            # remove unclose child if it is already end
            self._unclose_child = None

        # 重新让子节点接受 token.
        if self._unclose_child is not None:
            # otherwise let the unclose child to handle the token
            self._unclose_child.on_token(token)
            # 如果未结束的子节点已经运行结束, 则应该将子节点摘掉.
            if self._unclose_child.is_end:
                self._unclose_child = None
            return

        # 接受一个 start token.
        if token.type == CommandTokenType.START:
            self._on_cmd_start_token(token)
            return
        # 接受一个 end token
        elif token.type == CommandTokenType.END:
            self._on_cmd_end_token(token)
            return
        # 接受一个 delta 类型的 token.
        else:
            self._on_delta_token(token)
            return

    def _send_callback(self, task: Optional[CommandTask]) -> None:
        if task is None:
            # 节省一些冗余代码.
            return None
        if self._stop_event.is_set():
            # 停止了就啥也不干了.
            return None

        # 每个 element 的 task 只发送一次.
        if self._callback is not None:
            self._callback(task)

        if task is not self._current_task:
            # 添加 children tasks
            self._children_tasks.append(task)

    def _new_child_element(self, token: CommandToken) -> None:
        if token.type != CommandTokenType.START.value:
            # todo
            raise RuntimeError(f"invalid token {token}")

        command = self._commands.get(token.name, None)
        child = None
        if command is None:
            child = BaseCommandTaskElement(
                cid=token.command_id(),
                current_task=None,
                commands=self._commands,
                output=self._output,
                callback=self._callback,
                logger=self._logger,
                stop_event=self._stop_event,
            )
        else:
            meta = command.meta()
            task = BaseCommandTask(
                meta=meta,
                func=command.__call__,
                tokens=token.content,
                args=[],
                kwargs=token.kwargs,
                cid=token.command_id(),
            )
            if meta.delta_arg == CommandDeltaType.TOKENS.value:
                pass
            elif meta.delta_arg == CommandDeltaType.TEXT.value:
                pass

        if child is not None:
            self.children[child.cid] = child
            self._unclose_child = child

    @abstractmethod
    def _on_delta_token(self, token: CommandToken) -> None:
        pass

    @abstractmethod
    def _on_self_start(self) -> None:
        pass

    @abstractmethod
    def _on_cmd_start_token(self, token: CommandToken):
        pass

    @abstractmethod
    def _on_cmd_end_token(self, token: CommandToken):
        pass

    def is_end(self) -> bool:
        return self._end

    def destroy(self) -> None:
        del self._commands
        del self._output
        del self._tokens
        del self._unclose_child
        del self.children


class NoDeltaCommandTaskElement(BaseCommandTaskElement):
    """
    没有 delta 参数的 Command
    """
    _output_stream: Optional[OutputStream] = None

    def _on_delta_token(self, token: CommandToken) -> None:
        if self._output_stream is None:
            # 没有创建过 output stream, 则创建一个.
            _output_stream = self._output.new_stream(
                batch_id=token.command_part_id(),
            )
        elif self._output_stream.id != token.command_part_id():
            # 创建过 output_stream, 则需要比较是否是相同的 command part id.
            # 不是相同的 command part id, 则需要创建一个新的流, 这样可以分段感知到每一段 output 是否已经执行完了.
            task = self._output_stream.as_command_task()
            self._send_callback(task)
            # 于是新建一个 output stream.
            self._output_stream = None
            _output_stream = self._output.new_stream(
                batch_id=token.command_part_id(),
            )
        else:
            _output_stream = self._output_stream
        # 增加新的 stream delta
        _output_stream.add(token.content)
        self._output_stream = _output_stream

    def _on_self_start(self) -> None:
        # 直接发送命令自身.
        self._send_callback(self._current_task)

    def _on_cmd_start_token(self, token: CommandToken):
        # 如果子节点还是开标签, 不应该走到这一环.
        if self._unclose_child is not None:
            raise CommandTaskParseError(
                f"Start new child command {token} within unclosed command {self._unclose_child}"
            )
        self._new_child_element(token)

    def _on_cmd_end_token(self, token: CommandToken):
        if self._unclose_child is not None:
            # todo
            raise CommandTaskParseError("end current task with unclose child command")
        elif token.command_id() != self.cid:
            raise CommandTaskParseError("end current task with invalid command id")
        self._on_self_end()
        self._end = True

    def _on_self_end(self) -> None:
        if self._output_stream is not None:
            # 发送未发送的 output stream.
            output_stream_task = self._output_stream.as_command_task()
            self._send_callback(output_stream_task)
            self._output_stream = None

        if len(self._children_tasks) > 0:
            cancel_after_children_task = CancelAfterOthersTask(
                self.current,
                *self._children_tasks,
                tokens=f"</{self._current_task.meta.name}>",
            )
            # 等待所有 children tasks 完成, 如果自身还未完成, 则取消.
            self._send_callback(cancel_after_children_task)
        elif self.current is not None:
            # 按照 ctml 的规则, 修改规则.
            self._current_task.tokens = CMTLElement.make_start_mark(
                name=self._current_task.meta.name,
                attrs=self._current_task.kwargs,
                self_close=True,
            )


class TokenDeltaCommandTaskElement(BaseCommandTaskElement):

    def _on_self_start(self) -> None:
        sender, receiver = create_thread_safe_stream()
        self._token_sender = sender
        self._current_task.kwargs[CommandDeltaType.TOKENS.value] = receiver
        # 直接发送当前任务.
        self._send_callback(self._current_task)

    def _on_delta_token(self, token: CommandToken) -> None:
        self._token_sender.append(token)

    def _on_cmd_start_token(self, token: CommandToken):
        self._token_sender.append(token)

    def _on_cmd_end_token(self, token: CommandToken):
        if token.command_id() != self.cid:
            self._token_sender.append(token)
        else:
            self._end = True


class TextDeltaCommandTaskElement(BaseCommandTaskElement):
    _inner_content = ""

    def _on_delta_token(self, token: CommandToken) -> None:
        self._inner_content += token.content
        return

    def _on_self_start(self) -> None:
        # 开始时不要执行什么.
        return

    def _on_cmd_end_token(self, token: CommandToken):
        if token.command_id() != self.cid:
            self._inner_content += token.content
            return None
        if self.current is not None:
            self.current.kwargs[CommandDeltaType.TEXT.value] = self._inner_content
            self._send_callback(self._current_task)
        self._end = True

    def _on_cmd_start_token(self, token: CommandToken):
        self._inner_content += token.content
        return
