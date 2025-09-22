import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from ghoshell_moss.concepts.command import (
    CommandTask, Command, CommandToken, CommandTokenType, BaseCommandTask, CommandDeltaType,
    CancelAfterOthersTask,
)
from ghoshell_moss.concepts.interpreter import CommandTaskElement, CommandTaskCallback, CommandTaskParseError
from ghoshell_moss.concepts.shell import OutputStream, Output
from ghoshell_moss.helpers.stream import create_thread_safe_stream
from .token_parser import CMTLElement
from logging import Logger, getLogger
from threading import Event
from contextlib import contextmanager


class CommandTaskElementContext:
    """语法糖, 用来管理所有 element 共享的组件. """

    def __init__(
            self,
            commands: Dict[str, Command],
            output: Output,
            logger: Optional[Logger] = None,
            stop_event: Optional[Event] = None,
    ):
        self.commands = commands
        self.output = output
        self.logger = logger or getLogger("MOSShell")
        self.stop_event = stop_event or threading.Event()

    def new_root(self, callback: CommandTaskCallback, stream_id: str = "") -> CommandTaskElement:
        """
        创建解析树的根节点.
        """
        return NoDeltaCommandTaskElement(
            cid=stream_id,
            current_task=None,
            callback=callback,
            ctx=self
        )

    @contextmanager
    def new_parser(self, callback: CommandTaskCallback, stream_id: str = ""):
        """语法糖, 用来做上下文管理. """
        root = self.new_root(callback, stream_id)
        yield root
        root.destroy()


class BaseCommandTaskElement(CommandTaskElement, ABC):
    """
    标准的 command task 节点.
    """

    def __init__(
            self,
            cid: str,
            current_task: Optional[CommandTask],
            *,
            callback: Optional[CommandTaskCallback] = None,
            ctx: CommandTaskElementContext,
    ) -> None:
        self.cid = cid
        self.ctx = ctx
        self._current_task: Optional[CommandTask] = current_task
        """当前的 task"""

        self.children = {}
        """所有的子节点"""

        self._unclose_child: Optional[CommandTaskElement] = None
        """没有结束的子节点"""

        self._callback = callback
        """the command task callback method"""

        self._end = False
        """这个 element 是否已经结束了"""

        self._current_stream: Optional[OutputStream] = None
        """当前正在发送的 output stream"""

        self._children_tasks: List[CommandTask] = []
        """子节点发送的 tasks"""

        # 正式启动.
        self._destroyed = False
        self._on_self_start()

    def with_callback(self, callback: CommandTaskCallback) -> None:
        """设置变更 callback """
        self._callback = callback

    def on_token(self, token: CommandToken) -> None:
        if self.ctx.stop_event.is_set():
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
        if self.ctx.stop_event.is_set():
            # 停止了就啥也不干了.
            return None

        if task is not self._current_task:
            # 添加 children tasks
            self._children_tasks.append(task)

        if self._callback is not None:
            self._callback(task)

    def _new_child_element(self, token: CommandToken) -> None:
        """
        基于 start token 创建一个子节点.
        """
        if token.type != CommandTokenType.START.value:
            # todo
            raise RuntimeError(f"invalid token {token}")

        command = self.ctx.commands.get(token.name, None)
        if command is None:
            child = BaseCommandTaskElement(
                cid=token.command_id(),
                current_task=None,
                callback=self._callback,
                ctx=self.ctx,
            )
        else:
            meta = command.meta()
            task = BaseCommandTask(
                meta=meta,
                func=command.__call__,
                tokens=token.content,
                # ctml 语法不支持 args, 只支持 kwargs.
                args=[],
                kwargs=token.kwargs,
                cid=token.command_id(),
            )
            if meta.delta_arg == CommandDeltaType.TOKENS.value:
                child = DeltaTypeIsTokensCommandTaskElement(
                    cid=token.command_id(),
                    current_task=task,
                    callback=self._callback,
                    ctx=self.ctx,
                )
            elif meta.delta_arg == CommandDeltaType.TEXT.value:
                child = DeltaIsTextCommandTaskElement(
                    cid=token.command_id(),
                    current_task=task,
                    callback=self._callback,
                    ctx=self.ctx,
                )
            else:
                child = NoDeltaCommandTaskElement(
                    cid=token.command_id(),
                    current_task=task,
                    callback=self._callback,
                    ctx=self.ctx,
                )

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
        """
        手动清空依赖, 主要是避免存在循环依赖.
        """
        if self._destroyed:
            return
        self._destroyed = True
        # 递归清理所有的 element.
        for child in self.children.values():
            child.destroy()

        # 通常不需要手动清理. 但考虑到习惯性的意外, 还是处理一下. 防止内存泄漏.
        del self.ctx
        del self._unclose_child
        del self.children
        del self._current_stream
        del self._children_tasks
        del self._current_task


class NoDeltaCommandTaskElement(BaseCommandTaskElement):
    """
    没有 delta 参数的 Command
    """
    _output_stream: Optional[OutputStream] = None

    def _on_delta_token(self, token: CommandToken) -> None:
        if self._output_stream is None:
            # 没有创建过 output stream, 则创建一个.
            # 用来处理需要发送的 delta content.
            _output_stream = self.ctx.output.new_stream(
                batch_id=token.command_part_id(),
            )
        elif self._output_stream.id != token.command_part_id():
            # 创建过 output_stream, 则需要比较是否是相同的 command part id.
            # 不是相同的 command part id, 则需要创建一个新的流, 这样可以分段感知到每一段 output 是否已经执行完了.
            # 核心目标是, 当一个较长的 output 流被 command 分割成多段的话, 每一段都可以阻塞, 同时却可以提前生成 tts.
            # 这样生成 tts 的过程 add(token.content) 并不会被阻塞.
            task = self._output_stream.as_command_task()
            self._send_callback(task)
            # 于是新建一个 output stream.
            self._output_stream = None
            _output_stream = self.ctx.output.new_stream(
                batch_id=token.command_part_id(),
            )
        else:
            _output_stream = self._output_stream
        # 增加新的 stream delta
        _output_stream.buffer(token.content)
        self._output_stream = _output_stream

    def _on_self_start(self) -> None:
        # 直接发送命令自身.
        if self._current_task is not None:
            self._send_callback(self._current_task)

    def _on_cmd_start_token(self, token: CommandToken):
        # 如果子节点还是开标签, 不应该走到这一环.
        if self._unclose_child is not None:
            raise CommandTaskParseError(
                f"Start new child command {token} within unclosed command {self._unclose_child}"
            )
        elif self._output_stream is not None:
            # 结束未完成的 output 流.
            task = self._output_stream.as_command_task()
            self._send_callback(task)
            self._output_stream = None

        self._new_child_element(token)

    def _on_cmd_end_token(self, token: CommandToken):
        if self._unclose_child is not None:
            # 让子节点去处理.
            self._unclose_child.on_token(token)
            # 如果子节点处理完了, 自己也没了, 就清空.
            if self._unclose_child.is_end():
                self._unclose_child = None
            return
        elif token.command_id() != self.cid:
            # 自己来处理这个 token, 但 command id 不一致的情况.
            raise CommandTaskParseError("end current task with invalid command id")
        else:
            # 结束自身.
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


class DeltaTypeIsTokensCommandTaskElement(BaseCommandTaskElement):
    """
    当 delta type 是 tokens 时, 会自动拼装 tokens 为一个 Iterable / AsyncIterable 对象给目标 command.

    在并发运行的时候, 可能出现 command task 已经在运行, 但 delta tokens 没有生成完, 所以两者并行运行.
    这个功能的核心目标是实现并行的流式传输, 举例:

    1. LLM 在生成一个流, 传输给函数 foo
    2. 在 LLM 生成过程中, 函数 foo 已经拿到了 token, 并且在运行了.
    3. LLM 生成完所有 foo 的 tokens 时, foo 才能够结束.

    如果 foo 函数是运行在另一个通过双工通讯连接的 channel, 则这种做法能够达到最优的流式传输.
    """

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
            self._token_sender.end()
            self._end = True


class DeltaIsTextCommandTaskElement(BaseCommandTaskElement):
    """
    当 delta type 是 text 时, 这种解析逻辑是所有的中间 token 都视作文本
    等所有的文本都加载完, 才会发送这个 task.
    """

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
