from abc import ABC, abstractmethod
from contextlib import contextmanager
from logging import getLogger
from typing import Optional, Generic, Any, ClassVar

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.concepts.command import (
    BaseCommandTask,
    CancelAfterOthersTask,
    Command,
    CommandDeltaType,
    CommandDeltaValue,
    ValueOfCommandDeltaTypeMap,
    CommandTask,
    CommandToken,
    CommandTokenSeq,
    PyCommand,
)
from ghoshell_moss.core.concepts.errors import InterpretError, CommandErrorCode
from ghoshell_moss.core.concepts.interpreter import (
    CommandTaskCallback,
    CommandTokenParser,
)
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.concepts.speech import Speech, SpeechStream
from ghoshell_moss.core.helpers.stream import create_sender_and_receiver, ItemT

from .token_parser import CMTLSaxElement

__all__ = [
    "BaseCommandTokenParserElement",
    "CommandTaskElementContext",
    "DeltaIsTextElement",
    "DeltaIsCommandTokensElement",
    "EmptyCommandTaskElement",
    "NoDeltaCommandTaskElement",
    "RootCommandTaskElement",
]


async def invalid_command():
    task = ChannelCtx.task()
    raise CommandErrorCode.NOT_FOUND.error(f"command {task.caller_name()} not found")


invalid_command = PyCommand(invalid_command)


class CommandTaskElementContext:
    """语法糖, 用来管理所有 element 共享的组件."""

    instances_count: ClassVar[int] = 0

    def __init__(
            self,
            channel_commands: dict[str, dict[str, Command]],
            speech: Speech,
            logger: Optional[LoggerItf] = None,
            # stop_event: Optional[ThreadSafeEvent] = None,
            root_tag: str = "ctml",
            ignore_wrong_command: bool = False,
            callback: Optional[CommandTaskCallback] = None,
            delta_type_map: Optional[dict[str, Any]] = None,
    ):
        self.channel_commands_map = channel_commands
        # 主音频模块.
        self.speech = speech
        self.logger = logger or getLogger("moss")
        # self.stop_event = stop_event or ThreadSafeEvent()
        self.root_tag = root_tag
        self.ignore_wrong_command = ignore_wrong_command
        self.delta_type_map = delta_type_map or ValueOfCommandDeltaTypeMap.copy()
        self._callback = callback
        self._delivered_last_callback = False
        CommandTaskElementContext.instances_count += 1

    def __del__(self):
        self.speech = None
        self.channel_commands_map.clear()
        CommandTaskElementContext.instances_count -= 1

    def new_root(self, callback: CommandTaskCallback, stream_id: str = "") -> "RootCommandTaskElement":
        """
        创建解析树的根节点.
        """
        self.logger.info(
            "[CommandTaskElementContext] create root element, instances count %d, element instances count %d",
            CommandTaskElementContext.instances_count,
            BaseCommandTokenParserElement.instances_count,
        )
        return RootCommandTaskElement(
            self.root_tag,
            stream_id=stream_id,
            cid=stream_id,
            current_task=None,
            callback=callback,
            ctx=self,
        )

    def send_callback(self, task: CommandTask | None) -> None:
        if task is None:
            if not self._delivered_last_callback:
                self._send_callback(task)
                self._delivered_last_callback = True
            return
        if not isinstance(task, CommandTask):
            raise ValueError(f"task {task} is not a CommandTask")
        if self._delivered_last_callback:
            self.logger.error("[CommandTaskElementContext] delivered task %s after last already delivered", task)
            return
        self._send_callback(task)

    def _send_callback(self, task: CommandTask | None) -> None:
        if self._callback is not None:
            self._callback(task)

    @contextmanager
    def new_parser(self, callback: CommandTaskCallback, stream_id: str = ""):
        """语法糖, 用来做上下文管理."""
        root = self.new_root(callback, stream_id)
        yield root
        root.destroy()


class BaseCommandTokenParserElement(CommandTokenParser, ABC):
    """
    基础的 CommandToken 树形解析节点.
    解决共同的参数调用问题.
    """

    instances_count: ClassVar[int] = 0

    def __init__(
            self,
            name: str,
            stream_id: str,
            cid: str,
            current_task: Optional[CommandTask],
            *,
            depth: int = 0,
            callback: Optional[CommandTaskCallback] = None,
            ctx: CommandTaskElementContext,
    ) -> None:
        self._name = name
        self.stream_id = stream_id
        self.cid = cid
        self.ctx = ctx
        self.depth = depth
        self._current_task: Optional[CommandTask] = current_task
        """当前的 task. 每个节点默认都由一个 Task 创建. """

        self.inner_tasks: list[CommandTask] = []
        """在自己内部发送的各种 tasks."""

        self.children: list[BaseCommandTokenParserElement] = []
        """所有的子节点"""

        self._unclose_child: Optional[CommandTokenParser] = None
        """没有结束的子节点"""

        self._callback = callback
        """the command task callback method"""

        self._end = False
        """这个 element 是否已经结束了"""

        self._current_stream: Optional[SpeechStream] = None
        """当前正在发送的 output stream"""

        # 正式启动.
        self._destroyed = False
        self._done_is_delivered = False
        self._log_prefix = "[CommandTokenParser][cls=%s] sid=%s cid=%s depth=%d name=%s, " % (
            self.__class__.__name__,
            self.stream_id,
            cid,
            depth,
            self._name,
        )
        # 初始化自身节点.
        BaseCommandTokenParserElement.instances_count += 1

    def __del__(self):
        if not self._destroyed:
            self.destroy()
        BaseCommandTokenParserElement.instances_count -= 1

    def with_callback(self, callback: CommandTaskCallback) -> None:
        """设置变更 callback"""
        self._callback = callback

    def _send_callback(self, task: CommandTask | None) -> None:
        if task is None:
            if not self._done_is_delivered:
                self._done_is_delivered = True
            else:
                return
        elif not isinstance(task, CommandTask):
            raise TypeError(f"task must be CommandTask, got {type(task)}")

        if task is not None and task is not self._current_task:
            # 添加 children tasks
            self.inner_tasks.append(task)

        if self._callback:
            try:
                self._callback(task)
            except Exception as e:
                self.ctx.logger.exception("%s send callback failed: %s", self._log_prefix, e)
                raise e

    def is_end(self) -> bool:
        return self._end

    def raise_interrupt(self):
        raise InterpretError(f"Shell Interpreter failed due to system error")

    def on_token(self, token: CommandToken | None) -> list[CommandTask] | None:
        try:
            return self._on_token(token)
        except InterpretError as e:
            self.fail(e)
            raise e
        except Exception as e:
            self.ctx.logger.exception("%s on token failed: %s", self._log_prefix, e)
            self.fail(e)
            self.raise_interrupt()

    def fail(self, error: Exception) -> None:
        """
        递归处理异常.
        """
        if not self.is_end():
            self.on_own_end()
            self.ctx.logger.exception("%s failed: %s", self._log_prefix, error)
        if self._current_task is not None:
            self._current_task.fail(error)
        if isinstance(error, InterpretError):
            if len(self.inner_tasks) == 0:
                return
            for t in self.inner_tasks:
                if not t.done():
                    t.fail(error)

    def _on_token(self, token: CommandToken | None) -> list[CommandTask] | None:
        """
        当前节点得到了一个新的 command token.
        """
        if token is None:
            # 结束自己的生命.
            self._send_callback(None)
            return self.on_own_end()
        if self.is_end():
            self.ctx.logger.warning("%s receive token %s after element is end", self._log_prefix, token)
            return None

        # 如果有子节点状态已经变更, 但没有被更新, 临时更新一下. 容错.
        if self._unclose_child is not None:
            if self._unclose_child.is_end():
                # remove unclose child if it is already end
                self._unclose_child = None

        # 重新让子节点接受 token.
        # 简单来说, 一个子节点没结束的时候, 会把所有的 token 都发送给它.
        if self._unclose_child is not None:
            # otherwise let the unclose child to handle the token
            result = self._unclose_child.on_token(token)
            # 如果未结束的子节点已经运行结束, 则应该将子节点摘掉.
            # 这样在 Command Token 运行的时候, 出现了合法的子节点, 保留
            if self._unclose_child.is_end():
                self._unclose_child = None
            return result

        # 如果不是子节点去处理 token, 就轮到了自己来处理 token.
        # 接受一个 start token.
        if token.seq == CommandTokenSeq.DELTA:
            return self.on_delta_token(token)
        # 接受一个 end token
        elif token.seq == CommandTokenSeq.END:
            if token.command_id() == self.cid:
                # 结束自身.
                return self.on_own_end()
            return self.on_sub_end_token(token)
        # 接受一个 start token.
        elif token.seq == CommandTokenSeq.START:
            # 是自己就不太对了.
            if token.command_id() == self.cid:
                self.ctx.logger.error("%s received duplicated start command %s", self._log_prefix, token)
                self.raise_interrupt()
                return
            # 否则当成一个正常的 token.
            return self.on_sub_start_token(token)
        else:
            self.ctx.logger.error("%s received invalid command token %s", self._log_prefix, token)
            self.raise_interrupt()
            return

    def _find_command(self, chan: str, name: str) -> Optional[Command]:
        """
        寻找一个命令.
        """
        if chan not in self.ctx.channel_commands_map:
            return None
        channel_commands = self.ctx.channel_commands_map[chan]
        return channel_commands.get(name, None)

    def _is_root_token(self, token: CommandToken) -> bool:
        """
        是根节点的 Token.
        """
        if token is None:
            return False
        is_root_tag = token.chan == "" and token.name == self.ctx.root_tag
        return is_root_tag

    def _new_child_element(self, token: CommandToken) -> list[CommandTask] | None:
        """
        基于 start token 创建一个子节点. 策略树模式.
        """
        if token.seq != CommandTokenSeq.START.value:
            self.ctx.logger.error(
                "%s create new child but receive token which is not start: %s",
                self._log_prefix,
                token,
            )
            raise InterpretError(f"invalid tokens {token.content}")
        task = None
        # 判断这个 token 是不是 root token.
        command = self._find_command(token.chan, token.name)
        if command is None:
            if self.ctx.ignore_wrong_command:
                self.ctx.logger.warning(
                    "%s ignore wrong command %s, create empty one",
                    self._log_prefix,
                    token,
                )
                child = EmptyCommandTaskElement(
                    name=Command.make_uniquename(token.chan, token.name),
                    stream_id=self.stream_id,
                    cid=token.command_id(),
                    current_task=None,
                    # 提供递归的 task 传递路径.
                    callback=self._send_callback,
                    ctx=self.ctx,
                    depth=self.depth + 1,
                )
            else:
                # 抛出致命异常, 拒绝解析.
                err = f"command `{token.name}` from channel `{token.chan}` not found, use provided command only!"
                self.ctx.logger.error(
                    "%s receive invalid command token %s",
                    self._log_prefix,
                    token,
                )
                raise InterpretError(err)
        else:
            meta = command.meta()
            # 创建子节点的 Task.
            task = BaseCommandTask.from_command(
                command_=command,
                tokens_=token.content,
                args=token.args,
                kwargs=token.kwargs,
                cid=token.command_id(),
                chan_=token.chan,
                call_id=token.call_id,
            )
            # 根据不同 delta 类型, 来创建子节点的具体类型.
            if meta.delta_arg is not None:
                delta_value_type = self.ctx.delta_type_map.get(meta.delta_arg)
                # 接受 Tokens 作为流的类型.
                if delta_value_type is CommandDeltaValue.COMMAND_TOKEN_STREAM:
                    child = DeltaIsCommandTokensElement(
                        name=task.caller_name(),
                        stream_id=self.stream_id,
                        cid=token.command_id(),
                        current_task=task,
                        callback=self._send_callback,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )
                # 接受 AsyncIterable[Chunk] 的类型.
                elif delta_value_type is CommandDeltaValue.TEXT_CHUNKS_STREAM:
                    child = DeltaIsTextChunkElement(
                        name=task.caller_name(),
                        stream_id=self.stream_id,
                        cid=token.command_id(),
                        current_task=task,
                        callback=self._send_callback,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )
                # 接受 text__ 的类型.
                elif delta_value_type is CommandDeltaValue.TEXT:
                    child = DeltaIsTextElement(
                        name=task.caller_name(),
                        stream_id=token.command_id(),
                        cid=token.command_id(),
                        current_task=task,
                        callback=self._send_callback,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )
                else:
                    self.ctx.logger.error("%s command delta type %s is not implemented", meta.delta_arg)
                    child = NoDeltaCommandTaskElement(
                        name=task.caller_name(),
                        stream_id=self.stream_id,
                        cid=token.command_id(),
                        current_task=task,
                        callback=self._send_callback,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )

            else:
                child = NoDeltaCommandTaskElement(
                    name=task.caller_name(),
                    stream_id=self.stream_id,
                    cid=token.command_id(),
                    current_task=task,
                    callback=self._send_callback,
                    ctx=self.ctx,
                    depth=self.depth + 1,
                )

        if child is not None:
            # 把所有子孙都拿着. 恨不得....
            self.children.append(child)
            if not child.is_end():
                # 记录 unclose.
                self._unclose_child = child
            return child.on_init()
        return None

    @abstractmethod
    def on_delta_token(self, token: CommandToken) -> list[CommandTask] | None:
        """
        每个节点都要考虑, 拿到了属于自己的 delta token 怎么办.
        """
        pass

    @abstractmethod
    def on_init(self) -> list[CommandTask] | None:
        """
        每个节点初始化的逻辑.
        通常是在初始化时, 就发送 command task.
        """
        pass

    @abstractmethod
    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask] | None:
        """
        处理拿到了一个开始标记的 token. 这个不是来自自己的 Token.
        """
        pass

    @abstractmethod
    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask] | None:
        """
        拿到了一个结束标记的 Token. 不是自己的 Token.
        """
        pass

    def on_own_end(self) -> list[CommandTask] | None:
        """
        拿到了自身的结束 Token
        """
        self._end = True
        self.ctx.logger.debug("%s end self", self._log_prefix)
        return None

    def destroy(self) -> None:
        """
        手动清空依赖, 主要是避免存在循环依赖.
        """
        if self._destroyed:
            return
        self._destroyed = True
        # 递归清理所有的 element.
        for child in self.children:
            # 递归毁灭吧!!.
            child.destroy()

        # 通常不需要手动清理. 但考虑到习惯性的意外, 还是处理一下. 防止内存泄漏.
        del self.ctx
        del self._unclose_child
        del self.children
        del self._current_stream
        del self.inner_tasks
        del self._current_task


class NoDeltaCommandTaskElement(BaseCommandTokenParserElement):
    """
    没有 delta 参数的节点类型.
    也就是说这种类型的 Command 不支持 delta 数据, 也不支持子节点.
    不支持 Delta 数据的默认逻辑是, 将之视为音频片段.

    这种节点的 Cancel 标记理论上是无效的. 但我们隐藏一个防蠢规则:
    中间的数据仍然会生成节点, 而且自己结束时会生成一个尾标记任务.
    如果这个尾标记任务已经进入队列执行, 无论如何都会清空前一个任务. 技术上基于 Command Partial 来实现.

    相当于:
    - task start: 开启运行.
    - task end: cancel 它.
    """

    _speech_stream: Optional[SpeechStream] = None

    def on_delta_token(self, token: CommandToken) -> list[CommandTask] | None:
        output_stream_task = None
        if self._speech_stream is None:
            # 没有创建过 output stream, 则创建一个.
            # 用来处理需要发送的 delta content.
            _speech_stream = self.ctx.speech.new_stream(
                batch_id=token.command_part_id(),
            )
            output_stream_task = _speech_stream.as_command_task()
            self._send_callback(output_stream_task)
        elif self._speech_stream.id != token.command_part_id():
            # 创建过 output_stream, 则需要比较是否是相同的 command part id.
            # 不是相同的 command part id, 则需要创建一个新的流, 这样可以分段感知到每一段 output 是否已经执行完了.
            # 核心目标是, 当一个较长的 output 流被 command 分割成多段的话, 每一段都可以阻塞, 同时却可以提前生成 tts.
            # 这样生成 tts 的过程 add(token.content) 并不会被阻塞.
            self._clear_speech_stream()
            _speech_stream = self.ctx.speech.new_stream(
                batch_id=token.command_part_id(),
            )
            output_stream_task = _speech_stream.as_command_task()
            self._send_callback(output_stream_task)
        else:
            _speech_stream = self._speech_stream
        # 增加新的 stream delta
        _speech_stream.feed(token.content)
        self._speech_stream = _speech_stream
        if output_stream_task is not None:
            return [output_stream_task]
        return None

    def on_init(self) -> list[CommandTask] | None:
        # 直接发送命令自身.
        if self._current_task is not None:
            # 发送自己的 Task.
            self._send_callback(self._current_task)
            return [self._current_task]
        return None

    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask] | None:
        # 如果子节点还是开标签, 不应该走到这一环.
        if self._unclose_child is not None:
            self.ctx.logger.error(
                "%s Start new child command %s within unclosed command %s",
                self._log_prefix,
                token,
                self._unclose_child,
            )
            self.raise_interrupt()
            return
        self._clear_speech_stream()
        return self._new_child_element(token)

    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask] | None:
        self._clear_speech_stream()
        if self._unclose_child is not None:
            # 让子节点去处理.
            result = self._unclose_child.on_token(token)
            # 如果子节点处理完了, 自己也没了, 就清空.
            if self._unclose_child.is_end():
                self._unclose_child = None
            return result
        elif token.command_id() != self.cid:
            self.ctx.logger.error(
                "%s element end current task %s with invalid token %r", self._log_prefix, self._current_task, token
            )
            # 自己来处理这个 token, 但 command id 不一致的情况.
            self.raise_interrupt()
            return
        else:
            # 结束自身.
            # 理论上外部可以调用.
            return

    def _clear_speech_stream(self) -> None:
        if self._speech_stream is not None:
            # 发送未发送的 output stream.
            self._speech_stream.commit()
            self._speech_stream = None

    def on_own_end(self) -> list[CommandTask] | None:
        # 设置关闭.
        super().on_own_end()
        self._clear_speech_stream()
        if self._current_task is None:
            return None
        elif len(self.inner_tasks) > 0:
            cancel_after_children_task = CancelAfterOthersTask(
                self._current_task,
                *self.inner_tasks,
            )
            cancel_after_children_task.tokens = CMTLSaxElement.make_end_mark(
                self._current_task.chan,
                self._current_task.meta.name,
            )
            # 等待所有 children tasks 完成, 如果自身还未完成, 则取消.
            self._send_callback(cancel_after_children_task)
            return [cancel_after_children_task]
        else:
            # 按照 ctml 的规则, 修改 task 的开启标记. 用来做开标记逻辑.
            meta = self._current_task.meta
            self._current_task.tokens = CMTLSaxElement.make_start_mark(
                chan=meta.chan,
                name=meta.name,
                attrs=self._current_task.kwargs,
                self_close=True,
            )
            return None

    def destroy(self) -> None:
        self._clear_speech_stream()
        super().destroy()


class EmptyCommandTaskElement(NoDeltaCommandTaskElement):
    """
    一个空节点.
    """

    pass


class DeltaStreamElement(BaseCommandTokenParserElement, Generic[ItemT], ABC):
    """
    当 delta type 是 tokens 时, 会自动拼装 tokens 为一个 Iterable / AsyncIterable 对象给目标 command.

    在并发运行的时候, 可能出现 command task 已经在运行, 但 delta tokens 没有生成完, 所以两者并行运行.
    这个功能的核心目标是实现并行的流式传输, 举例:

    1. LLM 在生成一个流, 传输给函数 foo
    2. 在 LLM 生成过程中, 函数 foo 已经拿到了 token, 并且在运行了.
    3. LLM 生成完所有 foo 的 tokens 时, foo 才能够结束.

    如果 foo 函数是运行在另一个通过双工通讯连接的 channel, 则这种做法能够达到最优的流式传输.
    """

    def __init__(
            self,
            name: str,
            stream_id: str,
            cid: str,
            current_task: Optional[CommandTask],
            *,
            depth: int = 0,
            callback: Optional[CommandTaskCallback] = None,
            ctx: CommandTaskElementContext,
    ) -> None:
        sender, receiver = create_sender_and_receiver()
        self._sender = sender
        self._receiver = receiver
        self._deltas: str = ""
        self._exists_delta_value = None
        super().__init__(
            name,
            stream_id,
            cid,
            current_task,
            depth=depth,
            callback=callback,
            ctx=ctx,
        )

    def on_init(self) -> list[CommandTask] | None:
        delta_arg_name = self._current_task.meta.delta_arg
        self._exists_delta_value = self._current_task.kwargs.get(delta_arg_name, None)
        self._current_task.kwargs[delta_arg_name] = self._receiver
        # 直接发送当前任务.
        self._send_callback(self._current_task)
        return [self._current_task]

    def on_delta_token(self, token: CommandToken) -> list[CommandTask] | None:
        self._deltas += token.content
        parsed = self._parse_delta(token)
        self._sender.append(parsed)
        return None

    @abstractmethod
    def _parse_delta(self, token: CommandToken) -> ItemT:
        pass

    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask] | None:
        parsed = self._parse_delta(token)
        self._sender.append(parsed)
        self._deltas += token.content
        return None

    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask] | None:
        parsed = self._parse_delta(token)
        self._deltas += token.content
        self._deltas += token.content
        self._sender.append(parsed)
        return None

    def on_own_end(self) -> list[CommandTask] | None:
        result = super().on_own_end()
        if len(self._deltas) == 0 and self._exists_delta_value:
            self._sender.append(self._exists_delta_value)
        self._sender.commit()
        return result

    def fail(self, error: Exception) -> None:
        super().fail(error)
        if self._sender:
            self._sender.fail(error)

    def destroy(self) -> None:
        if self._sender:
            self._sender.commit()
        super().destroy()


class DeltaIsCommandTokensElement(DeltaStreamElement[CommandToken]):
    def _parse_delta(self, token: CommandToken) -> ItemT:
        if token is None:
            raise RuntimeError("why token is None")
        return token


class DeltaIsTextChunkElement(DeltaStreamElement[CommandToken]):
    def _parse_delta(self, token: CommandToken) -> ItemT:
        if token is None:
            raise RuntimeError("why token is None")
        return token.content


class DeltaIsTextElement(BaseCommandTokenParserElement):
    """
    当 delta type 是 text 时, 这种解析逻辑是所有的中间 token 都视作文本
    等所有的文本都加载完, 才会发送这个 task.
    """

    _inner_content = ""

    def on_delta_token(self, token: CommandToken) -> None:
        self._inner_content += token.content

    def on_init(self) -> list[CommandTask] | None:
        # 开始时不要执行什么.
        return None

    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask] | None:
        self._inner_content += token.content
        return None

    def on_own_end(self) -> list[CommandTask] | None:
        result = super().on_own_end()
        if self._current_task is not None:
            current_task_meta = self._current_task.meta
            delta_arg_name = current_task_meta.delta_arg
            deltas_exists_value = self._current_task.kwargs.get(delta_arg_name, "")
            # 做全文赋值.
            deltas_value = deltas_exists_value
            if len(self._inner_content) > 0:
                deltas_value = self._inner_content
            self._current_task.kwargs[CommandDeltaType.TEXT.value] = deltas_value
            if not self._inner_content:
                attrs = self._current_task.kwargs.copy()
                del attrs[CommandDeltaType.TEXT.value]
                self._current_task.tokens = CMTLSaxElement.make_start_mark(
                    self._current_task.chan,
                    current_task_meta.name,
                    attrs=attrs,
                    self_close=True,
                )
            else:
                start_tokens = self._current_task.tokens
                self._current_task.tokens = start_tokens + self._inner_content + f"</{self._current_task.meta.name}>"
            self._send_callback(self._current_task)
        self._end = True
        result = result or []
        result.append(self._current_task)
        return result

    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask] | None:
        self._inner_content += token.content
        return None


class RootCommandTaskElement(NoDeltaCommandTaskElement):
    def on_token(self, token: CommandToken | None) -> None:
        if self._is_root_token(token):
            if token.seq == "start":
                return
            elif token.seq == "end":
                self._send_callback(None)
                self.on_own_end()
                return
        return super().on_token(token)
