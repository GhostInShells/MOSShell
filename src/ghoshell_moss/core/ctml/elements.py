from abc import ABC, abstractmethod
from contextlib import contextmanager
from logging import getLogger
from typing import Optional, Generic, Any, ClassVar, AsyncIterator, Callable

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.concepts.command import (
    BaseCommandTask,
    CancelAfterOthersTask,
    Command,
    CommandDeltaArgName,
    CommandDeltaArgType,
    CommandDeltaArgName2TypeMap,
    CommandTask,
    CommandToken,
    CommandTokenSeq,
    PyCommand,
    CommandMeta,
    TaskScope,
)
from ghoshell_moss.core.concepts.errors import InterpretError, CommandErrorCode
from ghoshell_moss.core.concepts.interpreter import (
    CommandTaskCallback,
    CommandTokenParser,
)
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.contracts.speech import Speech, SpeechStream
from ghoshell_moss.core.helpers.stream import create_sender_and_receiver, ItemT, ThreadSafeStreamSender
from ghoshell_moss.core.ctml.v1_0_0.constants import (
    CONTENT_COMMAND_NAME, SCOPE_COMMAND_NAME,
    SCOPE_SHORTCUT, SCOPE_ENTER_COMMAND_NAME, SCOPE_EXIT_COMMAND_NAME,
)
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


# !!! 这是项目里最大的屎山 (之一?)
# 在晕头转向的熬夜中开发完, 有价值的是 feature 和单元测试.
# 只有保持单元测试向前兼容时, 才可以改动....
# 除非重写.

async def invalid_command():
    task = ChannelCtx.task()
    raise CommandErrorCode.NOT_FOUND.error(f"command {task.caller_name()} not found")


invalid_command = PyCommand(invalid_command)


class ScopeOpenTask(BaseCommandTask[None]):
    """
    start a channel scope
    """

    def __init__(self, group: TaskScope, tag: str = ''):
        self._group = group
        meta = CommandMeta(
            name=SCOPE_ENTER_COMMAND_NAME,
            chan=group.channel,
            blocking=True,
        )
        if tag:
            attrs_lines = []
            if group.channel:
                attrs_lines.append(f'channel="{group.channel}"')
            if group.until and group.until != group.default_until:
                attrs_lines.append(f'until="{group.until}"')
            if group.timeout is not None and group.timeout > 0.0:
                attrs_lines.append(f'timeout="{group.timeout}"')
            attrs_str = ' '.join(attrs_lines)
            tokens = f"<{tag}{attrs_str}>"
        else:
            tokens = ""
        super().__init__(
            chan=group.channel,
            meta=meta,
            func=self.start_scope,
            partial=None,
            tokens=tokens,
            args=[],
            kwargs={},
        )
        # 被持有的对象也包括自身.
        group.add(self)

    async def start_scope(self):
        # 首次被执行时, 正式开始记账.
        _ = self._group.tick()


class ScopeCloseTask(BaseCommandTask[str]):
    """
    close a channel scope
    """

    def __init__(self, group: TaskScope, tag: str = ''):
        self._group = group
        meta = CommandMeta(
            name=SCOPE_EXIT_COMMAND_NAME,
            chan=group.channel,
            blocking=True,
        )
        tokens = f"</{tag}>" if tag else ""
        super().__init__(
            chan=group.channel,
            meta=meta,
            func=self.end_scope,
            partial=None,
            tokens=tokens,
            args=[],
            kwargs={},
        )
        group.compiled()

        def _cancel_group(task: CommandTask) -> None:
            nonlocal group
            group.cancel()

        # 自己结束时, 也会 cancel 整个 group
        self.add_done_callback(_cancel_group)

    async def end_scope(self) -> None:
        try:
            await self._group.wait()
        finally:
            self._group.cancel()


class EmptyContentTask(BaseCommandTask[None]):

    def __init__(
            self,
            cid: str,
            channel: str,
            chunks__: AsyncIterator[str],
            call_id: str | int | None = None,
    ):
        meta = CommandMeta(
            name=CONTENT_COMMAND_NAME,
            chan=channel,
            blocking=True,
        )
        super().__init__(
            chan=channel,
            meta=meta,
            partial=None,
            func=self.__content__,
            tokens='',
            args=[],
            cid=cid,
            call_id=call_id,
            kwargs={'chunks__': chunks__},
        )

    @staticmethod
    async def __content__(chunks__: AsyncIterator[str]) -> tuple[list, dict]:
        async for chunk in chunks__:
            pass
        return [], {}


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
        self.delta_type_map = delta_type_map or CommandDeltaArgName2TypeMap.copy()
        self._callback = callback
        self._delivered_last_callback = False
        CommandTaskElementContext.instances_count += 1

    def __del__(self):
        self.speech = None
        self.channel_commands_map.clear()
        CommandTaskElementContext.instances_count -= 1

    def new_root(self, callback: CommandTaskCallback | None, stream_id: str = "") -> "RootCommandTaskElement":
        """
        创建解析树的根节点.
        """
        self.logger.info(
            "[CommandTaskElementContext] create root element, instances count %d, element instances count %d",
            CommandTaskElementContext.instances_count,
            BaseCommandTokenParserElement.instances_count,
        )
        root = RootCommandTaskElement(
            self.root_tag,
            parent_add_inner_task=None,
            chan="",
            stream_id=stream_id,
            cid=stream_id,
            current_task=None,
            ctx=self,
        )
        if callback is not None:
            root.with_callback(callback)
        return root

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
    def new_parser(self, callback: CommandTaskCallback | None, stream_id: str = ""):
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
            parent_add_inner_task: Callable[[CommandTask], None] | None,
            *,
            stream_id: str,
            cid: str,
            chan: str,
            current_task: Optional[CommandTask],
            depth: int = 0,
            ctx: CommandTaskElementContext,
            scope: TaskScope = None,
    ) -> None:
        self._name = name
        self.chan = chan
        self._parent_add_inner_task = parent_add_inner_task
        self.stream_id = stream_id
        self.cid = cid
        self.ctx = ctx
        self.depth = depth
        self.scope = scope or None
        self.current_task: Optional[CommandTask] = current_task
        """当前的 task. 每个节点默认都由一个 Task 创建. """

        self.inner_tasks: list[CommandTask] = []
        """在自己内部发送的各种 tasks."""

        self.children: list[BaseCommandTokenParserElement] = []
        """所有的子节点"""

        self._unclose_child: Optional[CommandTokenParser] = None
        """没有结束的子节点"""

        self._end = False
        """这个 element 是否已经结束了"""

        self._current_stream: Optional[SpeechStream] = None
        """当前正在发送的 output stream"""

        # 正式启动.
        self._has_inner_tokens = False
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
        self.destroy()
        BaseCommandTokenParserElement.instances_count -= 1

    def _add_to_parent(self, task: CommandTask) -> None:
        if task is None:
            return None
        if self._parent_add_inner_task is not None:
            self._parent_add_inner_task(task)
        return None

    def _add_inner_task(self, task: CommandTask) -> None:
        if task is not None:
            # 添加 children tasks
            self.inner_tasks.append(task)
            if self.scope:
                self.scope.add(task)

    def is_end(self) -> bool:
        return self._end

    def raise_interrupt(self, err: Exception | str = ''):
        raise InterpretError(f"Command Task parse failed: {err}")

    def on_token(self, token: CommandToken | None) -> list[CommandTask] | None:
        try:
            if token is None:
                return None
            result = self._on_token(token)
            if len(result) == 0:
                return []
            return result
        except InterpretError as e:
            self.ctx.logger.exception("%s on_token %s failed: %s", self._log_prefix, token, e)
            self.fail(e)
            raise e
        except Exception as e:
            self.ctx.logger.exception("%s on token %s failed: %s", self._log_prefix, token, e)
            self.fail(e)
            self.raise_interrupt("system error")
            return []

    def fail(self, error: Exception) -> None:
        """
        递归处理异常.
        """
        if not self.is_end():
            self._end = True
            self.ctx.logger.exception("%s failed: %s", self._log_prefix, error)
        if self.current_task is not None:
            self.current_task.fail(error)
        if self.scope is not None:
            self.scope.cancel("failed")
        if isinstance(error, InterpretError):
            if len(self.inner_tasks) == 0:
                return
            for t in self.inner_tasks:
                if not t.done():
                    t.fail(error)

    def _on_token(self, token: CommandToken | None) -> list[CommandTask]:
        """
        当前节点得到了一个新的 command token.
        """
        if token is None:
            # 结束自己的生命.
            return self.on_own_end()
        if self.is_end():
            self.ctx.logger.warning("%s receive token %s after element is end", self._log_prefix, token)
            return []

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
            self._has_inner_tokens = True
            return self.on_delta_token(token)
        # 接受一个 end token
        elif token.seq == CommandTokenSeq.END:
            if token.command_id() == self.cid:
                # 结束自身.
                return self.on_own_end()
            return self.on_sub_end_token(token)
        # 接受一个 start token.
        elif token.seq == CommandTokenSeq.START:
            self._has_inner_tokens = True
            # 是自己就不太对了.
            if token.command_id() == self.cid:
                self.ctx.logger.error("%s received duplicated start command %s", self._log_prefix, token)
                self.raise_interrupt()
                return []
            # 否则当成一个正常的 token.
            return self.on_sub_start_token(token)
        else:
            self.ctx.logger.error("%s received invalid command token %s", self._log_prefix, token)
            self.raise_interrupt()
            return []

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

    def _new_child_element(self, token: CommandToken) -> list[CommandTask]:
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
        # 判断这个 token 是不是 scope .
        command = self._find_command(token.chan, token.name)
        if token.name == SCOPE_COMMAND_NAME or token.name == SCOPE_SHORTCUT:
            timeout = token.kwargs.get("timeout", None)
            if timeout is not None:
                timeout = float(timeout)
            scope = TaskScope(
                channel=token.chan,
                until=token.kwargs.get("until", "flow"),
                timeout=timeout,
            )
            child = CommandWithoutDeltaArgElement(
                name=Command.make_unique_name(token.chan, SCOPE_COMMAND_NAME),
                parent_add_inner_task=self._add_inner_task,
                chan=token.chan,
                stream_id=self.stream_id,
                cid=token.command_id(),
                current_task=None,
                scope=scope,
                ctx=self.ctx,
                depth=self.depth + 1,
            )
        elif command is None:
            if self.ctx.ignore_wrong_command:
                self.ctx.logger.warning(
                    "%s ignore wrong command %s, create empty one",
                    self._log_prefix,
                    token,
                )
                child = CommandWithoutDeltaArgElement(
                    name=Command.make_unique_name(token.chan, token.name),
                    parent_add_inner_task=self._add_inner_task,
                    chan=token.chan,
                    stream_id=self.stream_id,
                    cid=token.command_id(),
                    current_task=None,
                    # 提供递归的 task 传递路径.
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
                if delta_value_type is CommandDeltaArgType.COMMAND_TOKEN_STREAM:
                    child = DeltaIsCommandTokensElement(
                        name=task.caller_name(),
                        parent_add_inner_task=self._add_inner_task,
                        chan=token.chan,
                        stream_id=self.stream_id,
                        cid=token.command_id(),
                        current_task=task,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )
                # 接受 AsyncIterable[Chunk] 的类型.
                elif delta_value_type is CommandDeltaArgType.TEXT_CHUNKS_STREAM:
                    child = DeltaIsTextChunkElement(
                        name=task.caller_name(),
                        parent_add_inner_task=self._add_inner_task,
                        chan=token.chan,
                        stream_id=self.stream_id,
                        cid=token.command_id(),
                        current_task=task,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )
                # 接受 text__ 的类型.
                elif delta_value_type is CommandDeltaArgType.TEXT:
                    child = DeltaIsTextElement(
                        name=task.caller_name(),
                        parent_add_inner_task=self._add_inner_task,
                        chan=token.chan,
                        stream_id=token.command_id(),
                        cid=token.command_id(),
                        current_task=task,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )
                else:
                    self.ctx.logger.error("%s command delta type %s is not implemented", meta.delta_arg)
                    child = CommandWithoutDeltaArgElement(
                        name=task.caller_name(),
                        parent_add_inner_task=self._add_inner_task,
                        chan=token.chan,
                        stream_id=self.stream_id,
                        cid=token.command_id(),
                        current_task=task,
                        ctx=self.ctx,
                        depth=self.depth + 1,
                    )

            else:
                child = CommandWithoutDeltaArgElement(
                    name=task.caller_name(),
                    parent_add_inner_task=self._add_inner_task,
                    chan=token.chan,
                    stream_id=self.stream_id,
                    cid=token.command_id(),
                    current_task=task,
                    ctx=self.ctx,
                    depth=self.depth + 1,
                )

        if child is not None:
            # 把所有子孙都拿着. 恨不得....
            self.children.append(child)
            if not child.is_end():
                # 记录 unclose.
                self._unclose_child = child
            # 如果
            if self.scope and child.current_task is not None:
                # 添加到 scope 里.
                self.scope.add(child.current_task)
            return child.on_init()
        return []

    @abstractmethod
    def on_delta_token(self, token: CommandToken) -> list[CommandTask]:
        """
        每个节点都要考虑, 拿到了属于自己的 delta token 怎么办.
        """
        pass

    @abstractmethod
    def on_init(self) -> list[CommandTask]:
        """
        每个节点初始化的逻辑.
        通常是在初始化时, 就发送 command task.
        """
        pass

    @abstractmethod
    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask]:
        """
        处理拿到了一个开始标记的 token. 这个不是来自自己的 Token.
        """
        pass

    @abstractmethod
    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask]:
        """
        拿到了一个结束标记的 Token. 不是自己的 Token.
        """
        pass

    def on_own_end(self) -> list[CommandTask]:
        """
        拿到了自身的结束 Token
        """
        self._end = True
        result = []
        self.ctx.logger.debug("%s end self", self._log_prefix)
        return result

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
        del self.current_task


# 已经废弃的实现, 用 ChannelScopeElement 替代.
class NoDeltaCommandTaskElement(BaseCommandTokenParserElement):
    """
    没有 delta 参数的节点类型.
    也就是说这种类型的 Command 不支持 delta 数据, 也不支持子节点.
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
            self._add_inner_task(output_stream_task)
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
            self._add_inner_task(output_stream_task)
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
        if self.current_task is not None:
            # 发送自己的 Task.
            return [self.current_task]
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
            return None
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
                "%s element end current task %s with invalid token %r", self._log_prefix, self.current_task, token
            )
            # 自己来处理这个 token, 但 command id 不一致的情况.
            self.raise_interrupt()
            return None
        else:
            # 结束自身.
            # 理论上外部可以调用.
            return None

    def _clear_speech_stream(self) -> None:
        if self._speech_stream is not None:
            # 发送未发送的 output stream.
            self._speech_stream.commit()
            self._speech_stream = None

    def on_own_end(self) -> list[CommandTask]:
        # 设置关闭.
        result = super().on_own_end()
        self._clear_speech_stream()
        if self.current_task is None:
            return result
        elif len(self.inner_tasks) > 0:
            cancel_after_children_task = CancelAfterOthersTask(
                self.current_task,
                *self.inner_tasks,
            )
            cancel_after_children_task.tokens = CMTLSaxElement.make_end_mark(
                self.current_task.chan,
                self.current_task.meta.name,
            )
            # 等待所有 children tasks 完成, 如果自身还未完成, 则取消.
            result.append(cancel_after_children_task)
            return result
        else:
            # 按照 ctml 的规则, 修改 task 的开启标记. 用来做开标记逻辑.
            meta = self.current_task.meta
            self.current_task.tokens = CMTLSaxElement.make_start_mark(
                chan=meta.chan,
                name=meta.name,
                attrs=self.current_task.kwargs,
                self_close=True,
            )
            return result

    def destroy(self) -> None:
        self._clear_speech_stream()
        super().destroy()


class CommandWithoutDeltaArgElement(BaseCommandTokenParserElement):
    """
    没有 delta 参数的节点类型.
    也就是说这种类型的 Command 不支持 delta 数据, 也不支持子节点.
    基于 CTML 1.0 的规则, 我们把这种
    """

    _current_content_stream_sender: ThreadSafeStreamSender | None = None
    _current_content_task: CommandTask | None = None
    _current_content_task_delivered: bool = False
    _buffer_stream_content: str = ""
    _self_task_delivered: bool = False

    def _create_new_content_task(self, token: CommandToken) -> tuple[ThreadSafeStreamSender, CommandTask]:
        sender, receiver = create_sender_and_receiver()
        command = self._find_command(token.chan, CONTENT_COMMAND_NAME)
        if command is not None:
            task = BaseCommandTask.from_command(
                command,
                kwargs={CommandDeltaArgName.CHUNKS.value: receiver},
                cid=token.command_part_id(),
                call_id=token.call_id,
            )
        else:
            task = EmptyContentTask(
                channel=token.chan,
                chunks__=receiver,
                cid=token.command_part_id(),
                call_id=token.call_id,
            )
        return sender, task

    def on_delta_token(self, token: CommandToken) -> list[CommandTask]:
        """
        接受到中间的 token 比如当前是 foo
        <foo> hello <bar/> world </foo>
        会接收到的 delta token 有 hello 和 world.
        """
        result = self._deliver_self(with_scope=True)
        new_task = None
        # 没有创建过 content stream.
        if self._current_content_task is None:
            # 没有创建过 content stream, 则创建一个.
            # 用来处理需要发送的 delta content.
            self._buffer_stream_content += token.content
            sender, new_task = self._create_new_content_task(token)
            sender.append(token.content)
            self._current_content_stream_sender = sender
            self._current_content_task = new_task

        # 如果不是同一个流了.
        elif self._current_content_task.cid != token.command_part_id():
            # 创建过 output_stream, 则需要比较是否是相同的 command part id.
            # 不是相同的 command part id, 则需要创建一个新的流, 这样可以分段感知到每一段 output 是否已经执行完了.
            # 核心目标是, 当一个较长的 output 流被 command 分割成多段的话, 每一段都可以阻塞, 同时却可以提前生成 tts.
            # 这样生成 tts 的过程 add(token.content) 并不会被阻塞.
            result.extend(self._clear_content_stream())
            self._buffer_stream_content += token.content
            sender, new_task = self._create_new_content_task(token)
            sender.append(token.content)
            self._current_content_stream_sender = sender
            self._current_content_task = new_task
        else:
            # task 存在, 而且正好 buffer.
            self._current_content_stream_sender.append(token.content)
            self._buffer_stream_content += token.content
            if self._current_content_task:
                self._current_content_task.tokens = self._buffer_stream_content
            if not self._current_content_task_delivered:
                new_task = self._current_content_task

        # 消息终于不为空了, 才会第一次发送.
        if new_task is not None and self._buffer_stream_content.strip() != "":
            self._add_inner_task(new_task)
            self._current_content_task_delivered = True
            result.append(new_task)
        return result

    def on_init(self) -> list[CommandTask]:
        # 不着急发送命令.
        if self.scope is None:
            self.scope = TaskScope(
                channel=self.chan,
                until='flow',
                timeout=None,
            )
        return []

    def _deliver_self(self, with_scope: bool) -> list[CommandTask]:
        if self._self_task_delivered:
            return []
        self._self_task_delivered = True
        tasks = []
        # 有 scope 的情况下, 先发送 scope.
        if self.scope is not None and self._has_inner_tokens:
            # 如果是隐藏节点, tag 是 None
            tag = SCOPE_SHORTCUT if self.current_task is None else ''
            scope_task = ScopeOpenTask(self.scope, tag=tag)
            # 隐藏节点, 所以不对外暴露 token.
            self._add_inner_task(scope_task)
            tasks.append(scope_task)
        if self.current_task is not None:
            if not self._has_inner_tokens:
                self.current_task.tokens = f"<{self.current_task.caller_name()}/>"
            self._add_to_parent(self.current_task)
            tasks.append(self.current_task)
        return tasks

    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask]:
        result = self._deliver_self(with_scope=True)
        # 如果子节点还是开标签, 不应该走到这一环.
        if self._unclose_child is not None:
            self.ctx.logger.error(
                "%s Start new child command %s within unclosed command %s",
                self._log_prefix,
                token,
                self._unclose_child,
            )
            self.raise_interrupt()
            return result
        result.extend(self._clear_content_stream())
        result.extend(self._new_child_element(token))
        return result

    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask]:
        self._clear_content_stream()
        if self._unclose_child is not None:
            # 让子节点去处理.
            result = self._unclose_child.on_token(token)
            # 如果子节点处理完了, 自己也没了, 就清空.
            if self._unclose_child.is_end():
                self._unclose_child = None
            return result
        elif token.command_id() != self.cid:
            self.ctx.logger.error(
                "%s element end current task %s with invalid token %r", self._log_prefix, self.current_task, token
            )
            # 自己来处理这个 token, 但 command id 不一致的情况.
            self.raise_interrupt()
            return []
        else:
            # 结束自身.
            # 理论上外部可以调用.
            return []

    def _clear_content_stream(self) -> list[CommandTask]:
        result = []
        if self._current_content_task is not None:
            if not self._current_content_task_delivered and self._buffer_stream_content.strip() != "":
                result = [self._current_content_task]
            self._current_content_task.tokens = self._buffer_stream_content
        self._current_content_task = None
        self._current_content_task_delivered = False
        self._buffer_stream_content = ""
        if self._current_content_stream_sender is not None:
            # 发送未发送的 output stream.
            self._current_content_stream_sender.commit()
            self._current_content_stream_sender = None
        return result

    def on_own_end(self) -> list[CommandTask]:
        result = self._deliver_self(with_scope=False)
        result.extend(self._clear_content_stream())
        # 确认一下处理逻辑. 如果 scope 存在的话, 需要发送 scope 的闭包.
        if self.scope and self._has_inner_tokens:
            # 如果有任务存在, 则 scope exit 的 tokens 用 caller 来做.
            tag = SCOPE_SHORTCUT if self.current_task is None else self.current_task.caller_name()
            scope_close_task = ScopeCloseTask(self.scope, tag=tag)
            result.append(scope_close_task)
            self._add_to_parent(scope_close_task)
        # 设置关闭.
        result.extend(super().on_own_end())
        return result

    def destroy(self) -> None:
        self._clear_content_stream()
        super().destroy()


class EmptyCommandTaskElement(CommandWithoutDeltaArgElement):
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
            parent_add_inner_task: Callable[[CommandTask], None] | None,
            *,
            chan: str,
            stream_id: str,
            cid: str,
            current_task: Optional[CommandTask],
            depth: int = 0,
            ctx: CommandTaskElementContext,
    ) -> None:
        sender, receiver = create_sender_and_receiver()
        self._sender = sender
        self._receiver = receiver
        self._deltas: str = ""
        self._exists_delta_value = None
        super().__init__(
            name=name,
            parent_add_inner_task=parent_add_inner_task,
            stream_id=stream_id,
            cid=cid,
            current_task=current_task,
            chan=chan,
            depth=depth,
            ctx=ctx,
        )

    def on_init(self) -> list[CommandTask]:
        delta_arg_name = self.current_task.meta.delta_arg
        self._exists_delta_value = self.current_task.kwargs.get(delta_arg_name, None)
        self.current_task.kwargs[delta_arg_name] = self._receiver
        # 直接发送当前任务.
        self._add_to_parent(self.current_task)
        return [self.current_task]

    def on_delta_token(self, token: CommandToken) -> list[CommandTask]:
        self._deltas += token.content
        parsed = self._parse_delta(token)
        self._sender.append(parsed)
        return []

    @abstractmethod
    def _parse_delta(self, token: CommandToken) -> ItemT:
        pass

    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask]:
        parsed = self._parse_delta(token)
        self._sender.append(parsed)
        self._deltas += token.content
        return []

    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask]:
        parsed = self._parse_delta(token)
        self._deltas += token.content
        self._deltas += token.content
        self._sender.append(parsed)
        return []

    def on_own_end(self) -> list[CommandTask]:
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
        if token.seq == "start":
            self.ctx.logger.error("%s text chunks__ receive ctml token %s", self._log_prefix, token)
            raise InterpretError(f"`chunks__` do not allow ctml inside, and remember use CDATA to escape xml mark!")
        return token.content


class DeltaIsTextElement(BaseCommandTokenParserElement):
    """
    当 delta type 是 text 时, 这种解析逻辑是所有的中间 token 都视作文本
    等所有的文本都加载完, 才会发送这个 task.
    """

    _inner_content = ""

    def on_delta_token(self, token: CommandToken) -> list[CommandTask]:
        self._inner_content += token.content
        return []

    def on_init(self) -> list[CommandTask]:
        # 开始时不要执行什么.
        return []

    def on_sub_start_token(self, token: CommandToken) -> list[CommandTask]:
        self.ctx.logger.error("%s text text__ receive ctml token %s", self._log_prefix, token)
        raise InterpretError(f"`text__` do not allow ctml inside, and remember use CDATA to escape xml mark!")

    def on_sub_end_token(self, token: CommandToken) -> list[CommandTask]:
        self.ctx.logger.error("%s text text__ receive ctml token %s", self._log_prefix, token)
        raise InterpretError(f"`text__` do not allow ctml inside, and remember use CDATA to escape xml mark!")

    def on_own_end(self) -> list[CommandTask]:
        result = super().on_own_end()
        if self.current_task is not None:
            current_task_meta = self.current_task.meta
            delta_arg_name = current_task_meta.delta_arg
            deltas_exists_value = self.current_task.kwargs.get(delta_arg_name, "")
            # 做全文赋值.
            deltas_value = deltas_exists_value
            if len(self._inner_content) > 0:
                deltas_value = self._inner_content
            self.current_task.kwargs[CommandDeltaArgName.TEXT.value] = deltas_value
            if not self._inner_content:
                attrs = self.current_task.kwargs.copy()
                if delta_arg_name in attrs:
                    del attrs[delta_arg_name]
                self.current_task.tokens = CMTLSaxElement.make_start_mark(
                    self.current_task.chan,
                    current_task_meta.name,
                    attrs=attrs,
                    self_close=True,
                )
            else:
                start_tokens = self.current_task.tokens
                self.current_task.tokens = start_tokens + self._inner_content + f"</{self.current_task.meta.name}>"
        self._end = True
        result = result or []
        result.append(self.current_task)
        for t in result:
            self._add_to_parent(t)
        return result


class RootCommandTaskElement(CommandWithoutDeltaArgElement):
    _callback: Callable[[CommandTask | None], None] | None = None

    def with_callback(self, callback: Callable[[CommandTask | None], None]):
        self._callback = callback

    def on_token(self, token: CommandToken | None) -> list[CommandTask] | None:
        if self._is_root_token(token):
            if token.seq == "start":
                return []
            elif token.seq == "end":
                self.on_own_end()
                return []
        result = super().on_token(token)
        if self._callback is not None:
            if result is None:
                self._callback(None)
            else:
                for t in result:
                    self._callback(t)
        return result

    def _deliver_self(self, with_scope: bool) -> list[CommandTask]:
        return []
