import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Callable, Iterable, AsyncIterable
from typing_extensions import Self
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.concepts.command import CommandTask, CommandToken
from ghoshell_moss.core.concepts.channel import ChannelFullPath, ChannelMeta
from ghoshell_moss.core.concepts.tools import ToolMeta, CommandAsTool
from ghoshell_moss.message import Message
from ghoshell_common.contracts import LoggerItf
from pydantic import BaseModel, Field
import queue

__all__ = [
    "CommandTaskCallback",
    "CommandTokenParser",
    "CommandTokenCallback",
    "TextTokenParser",
    "Interpreter",
    "Interpretation",
]

CommandTokenCallback = Callable[[CommandToken | None], None]
CommandTaskCallback = Callable[[CommandTask | None], None]


class TextTokenParser(ABC):
    """
    parse from string stream into command tokens
    """

    @abstractmethod
    def with_callback(self, *callbacks: CommandTokenCallback) -> None:
        """
        注册生成 command token 的回调.
        send command token to callback method
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """weather this parser is done parsing."""
        pass

    @abstractmethod
    def feed(self, delta: str) -> None:
        """feed this parser with the stream delta"""
        pass

    @abstractmethod
    def commit(self) -> None:
        """notify the parser that the stream is done"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        立刻停止解析, 也不会抛出异常.
        """
        pass

    @abstractmethod
    def buffered(self) -> str:
        """
        返回粘包后的输入文本.
        """
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandToken]:
        """返回已经生成的 command token"""
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        example for how to use parser manually
        """
        pass


class CommandTokenParser(ABC):
    """
    CommandTaskElement works like AST but in realtime.
    It accepts command token from a stream, and generate command task concurrently.

    The keypoint is, the command tokens are organized in the recursive pattern,
    that one command can embrace many children command within it, and handle them by its own means,
    just like a function call other functions inside it.

    So we need an Element Tree to parse the tokens into command tasks, and send the tasks immediately
    """

    @abstractmethod
    def on_token(self, token: CommandToken | None) -> list[CommandTask] | None:
        """
        接受一个 command token
        :param token: 如果为 None, 表示 command token 流已经结束.
        """
        pass

    @abstractmethod
    def is_end(self) -> bool:
        """是否解析已经完成了."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    @abstractmethod
    def destroy(self) -> None:
        """手动清理数据结构, 加快垃圾回收, 避免内存泄漏"""
        pass


class Interpretation(BaseModel):
    """
    Interpreter 一次运行的结果.
    """

    done: bool = Field(default=False, description="是否已经运行结束.")
    id: str = Field(description="interpretation id")
    observe: bool = Field(
        default=False,
        description="这个运行结果是否需要 AI 观察",
    )
    feed_inputs: list[str] = Field(default_factory=list, description="通过 interpreter feed 输入的文本")
    command_tokens: list[CommandToken] = Field(
        default_factory=list,
        description="运行时解析生成的 command tokens",
    )
    executed_inputs: list[str] = Field(default_factory=list, description="被执行过的输入文本.")

    compiled_tasks: dict[str, str] = Field(default_factory=dict, description="解析生成的 task 的 cid => task caller")
    pending_tasks: dict[str, str] = Field(default_factory=dict, description="未完成的 task 的 cid => task caller")
    cancelled_tasks: dict[str, str] = Field(
        default_factory=dict,
        description="运行结束的 task cid => task caller",
    )
    failed_tasks: dict[str, str] = Field(
        default_factory=dict,
        description="运行结束, 失败的 task cid => task caller",
    )
    success_tasks: dict[str, str] = Field(
        default_factory=dict, description="运行结束, 并且运行成功的 task cid => task caller"
    )
    output: list[Message] = Field(default_factory=list, description="运行结果中需要输出的消息体. ")
    messages: list[Message] = Field(default_factory=list, description="运行结果中需要观察的消息体.")
    interrupted: bool = Field(default=False, description="是否被强行打断")
    exception: str = Field(
        default="",
        description="运行的异常",
    )

    def executed_logos(self) -> str:
        return "".join(self.executed_inputs)

    def on_task_compiled(self, task: CommandTask | None) -> None:
        """注册 task 编译状态. """
        if task is None or task.meta.name.startswith("_"):
            return
        self.compiled_tasks[task.cid] = task.caller_name()
        self.pending_tasks[task.cid] = task.caller_name()

    def on_done_task(self, task: CommandTask) -> None:
        """注册 task 的回调. """
        if not task.done() or task.meta.name.startswith("_"):
            return
        if self.done:
            return
        task_id = task.cid
        if task_id in self.pending_tasks:
            self.pending_tasks.pop(task_id)
        # 注册执行成功的 tokens.
        if task.success():
            self.executed_inputs.append(task.tokens)
            self.success_tasks[task_id] = task.caller_name()
        # 记录 cancel 类别的.
        elif CommandErrorCode.is_cancelled(task.errcode):
            self.cancelled_tasks[task_id] = task.caller_name()
        # 记录异常的.
        else:
            self.failed_tasks[task_id] = task.caller_name()

        # 合并 task 运行结果.
        result = task.task_result()
        # 根据协议判定要 observe.
        if result.observe or CommandErrorCode.is_critical(task.errcode):
            self.observe = True
        if len(result.output) > 0:
            self.output.extend(result.output)
        result_messages = result.as_messages()
        if len(result_messages) > 0:
            self.messages.extend(result_messages)

    def output_messages(self) -> list[Message]:
        """
        提供给对客户端输出的消息.
        """
        return self.output.copy()

    def status_messages(self) -> list[Message]:
        """当前运行状态的描述. """
        status_message = Message.new()
        lines = []
        if self.interrupted:
            lines.append("Interrupted!")
        if self.exception:
            lines.append("Exception: %s" % self.exception)
        if len(self.success_tasks) > 0:
            lines.append("success: %d" % len(self.success_tasks))
        if len(self.cancelled_tasks) > 0:
            lines.append("canceled: %d" % len(self.cancelled_tasks))
        if len(self.failed_tasks) > 0:
            lines.append("failed: %d" % len(self.failed_tasks))
        if len(self.pending_tasks) > 0:
            lines.append("pending: %s" % ",".join(self.pending_tasks.values()))
        if len(lines) > 0:
            status_message.with_content("\n".join(lines))
            return [status_message]
        else:
            return []

    def executed_messages(self) -> list[Message]:
        """运行结果的描述"""
        messages = self.messages.copy()
        return messages

    def as_messages(self) -> list[Message]:
        messages = self.status_messages()
        messages.extend(self.executed_messages())
        return messages


class Interpreter(ABC):
    """
    命令解释器, 从一个文本流中解析 command token.
    同时将流式的 command token 解析为流式的 command task, 然后回调给执行器.

    它本身可以认为是 Shell 运行状态的关键帧.
    Shell 同一个时间只会创建一个有状态的 Interpreter, 如果上一个还未运行结束, 则会中断它.

    中断的方式有两种, clear / append
    clear 会清空上一个 Interpreter 所有的状态.
    append 则只会中断上一个 Interpreter 的运行.

    上一个 interpreter 是被临时中断的, 它的运行结果, 会传递给下一个 interpreter
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """each time stream interpretation has a unique id"""
        pass

    @property
    @abstractmethod
    def kind(self) -> str:
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        pass

    @abstractmethod
    def previews(self) -> Interpretation | None:
        """
        上一轮被中断的解释结果.
        """
        pass

    @abstractmethod
    def interpretation(self) -> Interpretation:
        """
        返回当前的 interpretation
        它可能仍然在运行中, 会不断添加新信息.
        """
        pass

    @abstractmethod
    def channels(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回当前 interpreter 的所有 channels.
        """
        pass

    @abstractmethod
    def meta_instruction(self) -> str:
        """
        给大模型使用 MOSS 的元规则.
        具体的 interpreter 可以定义不同的规则.
        举例: CTMLInterpreter 定义的是 CTML 规则.
        """
        pass

    @abstractmethod
    def static_messages(self) -> str:
        """
        当前 interpreter 状态下, channels 的完整提示词. 用于呈现给大模型.
        """
        pass

    def instruction(self, prompts: list[str] | None = None) -> str:
        """
        MOSS 架构默认的 system prompt.
        """
        instructions = [self.meta_instruction()]
        channel_instructions = self.static_messages()
        instructions.append(channel_instructions)
        if prompts:
            instructions.extend(prompts)
        return '\n'.join(instructions)

    @abstractmethod
    def dynamic_messages(self) -> list[Message]:
        """
        返回 interpreter 作为快照拿到的动态上下文.
        """
        pass

    def merge_messages(self, history: list[Message | dict], inputs: list[Message | dict]) -> list[Message]:
        """
        遵循系统规则合并消息体, 生成一个模型上下文.
        此处也是提示如何使用 interpreter 来定义上下文.

        在 Model Context 对话历史中, 可以认为最简单的上下文拓扑是:

        - instructions: 提示和指令. 尽可能少变更, 而且需要合并.
        - conversations: 对话历史.
        - last turn: 上一轮的输入和输出消息.
        - context: 当前的状态, 可变的部分. 而且要让模型理解这块是随时变化的.
        + new turn:
            - inputs: turn-based Model 本轮的输入.
            - recall: 结合上下文, 自动生成的 recall
            - reasoning: 思考过程
            - actions: 行动过程.
            - outputs: 输出
            - observation: 需要观察的讯息.
        """
        instructions = self.instruction()
        messages = [Message.new(tag="").with_content(instructions)]
        messages.extend(history)
        messages.extend(self.dynamic_messages())
        messages.extend(inputs)
        return messages

    @abstractmethod
    def feed(self, delta: str, throw: bool = True) -> bool:
        """
        向 interpreter 提交文本片段, interpreter 会异步解析这些输入流, 并且执行调度逻辑.
        >>> async def run_interpreter(interpreter: Interpreter, items: AsyncIterable[str]):
        >>>     async with interpreter:
        >>>         async for item in items:
        >>>             interpreter.feed(item)
        >>>         interpreter.commit()

        :param delta: 传输的文本片段.
        :param throw: 设置为 True, 如果解析过程异常, 会抛出 error. 可以用来做中断.
        :raise InterpreterError:
        :return: 如果状态正常, 提交成功返回 True, 否则返回 False.
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        标记所有的输入已经结束. 后续的 feed 不再生效.
        注意, 这时 interpreter 的解析流程, 执行流程可能尚未完成.
        """
        pass

    async def interpret(self, deltas: AsyncIterable[str]) -> None:
        """
        语法糖, 一个完整的解析过程, 需要包含 feed 和 commit.
        """
        async for delta in deltas:
            if not self.feed(delta):
                break
        self.commit()

    @abstractmethod
    def on_task_compiled(self, *callbacks: CommandTaskCallback) -> None:
        """
        注册 task 被创建时候的回调.
        """
        pass

    @abstractmethod
    def on_task_done(self, *callbacks: CommandTaskCallback) -> None:
        """
        注册 task 运行完毕时的回调.
        """
        pass

    @abstractmethod
    def text_token_parser(self) -> TextTokenParser:
        """
        interpreter 持有的 Token 解析器. 将文本输入解析成 command token, 同时将 command token 解析成 command task.
        command task 会自动回调 interpreter 执行.

        >>> def example(interpreter: Interpreter, deltas: AsyncIterable[str]) -> None:
        >>>     with interpreter.text_token_parser() as parser:
        >>>         async for delta in deltas:
        >>>             parser.feed(delta)

        注意 Parser 是同步阻塞的, 因此正确的做法是使用 interpreter 自带的 feed 函数实现非阻塞.
        通常 parser 运行在独立的线程池中.
        """
        pass

    @abstractmethod
    def command_token_parser(self) -> CommandTokenParser:
        """
        当前 Interpreter 做树形 Command Token 解析时使用的 Element 对象. debug 用.
        通常运行在独立的线程池中.
        """
        pass

    @abstractmethod
    def parsed_tokens(self) -> Iterable[CommandToken]:
        """
        已经解析生成的 command tokens.
        """
        pass

    @abstractmethod
    def received_text(self) -> str:
        """
        返回已经完成输入的文本内容. 必须通过 feed 输入.
        """
        pass

    @abstractmethod
    def compiled_tasks(self) -> dict[str, CommandTask]:
        """
        已经解析生成的 tasks.
        """
        pass

    @abstractmethod
    def managing_tasks(self) -> dict[str, CommandTask]:
        """
        管理的 tasks, 可能包含上一轮生成的.
        """
        pass

    def done_tasks(self) -> list[CommandTask]:
        """
        返回已经被执行的 tasks. 包含被取消或者出错的.
        """
        tasks = self.managing_tasks().copy()
        executed = []
        for task in tasks.values():
            if not task.done():
                continue
            executed.append(task)
        return executed

    def incomplete_tasks(self) -> list[CommandTask]:
        """
        返回已经解析成功, 但没有被执行完的 tasks.
        """
        tasks = self.managing_tasks().copy()
        pending = []
        for task in tasks.values():
            if not task.done():
                pending.append(task)
        return pending

    def executed_tokens(self) -> str:
        """
        返回当前已经执行完毕的 tokens.
        """
        tokens = []
        for task in self.done_tasks():
            tokens.append(task.tokens)
        return "".join(tokens)

    @abstractmethod
    async def close(
            self,
            cancel_executing: bool = True,
    ) -> Interpretation | None:
        """
        stop the interpretation
        :param cancel_executing: 是否同时清空解析出来的任务. 不清空的话, 任务本身并不会被中断.
        :return: 如果中断了一个未完成的 Interpreter, 返回已经执行的解释状态. 如果已经完成了, 则返回 None.
        """
        pass

    @abstractmethod
    def is_stopped(self) -> bool:
        """
        判断解释过程是否还在执行中.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否正在运行中: start -> stop 中间.
        """
        pass

    @abstractmethod
    def is_interrupted(self) -> bool:
        """
        解释过程是否被中断.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        example to use the interpreter:
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def exception(self) -> Optional[Exception]:
        """
        返回运行过程中产生的异常.
        """
        pass

    def raise_exception(self):
        if exp := self.exception():
            raise exp

    @abstractmethod
    async def wait_compiled(self, timeout: float | None = None) -> None:
        """
        等待解释过程完成. 完成有两种情况:
        1. 输入已经完备.
        2. 被中断.
        """
        pass

    @abstractmethod
    async def wait_stopped(self) -> Interpretation:
        """
        阻塞等待到运行结束或者系统被中断.
        然后返回 interpretation.
        不意味着它生成的 tasks 已经都被执行完毕了.
        """
        pass

    @abstractmethod
    async def wait_tasks(
            self,
            timeout: float | None = None,
            *,
            return_when: str = asyncio.ALL_COMPLETED,
            throw: bool = False,
            clear_undone: bool = True,
    ) -> dict[str, CommandTask]:
        """
        阻塞等待所有生成的 task, 并且按 return when 的规则返回.
        :param timeout: 设置等待的超时时间.
        :param throw: 如果 task 运行遇到异常了, 是否对外抛出.
        :param return_when: 退出 wait execution done 的时机.
        :param clear_undone: 退出这个函数时, 是否要设置未完成的 Task 为 Cleared
        """
        pass

    # --- tools 兼容.  --- #

    @abstractmethod
    def tools(self) -> Iterable[CommandAsTool]:
        """
        openai & anthropic & pydantic ai compatible tool
        """
        pass

    # --- interpreter 的无状态解析函数 --- #

    async def aparse_text_to_command_tokens(
            self,
            texts: AsyncIterable[str],
            *,
            stopped: Callable[[], bool] | None = None,
    ) -> AsyncIterable[CommandToken]:
        """
        将同步函数封装成异步函数, 同时仍然能正确抛出异常.
        """
        text_queue = queue.Queue()
        token_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def callback(token: CommandToken | None) -> None:
            loop.call_soon_threadsafe(token_queue.put_nowait, token)

        def real_stop():
            """
            判定强行中断实机.
            """
            nonlocal stop_event
            if stop_event.is_set():
                return True
            if stopped and stopped():
                return True
            return False

        async def consume():
            """
            消费传入的 texts.
            """
            nonlocal texts
            async for text in texts:
                text_queue.put(text)
            text_queue.put(None)

        cor = asyncio.to_thread(self.parse_text_to_command_tokens, text_queue, callback, stopped=real_stop)
        parsing_task = asyncio.create_task(cor)

        async def read_from():
            """
            读取消息.
            """
            while not real_stop():
                item = await token_queue.get()
                if item is None:
                    break
                yield item
            await parsing_task

        consume_task = asyncio.create_task(consume())
        try:
            async for got in read_from():
                yield got
        except asyncio.CancelledError:
            raise
        except Exception as e:
            text_queue.put(None)
            stop_event.set()
            self.logger.exception(
                "[Interpreter][%s] failed parsing text into command tokens: %r", self.__class__.__name__, e
            )
            raise e
        finally:
            # 冗余的回收.
            if not parsing_task.done():
                parsing_task.cancel()
            if not consume_task.done():
                consume_task.cancel()

    async def parse_tokens_to_command_tasks(
            self,
            tokens_queue: asyncio.Queue[CommandToken | None],
            task_callback: Callable[[CommandTask | None], None],
            *,
            stopped: Callable[[], bool] | None = None,
    ):
        """
        可以运行在协程中, 解析输入的 tokens 流, 返回 Command Tasks. 用毒丸做判断.
        raise InterpreterError
        """
        parser = self.command_token_parser()
        # parser.with_callback(task_callback)
        if stopped is None:
            def empty_stopped():
                return False

            stopped = empty_stopped
        try:
            with parser:
                while not stopped() and not parser.is_end():
                    try:
                        item = await asyncio.wait_for(tokens_queue.get(), 0.2)
                    except asyncio.TimeoutError:
                        continue
                    if item is None:
                        break
                    tasks = parser.on_token(item)
                    if tasks is not None:
                        for task in tasks:
                            task.on_compiled()
                            task_callback(task)
                    await asyncio.sleep(0.0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception(
                "[Interpreter][%s] failed parsing tokens into command tasks: %r", self.__class__.__name__, e
            )
            raise e
        finally:
            task_callback(None)
            parser.destroy()

    def parse_text_to_command_tokens(
            self,
            text_queue: queue.Queue[str | None],
            command_token_callback: Callable[[CommandToken | None], None],
            *,
            stopped: Callable[[], bool] | None = None,
    ):
        """
        通常运行在独立线程中, 解析输入的 Text 流, 返回 Command Token 流. 用毒丸做判断.
        raise InterpreterError
        """
        text_token_parser = self.text_token_parser()
        text_token_parser.with_callback(command_token_callback)
        if stopped is None:
            def empty_stopped():
                return False

            stopped = empty_stopped
        with text_token_parser:
            while not text_token_parser.is_done():
                if stopped():
                    text_token_parser.stop()
                    break
                try:
                    # check every 0.1 second if the loop is stopped.
                    item = text_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    text_token_parser.commit()
                    break
                text_token_parser.feed(item)
