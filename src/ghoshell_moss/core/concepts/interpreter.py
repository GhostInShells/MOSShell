import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Optional, Callable, Iterable, AsyncIterable

from typing_extensions import Self
from ghoshell_moss.core.concepts.command import CommandTask, CommandToken
from ghoshell_moss.core.concepts.channel import ChannelFullPath, ChannelMeta
from ghoshell_moss.message import Message

__all__ = [
    "CommandTaskCallback",
    "CommandTaskParseError",
    "CommandTokenParserElement",
    "CommandTokenCallback",
    "StringTokenParser",
    "Interpreter",
]

CommandTokenCallback = Callable[[CommandToken | None], None]
CommandTaskCallback = Callable[[CommandTask | None], None]


class CommandTaskParseError(Exception):
    pass


class StringTokenParser(ABC):
    """
    parse from string stream into command tokens
    """

    @abstractmethod
    def with_callback(self, *callbacks: CommandTokenCallback) -> None:
        """
        send command token to callback method
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """weather this parser is done parsing."""
        pass

    @abstractmethod
    def start(self) -> None:
        """start this parser"""
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
    def close(self) -> None:
        """
        stop the parser and clear the resources.
        """
        pass

    @abstractmethod
    def buffer(self) -> str:
        """
        return the buffered stream content
        """
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandToken]:
        """返回已经生成的 command token"""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        example for how to use parser manually
        """
        if exc_val is None:
            # ending is needed if parse success
            self.commit()
        self.close()


class CommandTokenParserElement(ABC):
    """
    CommandTaskElement works like AST but in realtime.
    It accepts command token from a stream, and generate command task concurrently.

    The keypoint is, the command tokens are organized in the recursive pattern,
    that one command can embrace many children command within it, and handle them by its own means,
    just like a function call other functions inside it.

    So we need an Element Tree to parse the tokens into command tasks, and send the tasks immediately
    """

    depth: int

    current: Optional[CommandTask] = None
    """the current command task of this element, created by `start` type command token"""

    children: dict[str, "CommandTokenParserElement"]
    """the children element of this element"""

    @abstractmethod
    def with_callback(self, callback: CommandTaskCallback) -> None:
        """设置一个 callback, 替换默认的 callback. 通常不需要使用."""
        pass

    @abstractmethod
    def on_token(self, token: CommandToken | None) -> None:
        """
        接受一个 command token
        :param token: 如果为 None, 表示 command token 流已经结束.
        """
        pass

    @abstractmethod
    def is_end(self) -> bool:
        """是否解析已经完成了."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """手动清理数据结构, 加快垃圾回收, 避免内存泄漏"""
        pass


class Interpreter(ABC):
    """
    命令解释器, 从一个文本流中解析 command token.
    同时将流式的 command token 解析为流式的 command task, 然后回调给执行器.

    The Command Interpreter that parse the LLM-generated streaming tokens into Command Tokens,
    and send the compiled command tasks into the shell executor.

    Consider it a one-time command parser + command executor
    """

    id: str
    """each time stream interpretation has a unique id"""

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
    def instruction_messages(self) -> list[Message]:
        """
        当前 interpreter 状态下, channels 的完整提示词. 用于呈现给大模型.
        在 Model Context 对话历史中, 可以认为最简单的上下文拓扑是:

        - instructions: 提示和指令. 尽可能少变更, 而且需要合并.
        - conversations: 对话历史.
        - context: 当前的状态, 可变的部分. 而且要让模型理解这块是随时变化的.
        + new turn:
            - inputs: turn-based Model 本轮的输入.
            - recall: 结合上下文, 自动生成的 recall
            - reasoning: 思考过程
            - actions: 行动过程.
            - outputs: 输出
            - observation: 需要观察的讯息.

        """
        pass

    @abstractmethod
    def context_messages(self, *, channel_names: list[str] | None = None) -> list[Message]:
        """
        返回 interpreter 的关联上下文.
        对应 Model Context 中的 conversation 部分.
        """
        pass

    def merge_messages(self, history: list[Message], inputs: list[Message]) -> list[Message]:
        """
        遵循系统规则合并消息体.
        """
        meta_message = Message.new(role="system").with_content(self.meta_instruction()).as_completed()
        messages = [meta_message]
        messages.extend(self.instruction_messages())
        messages.extend(history)
        messages.extend(self.context_messages())
        messages.extend(inputs)
        return messages

    @abstractmethod
    def feed(self, delta: str) -> None:
        """
        向 interpreter 提交文本片段, 会自动触发其它流程.

        example:
        async with interpreter:
            async for item in async_iterable_texts:
                interpreter.feed(item)
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        commit the inputs
        """
        pass

    async def interpret(self, deltas: AsyncIterable[str]) -> None:
        """
        一个完整的解析过程, 需要包含 feed 和 commit.
        """
        async for delta in deltas:
            self.feed(delta)
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
    def string_token_parser(self) -> StringTokenParser:
        """
        interpreter 持有的 Token 解析器. 将文本输入解析成 command token, 同时将 command token 解析成 command task.

        >>> def example(interpreter: Interpreter, deltas: AsyncIterable[str]) -> None:
        >>>     with interpreter.string_token_parser() as parser:
        >>>         async for delta in deltas:
        >>>         parser.feed(delta)

        注意 Parser 是同步阻塞的, 因此正确的做法是使用 interpreter 自带的 feed 函数实现非阻塞.
        通常 parser 运行在独立的线程池中.
        """
        pass

    @abstractmethod
    def command_token_parser(self) -> CommandTokenParserElement:
        """
        当前 Interpreter 做树形 Command Token 解析时使用的 Element 对象. debug 用.
        通常运行在独立的线程池中.
        """
        pass

    @abstractmethod
    def parsed_tokens(self) -> Iterable[CommandToken]:
        """
        已经解析生成的 tokens.
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
    def outputted(self) -> Iterable[str]:
        """已经对外输出的文本内容. todo: 删除这个函数. """
        pass

    @abstractmethod
    def executed(self) -> list[CommandTask]:
        """
        返回已经被执行的 tasks.
        """
        pass

    def executed_tokens(self) -> str:
        """
        返回当前已经执行完毕的 tokens.
        """
        tokens = []
        for task in self.executed():
            tokens.append(task.tokens)
        return "".join(tokens)

    @abstractmethod
    async def start(self) -> None:
        """
        启动解释过程.

        start the interpretation, allowed to push the tokens.
        """
        pass

    @abstractmethod
    async def stop(
            self,
            cancel_executing: bool = False,
    ) -> None:
        """
        stop the interpretation
        :param cancel_executing: 是否同时清空解析出来的任务. 不清空的话, 任务本身并不会被中断.
        """
        pass

    @abstractmethod
    def is_stopped(self) -> bool:
        """
        判断解释过程是否还在执行中.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否正在运行中: start -> end 中间.
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

        async with interpreter as itp:
            # the interpreter started
            async for item in async_iterable_texts:
                # 判断是否被中断. 如果被中断可以 break.
                if not itp.is_stopped():
                    itp.feed(item)

            await itp.wait_until_done()

        result = itp.results()
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def exception(self) -> Optional[Exception]:
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

        wait until the interpretation of command tasks are done (finish, failed or cancelled).
        """
        pass

    @abstractmethod
    async def wait_results(self) -> dict[str, str]:
        """
        将所有已经执行完的 task 的 result 作为有序的字符串字典输出
        知道第一个运行失败的.
        其中返回值为 None 或空字符串的不会展示.

        todo: 这是一个 alpha 版为了方便快速实现 react 做的临时机制. 不是正式机制.

        :return: key is the task name and attrs, value is the result or error of the command
                 if command task return None, ignore the result of it.
        """
        pass

    @abstractmethod
    async def wait_execution_done(
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
