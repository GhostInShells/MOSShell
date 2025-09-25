from ghoshell_moss.concepts.command import CommandToken, CommandTask
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, Dict

CommandTokenCallback = Callable[[CommandToken | None], None]
CommandTaskCallback = Callable[[CommandTask | None], None]


class CommandTokenParseError(Exception):
    """converge the exceptions during command token parsing into this type"""
    pass


class CommandTaskParseError(Exception):
    pass


class CommandTokenParser(ABC):
    """
    parse from string stream into command tokens
    """

    @abstractmethod
    def is_running(self) -> bool:
        """weather this command is running"""
        pass

    @abstractmethod
    def with_callback(self, callback: CommandTokenCallback) -> None:
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
    def stop(self) -> None:
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
        self.stop()


class CommandTaskElement(ABC):
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

    children: Dict[str, "CommandTaskElement"]
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
        """是否解析已经完成了. """
        pass

    @abstractmethod
    def destroy(self) -> None:
        """手动清理数据结构, 加快垃圾回收, 避免内存泄漏"""
        pass


class Interpreter(ABC):
    """
    命令解释器, 从一个文本流中解析 command token, 同时将流式的 command token 解析为流式的 command task, 然后回调给执行器.

    The Command Interpreter that parse the LLM-generated streaming tokens into Command Tokens,
    and send the compiled command tasks into the shell executor.

    Consider it a one-time command parser + command executor
    """

    id: str
    """each time stream interpretation has a unique id"""

    @abstractmethod
    async def feed(self, delta: str) -> None:
        """
        向 interpreter 提交文本片段, 会自动触发其它流程.

        example:
        async with interpreter:
            async for item in async_iterable_texts:
                interpreter.feed(item)
        """
        pass

    @abstractmethod
    def with_callback(self, callback: CommandTaskCallback) -> None:
        pass

    @abstractmethod
    def parser(self) -> CommandTokenParser:
        """
        interpreter 持有的 Token 解析器. 将文本输入解析成 command token, 同时将 command token 解析成 command task.

        example:
        with interpreter.parser() as parser:
            async for item in async_iterable_texts:
            paser.feed(item)
        注意 Parser 是同步阻塞的, 因此正确的做法是使用 interpreter 自带的 feed 函数实现非阻塞.
        通常 parser 运行在独立的线程池中.
        """
        pass

    @abstractmethod
    def root_task_element(self) -> CommandTaskElement:
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
    def parsed_tasks(self) -> Dict[str, CommandTask]:
        """
        已经解析生成的 tasks.
        """
        pass

    @abstractmethod
    def outputted(self) -> Iterable[str]:
        """已经对外输出的文本内容. """
        pass

    @abstractmethod
    def results(self) -> Dict[str, str]:
        """
        将所有已经执行完的 task 的 result 作为有序的字符串字典输出

        :return: key is the task name and attrs, value is the result or error of the command
                 if command task return None, ignore the result of it.
        """
        pass

    @abstractmethod
    def executed(self) -> str:
        """
        返回已经被执行的 tokens.
        """
        pass

    @abstractmethod
    def inputted(self) -> str:
        """
        返回已经完成输入的文本内容. 必须通过 feed 输入.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动解释过程.

        start the interpretation, allowed to push the tokens.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        中断解释过程. 有可能由其它的并行任务来触发, 触发后 feed 不会抛出异常.

        stop the interpretation and cancel all the running tasks.
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
        pass

    @abstractmethod
    def is_interrupted(self) -> bool:
        """
        解释过程是否被中断.
        """
        pass

    async def __aenter__(self):
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
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @abstractmethod
    async def wait_parse_done(self) -> None:
        """
        等待解释过程完成. 完成有两种情况:
        1. 输入已经完备.
        2. 被中断.

        wait until the interpretation of command tasks are done (finish, failed or cancelled).
        :return: True if the interpretation is fully finished.
        """
        pass

    @abstractmethod
    def destroy(self) -> None:
        """
        为了防止内存泄漏, 增加一个手动清空的方法.
        """
        pass


class SyncInterpreter(CommandTokenParser, ABC):
    """
    The sync interface of the Command Interpreter,
    if we have to use the interface in another thread.
    """
    id: str

    @abstractmethod
    def wait_until_done(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
