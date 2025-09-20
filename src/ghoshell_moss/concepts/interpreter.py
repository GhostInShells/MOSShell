import threading
from .command import CommandToken, CommandTask, CommandMeta
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, Coroutine, List, Dict

CommandTokenCallback = Callable[[CommandToken], None]
CommandTaskCallback = Callable[[CommandTask], None]


class CommandTokenParseError(Exception):
    pass


class CommandTokenParser(ABC):
    """
    parse string stream into command tokens
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
    def end(self) -> None:
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
            self.end()
        self.stop()


class CommandTaskParseError(Exception):
    pass


class CommandTaskElement(ABC):
    current: Optional[CommandTask] = None
    children: Dict[str, "CommandTaskElement"]

    @abstractmethod
    def with_callback(self, callback: CommandTaskCallback) -> None:
        pass

    @abstractmethod
    def on_token(self, token: CommandToken) -> None:
        pass

    @abstractmethod
    def is_end(self) -> bool:
        pass

    def all_tasks(self, only_children: bool = False) -> Iterable[CommandTask]:
        """
        先序遍历所有子节点已经生成的 tasks.
        """
        if self.current is not None and not only_children:
            yield self.current
        for child in self.children.values():
            yield from child.all_tasks()

    @abstractmethod
    def destroy(self) -> None:
        pass


class CommandTaskParser(ABC):

    @abstractmethod
    def with_callback(self, callback: Callable[[CommandTask], None]):
        """
        send parsed command task to callback method
        """
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def parse(self, *tokens: CommandToken) -> None:
        """
        parse command tokens into command tasks
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandTask]:
        """
        parsed command tasks in order
        """
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class Interpreter(ABC):
    """
    The Command Interpreter that parse the LLM-generated streaming tokens into Command Tokens,
    and send the compiled command tasks into the shell executor.

    Consider it a one-time command parser + command executor
    """

    id: str
    """each time interpretation has a unique id with a stream"""

    @abstractmethod
    async def start(self) -> None:
        """
        start the interpretation, allowed to push the tokens.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        stop the interpretation and cancel all the running tasks.
        """
        pass

    async def __aenter__(self):
        """
        example to use the interpreter:

        async with interpreter as itp:
            # the interpreter started
            itp.put(text)
            itp.wait_until_done()
            # exit the interpretation

        """
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @abstractmethod
    async def wait_until_done(self) -> bool:
        """
        wait until the interpretation of command tasks are done (finish, failed or cancelled).
        :return: True if the interpretation is fully finished.
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
