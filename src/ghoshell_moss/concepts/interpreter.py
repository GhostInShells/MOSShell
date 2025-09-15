import threading
from .command import CommandToken, CommandTask, CommandMeta
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, Coroutine, List, Dict

CommandTokenCallback = Callable[[CommandToken], None]


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


class CommandElementTree(ABC):
    """
    The way to parse command token stream into command task, is element tree, a bit like AST.
    Parent node hold children elements,
    when add tokens, check if undone child exists and parse tokens into it, until itself reaches end.
    This abstract just show the parsing logic to you.
    """
    children: Dict[str, "CommandElementTree"]

    @abstractmethod
    def id(self) -> str:
        """
        return the command id (stream_id + cmd_id) of the command
        """
        pass

    @abstractmethod
    def add(self, token: CommandToken) -> None:
        """
        notice the token may be delta, end or children command tokens.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandTask]:
        """
        return the tasks from the current command. one command may generate multiple command tasks.
        because delta parts of the command may generate different command task for each part.
        """
        pass

    def all_parsed(self) -> Iterable[CommandTask]:
        """
        return the parsed command task in deep-first order.
        Notice element parsed tasks order may not be the same as the real command tasks order,
        since sometimes a parent command task will be called after all the children tasks are done.
        """
        yield from self.parsed()
        for child in self.children.values():
            yield from child.all_parsed()


class CommandTaskParser(ABC):
    """
    parse the command token stream into command task stream, by CommandElement tree
    """

    @abstractmethod
    def with_callback(self, callback: Callable[[CommandToken], None]) -> None:
        """
        register a new callback
        """
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def feed(self, token: CommandToken) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def buffer(self) -> Iterable[CommandTask]:
        pass


class Interpreter(CommandTokenParser, ABC):
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
