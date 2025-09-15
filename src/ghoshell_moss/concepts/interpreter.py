import threading
from .command import CommandToken, CommandTask, CommandMeta
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, Coroutine

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
        if exc_val is None:
            self.end()
        self.stop()


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
