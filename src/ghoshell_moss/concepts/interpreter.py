import threading
from .command import CommandToken, CommandTask, CommandMeta
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, Coroutine


class CommandParser(ABC):

    @abstractmethod
    def put(self, delta: str) -> None:
        """
        push the streaming tokens delta into the interpreter.
        """
        pass

    @abstractmethod
    def cancel(self) -> None:
        """
        thread-safe way to cancel the running interpretation
        """
        pass

    @abstractmethod
    def with_command_token_callback(self, callback: Callable[[CommandToken], None]) -> None:
        """
        register the callbacks for each parsed command token.
        usually the put_nowait method of a queue.
        """
        pass

    @abstractmethod
    def with_command_task_callback(self, callback: Callable[[CommandTask], None]) -> None:
        """
        register the callbacks for each parsed command task.
        usually the put_nowait method of a queue.
        """
        pass

    @abstractmethod
    def parsed_tokens(self) -> Iterable[CommandToken]:
        """
        return the parsed command tokens
        usually the command token parsing is much faster than the execution.
        """
        pass

    @abstractmethod
    def parsed_tasks(self) -> Iterable[CommandTask]:
        """
        return the parsed command tasks in compiling order
        """
        pass


class Interpreter(CommandParser, ABC):
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


class SyncInterpreter(CommandParser, ABC):
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
