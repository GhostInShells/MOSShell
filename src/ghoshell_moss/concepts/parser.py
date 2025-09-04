import threading

from .command import CommandToken, CommandTokenStream
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional
from threading import Thread

ParserCallback = Callable[[CommandTokenStream | None], None]


class Parser(ABC):
    """
    Parse a string into a CommandToken.
    """

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def add(self, tokens: str) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def wait(self) -> Iterable[CommandToken]:
        pass

    @abstractmethod
    def with_callback(self, callback: ParserCallback) -> None:
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def pipe(parser: "Parser", tokens: Iterable[str]) -> Iterable[str]:
        """
        use parser as iterable str pipe
        """


class Interpreter(ABC):
    @abstractmethod
    def new_parser(self, callback: Optional[ParserCallback] = None) -> Parser:
        pass

    def parse(self, tokens: Iterable[str], callback: Optional[ParserCallback] = None) -> Iterable[CommandToken]:
        parser = self.new_parser(callback)
        stream = CommandTokenStream()
        parser.with_callback(stream.append)

        def consumer():
            with parser:
                for token in tokens:
                    parser.add(token)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        return stream

    def parse_string(self, tokens: str) -> Iterable[CommandToken]:
        parser = self.new_parser()
        buffer = []

        def callback(token: CommandToken | None) -> None:
            if token is not None:
                buffer.append(token)

        parser.with_callback(callback)
        with parser:
            parser.add(tokens)
        return buffer

    def pipe(self, tokens: Iterable[str], callback: Optional[ParserCallback] = None) -> Iterable[str]:
        parser = self.new_parser(callback)
        with parser:
            for token in tokens:
                parser.add(token)
                yield token
