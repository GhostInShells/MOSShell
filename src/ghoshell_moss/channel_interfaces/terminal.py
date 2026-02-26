from abc import ABC, abstractmethod
from ghoshell_moss.core import Channel, PyChannel, ChannelInterface
from ghoshell_moss.message import Message

EXIT_CODE = int
STDOUT = str
STDERR = str


class Terminal(ChannelInterface, ABC):
    """
    定义一个标准的 Listener 模块, 用来管理 AI 的聆听模式.
    """

    @abstractmethod
    async def exec(
            self,
            command: str,
            timeout: float = 10.0,
    ) -> tuple[EXIT_CODE, STDOUT, STDERR]:
        """
        Execute a shell command and return structured results.
        :param command: Command full line to execute.
            (Note: Implementation should handle proper shell escaping)
        :param timeout: Timeout in seconds
        :return: EXIT_CODE, STDOUT, STDERR
        """
        pass

    @abstractmethod
    async def context_messages(self) -> list[Message]:
        """
        Compile environmental context into natural language prompt.
        """
        pass

    def as_channel(self, name: str = "", description: str = "") -> Channel:
        channel = PyChannel(
            name=name or "terminal",
            description=description or "able to execute command in terminal",
            blocking=True,
        )
        channel.build.command()(self.exec)
        channel.build.context_messages(self.context_messages)
        return channel
