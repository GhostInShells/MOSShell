from abc import ABC, abstractmethod
from ghoshell_moss.core import Channel, PyChannel, ChannelInterface
from ghoshell_moss.message import Message


class MarkdownDocs(ChannelInterface, ABC):
    """
    文档阅读和管理的功能.
    计划是能管理一个文档库, 阅读, 创建和修改.

    它应该是文件管理器的一个子实现.

    基本原理是:
    0. 指定文档的根目录, 创建目标文档.
    1. Documents 提供目录和文件索引 (扫描指定目录的 markdown 文件). 包含目录级的摘要.
    2. 在每个目录内维护一个 yaml 文件, 可以往里面添加 目录 和 文档的摘要.
    3. 通过搜索关键字来定位文档内容.
    4. pin 指定的文档(用 foo/bar/baz.md) 到 context messages 中. 下一个回合才可以看到详细的内容.
    5. unpin
    6. create 创建一个文档.
    7. edit 一个文档. context messages 中展示被 edit 的文档, 标记行号.
    8. 增加文档内容, 替代文档内容, 删除文档内容.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def is_editing(self) -> bool:
        pass

    @abstractmethod
    def context_messages(self) -> list[Message]:
        pass

    @abstractmethod
    async def pin(self, docs: list[str]) -> None:
        pass

    @abstractmethod
    async def unpin(self, docs: list[str]) -> None:
        pass

    @abstractmethod
    async def create(self, doc: str) -> None:
        pass

    @abstractmethod
    async def edit(self, doc: str) -> None:
        pass

    @abstractmethod
    async def append_content(self, text__: str) -> None:
        pass

    @abstractmethod
    async def delete_content(self, start_line: int, end_line: int) -> None:
        pass

    @abstractmethod
    async def replace_content(self, target: str, limit: int = 0, text__: str = "") -> None:
        pass

    @abstractmethod
    async def insert_content(self, start_line: int, text__: str) -> None:
        pass

    @abstractmethod
    async def rewrite(self, start_line: int = 0, end_line: int = -1, text__: str = ""):
        pass

    def as_channel(self, name: str = "", description: str = "") -> Channel:
        channel = PyChannel(
            name=name or self.name(),
            description=description or self.description(),
        )

        channel.build.context_messages(self.context_messages)
        channel.build.command()(self.pin)
        channel.build.command()(self.unpin)
        channel.build.command()(self.create)
        channel.build.command()(self.edit)
        channel.build.command(available=self.is_editing)(self.append_content)
        channel.build.command(available=self.is_editing)(self.replace_content)
        channel.build.command(available=self.is_editing)(self.delete_content)
        channel.build.command(available=self.is_editing)(self.insert_content)
        channel.build.command(available=self.is_editing)(self.rewrite)
        return channel
