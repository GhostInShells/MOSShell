from abc import ABC, abstractmethod
from ghoshell_moss.core import Channel, PyChannel, ChannelInterface
from ghoshell_moss.message import Message
from .terminal import Terminal


class ProjectManager(ChannelInterface, ABC):
    """
    项目管理模块.
    基本原理是
    0. 可以进入到一个指定目录 (project)
    1. 可以在这个目录里使用 terminal 进行基础的操作.
    2. 可以默认看到 n 层的目录 (基于 gitignore 排除).
    3. 可以进入具体的目录, 从而看到目录里的文件列表 (基于 gitignore 排除)
    4. 可以在目录里创建一个 yaml 文件, 记录必要的讯息
    5. 可以修改指定的文件.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def terminal(self) -> Terminal:
        pass
