import asyncio
from typing import Literal, Callable, Iterable, AsyncIterable, AsyncIterator

from ghoshell_common.contracts import LoggerItf
from typing_extensions import Self
from abc import ABC, abstractmethod

from .manifests import Manifest
from .matrix import Matrix
from .app import AppStore
from ghoshell_moss.core.concepts.session import Session, OutputItem
from ghoshell_moss.core.concepts.mindflow import Mindflow
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.blueprint.states import PrimeChannel
from ghoshell_moss.message import Message
from ghoshell_container import IoCContainer
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field, AwareDatetime
from datetime import datetime
from dateutil import tz
import frontmatter
from pathlib import Path


class ToolSet(ABC):
    """
    将 MOSS runtime 包装成 tools, 希望可以被作为工具提供给别的框架.
    不过需要目标框架自行兼容输出协议.
    """

    @abstractmethod
    def moss_instruction(self) -> str:
        """
        返回所有的 instruction, 信息, 可以加入到 agent 的 instruction.
        """
        pass

    @abstractmethod
    def moss_dynamic_messages(self) -> list[Message]:
        """
        返回 moss 运行时的动态信息,
        仅包含组件的 interface, context messages 等等.
        """
        pass

    @abstractmethod
    async def moss_exec(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        """
        向 MOSS 的运行时添加新的指令. 通常是 CTML.
        :param logos: 基于 ctml 语法提供的 command 字符串.
        :param call_soon: 如果为 True, 会立刻中断任何运行中的命令, 否则只是追加新的指令.
        :param wait_done: 为 True 的话, 阻塞到运行结束后, 拿到观察的结果.
        """
        pass

    @abstractmethod
    async def moss_observe(
            self,
            timeout: float | None = None,
            with_dynamic: bool = True,
    ) -> list[Message]:
        """
        观察等待到 moss 运行状态变更.
        通常包含:
        1. 新的高优消息输入
        2. 当前有命令在执行, 并且已经执行完或发生了异常.
        3. 等待超时, 仍然返回最新的观察结果.

        :param timeout: 指定一个等待时间, 否则会持续等待到有任何事件为止.
        :param with_dynamic: 观察的结果里是否包含最新的 moss dynamic 信息.
        """
        pass

    @abstractmethod
    async def moss_interrupt(self) -> list[Message]:
        """
        立刻中断所有运行中的命令. 并且返回中断的情况.
        """
        pass

    @property
    @abstractmethod
    def apps(self) -> AppStore:
        """
        管理 moss 架构下的 app 体系.
        可以启动/关闭 app.
        """
        pass

    @property
    @abstractmethod
    def shell(self) -> MOSShell:
        """
        全双工运行时.
        可以在它没启动时做一些操作.
        运行时可以直接通过它的 API 去控制 clear / pause 等操作.
        """
        pass

    @property
    @abstractmethod
    def matrix(self) -> Matrix:
        """
        环境通讯的总线.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """正式启动"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """运行结束"""
        pass


class MossRuntime(ABC):
    """
    MOSS 架构的主运行时, 环境中的单例.
    """

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        当前所处的模式.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否在运行中.
        """
        pass

    @property
    def logger(self) -> LoggerItf:
        return self.matrix.logger

    @abstractmethod
    def wait_close_sync(self, timeout: float | None = None) -> bool:
        """
        同步阻塞.
        """
        pass

    @abstractmethod
    async def wait_close(self) -> None:
        """
        异步阻塞到接受到停止讯号或者系统已经退出.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        发送关闭信号, 中断 Runtime.
        """
        pass

    @abstractmethod
    def pause(self, toggle: bool = True) -> None:
        """
        pause the runtime immediately
        产生的效果: 停止所有运行中逻辑, 中断循环, clear & pause shell, 除非 unpause 否则不接受新命令.
        """
        pass

    @property
    def container(self) -> IoCContainer:
        """
        运行时 ioc 容器.
        Runtime 相关所有单例都在里面.
        """
        return self.matrix.container

    def contracts(self) -> Iterable[type]:
        """
        返回 IoC 容器里绑定的所有对象.
        """
        return self.matrix.container.contracts(recursively=True)

    @property
    @abstractmethod
    def apps(self) -> AppStore:
        """
        管理 moss 架构下的 app 体系.
        可以启动/关闭 app.
        """
        pass

    @property
    @abstractmethod
    def shell(self) -> MOSShell:
        """
        全双工运行时.
        可以在它没启动时做一些操作.
        运行时可以直接通过它的 API 去控制 clear / pause 等操作.
        """
        pass

    @property
    def main_channel(self) -> PrimeChannel:
        """
        shell 的 main channel, 可以
        """
        return self.shell.main_channel

    @property
    @abstractmethod
    def matrix(self) -> Matrix:
        """
        MOSS 架构下, 多节点并行运行时的交互总线.
        """
        pass

    @property
    def session(self) -> Session:
        """
        runtime 当前所处的 Session.
        可以管理 input 和 output.

        这个函数缩短路径并声明它的存在.
        """
        return self.matrix.session

    def add_input(self, *messages: Message, priority: int = 0) -> None:
        """
        立刻添加新的输入到 Runtime 中.
        这些输入会发送给 on_output, 同时判断是否中断正在运行的 loop, 并且新起一个消费 inputs 的 loop.
        如果不能中断的话, 则会被 buffer 或丢弃.
        """
        pass

    def output(self, *items: OutputItem) -> None:
        """
        输出 output item. 由于这是 moss 的 output, 所以里面其实包含 input.
        """
        return self.matrix.session.output(*items)

    def on_output(self, callback: Callable[[OutputItem], None]):
        """
        接受 output item 并考虑渲染.
        """
        pass

    def run_until_closed(self) -> None:
        import uvloop
        asyncio.set_event_loop(uvloop.new_event_loop())

        async def _main() -> None:
            async with self:
                await self.wait_close()

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MossMode(BaseModel):
    """
    指定的运行模式.
    用来管理 MOSS Runtime 的运行时可发现资源.
    不使用 Mode 仍然可以启动 MOSS.
    """

    name: str = Field(
        description="模式的名称."
    )

    instruction: str = Field(
        default='',
        description="模式的详细介绍. 也会作为模式的专属 instruction"
    )

    description: str = Field(
        description="模式的一句话简介, 通常是 docstring 的第一句. 也支持独立定义",
    )

    apps: list[str] = Field(
        default_factory=lambda: ['*'],
        description="允许加载的 apps, 用 `group/name` 或者 `group/*` 的方式定义. 如果为 ['*']  则表示所有 apps 下的都允许加载."
    )

    bringup: list[str] = Field(
        default_factory=list,
        description="启动时允许自动启动的 apps, 规则和 apps 相同. 默认为空. "
    )

    import_path: str = Field(
        default="",
        description="找到模式实例的 python module path, 如果是从 markdown 文件找到的, 则为空."
    )

    file: str = Field(
        default="",
        description="找到模式实例的文件绝对路径. 比如 xxxx/src/MOSS/modes/default/MODE.md "
    )

    __manifest__: Manifest | None = None

    @classmethod
    def from_markdown(cls, file: Path) -> Self:
        """
        from a markdown file discover Mode.
        """
        if not file.exists():
            raise FileNotFoundError(f"{file} not found")
        post = frontmatter.loads(file.read_text())
        data = post.metadata
        docstring = post.content
        if "description" not in data:
            description = docstring.split("\n", 1)[0]
            data['description'] = description
        data['docstring'] = docstring
        result = cls(**data)
        result.file = str(file)
        return result

    def to_markdown(self) -> str:
        """
        to markdown format content.
        """
        meta_data = self.model_dump(
            exclude_none=True,
            exclude_defaults=False,
            exclude={'import_path', 'file', 'instruction'},
        )
        post = frontmatter.Post(content=self.instruction, **meta_data)
        return frontmatter.dumps(post)

    def with_manifest(self, manifest: Manifest, override: bool = False) -> Self:
        """
        define manifest
        """
        if override or self.__manifest__ is None:
            self.__manifest__ = manifest
        return self

    @property
    def manifest(self) -> Manifest:
        if self.__manifest__ is None:
            self.__manifest__ = Manifest()
        return self.__manifest__


class MossHost(ABC):
    """
    MOSS (model-oriented operating system shell) 的高阶抽象.
    Host 用来管理和发现环境, 从环境中创建 Moss 的一切.

    1. 它屏蔽了 shell/interpreter 等内核模块.
    2. 它管理 Shell 的环境发现与运行.
    3. 它解决并行思考网络内的通讯体系.
    4. 它缝合 Ghost 和 Shell. 作为一个独立的认知实体架构.

    架构拓扑的设计, 延续自 2019~2020 年的实现.
    https://github.com/thirdgerb/chatbot/blob/dba62e1337559c327d27ec4300366cd890a18ebc/src/Host/IHost.php#L4
    """

    @classmethod
    def discover(cls) -> Self:
        """
        环境发现的标准实现.
        """
        from ghoshell_moss.host import Host
        return Host.discover()

    @property
    @abstractmethod
    def manifest(self) -> Manifest:
        """
        返回当前环境下发现的 Matrix 实例.
        可以直接用于开发一个节点.
        """
        pass

    @property
    @abstractmethod
    def mode(self) -> MossMode:
        """
        current mode.
        """
        pass

    @abstractmethod
    def all_modes(self) -> dict[str, MossMode]:
        """
        当前环境中可用的运行时模式, 用于管理不同模式下的差异化资源.
        比如 mac 模式, 机器人模式, 就可以完全隔离开.
        """
        pass

    @abstractmethod
    def matrix(self) -> Matrix:
        """
        返回当前环境下发现的 Matrix 实例.
        可以直接用于开发一个节点.
        >>> async def main(moss: MossHost):
        >>>     async with moss.matrix():
        >>>         ...
        """
        pass

    @abstractmethod
    def run_toolset(self) -> ToolSet:
        """
        run as toolset.
        """
        pass
