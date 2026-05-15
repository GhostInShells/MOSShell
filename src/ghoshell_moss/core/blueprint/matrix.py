from typing import Literal, Callable, Awaitable, Any, Coroutine, Iterable, TypeVar, Type, Protocol

from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.contracts import LoggerItf, ConfigStore, Workspace, SystemPrompter, ResourceRegistry
from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.manifests import Manifests
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import frontmatter

__all__ = ['Matrix', 'Cell', 'SystemPrompter', 'ScopesKey', 'MatrixLifecycleObject', 'Mode']

CellTypes = Literal[
    'host',  # 表示为启动网络的主进程节点.
    'app',  # 表示在相同的 workspace 下的 App 节点. 由 main 节点管理生命周期.
    'fractal',  # Matrix 的分形通讯机制下, 其它 Matrix 连接到当前 Matrix, 所形成的 cell 节点.
]


class Cell(ABC):
    """
    在 matrix 中可以并行独立运行的单元, 拥有独立的进程.

    比如并行思考模块, channel provider 等等.
    不需要实现它, Matrix 的实现会包含 Cell 的定义.
    合法的 Cell 在 Matrix 体系中自动被感知和发现.
    """
    name: str  # 节点的名称.
    description: str  # 节点的描述.
    type: CellTypes | str
    where: str  # 这个节点自身的工作目录.

    @property
    def address(self) -> str:
        """节点的地址. 通常作为节点的各种通讯机制的前缀或关键环节."""
        # 遵循路径模式, 方便 fn match 做匹配.
        return self.make_address(self.type, self.name)

    @classmethod
    def make_address(cls, type: str, fullname: str) -> str:
        return '/'.join([type, fullname])

    @property
    def log_name(self) -> str:
        return '.'.join(['moss', self.type, self.name.replace('/', '.')])

    @abstractmethod
    def is_alive(self) -> bool:
        """
        节点是否在运行中.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "where": self.where,
            "log_name": self.log_name,
            "is_alive": self.is_alive(),
        }


CellAddress = str
_ThisCellName = None
_ThisCellType = None
_MatrixMainCellAddress = None

ScopesKey = Literal[
    'mode',  # 对环境中所有资源的隔离形式, 通过不同的 mode 隔离不同的资源组合. 使得资源如 provider, config 等可以复用.
    'session_scope',  # 运行时隔离的基本维度, 使用不同的 scope 启动, 可以用来隔离通讯/存储等. 前提是对应组件使用了这个隔离级别.
    'session_id',  # 运行时的唯一 Id. 如果一些资源或状态希望在系统关闭时就丢弃, 可以基于 session_id 构建隔离级别来通讯或存储.
    'cell_address',  # Matrix 实例作为通讯架构, 运行在每个不同的 Cell 内. 同时可以有很多个 cell 并行运行组网.
]

INSTANCE = TypeVar('INSTANCE')


class MatrixLifecycleObject(Protocol):
    """关键的运行时对象, 注册到生命周期中, 按次序启动. """

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class Mode(BaseModel):
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
    ctml_version: str = Field(
        default='',
        description='模式选择独立的 ctml version. '
    )

    description: str = Field(
        description="模式的一句话简介, 通常是 docstring 的第一句. 也支持独立定义",
    )

    apps: list[str] = Field(
        default_factory=lambda: ['*/*'],
        description="允许加载的 apps, 用 `group/name` 或者 `group/*` 的方式定义. 如果为 ['*']  则表示所有 apps 下的都允许加载."
    )

    bringup_apps: list[str] = Field(
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

    __manifest__: Manifests | None = None

    @classmethod
    def from_markdown(cls, file: Path, *, mode_name: str = None) -> Self:
        """
        from a markdown file discover Mode.
        """
        if not file.exists():
            raise FileNotFoundError(f"{file} not found")
        post = frontmatter.loads(file.read_text())
        data = post.metadata
        docstring = post.content
        if mode_name is not None and mode_name:
            data['name'] = mode_name
        elif 'name' in data:
            pass
        else:
            data['name'] = file.name.split('.', 1)[0]

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

    def with_manifest(self, manifest: Manifests, override: bool = False) -> Self:
        """
        define manifest
        """
        if override or self.__manifest__ is None:
            self.__manifest__ = manifest
        return self

    @property
    def manifest(self) -> Manifests:
        if self.__manifest__ is None:
            self.__manifest__ = Manifests()
        return self.__manifest__


class Matrix(ABC):
    """
    MOSS 架构下多节点组网后形成的通讯矩阵的客户端.

    持有矩阵的抽象可以通过矩阵通讯, 本身应该是进程级别单例.
    Matrix 是用于构建可跨进程通讯的基本抽象, 并且从环境中自我发现.
    """

    @classmethod
    def discover(cls) -> Self:
        """
        约定的环境发现逻辑.
        基于 Matrix 默认实现创建应用, 只需要调用 Matrix.discover() 根据抽象提供的能力即可.
        """
        # moss 架构的默认实现.
        # 这里使用了反范式, discover 包含了默认实现.
        from ghoshell_moss.host import Host
        return Host.discover().matrix()

    # --- 自解释信息 --- #

    @property
    @abstractmethod
    def mode(self) -> Mode:
        """
        返回当前 MOSS 运行的模式.
        """
        pass

    @abstractmethod
    def cell_env(self) -> dict[str, str]:
        """
        Cell 自身相关的环境变量.

        通常基于这些环境变量来还原 matrix 运行时, 自身所处的 cell.
        """
        # matrix 不依赖 Environment 对象, 避免发现逻辑永远不可重写.
        pass

    @property
    @abstractmethod
    def this(self) -> Cell:
        """
        返回当前节点自身的讯息. 节点之间通讯仅仅通过 topics / parameter / action 等.
        自身的 cell 类型是不需要定义的, Matrix 在环境中发现, 启动时, 自动会生成描述.
        """
        pass

    # -- 运行前 注册函数 -- #

    def register(
            self,
            abstract: Type[INSTANCE],
            binding: INSTANCE | Callable[[IoCContainer], INSTANCE],
    ) -> None:
        """
        ioc 容器注册方式.
        """
        # 为方便立刻理解 ioc 容器注册, 提供这个语法糖, 作为自解释方式.
        # 如果要全功能的 provider, 需要查看 ghoshell_container:Provider
        # 并不推荐用这种方式做注册, 因为没有环境发现声明. 更好的方式是
        #   1. 基于 Manifests 在 (workspace.src) MOSS.manifests.providers package里定义 provider 实例.
        #   2. 在指定 Mode, 如 (workspace.src) MOSS.modes.default.providers 里定义 provider 实例.
        #   注册方式具体查看 ghoshell_moss.host.manifests 和 ghoshell_moss.core.blueprint.environment
        from ghoshell_container import provide
        provider = provide(abstract, singleton=True)(binding)
        self.container.register(provider)

    @abstractmethod
    def register_lifecycle_objects(self, obj: MatrixLifecycleObject | Type[MatrixLifecycleObject]) -> None:
        """注册会和 matrix 同步启动的对象. 会依次序启动, 绑定生命周期, 不会做容错. """
        pass

    # -- 运行时 API -- #

    def resources(self) -> ResourceRegistry:
        """返回 matrix 共享的资源中心. """
        return self.container.force_fetch(ResourceRegistry)

    @abstractmethod
    def moss_system_prompter(self) -> SystemPrompter:
        """
        moss 全局的 system prompter.
        matrix 必须完成全局 prompter 的定义, 并注册到 IoC 容器中.
        """
        pass

    @property
    @abstractmethod
    def moss_mode_name(self) -> str:
        pass

    def scopes(self) -> dict[ScopesKey, str]:
        """返回 Matrix 运行时的维度座标. 用来构建不同的隔离级别. """
        return {
            'session_id': self.session.session_id,
            'session_scope': self.session.session_scope,
            'mode': self.mode.name,
            'cell_address': self.this.address,
        }

    @abstractmethod
    def ctml_version(self) -> str:
        """
        当前环境定义的 ctml version.
        """
        pass

    @abstractmethod
    def get_ctml_prompt(self, version: str | None = None) -> str:
        """
        返回环境中定义的系统提示词.
        """
        pass

    @abstractmethod
    def list_cells(self) -> dict[CellAddress, Cell]:
        """
        返回环境里的所有节点, 以及这些节点是否在运行.
        """
        pass

    @property
    @abstractmethod
    def session(self) -> Session:
        """
        共享的 Session Store.
        """
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        环境中共享的 IoC 容器. 只包含进程级别的服务.
        主要是 manifests 里提供的服务.
        """
        pass

    @property
    @abstractmethod
    def manifests(self) -> Manifests:
        """
        运行环境中各种能力的声明.
        """
        pass

    @property
    @abstractmethod
    def configs(self) -> ConfigStore:
        """
        基于环境发现的配置中心.
        """
        pass

    def show_configs(self) -> Iterable[dict[str, str]]:
        """
        不返回配置值的情况下, 返回配置的介绍.
        """
        store = self.configs
        for config_info in self.manifests.configs().values():
            info = {
                "name": config_info.name,
                "description": config_info.description,
                "file": config_info.file(store),
                "type": config_info.model_path,
            }
            yield info

    @abstractmethod
    def provide_channel(
            self,
            channel: Channel,
            *,
            cell_type: Literal['app', 'main'] | _ThisCellType = _ThisCellType,
            cell_name: str | _ThisCellName = _ThisCellName,
    ) -> asyncio.Future[None]:
        """
        将 Channel 通过当前节点提供到整个 Matrix 网络中,
        :param channel: 需要提供到 matrix 体系里的根节点.
        :param cell_type: 需要定义 channel 提供出去时的 cell 类型. 通常不需要传参, 按约定定义.
        :param cell_name: 提供出去时使用的 cell 名称. 不传参时就使用自己的名字.
        """
        # 在 AppCell 内通过 Matrix 调用 provide channel,
        # 一个进程只能调用一个 provide channel, 可以提供树形的 channel.
        pass

    @abstractmethod
    def channel_proxy(
            self,
            address: str,
            name: str,
            description: str = '',
            id: str | None = None,
            only_allowed_in_main_cell: bool = True,
    ) -> ChannelProxy:
        """
        搭建一个 proxy 获取另一个节点里通过 address (通常是 cell address) 提供的 channel. 进行跨网络同构.

        这个函数除特殊情况外, 不需要手动使用. Host 节点启动时会提供 apps 的自动发现.

        :param address: cell address where providing a channel tree
        :param name: channel name which rewrite the providing channel.
        :param description: channel description which rewrite the providing channel.
        :param id: channel uid if given, otherwise will generate a uuid for the proxy.
        :param only_allowed_in_main_cell: if true, check this cell is host main cell or raise error.
        :raise RuntimeError: if the current cell is not the main cell of the matrix runtime.
        """
        # 通常只允许 Matrix 里的 main cell 使用 proxy 连接 channel. 因为 channel 是 matrix 内唯一的.
        # 多个 proxy 连接会导致 channel 频繁地重启.
        # 仍然允许用这个方式进行测试.
        #
        # Matrix 底层有跨环境的通讯总线, 比如 redis / ws / mqtt 等等. 默认的 Host 使用的 zenoh 来组网.
        # 进入这个网络后, 可以通过 address 的方式来组建 proxy => provider 的通讯.
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        """
        日志模块. 从属于当前节点.
        """
        pass

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """
        workspace 管理.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        matrix 自身是否在运行.
        """
        pass

    @abstractmethod
    def is_host_running(self) -> bool:
        """
        判断 moss 是否在运行中.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭自身, 用于优雅退出.
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        阻塞等待自身运行退出.
        所有的功能都会关闭.
        """
        pass

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """
        阻塞等待自身退出.
        该方法仅限同步上下文调用
        """
        pass

    @abstractmethod
    def create_task(self, cor: Coroutine) -> asyncio.Task:
        """
        创建包含在 Matrix 生命周期内的 Task
        """
        pass

    # --- 启动函数, 并非必要, 基于 code as prompt 原则提示如何使用 --- #

    async def arun(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        Matrix 运行的基本逻辑.
        可参考或直接基于这个函数运行基于 Matrix 的应用.
        如果将它包裹成 Asyncio.Task, 也可以和主协程并行运行.
        """
        if self.is_running():
            raise RuntimeError(f'Matrix already running.')

        async with self:
            loop = asyncio.get_running_loop()

            # 1. 先执行获取 Awaitable 对象
            result_or_coro = main_coro(self)

            # 2. 判断是否是协程（需要被包装成 Task 才能并发）
            if asyncio.iscoroutine(result_or_coro):
                task = loop.create_task(result_or_coro)
                exit_signal = loop.create_task(self.wait_closed())

                try:
                    done, pending = await asyncio.wait(
                        [task, exit_signal],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    if task in done:
                        return await task
                    raise asyncio.CancelledError("Matrix identity is closing")
                finally:
                    # 3. 这里的清理逻辑必须覆盖到位
                    for t in [task, exit_signal]:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(task, exit_signal, return_exceptions=True)
            else:
                # 如果用户传的是普通 Awaitable 或已完成的结果
                return await result_or_coro

    def run(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        同步阻塞入口。内部自动拉起事件循环并治理生命周期。
        兼容 Python 3.10 的顶层入口。
        """
        try:
            import uvloop
        except ImportError:
            # 如果不能支持.
            uvloop = None

        try:
            if uvloop is not None:
                asyncio.set_event_loop(uvloop.new_event_loop())
            return asyncio.run(self.arun(main_coro))
        except KeyboardInterrupt:
            pass  # 底层 arun 已经处理了清理

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
