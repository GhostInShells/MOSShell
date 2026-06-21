from typing import Literal, Callable, Awaitable, Any, Coroutine, TypeVar, Type, Protocol, TypeAlias
from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.cell import Cell as MossCell, CellType, CellNetwork
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.manifests import Manifests
from ghoshell_moss.contracts import ConfigStore, Workspace, SystemPrompter, ResourceRegistry, Storage
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import frontmatter
import logging

__all__ = ['Matrix', 'SystemPrompter', 'MatrixLifecycleObject', 'Mode', 'ScopeKey']

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

    # --- todo app 体系都要移除 --- #
    apps: list[str] = Field(
        default_factory=lambda: ['*/*'],
        description="允许加载的 apps, 用 `group/name` 或者 `group/*` 的方式定义. 如果为 ['*']  则表示所有 apps 下的都允许加载."
    )
    bringup_apps: list[str] = Field(
        default_factory=list,
        description="启动时允许自动启动的 apps, 规则和 apps 相同. 默认为空. "
    )

    include_cells: list[str] = Field(
        default_factory=lambda: ['*/*'],
        description="允许通过环境发现的 cells 所处的相对路径. ",
    )
    exclude_cells: list[str] = Field(
        default_factory=list,
        description="指定排除掉的 cells 相关对路径",
    )
    bringup_cells: list[str] = Field(
        default_factory=list,
        description="模式启动时, 会自动开启的 cells. "
    )

    # --- mode 发现路径 --- #

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


ScopeKey: TypeAlias = Literal['ghost', 'mode', 'session_scope', 'session_id', 'cell']


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
        from ghoshell_moss.facade import discover_host
        host = discover_host()
        return host.matrix()

    @property
    @abstractmethod
    def env(self) -> Environment:
        pass

    # --- 自解释信息 --- #

    @property
    @abstractmethod
    def mode(self) -> Mode:
        """
        返回当前 MOSS 运行的模式.
        """
        pass

    # --- cells - Matrix 可以管理多个节点的通讯, 每个节点称之为 Cell --- #

    @property
    @abstractmethod
    def this(self) -> MossCell:
        """
        返回当前节点自身的讯息. 节点之间通讯仅仅通过 topics / parameter / action 等.
        自身的 cell 类型是不需要定义的, Matrix 在环境中发现, 启动时, 自动会生成描述.
        """
        pass

    @property
    def parent_cell_address(self) -> str:
        """启动当亲 Matrix 的父进程 Cell 对应的 Address """
        return self.env.parent_cell_address

    # --- 环境通讯总线 --- #

    @property
    @abstractmethod
    def cells(self) -> CellNetwork:
        """
        当前通讯网络下 Cells 的发现与管理.
        """
        pass

    @property
    @abstractmethod
    def session(self) -> Session:
        """
        所有 Matrix 共享的通讯总线
        同时分享会话级别的存储空间.
        """
        pass

    # --- channel --- #

    def provide_channel(
            self,
            channel: Channel,
            *,
            bridge_address: str | None = None,
    ) -> asyncio.Future[None]:
        """
        将 Channel 通过当前节点提供到整个 Matrix 网络中,
        :param channel: 需要提供到 matrix 体系里的根节点.
        :param bridge_address: 提供时声明通讯时的地址.
        """
        if bridge_address is None:
            bridge_address = self.this.bridge_address
        provider = self.cells.provide(bridge_address, channel)
        return self.create_task(provider.arun_until_closed(channel))

    def cell_channel_proxy(
            self,
            address: str,
            *,
            name: str = '',
            description: str = '',
            only_allowed_in_host_cell: bool = True,
    ) -> ChannelProxy:
        """
        搭建一个 proxy 获取另一个节点里通过 address (通常是 cell address) 提供的 channel. 进行跨网络同构.

        一个节点 provider, 另一个节点 proxy, 就可以形成 channel 基于 matrix 的通讯体系.
        通常情况下, proxy 只由 Matrix 的 Host 节点管理.

        :param address: cell address where providing a channel tree
        :param name: channel name which rewrite the providing channel.
        :param description: channel description which rewrite the providing channel.
        :param only_allowed_in_host_cell: if true, check this cell is host main cell or raise error.

        :raise RuntimeError: if the current cell is not the main cell of the matrix runtime.
        """
        # 通常只允许 Matrix 里的 host cell 使用 proxy 连接 channel. 因为 channel 是 matrix 内唯一的.
        # 多个 proxy 连接会导致 channel 频繁地重启.
        # 仍然允许用这个方式进行测试.
        # Matrix 底层有跨环境的通讯总线, 比如 redis / ws / mqtt 等等. 默认的 Host 使用的 zenoh 来组网.
        # 进入这个网络后, 可以通过 address 的方式来组建 proxy => provider 的通讯.
        if only_allowed_in_host_cell and self.this.type != CellType.host.value:
            raise RuntimeError(f"Current cell {self.this.address} is not host cell.")
        # 必须是为已经
        cell = self.cells.cached_cells().get(address)
        if cell is None:
            raise LookupError(f"Cell {address} is not found")
        bridge_address = self.this.bridge_address
        name = name or cell.normalized_name()
        description = description or cell.meta.description
        return self.cells.create_proxy(bridge_address, name, description)

    # --- Matrix 提供的文件存储区汇总 --- #

    @property
    def workspace(self) -> Workspace:
        """
        workspace 管理.
        """
        return self.env.workspace

    @property
    @abstractmethod
    def cell_workspace(self) -> Workspace:
        """
        cell 独立的 workspace. 基于约定返回.
        如果 cell 声明里有约定, 使用声明的地址.
        否则基于 moss workspace 创建.
        """
        pass

    @property
    def ghosts_storage(self) -> Storage:
        """
        workspace 里所有 ghosts 持久化存储所在的空间.
        """
        return self.workspace.root().sub_storage('ghosts')

    @property
    def modes_storage(self) -> Storage:
        """
        workspace 里所有 moss 模式的持久化存储空间.
        """
        return self.workspace.root().sub_storage('modes')

    def get_ghost_storage(self, ghost_name: str) -> Storage:
        """不同的 ghost 独享的存储空间. """
        return self.ghosts_storage.sub_storage(ghost_name)

    def get_modes_storage(self, mode_name: str) -> Storage:
        """不同模式独享的存储空间"""
        return self.modes_storage.sub_storage(mode_name)

    @property
    def ghost_home(self) -> Storage:
        """当前 ghost 持久化存储的根目录. """
        return self.get_ghost_storage(self.ghost_name)

    @property
    def mode_home(self) -> Storage:
        """当前模式持久化存储的根目录. """
        return self.get_modes_storage(self.mode_name)

    def storages(self) -> dict[str, Storage]:
        """
        Matrix 可提供的各种不同隔离级别的持久化存储路径, 显式声明定义.
        此处不建议直接使用, 而是提示项目的基础约定.
        """
        return {
            # 项目的 workspace.
            'workspace': self.workspace.root(),
            # 运行时文件的目录.
            'runtime': self.workspace.runtime(),
            # 日志文件的目录.
            'logs': self.workspace.logs(),
            # 配置文件的目录.
            'configs': self.workspace.configs(),
            # 全局资源文件的目录.
            'assets': self.workspace.assets(),
            # 所有的 ghosts 持久存储的目录.
            'ghosts': self.ghosts_storage,
            # 当前 ghost 的 home 目录.
            'ghost_home': self.ghost_home,
            # 所有 moss 运行模式的目录.
            'modes': self.modes_storage,
            # 当前 moss 运行模式的目录.
            'mode_home': self.mode_home,
            # 当前 session 的持久化存储目录. 是 session id 级别的.
            'session': self.session.storage,
            # cell 独有的 workspace 位置.
            'cell': self.cell_workspace.root(),
            'cell_runtime': self.cell_workspace.runtime(),
            'cell_assets': self.cell_workspace.assets(),
            'cell_logs': self.cell_workspace.logs(),
            # 所有临时存储空间使用, 都应该基于 tmp
            'tmp': self.session.tmp_storage
        }

    def scopes(self) -> dict[ScopeKey, str]:
        """返回 Matrix 运行时的维度座标. 用来构建不同的隔离级别. """
        return {
            'session_id': self.session_id,
            'session_scope': self.session_scope,
            'mode': self.mode_name,
            'ghost': self.ghost_name,
            'cell': self.this.address,
        }

    def get_scoped_url(self, *scopes: ScopeKey, **kwargs: str) -> str:
        """
        基于作用域生成一个 URL 形式的资源路径.
        可以用这种形式生成字符串唯一 id, 用来管理各种可复用的资源.

        举个例子: get_scoped_url('ghost', 'mode', user=name) 会生成一个 指定Ghost在指定模式下对特定用户 的唯一id,
        配合后缀, 可以提供资源管理的不同隔离级别.
        """
        scope_values = self.scopes()
        for scope in scopes:
            if scope in scope_values:
                kwargs[scope] = scope_values[scope]
        result = []
        for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
            result.append(k.strip('/'))
            result.append(v.strip('/'))
        return '/'.join(result)

    def get_scoped_storage(self, scope: ScopeKey, *scopes: ScopeKey) -> Storage:
        """
        基于指定的作用域获取一个持久化存储的 Storage 位置. 举例:
        - get_scoped_storage('ghost', 'mode') : 当前 Ghost X MOSS 不同模式独立的存储空间.
        - get_scoped_storage('ghost') : 当前 Ghost 的持久化存储空间..
        - get_scoped_storage('session_id', 'ghost'): 在当前 session id 下, 为当前 ghost 准备的存储空间.
        """
        if scope == 'ghost':
            root = self.get_ghost_storage(self.ghost_name)
        elif scope == 'mode':
            root = self.get_modes_storage(self.mode_name)
        elif scope == 'session_id':
            root = self.session.storage
        elif scope == 'cell':
            root = self.cell_workspace.root()
        elif scope == 'session_scope':
            root = self.session.scope_storage
        else:
            raise KeyError(f"scope {scope} is not supported")
        storage = root
        scope_values = self.scopes()
        for scope in scopes:
            if scope not in scope_values:
                raise KeyError(f"scope {scope} not in scopes")
            sub_storage_path = f"{scope}-{scope_values[scope]}"
            storage = storage.sub_storage(sub_storage_path)
        return storage

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

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """
        日志模块. 从属于当前节点.
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
        优先走 mode, 其次走全局发现.
        """
        pass

    @property
    @abstractmethod
    def configs(self) -> ConfigStore:
        """
        基于环境发现的配置中心.
        """
        pass

    # --- scopes. 运行时的作用域信息. --- #

    @property
    def mode_name(self) -> str:
        """当前模式的名称. """
        return self.mode.name

    @property
    def ghost_name(self) -> str | Literal['none']:
        """
        如果当前的 Host 节点是用 GhostRuntime 运行的, 则返回 ghost name, 否则是 'None'
        """
        return self.env.ghost_name

    @property
    def session_scope(self) -> str:
        return self.env.session_scope

    @property
    def session_id(self) -> str:
        return self.env.session_id

    # ---- 状态描述 ---- #

    @abstractmethod
    def is_running(self) -> bool:
        """
        matrix 自身是否在运行.
        """
        pass

    def is_host(self) -> bool:
        return self.this.type == CellType.host.value

    @abstractmethod
    def is_host_running(self) -> bool:
        """
        判断 moss 是否在运行中.
        """
        pass

    # --- 生命周期管理 --- #

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
    def create_task(
            self,
            cor: Coroutine,
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        """
        创建包含在 Matrix 生命周期内的 Task
        """
        pass

    # --- 子进程 spawn --- #

    @abstractmethod
    async def spawn(
            self,
            *args: str,
            cell_address: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
    ) -> asyncio.subprocess.Process:
        """
        Spawn a subprocess with MOSS environment context.

        The child inherits Matrix scope identity (workspace, scope, session_id...)
        via environment variables.

        The child runs in its own process group (``start_new_session=True``)
        so terminal signals to the parent do not propagate.

        *stdin*, *stdout*, *stderr* — pass ``asyncio.subprocess.PIPE``
        for async stream I/O, ``asyncio.subprocess.DEVNULL`` to suppress,
        or an fd for file redirection.

        when matrix is stopped, the process will be killed.
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
                    _ = await asyncio.gather(task, exit_signal, return_exceptions=True)
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
    async def add_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """
        可以在运行时动态添加 lifecycle object, 会绑定到 exit stack 启动, 退出时清空.
        """
        pass

    @abstractmethod
    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """注册 lifecycle object, 只有在运行前可以注册. """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
