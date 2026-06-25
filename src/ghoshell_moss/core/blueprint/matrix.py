from typing import Literal, Callable, Awaitable, Any, Coroutine, TypeVar, Type, Protocol, TypeAlias
from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.cell import Cell as MossCell, CellType, CellNetwork, CellRegistry
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.contracts import ConfigStore, Workspace, SystemPrompter, ResourceRegistry, Storage
from ghoshell_container import IoCContainer
from pathlib import Path
import asyncio
import logging

__all__ = ['Matrix', 'SystemPrompter', 'MatrixLifecycleObject', 'RuntimeScopeKey']

INSTANCE = TypeVar('INSTANCE')


class MatrixLifecycleObject(Protocol):
    """关键的运行时对象, 注册到生命周期中, 按次序启动. """

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RuntimeScopeKey: TypeAlias = Literal['ghost', 'mode', 'network', 'cell']


class Matrix(ABC):
    """
    MOSS 架构下多节点组网后形成的通讯矩阵的客户端.

    持有矩阵的抽象可以通过矩阵通讯, 本身应该是进程级别单例.
    Matrix 是用于构建可跨进程通讯的基本抽象, 并且从环境中自我发现.
    """

    @classmethod
    def discover(
            cls,
            *,
            env: Environment | None = None,
    ) -> Self:
        """
        约定的环境发现逻辑.
        基于 Matrix 默认实现创建应用, 只需要调用 Matrix.discover() 根据抽象提供的能力即可.
        """
        # moss 架构的默认实现.
        # 这里使用了反范式, discover 包含了默认实现.
        from ghoshell_moss.factory import create_matrix, create_project
        env = env or Environment.discover()
        # 初始化 env
        env.bootstrap()
        project = create_project(env)
        # 初始化 project.
        project.bootstrap()
        return create_matrix(env, project)

    @property
    @abstractmethod
    def env(self) -> Environment:
        """环境"""
        pass

    @property
    @abstractmethod
    def project(self) -> Project:
        """当前项目所在. """
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
        """启动当前 Matrix 的父进程 Cell 对应的 Address """
        return self.env.parent_cell_address

    # --- 环境通讯总线 --- #

    @property
    @abstractmethod
    def network(self) -> CellNetwork:
        """
        当前通讯网络下 Cells 的发现与管理.
        """
        pass

    @property
    @abstractmethod
    def cells(self) -> CellRegistry:
        """
        当前状态的 cells.
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

    @abstractmethod
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
        pass

    @abstractmethod
    def channel_proxy(
            self,
            address: str,
            *,
            name: str = '',
            description: str = '',
            only_allowed_in_host_cell: bool = True,
    ) -> ChannelProxy:
        """
        搭建一个 proxy 获取另一个节点里通过 address (通常是 cell address) 提供的 channel. 进行跨网络同构.
        :param address: cell address where providing a channel tree
        :param name: channel name which rewrite the providing channel.
        :param description: channel description which rewrite the providing channel.
        :param only_allowed_in_host_cell: if true, check this cell is host main cell or raise error.

        :raise RuntimeError: if the current cell is not the main cell of the matrix runtime.
        """
        pass

    # --- Matrix 提供的文件存储区汇总 --- #

    @property
    def workspace(self) -> Workspace:
        """
        workspace 管理.
        """
        return self.project.workspace

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
    @abstractmethod
    def project_root(self) -> Storage:
        """
        项目的 storage.
        """
        pass

    @property
    def ghosts_home_root(self) -> Storage:
        """
        workspace 里所有 ghosts 持久化存储所在的空间.
        """
        return self.workspace.root().sub_storage(self.project.ghosts_home)

    @property
    def modes_home_root(self) -> Storage:
        """
        workspace 里所有 moss 模式的持久化存储空间.
        """
        return self.workspace.root().sub_storage(self.project.modes_home)

    def get_ghost_home(self, ghost_name: str) -> Storage:
        """不同的 ghost 独享的存储空间. """
        return self.ghosts_home_root.sub_storage(ghost_name)

    def get_mode_home(self, mode_name: str) -> Storage:
        """不同模式独享的存储空间"""
        return self.modes_home_root.sub_storage(mode_name)

    @property
    def ghost_home(self) -> Storage:
        """当前 ghost 持久化存储的根目录. """
        return self.get_ghost_home(self.ghost_name)

    @property
    def mode_home(self) -> Storage:
        """当前模式持久化存储的根目录. """
        return self.get_mode_home(self.mode_name)

    @property
    def network_home(self) -> Storage:
        return self.workspace.runtime().sub_storage('networks').sub_storage(self.network_name)

    def storages(self) -> dict[str, Storage]:
        """
        Matrix 可提供的各种不同隔离级别的持久化存储路径, 显式声明定义.
        此处不建议直接使用, 而是提示项目的基础约定.
        """
        return {
            # project
            "project": self.project_root,
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
            'ghosts': self.ghosts_home_root,
            # 所有 moss 运行模式的目录.
            'modes': self.modes_home_root,
            # 当前 ghost 的 home 目录.
            'ghost_home': self.ghost_home,
            # 当前 moss 运行模式的目录.
            'mode_home': self.mode_home,
            'network_home': self.network_home,
            # 当前 session 的持久化存储目录.
            'session_scope': self.session.storage,
            # cell 独有的 workspace 位置.
            'cell': self.cell_workspace.root(),
            # session id (通常是 cell runtime id) 对应的 storage.
            'tmp': self.session.tmp_storage,
        }

    def runtime_scopes(self) -> dict[RuntimeScopeKey, str]:
        """返回 Matrix 运行时的维度座标. 用来构建不同的隔离级别. """
        return {
            'mode': self.mode_name,
            'ghost': self.ghost_name,
            'cell': self.this.address,
            'network': self.network_name,
        }

    def get_runtime_url(self, *scopes: RuntimeScopeKey, **kwargs: str) -> str:
        """
        基于作用域生成一个 URL 形式的资源路径.
        可以用这种形式生成字符串唯一 id, 用来管理各种可复用的资源.

        举个例子: get_scoped_url('ghost', 'mode', user=name) 会生成一个 指定Ghost在指定模式下对特定用户 的唯一id,
        配合后缀, 可以提供资源管理的不同隔离级别.
        """
        scope_values = self.runtime_scopes()
        for scope in scopes:
            if scope in scope_values:
                kwargs[scope] = scope_values[scope]
        result = []
        for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
            result.append(k.strip('/'))
            result.append(v.strip('/'))
        return '/'.join(result)

    def get_runtime_scope_storage(self, ns_key: RuntimeScopeKey, *ns_keys: RuntimeScopeKey) -> Storage:
        """
        基于指定的作用域获取一个持久化存储的 Storage 位置. 举例:
        - get_scoped_storage('ghost', 'mode') : 当前 Ghost X MOSS 不同模式独立的存储空间.
        - get_scoped_storage('ghost') : 当前 Ghost 的持久化存储空间..
        - get_scoped_storage('session_id', 'ghost'): 在当前 session id 下, 为当前 ghost 准备的存储空间.
        """
        if ns_key == 'ghost':
            root = self.ghost_home
        elif ns_key == 'mode':
            root = self.mode_home
        elif ns_key == 'cell':
            root = self.cell_workspace.root()
        elif ns_key == 'network':
            root = self.network_home
        else:
            raise KeyError(f"scope {ns_key} is not supported")
        storage = root
        scope_values = self.runtime_scopes()
        for ns_key in ns_keys:
            if ns_key not in scope_values:
                raise KeyError(f"scope {ns_key} not in scopes")
            sub_storage_path = f"{ns_key}-{scope_values[ns_key]}"
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

    @property
    @abstractmethod
    def resources(self) -> ResourceRegistry:
        """返回 matrix 共享的资源中心. """
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
    def configs(self) -> ConfigStore:
        """
        基于环境发现的配置中心.
        """
        pass

    # --- scopes. 运行时的作用域信息. --- #

    @property
    def mode_name(self) -> str:
        """当前模式的名称. """
        return self.env.mode_name

    @property
    def ghost_name(self) -> str | Literal['none']:
        """
        如果当前的 Host 节点是用 GhostRuntime 运行的, 则返回 ghost name, 否则是 'None'
        """
        return self.env.ghost_name

    @property
    def network_scope(self) -> str:
        return self.network.scope

    @property
    def network_name(self) -> str:
        return self.network.name

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def session_scope(self) -> str:
        """session scope 是 mode-ghost-network-scope 的组合. """
        return self.session.session_scope

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
