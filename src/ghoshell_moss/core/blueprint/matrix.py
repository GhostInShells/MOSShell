"""
MOSS 实例运行在 Matrix 的网络中.
一个网络可能同时有很多套 MOSS 的实例 (Host) 和能力单元 (Cell) 在运行.

Matrix 网络投影到 Cell 内部的形式是 Matrix 抽象.
cell 经由它持有身份、暴露膜、观察网络、拉起并治理新的进程.

命名的哲学锚点: Matrix 实例是 "整体在局部的投影" —
洞穴之光投影出蜂巢的形状. 人类语言经常用投影指代实体 (指着屏幕里的代码流
叫 matrix), 这种指代等价具有哲学实在性. Matrix 不是 mesh, 也不是 mesh 的
客户端 — mesh 只是它投影的来源之一.
"""
import dataclasses
from typing import Literal, Callable, Awaitable, Any, Coroutine, Protocol, TypeAlias, Type, TYPE_CHECKING
from typing_extensions import Self
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import Warrant
from ghoshell_moss.core.blueprint.cell import Cell, CellNetwork, CellAddress, CellRuntimeInfo
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata
from ghoshell_moss.core.blueprint.service import ServiceOperator, ServiceClient, ServiceServer
from ghoshell_moss.contracts import Workspace, ResourceRegistry, ConfigStore
from ghoshell_moss.contracts.subprocesses import Subprocesses, ManagedProcess, ProcessMeta
from ghoshell_moss.contracts.job_supervisor import JobSupervisor
from ghoshell_container import IoCContainer, Contracts
from pathlib import Path
import asyncio
import logging

Facade = ABC
"""Facade 标记"消费面"抽象 — 你持有并调用它, 而非继承它。"""

__all__ = ['Matrix', 'MatrixLifecycleObject', 'RuntimeScopeKey', 'CellHandle']


class MatrixLifecycleObject(Protocol):
    """关键的运行时对象, 生命周期注册到 Matrix 中, Matrix 托管其启动和关闭, 按次序启动. """

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RuntimeScopeKey: TypeAlias = Literal['ghost', 'mode', 'network', 'cell']


@dataclasses.dataclass
class CellHandle:
    """
    父进程视角的 cell 句柄: cell 身份 (runtime) + 子进程句柄 (process) 的组合.

    通过 matrix.run_node 拉起, 由 matrix.handled_cells() 追踪.
    stop / wait 是 process 侧的转发糖, 直接操作 process 也合法.
    """
    runtime: CellRuntimeInfo
    process: ManagedProcess

    @property
    def address(self) -> CellAddress:
        return self.runtime.address

    async def stop(self, timeout: float = 5.0) -> None:
        """优雅停子进程 (SIGTERM → 超时 killpg). 语义等同 process.stop."""
        await self.process.stop(timeout)

    async def wait(self) -> ProcessMeta:
        """阻塞等子进程退出, 返回 exit meta."""
        return await self.process.wait()


class Matrix(Facade):
    """
    MOSS 通讯矩阵在本进程内的投影. 进程级别单例, 从环境中自我发现.

    首页成员即认知地图 (code as prompt):
    身份 (this/env/project/network), 能力组织 (provide_channel), 调试声明 (publish_event)
    观察 (mesh), 治理 (run_cell), 通用功能模块 (processes/jobs),
    关键协议入口 (session/workspace/home/container), 生命周期 (arun/run/close/...).

    开发 Cell 时请遵循 Matrix 展示的能力地图, 按需选用能力, 适当扩大探索.
    """

    # -- composition root -- #

    @classmethod
    def discover(
            cls,
            *,
            env: Environment | None = None,
    ) -> Self:
        """
        获取当前进程的 Matrix 实例.

        进程内同 cell 身份只建一个 (防止重复入网 / workspace 锁争抢).
        `async with` 退出后再次 discover 会新建 — 不返回已关闭的 matrix.
        开发时可专注于提供的 API, 不关心如何构建 Matrix.
        """
        # 反范式实现抽象可执行. 在你需要了解细节时, 可以追踪到真实的工厂代码.
        # 工厂函数可 patch.
        from ghoshell_moss.factory import create_matrix, create_project
        global _instance
        if _instance is not None:
            return _instance
        env = env or Environment.discover()
        project = create_project(env)
        project.bootstrap()
        _instance = create_matrix(env, project)
        return _instance

    @classmethod
    def new(
            cls,
            node_name: str,
            *,
            description: str = '',
            category: str = '',
            env: Environment | None = None,
            persist: bool = False,
    ) -> Self:
        """在运行时环境中, 用指明的方式定义一个 Node, 用它获得 matrix 实例"""
        from ghoshell_moss.factory import create_matrix, create_project
        from ghoshell_moss.core.blueprint.cell import NodeManifest, build_cell_from_node, CellRuntimeInfo
        global _instance
        if _instance is not None:
            raise RuntimeError(f"The Matrix is already running: %s", _instance.this)
        env = env or Environment.discover()
        node = NodeManifest.new(node_name, description=description, category=category)
        # 是否是一个持久运行节点.
        node.persist = persist
        cell = build_cell_from_node(env, node)
        runtime_info = CellRuntimeInfo.from_cell(cell)
        project = create_project(env)
        project.bootstrap()
        _instance = create_matrix(env, project, runtime_info)
        return _instance

    @classmethod
    def reset_discover_instance(cls, instance: 'Matrix') -> None:
        """矩阵退出后复位 discover 缓存 — 已关闭的实例不再被后续 discover 返回."""
        global _instance
        if _instance is instance:
            _instance = None

    def contracts(self) -> 'Contracts':
        """
        matrix 承诺的 IoC 依赖集合 — 装配时 (MatrixImpl.__init__) 校验.

        缺失任何一项即构造失败, 新增基建依赖进这个集合.
        环境里没有会造成 fail-fast, 不等到首次 force_fetch 才暴露.

        三类:
        1. blueprint 架构基建 — Project/Environment/Matrix, 加 Cell/CellAddress
           (matrix 在 __init__ 直接 set, 实例已知)
        2. session 配套 — Session/TopicService/QAManager.
           Cache/Parameter 不走 IoC — Session 直接暴露 session.cache/session.parameters.
        3. contracts 配套 — Workspace/Subprocesses/JobSupervisor/LoggerItf/
           logging.Logger/ConfigStore/ResourceRegistry. 不含 FileEditor (依赖 cwd,
           matrix 级暴露越权), 不含 asr/audio 等重能力.
        """
        from ghoshell_common.contracts import LoggerItf
        from ghoshell_moss.core.blueprint.project import Project
        from ghoshell_moss.core.blueprint.environment import Environment
        from ghoshell_moss.core.blueprint.cell import Cell, CellAddress
        from ghoshell_moss.core.blueprint.session import Session
        from ghoshell_moss.core.concepts.topic import TopicService
        from ghoshell_moss.core.concepts.qa import QAManager
        from ghoshell_moss.contracts.workspace import Workspace
        from ghoshell_moss.contracts.subprocesses import Subprocesses
        from ghoshell_moss.contracts.job_supervisor import JobSupervisor
        from ghoshell_moss.contracts.configs import ConfigStore
        from ghoshell_moss.contracts.resource import ResourceRegistry
        import logging
        contracts = [
            # 1. blueprint 架构基建
            Project, Environment, Matrix, Cell, CellAddress,
            # 2. session 配套
            Session, TopicService, QAManager,
            # 3. contracts 配套
            Workspace, Subprocesses, JobSupervisor, LoggerItf, logging.Logger,
            ConfigStore, ResourceRegistry,
        ]

        return Contracts.new(*contracts)

    # -- 身份 -- #

    @property
    @abstractmethod
    def env(self) -> Environment:
        """
        matrix 所处的进程环境信息. 通常不需要了解细节, 仅在开发逻辑与环境变量等相关时探索.
        """
        pass

    @property
    @abstractmethod
    def project(self) -> Project:
        """ 当前 Matrix 节点 (cell) 所处的目位置, 包含相关文件路径和环境发现的能力. 仅当要开发项目级别能力时探索. """
        pass

    @property
    def project_home(self) -> Path:
        """moss 所在项目的根目录. """
        return self.project.root

    @property
    def workspace(self) -> Workspace:
        """当前 project 内, moss 自身的 workspace. 相同 project 下的 Cell 共享的空间. """
        return self.project.workspace

    @property
    @abstractmethod
    def this(self) -> Cell:
        """
        当前进程 - Cell - 在 Matrix 网络内的身份讯息. — 凡入网皆有身份.
        将 Matrix 看作一个城市的话, Cell 就是当前进程自己所处的房间.
        """
        pass

    @property
    def home(self) -> Path:
        """
        本 cell 的持久领地 — 跨次运行存续的状态 (记忆/配置/数据) 的归宿.

        默认取 self.this.home. cell 需要更完整的目录结构时, 通常自己就是一个
        独立的 project (自带 .moss), 从 Environment 重新 discover.
        """
        return Path(self.this.home)

    @property
    def mode_home(self) -> Path:
        """ moss 当前模式 (moss host mode) 的工作空间. """
        return self.env.mode_home

    @property
    def ghost_home(self) -> Path:
        """ moss 当前 ghost 的工作空间. ghost 和 mode 是正交关系, 各自有独立的工作区. """
        return self.env.ghost_home

    @property
    @abstractmethod
    def cell_workspace(self) -> Workspace:
        """
        本 cell 自身的独立 workspace — 根目录为 cell home.

        与 workspace (project 级共享) 不同, cell_workspace 提供 cell 隔离的
        配置、资产、运行时数据. configs() 读取 cell 自己目录下的 configs/.
        """
        ...

    @property
    def configs(self) -> ConfigStore:
        return self.project.configs

    @property
    @abstractmethod
    def network_info(self) -> NetworkMetadata:
        """
        本 matrix 所接入网络的配置元信息.
        通常 Cell 进程不需要关注具体信息. 除了运行逻辑和所处网络本身有关时查看.
        """
        pass

    # -- 本 cell 的入网侧 -- #

    @abstractmethod
    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        """
        将当前进程内的能力, 通过 moss channel 提供到 network 中, 供 Ghost (持久智能体) 使用.
        channel 的能力提供方式详见 channel_builder
        模型操控 channel 的方式详见 ctml.

        一个 Cell 只能提供一个 channel 根节点 (树结构). 此方法只能调用一次, wait 它可以阻塞到进程被外部关闭.
        会自动声明提供了能力到网络中.
        """
        pass

    @abstractmethod
    async def publish_event(self, content: str) -> None:
        """
        向网络广播本 cell 的轻量事件. 网络中的主宰 (Ghost) 可以感知到事件的发生.
        """
        pass

    # -- 观察: 网络的延迟视图 (惰性门) -- #

    @abstractmethod
    async def network(self) -> CellNetwork:
        """
        提供 API 观察网络中所有 Cell 的相关讯息.
        只有在运行时动态反映 cell 状态时, 才需要获取.
        """
        # 懒加载模块, 首次使用需要用 async 创建.
        pass

    @abstractmethod
    async def run_node(
            self,
            target: Path,
            *,
            extra_env: dict[str, str] | None = None,
    ) -> CellHandle:
        """
        以本 matrix 为治理域拉起一个 node cell 子进程.

        :param target: 指向 node 的路径, 支持:
            - 绝对路径: 直接使用
            - 相对路径: 相对 project.root 解析并立即绝对化
            指向 NODE.md → 直接用声明入口;
            指向目录 → 找目录下的 NODE.md;
            指向脚本 → NodeManifest.from_script 向上认亲.
        :param extra_env: 追加注入子进程的特殊环境变量. MOSS 运行时环境变量默认继承.
        :return CellHandle: cell 身份 + 子进程句柄. wait/stop 走 handle;
            子进程是否入网 (跑 matrix 且 announce) 由 mesh 观察, 不由本方法保证.

        :raise FileNotFoundError: target 解析后不存在.
        :raise RuntimeError: node 未安装 (错误信息给出 INSTALL.md 绝对路径).
        :raise DuplicatedError: singleton node 已有活实例.

        子进程运行失败通过 CellHandle.process 的 done callback 获取.
        """
        ...

    @abstractmethod
    def handled_cells(self) -> dict[CellAddress, CellHandle]:
        """
        本 matrix 当前**活着**的 cell handle, 按 address 索引.

        与 Subprocesses.executing() 同构 — 只含 process 未退出的条目.
        crash / 正常退出的 handle 会移出本 dict, 进入 dead_cells() FIFO.

        :return: dict 快照, 调用方不应 mutate 返回值.
        """
        ...

    @abstractmethod
    def dead_cells(self) -> list[CellHandle]:
        """
        最近死亡的 cell handle FIFO (bounded).

        与 Subprocesses.executed() 同构 — 保留有限条数, 溢出丢最老的.
        debug 视角: 通过 handle.process 拿 exit code / stderr 尾部.

        :return: list 快照, 最新的在末尾. 调用方不应 mutate 返回值.
        """
        ...

    # -- 子进程管理 -- #

    @property
    @abstractmethod
    def processes(self) -> Subprocesses:
        """
        当前进程的子进程管理器. 可以用 shell / execute 机制起子进程, 当前 Matrix Cell 进程托管生命周期. 避免孤儿.
        同时可以拿到所有通过它托管的子进程. run cell 底层的子进程管理基于此.
        """
        pass

    @property
    @abstractmethod
    def jobs(self) -> JobSupervisor:
        """
        基于子进程的后台任务管理器. 通过它可以托管和治理后台任务.
        底层通过 subprocess 托管子进程.
        """
        pass

    @abstractmethod
    def new_jobs(self) -> JobSupervisor:
        """
        创建一个新的 JobSupervisor 实例, 手动治理起 async with 生命周期, 独立使用. matrix 不会托管它的治理.
        """
        pass

    # -- Matrix 网络通讯协议底座 -- #

    @property
    @abstractmethod
    def session(self) -> Session:
        """
        面向 Matrix 全网的通讯总线
        五种通讯原语 (topic / stream / signal / ...) 是 Network 内部一切实现之间的通讯桥梁.
        """
        pass

    # @abstractmethod
    async def service_operator(self) -> ServiceOperator:
        """
        Matrix cell 之间的服务化通讯底座, 由 Cell 提供 Service, 其它 Cell 可以访问.
        屏蔽 Cell 之间点对点和一对多的通讯底层复杂度 (生命周期同步, 通讯层协议 --zenoh 等-- 屏蔽)
        方便在此基础上快速搭建业务层协议.
        """
        ...

    async def serve_service(self, service_cls: Type[ServiceServer]) -> ServiceServer:
        """通过 Service Server 的 Facade 或 Adapter 启动它, 注册到 Matrix 生命周期中. """
        server = service_cls.new(self)
        await self.add_lifecycle_object(server)
        return server

    async def connect_service(self, client_cls: Type[ServiceClient]) -> ServiceClient:
        """通过 Service Client 的 Facade 或 Adapter 启动它, 注册到 Matrix 生命周期中. """
        client = client_cls.new(self)
        await self.add_lifecycle_object(client)
        return client

    # -- 基础模块 -- #

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        IoC 容器的门 — 进程级共享服务 (manifests 声明的 providers).

        configs / resources 等运行时服务从这里 fetch;
        注册新服务优先走 manifests 声明 (环境发现自解释), 而非运行时 register.
        """
        pass

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """日志模块, 从属于当前节点."""
        pass

    @property
    @abstractmethod
    def resources(self) -> ResourceRegistry:
        """
        跨 scheme+host 的资源路由层 (VFS).
        """
        ...

    @property
    @abstractmethod
    def warrant(self) -> Warrant:
        """
        Matrix 级通用授权机制 — 交互式审批 (host 写 storage / 非 host topic 模式).

        消费方 `matrix.warrant.require(permission)` 做审批 (如 node 启动时授权).
        软授权边界 (非安全机制, 见 warrant FEATURE.md KD14): 模型可自迭代自授权.
        """
        pass

    # -- scoped 身份族: 运行时座标 → 存储隔离级别 -- #

    def runtime_scopes(self) -> dict[RuntimeScopeKey, str]:
        """返回 Matrix 运行时的维度座标, 用来构建不同的隔离级别."""
        # scoped 概念只能是运行时的 (mode × ghost × network × cell 四维座标, 在运行前不存在)
        # 它可以在 Project 范围内治理一块特殊的存储领地.
        return {
            'mode': self.env.mode_name,
            'ghost': self.env.ghost_name,
            'cell': self.this.address,
            'network': self.network_info.name,
        }

    def get_runtime_url_path(self, *scopes: RuntimeScopeKey, **kwargs: str) -> str:
        """
        基于作用域生成一个 URL 形式的资源路径 — 可作为唯一 id 管理可复用资源.

        例: get_runtime_url('ghost', 'mode', user=name) 生成
        "指定 Ghost 在指定模式下对特定用户" 的唯一 id.
        用于组装资源声明, 形如: scheme://cell_address/scoped/path/resource
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

    # -- 状态描述 -- #

    @abstractmethod
    def is_running(self) -> bool:
        """matrix 自身是否在运行."""
        pass

    def is_host(self) -> bool:
        """本 cell 是否是当前网络的 host — 通常对一些 Cell 治理包提供. """
        return self.this.is_host

    # -- 生命周期 -- #

    @abstractmethod
    def close(self) -> None:
        """关闭自身, 用于优雅退出."""
        ...

    @abstractmethod
    async def wait_closed(self) -> None:
        """阻塞等待自身运行退出, 所有功能都会关闭."""
        ...

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """阻塞等待自身退出. 仅限同步上下文调用."""
        ...

    @abstractmethod
    def create_task(
            self,
            cor: Awaitable[Any],
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        """创建包含在 Matrix 生命周期内的 Task."""
        ...

    @abstractmethod
    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """注册与 matrix 同步启动的对象. 依次序启动, 绑定生命周期, 不做容错. 仅运行前可调用."""
        ...

    @abstractmethod
    async def add_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """运行时动态添加 lifecycle object, 绑定到 exit stack, 退出时清空."""
        ...

    # -- 启动函数. 并非必要, 基于 code as prompt 原则提示如何使用 -- #

    async def arun(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        Matrix 运行的基本逻辑. 可参考或直接基于这个函数运行基于 Matrix 的应用.
        如果将它包裹成 asyncio.Task, 也可以和主协程并行运行.
        """
        if self.is_running():
            raise RuntimeError('Matrix already running.')

        async with self:
            loop = asyncio.get_running_loop()
            result_or_coro = main_coro(self)

            if asyncio.iscoroutine(result_or_coro):
                task = loop.create_task(result_or_coro)
                exit_signal = loop.create_task(self.wait_closed())
                try:
                    done, pending = await asyncio.wait(
                        [task, exit_signal],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if task in done:
                        return await task
                    raise asyncio.CancelledError("Matrix is closing")
                except asyncio.CancelledError:
                    pass  # 外部取消 (KeyboardInterrupt → asyncio.run 取消 task) 或内部关闭, 静默退出
                finally:
                    for t in [task, exit_signal]:
                        if not t.done():
                            t.cancel()
                    _ = await asyncio.gather(task, exit_signal, return_exceptions=True)
            else:
                return await result_or_coro

    def run(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        同步阻塞入口. 内部自动拉起事件循环并治理生命周期.
        兼容 Python 3.10 的顶层入口.
        """
        try:
            import uvloop
        except ImportError:
            uvloop = None

        try:
            if uvloop is not None:
                asyncio.set_event_loop(uvloop.new_event_loop())
            return asyncio.run(self.arun(main_coro))
        except KeyboardInterrupt:
            pass  # arun 已处理清理

    # -- serve_mcp: code-as-prompt 糖, 在 matrix 生命周期内 serve MCP server -- #

    async def aserve_mcp(
        self,
        mcp: 'MCPServer',
        *,
        host: str = '127.0.0.1',
        port: int = 0,
    ) -> None:
        """
        在已运行的 Matrix 内 serve 一个 MCP server.

        调用前 matrix 必须已进入上下文 (``async with matrix`` 内 或
        ``matrix.run()`` 回调内). 只负责 transport, 不管理 matrix 生命周期.

        serve 用 ``run_streamable_http_async`` (async), 绝不 ``mcp.run()``
        (后者内部 ``anyio.run()`` 开新 event loop, 跟 matrix 的 loop 打架).
        mcp 实例的 tools 在调用前已注册完毕, 本方法不重新注册.
        """
        if not self.is_running():
            raise RuntimeError('Matrix not running.  Use serve_mcp() or enter matrix context first.')

        self.logger.info(
            'MCP server serving on %s:%s (stateless streamable-http)',
            host, port,
        )
        await mcp.run_streamable_http_async(
            host=host, port=port, stateless_http=True,
        )

    def serve_mcp(
        self,
        mcp: 'MCPServer',
        *,
        host: str = '127.0.0.1',
        port: int = 0,
    ) -> None:
        """
        同步阻塞入口: 在 Matrix 生命周期内 serve 一个 MCP server.

        code-as-prompt 糖, 对标 :meth:`run`. 内部自动拉起事件循环、进入 matrix
        上下文、serve、退出清理. mcp 实例的 tools 在调用前已注册完毕, 本方法只
        负责 transport 生命周期, 不重新注册 tool::

            mcp = MCPServer("my-node")

            @mcp.tool()
            async def hello(name: str) -> str:
                return f"hi {name}"

            matrix.serve_mcp(mcp, port=8080)
        """
        try:
            import uvloop
        except ImportError:
            uvloop = None

        if uvloop is not None:
            asyncio.set_event_loop(uvloop.new_event_loop())

        async def _run():
            async with self:
                await self.aserve_mcp(mcp, host=host, port=port)

        return asyncio.run(_run())

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


_instance: Matrix | None = None
