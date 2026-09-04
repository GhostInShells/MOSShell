"""
MatrixImpl — Matrix ABC 的实现

骨架承 host/matrix.py 老代码 (人类手写的艰难草创产物, 生命周期/exit_stack/
ThreadSafeEvent 模式经过考验), 表面按 Matrix ABC 重整.
"""

import asyncio
import contextlib
import logging
from collections import deque
from pathlib import Path
from typing import Coroutine, Iterable, Type, Callable, Awaitable
from typing_extensions import Self

from ghoshell_container import IoCContainer, Provider
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts import Workspace, LocalWorkspace, ResourceRegistry, ConfigStore
from ghoshell_moss.contracts.subprocesses import Subprocesses, ProcessMeta, CaptureSpec

from ghoshell_moss.core.blueprint.matrix import Matrix, MatrixLifecycleObject, CellHandle
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata
from ghoshell_moss.core.blueprint.cell import (
    CellAddress, Cell, CellRuntimeInfo, CellPresence, CellNetwork,
    NodeManager, normalize,
    enter_cell_lifecycle,
)
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import Warrant
from ghoshell_moss.core.blueprint.parameter import Parameters
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.helpers import ThreadSafeEvent

from ghoshell_moss.matrix.adapter import MatrixNetworkAdapter
from ghoshell_moss.matrix.operator import ZenohOperator
from ghoshell_moss.matrix.parameters import ZenohParameters
from ghoshell_moss.matrix.zenoh_helper import MatrixEnvNamespace

__all__ = ['MatrixImpl']


class MatrixImpl(Matrix):
    """
    Matrix ABC 的实现 (§YY/§ZZ).

    构造由 factory._create_matrix (worker cell) 或 Host 抽象走 concrete (host)
    完成, 不 discover — matrix.discover 只是糖 (blueprint/matrix.py L92-116).
    """

    def __init__(
            self,
            *,
            env: Environment,
            project: Project,
            runtime_info: CellRuntimeInfo,
            adapter: MatrixNetworkAdapter,
            network: NetworkMetadata,
            logger: logging.Logger | None = None,
    ):
        # runtime_info 是 Matrix 唯一的身份真相载体 — cell + pid/pgid/start_time
        # 全部由 caller (factory worker path / Host concrete) 显式构建后传入.
        # Matrix 层不自造, 不猜, 只消费.
        if not env.is_sealed:
            raise RuntimeError(
                'Matrix requires sealed Environment; '
                'caller (factory/Host) must call env.seal() before Matrix construction.'
            )
        self._env = env
        self._project = project
        # set to current.
        runtime_info.pid = self._env.pid
        self._runtime_info = runtime_info
        self._adapter = adapter
        self._network_metadata = network

        cell = runtime_info.cell

        # 默认 logger = moss.cell.{normalize(address)}, hierarchical name
        # 冒泡到顶层 moss.log (project.bootstrap 已挂 handler).
        self._logger: logging.Logger = logger or logging.getLogger(
            f'moss.cell.{normalize(cell.address)}',
        )

        # -- 生命周期原语 (承 host/matrix.py) -- #
        self._started = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._exit_stack = contextlib.ExitStack()
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # -- container -- #
        self._container: IoCContainer = self._prepare_container()

        # -- 网络三件 + operator, __aenter__ async 阶段填 -- #
        self._presence: CellPresence | None = None  # adapter.new_presence 产物
        self._operator: 'ZenohOperator | None' = None  # service_operator() 惰性创建
        self._watcher: CellNetwork | None = None  # mesh() 惰性创建
        self._parameters: Parameters | None = None  # parameters() 惰性创建

        # -- 治理: run_node 拉起的所有 cell handle (§YY handled_cells 契约) -- #
        # 与 Subprocesses.executing/executed 同构: 活着的 handle 在 dict,
        # 死掉的进 FIFO (bounded). on_exit callback 完成 dict → deque 转移.
        self._handled_cells: dict[CellAddress, CellHandle] = {}
        self._dead_cells: deque[CellHandle] = deque(maxlen=128)

        # -- 生命周期挂载对象 (承老代码) -- #
        # 运行前 register_lifecycle_object 塞入, __aenter__ async 阶段依次 enter.
        self._lifecycle_bound: list[MatrixLifecycleObject | Type[MatrixLifecycleObject]] = []

        # -- 任务组 -- #
        self._binding_futures: set[asyncio.Future] = set()
        # provide_channel 的单槽任务 — 一个 cell 只提供一个根 channel (§UU-2).
        # 收尾时 cancel 触发 provider.arun_until_closed 走 finally.
        self._channel_provider_task: asyncio.Task | None = None

        # -- 日志前缀 -- #
        self._log_prefix = (
            f"<Matrix address={cell.address} "
            f"scope={network.scope} is_host={cell.is_host}>"
        )

    # ==================================================================
    # 身份 (Matrix ABC properties)
    # ==================================================================

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def project(self) -> Project:
        return self._project

    @property
    def this(self) -> Cell:
        # §YY-1 第 2 条: this 是纯数据 (Cell), 入网机制对象 (Presence)
        # 藏在 self._presence 里 (provide_channel/publish_event 是它的糖).
        return self._runtime_info.cell

    @property
    def network_info(self) -> NetworkMetadata:
        # §YY-1 第 4 条: 运行时自解释 — cell 定义时不知道自己会被接进哪个网络.
        return self._network_metadata

    # ==================================================================
    # 膜: 本 cell 的入网侧
    # ==================================================================

    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        """
        把 channel 作为本 cell 的膜暴露到网络.

        -- 设计意图 (人类 2026-07-13 明示, 修正 refactor 漂移) --

        1. **单实例**: 每个 Matrix 只允许 provide 一次 root channel; 二次调用
           直接 raise. 不做替换语义, 不做集合化 — cell 只提供一个根 channel
           是硬约束 (§UU-2 膜承诺).

        2. **单根**: 一个 matrix + cell = 一个 channel 根节点. 若未来需要
           子挂 / fractal 视图, 那是 Watcher 侧的事, 不是这里第二次 provide.

        3. **Fire-and-forget, 但外面可以 await**. 事实上绝大多数 cell 的
           主循环就是 `await matrix.provide_channel(channel)` — 阻塞到膜下线
           为止. 因此**返回的 Future = 跑 arun_until_closed 的 Task 本身**
           (Task IS asyncio.Future[None]). 不另造 future, 不改语义.

        4. **副作用委托给 Presence**: membrane += 'channel' + updated 时间戳
           + publish 'channel added' event 全部由 `self._presence.provide(channel)`
           完成 — 这是 Presence ABC 的承诺 (cell.py L718, §UU-7). 本函数
           只负责启动 provider, 不重复副作用.

        5. **不猜 provider 的 ready 信号**. Provider 与 Proxy 之间是自动
           通讯的, 上层调用者不需要读底层 wait_connected. 试图"猜"一个
           ready 语义就是上一版翻车的直接病因 (见"漂移记").

        -- 漂移记 (探索备查, 后来者勿再入坑) --

        上一版 (被本次 commit 覆盖) 做了三件错事:

        a) 起了双 task 结构 (outer `_run` + inner `_wait_connected`) 并调用
           `provider.wait_connected()` — provider 端**根本没这个方法**, 只在
           `ChannelProxy` (duplex/proxy.py) 和 `ChannelRuntime` (concepts/channel.py)
           上有. AttributeError 被内层 try 吞进另造的 future, 表面无声,
           实际 future 永不 resolve. `await matrix.provide_channel(...)`
           作为 cell 主循环的写法就被打断.

        b) 字段 `_channel_provider_tasks: set[Task]` 假设多实例 →
           违反单根 channel 约束.

        c) 误读 ABC docstring "future 在膜可被远端连接时 resolve" 为
           "另造一个 ready 信号 future", 实际原意是 "future = task 生命周期,
           走完 = 膜下线". 一字之差, 语义倒过来.

        moss-shell mcp 的 TTS 之所以还能说话, 是因为外层 task 事实上跑起了
        `arun_until_closed`, provider 副作用生效; 只是所有 `await` 调用者
        拿到的 future 是坏的. 这种"表面工作 + 隐性坏 API"是最难诊断的漂移形态.
        """
        self._check_running()
        if (
                self._channel_provider_task is not None
                and not self._channel_provider_task.done()
        ):
            raise RuntimeError(
                'Matrix.provide_channel called twice; a cell provides exactly '
                'one root channel (§UU-2 膜承诺). If you need a different '
                'channel, close this Matrix and start a new one.'
            )
        loop = self._event_loop
        if loop is None:
            raise RuntimeError('Matrix event loop not ready')
        if self._presence is None:
            raise RuntimeError('Matrix presence not initialized (adapter not started)')

        async def _providing() -> None:
            # Presence.provide: 拿 bare ChannelProvider, 顺带完成
            # membrane 声明 + 'channel added' event 广播 (§UU-7, zenoh_presence.py).
            provider = await self._presence.provide_channel(channel)
            # cell 已被 Presence 就地 mutate (providing += 'channel', update 时间戳),
            # 同步 ledger 让 CLI (moss cells status/show) 看到最新真相.
            # 与网络 announce 是两个正交的更新目标 (§UU-3).
            await self._sync_cell_ledger()
            # 跑到关闭 — matrix __aexit__ 会 cancel 本 task, 触发 provider
            # 自身的 finally 收尾 (arun_until_closed 内部处理 CancelledError).
            await provider.arun_until_closed(channel)

        task = loop.create_task(
            _providing(),
            name=f'provide_channel:{channel.name}',
        )
        self._channel_provider_task = task
        return task

    async def publish_event(self, content: str) -> None:
        """向网络广播 CellEvent (refetch=True). 委托 self._presence."""
        self._check_running()
        if self._presence is None:
            raise RuntimeError('Matrix presence not initialized')
        await self._presence.publish_event(content)

    # ==================================================================
    # 观察: 惰性门 mesh() → Watcher (§UU-7 / §YY-1 第 3 条 opt-in by usage)
    # ==================================================================

    async def network(self) -> CellNetwork:
        """
        惰性门: 首次调用时 adapter.new_watcher + add_lifecycle_object;
        后续调用返回同一实例. worker cell 不调即 O(1) 不付 O(N) 观察成本.
        """
        self._check_running()
        if self._watcher is not None:
            return self._watcher
        # -- 惰性构造 -- #
        # env 传给 mesh 用于 cell.is_local(env) 判定 (§UU-7 local/foreign 分档).
        watcher = self._adapter.new_watcher(
            env=self._env,
            logger=self._logger,
        )
        # 加进 async exit stack, 触发 __aenter__ 完成订阅 + 初始 refresh.
        await self._async_exit_stack.enter_async_context(watcher)
        self._watcher = watcher
        self._logger.debug("%s watcher lazily created", self._log_prefix)
        return watcher

    # ==================================================================
    # service operator: 惰性门, 遵循 network() 的 lazy-gate 模式
    # ==================================================================

    async def service_operator(self) -> 'ZenohOperator':
        """
        惰性门: 首次调用时构造 ZenohOperator + enter_async_context;
        后续调用返回同一实例. 纯 worker cell 不调即不付发现成本.
        """
        self._check_running()
        if self._operator is not None:
            return self._operator
        import zenoh
        session = self._container.force_fetch(zenoh.Session)
        network_ns = MatrixEnvNamespace(self._env).network_ns
        operator = ZenohOperator(
            session=session,
            network_ns=network_ns,
            this_address=self.this.address,
            logger=self._logger,
        )
        await self._async_exit_stack.enter_async_context(operator)
        self._operator = operator
        self._logger.debug("%s operator lazily created", self._log_prefix)
        return operator

    # ==================================================================
    # parameter: 惰性门, 点对点 declare/subscribe (matrix 面, §D1)
    # ==================================================================

    async def parameters(self) -> Parameters:
        """
        惰性门: 首次调用时构造 ZenohParameters + enter_async_context;
        后续调用返回同一实例. 纯 worker cell 不调即不付 queryable/发布循环成本.
        """
        self._check_running()
        if self._parameters is not None:
            return self._parameters
        import zenoh
        session = self._container.force_fetch(zenoh.Session)
        parameters = ZenohParameters(
            session,
            MatrixEnvNamespace(self._env),
            self.this.address,
            logger=self._logger,
        )
        await self._async_exit_stack.enter_async_context(parameters)
        self._parameters = parameters
        self._logger.debug("%s parameters lazily created", self._log_prefix)
        return parameters

    # ==================================================================
    # 治理咽喉: run_node (六动词的 run, §YY blueprint/matrix.py L204+)
    # ==================================================================

    async def run_node(
            self,
            target: Path,
            *,
            extra_env: dict[str, str] | None = None,
    ) -> CellHandle:
        """
        拉起一个 node cell — 唯一 spawn 咽喉收敛到 NodeManager.spawn_node.

        只做: resolve target → spawn_node (installed/probe/singleton 预检/execute)
        → 组装 CellHandle.
        singleton 预检已由 spawn_node 同步判定 (撞锁抛 DuplicatedError, 本层不再重复);
        锁持有 / 账本清理 / pid·pgid 回填归 node 自身 (enter_cell_lifecycle) 或 host 清孤儿.

        :param target: 路径 (NODE.md / 目录 / 脚本 / 相对 project.root).
        :param extra_env: 追加注入子进程的环境变量.
        :return CellHandle: cell 身份 + 子进程句柄, 由 matrix.handled_cells() 追踪.
        """
        self._check_running()

        node_manager = self._container.force_fetch(NodeManager)
        manifest = await asyncio.to_thread(node_manager.resolve_node, target)

        runtime, managed = await node_manager.spawn_node(
            manifest,
            extra_env=extra_env,
            capture=self._cell_capture,
        )

        handle = CellHandle(runtime=runtime, process=managed)
        self._handled_cells[runtime.address] = handle
        managed.add_done_callback(self._on_cell_exit(runtime.address))
        self._logger.info(
            "%s run_node spawned: address=%s pid=%s home=%s",
            self._log_prefix, runtime.address, managed.meta.pid, runtime.cell.home,
        )
        return handle

    @staticmethod
    def _cell_capture(runtime: CellRuntimeInfo) -> CaptureSpec:
        """capture factory — 落盘 stdout/stderr 到 cell.home/runtime/ (路径依赖 address)."""
        runtime_dir = Path(runtime.cell.home) / CellRuntimeInfo.RUNTIME_SUBDIR
        return CaptureSpec(
            buffer_lines=200,
            stdout_file=CellRuntimeInfo.default_stdout_log(runtime_dir, runtime.address),
            stderr_file=CellRuntimeInfo.default_stderr_log(runtime_dir, runtime.address),
        )

    def _on_cell_exit(
            self,
            address: CellAddress,
    ) -> Callable[[ProcessMeta], None]:
        """构造 on_exit callback: 从 _handled_cells → _dead_cells FIFO.

        账本清理归 node 自身 (enter_cell_lifecycle finally) 或 host 清孤儿, 此处不重复.
        callback 在 asyncio loop 线程触发 (Subprocesses 承诺), 无需线程安全.
        """

        def _callback(meta: ProcessMeta) -> None:
            handle = self._handled_cells.pop(address, None)
            if handle is not None:
                self._dead_cells.append(handle)
            self._logger.info(
                "%s cell exited: address=%s exit_code=%s",
                self._log_prefix, address, meta.exit_code,
            )

        return _callback

    def handled_cells(self) -> dict[CellAddress, CellHandle]:
        """§YY: 当前活着的 cell handle. 返回 dict 快照, 调用方不应 mutate."""
        return dict(self._handled_cells)

    def dead_cells(self) -> list[CellHandle]:
        """§YY: 最近死亡的 cell handle FIFO 快照, 最新在末尾."""
        return list(self._dead_cells)

    def _kill_orphan_cell(self, info: CellRuntimeInfo) -> None:
        """host 侧 clear_cell_runtimes 的 kill 回调 — 走 NodeManager.kill_cell. 幂等."""
        try:
            node_manager = self._container.force_fetch(NodeManager)
            node_manager.kill_cell(info.address)
        except Exception:
            self._logger.exception(
                "%s failed to kill orphan cell %s",
                self._log_prefix, info.address,
            )

    def _sync_cell_ledger_impl(self) -> None:
        """sync 落盘 (给 to_thread 用). ledger 是 debug 便利, 写失败不阻断运行时."""
        try:
            self._runtime_info.write_to_runtime_dir(self._env.cell_runtimes_dir)
        except Exception:
            self._logger.exception(
                "%s failed to sync CellRuntimeInfo to ledger",
                self._log_prefix,
            )

    async def _sync_cell_ledger(self) -> None:
        """将 runtime_info 落盘, 反映 self._runtime_info.cell 的最新状态到 ledger.

        调用时机: 任何修改 self._runtime_info.cell 的操作之后 (今为 provide_channel;
        未来 providing 增删 / update() 触发同源写盘). 让 CLI (moss cells status/show)
        看到最新真相. 与 network announce 是两个正交的更新目标 (§UU-3 owner/network 两平面).

        走 to_thread — MOSS 全双工纪律: async 内任何 > 1ms 同步 IO 都是 stop-the-world.
        """
        await asyncio.to_thread(self._sync_cell_ledger_impl)

    def _closing_publish_hint(
            self, exc_type, exc_val,
    ) -> str | None:
        """决定关闭时 publish 的 content — §WW-5 四弧发送侧.

        None 表示不 publish (罕见, 目前 always publish).
        - 正常 exit → 弧④ "closing normally"
        - KeyboardInterrupt → 弧①的一种 "closing on keyboard interrupt"
        - CancelledError → "closing on cancel"
        - 其他 Exception → 弧② "crash: {type}: {msg}"

        content 里带 exc_type 名, 让远端 mesh 消费者在生成 signal 时有分类依据
        (event → signal → mindflow 的转换归 channel 层, 不在本 matrix 层做).
        """
        if exc_val is None:
            return 'closing normally'
        if isinstance(exc_val, KeyboardInterrupt):
            return 'closing on keyboard interrupt'
        if isinstance(exc_val, asyncio.CancelledError):
            return 'closing on cancel'
        # 其他 exception: crash 弧②. 截 200 字符防超长 payload.
        msg = str(exc_val)[:200] if str(exc_val) else ''
        exc_name = exc_type.__name__ if exc_type is not None else 'Exception'
        return f'crash: {exc_name}: {msg}' if msg else f'crash: {exc_name}'

    # ==================================================================
    # 灶台 (Subprocesses 从 IoC pull)
    # ==================================================================

    @property
    def processes(self) -> Subprocesses:
        self._check_running()
        return self._container.force_fetch(Subprocesses)

    # ==================================================================
    # 门: session / workspace / home / container / logger
    # ==================================================================

    @property
    def session(self) -> Session:
        self._check_running()
        return self._container.force_fetch(Session)

    # workspace 由 Matrix ABC 提供 concrete: return self.project.workspace.
    # (blueprint/matrix.py) 无需 override.
    #
    # home 由 Matrix ABC 提供 default: Path(self.this.home). 不 override.
    # build_node_from_manifest / build_host_cell 决定 Cell.home.

    @property
    def cell_workspace(self) -> Workspace:
        return LocalWorkspace(self.home)

    @property
    def resources(self) -> ResourceRegistry:
        return self._container.force_fetch(ResourceRegistry)

    @property
    def warrant(self) -> Warrant:
        self._check_running()
        return self._container.force_fetch(Warrant)

    @property
    def container(self) -> IoCContainer:
        if self._container is None:
            raise RuntimeError('Matrix container not initialized')
        return self._container

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # ==================================================================
    # 状态
    # ==================================================================

    def is_running(self) -> bool:
        return self._started and not (
                self._closing_event.is_set() or self._closed_event.is_set()
        )

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Matrix is not running (address={self.this.address})')

    # is_host_running 从 Matrix ABC 拿掉 (2026-07 review): sync 依赖惰性 mesh
    # 是设计瑕疵 (worker 未 mesh() 就无从判). 消费者需要判组网状态时走
    # `(await matrix.mesh()).has_host()` — host 在 network 级唯一, 不需 project_id 过滤.

    # ==================================================================
    # 生命周期基础 (承 host/matrix.py)
    # ==================================================================

    def close(self) -> None:
        self._closing_event.set()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        return self._closed_event.wait_sync(timeout)

    def create_task(
            self,
            cor: Awaitable,
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        self._check_running()

        async def _ensure_future_done():
            try:
                await cor
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._logger.error(
                    "%s inner task %s exception: %r",
                    self._log_prefix, name, e,
                )
                if stop_matrix_on_error and self.is_running():
                    self.close()
            finally:
                self._logger.debug("%s inner task %s done", self._log_prefix, name)

        task = self._event_loop.create_task(_ensure_future_done(), name=name)
        self._binding_futures.add(task)
        task.add_done_callback(lambda c: self._binding_futures.discard(c))
        return task

    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """运行前注册. 运行中调用抛错 (add_lifecycle_object 才是动态添加通道)."""
        if self._closing_event.is_set():
            raise RuntimeError('Matrix already closing')
        if self.is_running():
            self._event_loop.create_task(self._async_exit_stack.enter_async_context(obj))
        else:
            self._lifecycle_bound.append(obj)

    async def add_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """运行时动态添加. 已在 exit_stack 中的对象跳过."""
        self._check_running()
        for registered in self._lifecycle_bound:
            if obj is registered:
                return
        self._lifecycle_bound.append(obj)
        await self._async_exit_stack.enter_async_context(obj)
        self._logger.info("%s add lifecycle object: %s", self._log_prefix, obj)

    # ==================================================================
    # 装配: IoC 两阶段 (§ZZ-5)
    # ==================================================================

    def _prepare_container(self) -> IoCContainer:
        """
        sync 阶段: 注册全部 provider, 返回未 bootstrap 的 container.

        次序: project providers (MOSS.manifests) → matrix 实例 set →
        MATRIX.manifests (环境能力, mode 激活时) → adapter driver 装配 →
        matrix default providers → matrix 契约校验.
        """
        container = self.project.container
        # -- matrix 自身实例 -- #
        container.set(Matrix, self)
        container.set(MatrixImpl, self)
        container.set(Environment, self._env)
        container.set(Project, self._project)
        container.set(Workspace, self._project.workspace)
        container.set(MatrixNetworkAdapter, self._adapter)
        container.set(Cell, self._runtime_info.cell)
        container.set(CellAddress, self._runtime_info.cell.address)

        # -- MATRIX.manifests: mode 环境能力 (mode 激活时, 初始全空) -- #
        if not self._env.no_mode:
            mode = self._project.current_mode()
            if mode is not None:
                mode.bootstrap()
                matrix_m = mode.matrix_manifests()
                for p in matrix_m.providers():
                    if p.is_error():
                        continue
                    container.register(p.value())

        # -- adapter driver-specific 装配 -- #
        # bind_ioc: 通常注册 lazy provider (如 zenoh.Session), 捕捉 adapter 引用,
        # 首次 fetch 时读 adapter._session (那时 adapter.__aenter__ 已完成).
        self._adapter.bind_ioc(container)
        for provider in self._adapter.default_providers():
            if container.bound(provider.contract()):
                continue
            container.register(provider)

        # -- matrix default providers — if not bound (default 兜底) -- #
        for provider in self._default_providers():
            if container.bound(provider.contract()):
                continue
            container.register(provider)

        # -- matrix 装配契约校验 — 缺失任何一项即构造失败 (fail-fast) -- #
        self.contracts().validate(container)

        return container

    def _default_providers(self) -> Iterable[Provider]:
        """
        matrix 层的 default 接线 (default 兜底).

        仅剩 logger. 协议 default (topic/session/qa) 归 adapter;
        子进程/任务等 project 级能力归 Project._default_providers.
        """
        from ghoshell_moss.matrix.providers.logger_provider import MatrixLoggerProvider
        yield MatrixLoggerProvider()

    # ==================================================================
    # 生命周期 (承 host/matrix.py __aenter__/__aexit__ 骨架, 表面按 §ZZ-5)
    # ==================================================================

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError('Matrix already started')
        self._started = True
        self._event_loop = asyncio.get_running_loop()

        # -- sync 阶段 -- #
        self._exit_stack.__enter__()
        # bootstrap project first
        self._exit_stack.enter_context(self._project)

        # 本 cell 生命周期先落地 (§UU-6 单写者原则):
        #   1. singleton → workspace.lock() 争抢 (host 一律 singleton)
        #   2. host → 启动时 clear_cell_runtimes 清孤儿
        #   3. 写 self CellRuntimeInfo 到 env.cell_runtimes_dir (让 CLI/其他 cell 能看见)
        #   4. 反卷: 删自己的 runtime file, host 再清一遍孤儿
        # 位置在 container 之前 — 装配失败时 exit_stack 会反卷释放锁/删文件.
        # runtime_info 由 __init__ 注入 (caller 显式构建), 此处不自造.
        enter_cell_lifecycle(
            self._exit_stack,
            self._env,
            self._runtime_info,
            kill=self._kill_orphan_cell,
        )
        # pull LoggerItf from IoC 覆写 (§ZZ-6): 有覆写就用, 没有保 self._logger 兜底
        pulled = self._container.get(LoggerItf)
        if pulled is not None:
            # 只在 pulled 不是 root moss logger 时覆写 (避免层级混乱)
            self._logger = pulled if isinstance(pulled, logging.Logger) else self._logger
        else:
            self._container.set(LoggerItf, self._logger)

        # -- async 阶段 -- #
        try:
            await self._async_exit_stack.__aenter__()

            # 0. Subprocesses 生命周期接线 — 从 IoC 取出, 未运行则绑进 exit stack.
            #    enter 越早, __aexit__ 越后反卷 → 它在 matrix 所有子进程之后做最后清场
            #    (SIGINT→grace→SIGKILL), 兜住"子进程不比 owner 活得久"铁律.
            #    空转无成本: __aenter__ 只置位, __aexit__ 对空 _executing/_tasks 是空循环.
            subprocesses = self._container.force_fetch(Subprocesses)
            if not subprocesses.is_running():
                await self._async_exit_stack.enter_async_context(subprocesses)

            # channel provider tasks 收尾钩子 (承老 _ensure_channel_provider_task_cancelled)
            self._async_exit_stack.push_async_callback(self._cancel_channel_provider_tasks)

            # 1. adapter async 起 (driver 底层: zenoh session + hub)
            await self._async_exit_stack.enter_async_context(self._adapter)

            # 2. new_presence + async ctx (触发首次 announce)
            self._presence = self._adapter.new_presence(
                self._runtime_info.cell,
                logger=self._logger,
            )
            await self._async_exit_stack.enter_async_context(self._presence)

            # 3. force_fetch TopicService + Session 触发 provider chain,
            #    进 async ctx 让它们跑到关闭
            topic_service = self._container.force_fetch(TopicService)
            await self._async_exit_stack.enter_async_context(topic_service)

            session = self._container.force_fetch(Session)
            await self._async_exit_stack.enter_async_context(session)

            # 3.5 warrant 默认加载 (可选: 未注册 provider 则跳过, matrix.warrant 访问时再报错)
            warrant = self._container.get(Warrant)
            if warrant is not None:
                await self._async_exit_stack.enter_async_context(warrant)

            # 4. lifecycle_bound 依次启动 (承老代码)
            enter_order: list[MatrixLifecycleObject] = []
            for lc in self._lifecycle_bound:
                if isinstance(lc, type):
                    obj = self._container.get(lc)
                else:
                    obj = lc
                if obj is None:
                    self._logger.warning(
                        "%s lifecycle object %s not found in container",
                        self._log_prefix, lc,
                    )
                    continue
                await self._async_exit_stack.enter_async_context(obj)
                enter_order.append(obj)
            self._lifecycle_bound = enter_order

            # 5. task_group 收尾钩子
            self._async_exit_stack.push_async_callback(self._cancel_task_group)

            self._logger.info(
                "%s matrix started (driver=%s scope=%s)",
                self._log_prefix, self._network_metadata.driver, self._network_metadata.scope,
            )
            return self
        except Exception as e:
            self._logger.exception(
                "%s matrix failed to start: %s", self._log_prefix, e,
            )
            self._closing_event.set()
            raise

    async def _cancel_channel_provider_tasks(self) -> None:
        task = self._channel_provider_task
        self._channel_provider_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _cancel_task_group(self) -> None:
        tasks = list(self._binding_futures)
        self._binding_futures.clear()
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # -- 关闭 publish (§WW-5 四弧发送侧) -- #
        # matrix 层保障 event 发送成功即完成责任, 消费侧 (event → signal → mindflow)
        # 归 channel 层, 这里不等待送达. 必须在 async_exit_stack 反卷之前 —
        # 否则 presence 已 close, publisher 已释放.
        # updated=False: cell 即将下线, mesh 消费者 refetch 无意义, 只推送信号.
        if self._presence is not None:
            content = self._closing_publish_hint(exc_type, exc_val)
            if content is not None:
                try:
                    await self._presence.publish_event(content, updated=False)
                except asyncio.CancelledError:
                    self._logger.info(
                        "%s publish closing event cancelled: %s",
                        self._log_prefix, content,
                    )
                except Exception:
                    self._logger.exception(
                        "%s failed to publish closing event: %s",
                        self._log_prefix, content,
                    )

        try:
            if exc_val is not None:
                if isinstance(exc_val, KeyboardInterrupt):
                    self._logger.info("%s stop on keyboard interrupt", self._log_prefix)
                elif isinstance(exc_val, asyncio.CancelledError):
                    self._logger.info("%s stop on cancel", self._log_prefix)
                else:
                    self._logger.exception(
                        "%s stop on error: %s", self._log_prefix, exc_val,
                    )
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self._logger.exception("%s failed to aexit: %s", self._log_prefix, e)
        finally:
            self._closing_event.set()
            self._closed_event.set()

        # sync stack 反卷 (container shutdown + enter_cell_lifecycle finally 触发 —
        # runtime file 删除在此)
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
        # discover 缓存复位 — 已关闭的 matrix 不再被 Matrix.discover() 返回.
        Matrix.reset_discover_instance(self)
