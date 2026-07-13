"""
MatrixImpl — Matrix ABC 的实现 (§ZZ 全套决策的落地).

骨架承 host/matrix.py 老代码 (人类手写的艰难草创产物, 生命周期/exit_stack/
ThreadSafeEvent 模式经过考验), 表面按 §YY 的 Matrix ABC 重整.

装配次序 (§ZZ-5 IoC 两阶段纪律):
  sync 阶段 (Matrix.__aenter__ 前段):
    1. container.set(Matrix/Env/Project/Workspace/MatrixNetworkAdapter, ...)
    2. matrix_manifests().providers 全 register (workspace baseline)
    3. matrix _default_providers() (Subprocesses/JobSupervisor) — if not bound
    4. adapter.bind_ioc(container) → 注册 lazy zenoh.Session (§ZZ-5 provide 语法糖)
    5. adapter.default_providers() → 注册 TopicService/Session Provider (if not bound)
    6. logger provider — if not bound (default MatrixLoggerProvider)
    7. container.bootstrap() — 触发副作用改造, 不 fetch driver 底层对象, 不成环
    8. pull LoggerItf from IoC 覆写 self._logger (§ZZ-6 pull 不 push)
  async 阶段 (Matrix.__aenter__ 后段):
    9. async_exit_stack.__aenter__
    10. await adapter.__aenter__() → self._session 填充, hub 起
    11. adapter.new_presence(presence_data) → async ctx (触发首次 announce)
    12. force_fetch(TopicService) / force_fetch(Session) → 触发 lazy chain
    13. lifecycle_bound_objects 依次启动
    14. task_group cleanup + provide_channel task cleanup 钩子

不拆 HostMatrix / CellMatrix 两个类 (§ZZ-7): is_host 已通过 CellPresence 显式
确认非判断, 差异全在 adapter 起 driver 的 config 参数.
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Coroutine, Iterable, Type, Callable, Any
from typing_extensions import Self

from ghoshell_container import Container, IoCContainer, Provider, provide
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts import Workspace, ConfigStore
from ghoshell_moss.contracts.configs import WorkspaceYamlConfigStoreProvider
from ghoshell_moss.contracts.resource import ResourceStorageFactoryBootstrapper
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.contracts.job_supervisor import JobSupervisor

from ghoshell_moss.core.blueprint.matrix import Matrix, MatrixLifecycleObject
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata, MatrixManifest
from ghoshell_moss.core.blueprint.cell import (
    CellAddress, CellPresence, CellManifest, CellRecord, CellState,
    Presence, Watcher, ExecSpec, normalize, make_address, DuplicatedError,
)
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.helpers import ThreadSafeEvent

from ghoshell_moss.matrix.providers.subprocesses_provider import MatrixSubprocessesProvider
from ghoshell_moss.matrix.providers.job_supervisor_provider import MatrixJobSupervisorProvider
from ghoshell_moss.matrix.providers.logger_provider import MatrixLoggerProvider

from ghoshell_moss.matrix.adapter import MatrixNetworkAdapter

from ghoshell_moss.message import unique_id

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
            manifest: CellManifest,
            presence: CellPresence,
            adapter: MatrixNetworkAdapter,
            network: NetworkMetadata,
            logger: logging.Logger | None = None,
    ):
        # -- 显式入参 -- #
        # env: 环境载体 (身份/路径唯一信源, seal 后 discover 全局单例)
        # project: 治理域句柄
        # manifest: 本 cell 的稳定身份 (name 作 home 键, §YY-1 第 6 条双目录判决)
        # presence: 本 cell 网络身份 (address/is_host/alias 已定死, factory 或 Host 侧构造)
        # adapter: driver 私有 (§ZZ-3), 未 __aenter__
        # network: metadata (name/scope/driver/config), 运行时自解释
        # env 必须已 seal (factory / Host concrete 侧责任).
        # 老 host/matrix.py 里 __init__ 调 env.seal() 是遗迹 — seal 是一次性
        # 跃迁, 二次抛错. Matrix 层假设 env 已 sealed 到达.
        if not env.is_sealed:
            raise RuntimeError(
                'Matrix requires sealed Environment; '
                'caller (factory/Host) must call env.seal() before Matrix construction.'
            )
        self._env = env
        self._project = project
        self._manifest = manifest
        self._presence_data = presence
        self._adapter = adapter
        self._network_metadata = network

        # -- logger: §ZZ-6 命名层级 + pull 反绑 -- #
        # 默认 logger = moss.cell.{normalize(address)}, java log4j hierarchy
        # 冒泡到顶层 moss.log (project.bootstrap 已挂 handler).
        self._logger: logging.Logger = logger or logging.getLogger(
            f'moss.cell.{normalize(presence.address)}',
        )

        # -- 生命周期原语 (承 host/matrix.py) -- #
        self._started = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._exit_stack = contextlib.ExitStack()
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # -- container -- #
        self._container: Container | None = None    # sync 阶段填

        # -- 网络三件, __aenter__ async 阶段填 -- #
        self._presence: Presence | None = None   # adapter.new_presence 产物
        self._watcher: Watcher | None = None     # mesh() 惰性创建

        # -- 生命周期挂载对象 (承老代码) -- #
        # 运行前 register_lifecycle_object 塞入, __aenter__ async 阶段依次 enter.
        self._lifecycle_bound: list[MatrixLifecycleObject | Type[MatrixLifecycleObject]] = []

        # -- 任务组 -- #
        self._task_group: set[asyncio.Task] = set()
        # provide_channel 起的 channel provider tasks, 收尾时统一 cancel
        self._channel_provider_tasks: set[asyncio.Task] = set()

        # -- 日志前缀 -- #
        self._log_prefix = (
            f"<Matrix address={presence.address} "
            f"scope={network.scope} is_host={presence.is_host}>"
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
    def this(self) -> CellPresence:
        # §YY-1 第 2 条: this 是纯数据 (CellPresence), 入网机制对象 (Presence)
        # 藏在 self._presence 里 (provide_channel/publish_event 是它的糖).
        return self._presence_data

    @property
    def network(self) -> NetworkMetadata:
        # §YY-1 第 4 条: 运行时自解释 — cell 定义时不知道自己会被接进哪个网络.
        return self._network_metadata

    # ==================================================================
    # 膜: 本 cell 的入网侧
    # ==================================================================

    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        """
        把 channel 作为本 cell 的膜暴露到网络 (§YY blueprint/matrix.py).

        实现: 委托 self._presence.provide(channel) 拿 ChannelProvider,
        create_task(provider.arun_until_closed(channel)) 启动, 返回 future
        在 provider.wait_connected() 完成时 resolve.
        """
        self._check_running()
        loop = self._event_loop
        if loop is None:
            raise RuntimeError('Matrix event loop not ready')
        if self._presence is None:
            raise RuntimeError('Matrix presence not initialized (adapter not started)')

        future: asyncio.Future[None] = loop.create_future()

        async def _run() -> None:
            try:
                # 1. Presence.provide 返回 bare ChannelProvider (未 arun);
                #    副作用已完成: presence.membrane += ['channel'] + touch updated
                #    + 尝试 publish 'channel added' CellEvent (refetch=True).
                provider = await self._presence.provide(channel)
                # 2. 后台任务里跑 provider 到关闭. wait_connected 到达 = 膜可用,
                #    此时 resolve future.
                async def _wait_connected() -> None:
                    try:
                        await provider.wait_connected()
                        if not future.done():
                            future.set_result(None)
                    except Exception as e:
                        if not future.done():
                            future.set_exception(e)
                loop.create_task(_wait_connected())
                # 3. arun_until_closed 会跑到 provider 关闭 (matrix __aexit__ cancel).
                await provider.arun_until_closed(channel)
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as e:
                self._logger.exception(
                    "%s provide_channel task exception: %s", self._log_prefix, e,
                )
                if not future.done():
                    future.set_exception(e)

        task = loop.create_task(_run(), name=f'provide_channel:{channel.name if hasattr(channel, "name") else "?"}')
        self._channel_provider_tasks.add(task)
        task.add_done_callback(self._channel_provider_tasks.discard)
        return future

    async def publish_event(self, content: str) -> None:
        """向网络广播 CellEvent (refetch=True). 委托 self._presence."""
        self._check_running()
        if self._presence is None:
            raise RuntimeError('Matrix presence not initialized')
        await self._presence.publish_event(content)

    # ==================================================================
    # 观察: 惰性门 mesh() → Watcher (§UU-7 / §YY-1 第 3 条 opt-in by usage)
    # ==================================================================

    async def mesh(self) -> Watcher:
        """
        惰性门: 首次调用时 adapter.new_watcher + add_lifecycle_object;
        后续调用返回同一实例. worker cell 不调即 O(1) 不付 O(N) 观察成本.
        """
        self._check_running()
        if self._watcher is not None:
            return self._watcher
        # -- 惰性构造 -- #
        # self_project_id 用于 Watcher.view(project_id=...) 的本地/远端过滤
        # (UU-1.10 数据标签 + 视图过滤, 不做 namespace 硬切分).
        watcher = self._adapter.new_watcher(
            self_project_id=self._project.id,
            logger=self._logger,
        )
        # 加进 async exit stack, 触发 __aenter__ 完成订阅 + 初始 refresh.
        await self._async_exit_stack.enter_async_context(watcher)
        self._watcher = watcher
        self._logger.debug("%s watcher lazily created", self._log_prefix)
        return watcher

    # ==================================================================
    # 治理咽喉: run_cell (六动词的 run, ledger 域, §YY blueprint/matrix.py L204+)
    # ==================================================================

    async def run_cell(
            self,
            target: str,
            *,
            extra_env: dict[str, str] | None = None,
    ) -> CellPresence:
        """
        拉起一个 cell — 咽喉五步 (§YY blueprint/matrix.py run_cell docstring 中钉住).

        本期实现覆盖: target 解析 (name / path) → singleton 查重 (domain 档 owner
        内存态) → processes.execute spawn.

        本期简化 / TODO 记号:
          - host 档 flock 单例执法暂不实施 (§WW 判决 v2 落地)
          - 实例残迹目录修剪保留策略暂不实施 (每次 spawn 新 uid)
          - ledger append 仅 logger.info, 不落 JSON (§UU-6 ledger 无对象身份,
            咽喉写 best-effort 不回读, CLI 是唯一读者, 本期无 CLI 消费者)
          留 TODO 引 §UU/§WW 章节, 后续补.

        :raise LookupError: name 不在 inventory (含近似名提示)
        :raise FileNotFoundError: path 不存在
        :raise RuntimeError: cell 未安装 (给 INSTALL.md 绝对路径)
        :raise DuplicatedError: singleton 声明冲突
        """
        self._check_running()

        # -- 咽喉步骤 1: 解析 target → CellManifest + ExecSpec 绝对化 -- #
        manifest = self._resolve_target(target)
        if manifest.exec is None:
            raise RuntimeError(
                f"cell {manifest.name!r} has no exec spec (CELL.md missing `run:` "
                f"declaration); cannot spawn without explicit entrypoint. "
                f"Declare `run:` in {manifest.name}/CELL.md."
            )
        if not manifest.installed:
            # TT-12 错误信息即 prompt: 指向 INSTALL.md 让模型下一步知道去装
            # (INSTALL.md 路径 = CELL.md 同目录, 但 manifest 层没有 abs path,
            #  本期先给 name 提示, 后续补 abs)
            raise RuntimeError(
                f"cell {manifest.name!r} is not installed. See "
                f"{manifest.name}/{CellManifest.INSTALL_FILENAME} for install steps."
            )

        # -- 步骤 2: singleton 查重 (domain 档 owner 内存态) -- #
        new_address = make_address(
            'cell', manifest.name, uid=unique_id()[:8],
        )
        if manifest.singleton == 'domain':
            # owner 内存态查重: 遍历 processes.executing() 看有没有同 name 的
            # 已跑实例. 本期只做 name 匹配 (address 里 name 段作为 key).
            for meta in self._processes_snapshot():
                if meta.get('cell_name') == manifest.name:
                    raise DuplicatedError(
                        f"cell {manifest.name!r} declares singleton=domain and is "
                        f"already running (pid={meta.get('pid')}); "
                        f"stop the existing instance before running a new one."
                    )
        elif manifest.singleton == 'host':
            # host 档 = 机器级硬件单点 (机器人控制类), 走 flock 跨治理域互斥.
            # TODO: 本期暂不实施 flock 执法 (§WW-4 判决), 走 domain 档同样查重.
            self._logger.warning(
                "cell %r declares singleton=host; flock enforcement not yet implemented (§WW-4)",
                manifest.name,
            )

        # -- 步骤 3: 修剪同稳定身份的旧实例残迹目录 -- #
        # §YY-1 第 6 条: crash 现场留到下次 spawn 才清, 保留策略在 spawn 时执行.
        # TODO: 本期暂不实施, 每次 spawn 新 uid 目录不清理.

        # -- 步骤 4: processes.execute spawn -- #
        instance_cwd = self._instance_runtime_dir(new_address)
        instance_cwd.mkdir(parents=True, exist_ok=True)

        # 子进程环境: env.dump_cell_env + 用户 extra_env 覆盖
        child_env = self.env.dump_cell_env(
            cell_address=new_address,
            parent_cell_address=self.this.address,
            with_os_env=True,
        )
        if extra_env:
            child_env.update(extra_env)

        # 组装 argv: manifest.exec.command + args
        argv = [manifest.exec.command, *manifest.exec.args]
        managed = await self.processes.execute(
            *argv,
            name=f'cell:{manifest.name}',
            description=manifest.description or f'cell {manifest.name}',
            cwd=instance_cwd,
            extra_env={**manifest.exec.env, **child_env},
            with_os_env=False,   # child_env 已含 os_env 副本
        )

        # -- 步骤 5: append ledger CellRecord -- #
        # §UU-6: 咽喉 spawn 瞬间 append JSON best-effort, 单写者原则.
        # 本期只 logger.info 记录 (无 CLI 消费者, 落盘 TODO):
        record = CellRecord(
            address=new_address,
            alias=manifest.name,
            pid=managed.meta.pid if hasattr(managed.meta, 'pid') else 0,
            pgid=0,   # TODO: process group id from ManagedProcess
            start_time=0.0,   # TODO: 从 managed.meta 取
            project_id=self.project.id,
            cwd=str(instance_cwd.absolute()),
            spawner=self.this.address,
        )
        self._logger.info(
            "%s run_cell append record: %s",
            self._log_prefix, record.model_dump_json(exclude_defaults=True),
        )
        # TODO: 落盘 workspace/runtime/cells/ledger.jsonl (§UU-6 咽喉唯一写者)

        # -- 返回合成 SPAWNED 态 presence (WW-5 无 wait) -- #
        # is_host 从 address 推断 (§ZZ-10 property), 子 cell 一律 worker
        # (host 是顶层启动, 不经 run_cell → address='cell/...' → is_host=False).
        # 后续 ready / crash / normal exit / 永不入网 通过 CellEvent → Signal
        # 送 MossRuntime.mindflow 作 background hint (M7.5), 不在此处 wait.
        return CellPresence(
            address=new_address,
            alias=manifest.name,
            state=CellState.SPAWNED,
            project_id=self.project.id,
        )

    def _resolve_target(self, target: str) -> CellManifest:
        """
        解析 target → CellManifest (§YY run_cell docstring 步骤 1).

        name (不含路径分隔符) → project.cells 遍历查 name;
        path (含路径分隔符) → 按调用方 cwd 解析绝对化, 目录/CELL.md/脚本三种.
        """
        if '/' in target or '\\' in target or target.endswith('.py'):
            # path 形式
            path = Path(target).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"cell target path {target!r} does not exist "
                    f"(resolved to {path}). Check the path or use a name in inventory."
                )
            if path.is_dir():
                manifest = CellManifest.read_from_directory(path)
                if manifest is None:
                    raise LookupError(
                        f"no {CellManifest.MANIFEST_FILENAME} found in {path}. "
                        f"Either add CELL.md or point to a script file directly."
                    )
                return manifest
            # 脚本文件 → 向上认亲 (WW-4)
            return CellManifest.from_script(path)

        # name 形式: project.cells inventory 反查
        cells = self.project.cells.list_cell_manifests(refresh=False)
        # cells: dict[relative_path, CellManifest]. name 反查 = 遍历.
        for _rel, manifest in cells.items():
            if manifest.name == target:
                return manifest
        # 未找到 — 给近似名 (TT-12 错误信息即 prompt)
        names = sorted({m.name for m in cells.values()})
        raise LookupError(
            f"cell name {target!r} not found in project inventory. "
            f"Available: {names[:10]}{'...' if len(names) > 10 else ''}"
        )

    def _instance_runtime_dir(self, address: CellAddress) -> Path:
        """
        本次 spawn 的实例残迹目录 (§YY-1 第 6 条 uid 键).

        {workspace}/runtime/cells/{normalize(address)}/ — 保 spawn cwd + 日志 + scratch.
        与 home (稳定身份键) 分离.
        """
        return (
            self.workspace.runtime()
            .sub_storage('cells').sub_storage(normalize(address))
            .abspath()
        )

    def _processes_snapshot(self) -> Iterable[dict]:
        """占位: 返回 processes.executing() 里可能匹配 singleton 查重的 meta.

        Subprocesses.executing 返回 dict[int, ProcessMeta]. ProcessMeta 上有
        name 字段 (spawn 时 name=f'cell:{manifest.name}'), 用来 dedup.
        """
        for _idx, meta in self.processes.executing().items():
            name = getattr(meta, 'name', '') or ''
            cell_name = name[5:] if name.startswith('cell:') else ''
            yield {
                'cell_name': cell_name,
                'pid': getattr(meta, 'pid', 0),
            }

    # ==================================================================
    # 灶台 (§UU-2 / §YY: Subprocesses / JobSupervisor 从 IoC pull)
    # ==================================================================

    @property
    def processes(self) -> Subprocesses:
        return self._container.force_fetch(Subprocesses)

    @property
    def jobs(self) -> JobSupervisor:
        return self._container.force_fetch(JobSupervisor)

    # ==================================================================
    # 门: session / workspace / home / container / logger
    # ==================================================================

    @property
    def session(self) -> Session:
        # §YY-1 第 1 条 (session 永在首页) + §ZZ-5 (通过 IoC provider chain 拉起,
        # provider 内部 lazy). MOSS Session ≠ zenoh.Session — Session 是通讯总线抽象.
        return self._container.force_fetch(Session)

    # workspace 由 Matrix ABC 提供 concrete: return self.project.workspace.
    # (blueprint/matrix.py L298-301) 无需 override.

    @property
    def home(self) -> Workspace:
        """
        本 cell 的持久领地 (§YY-1 第 6 条双目录判决 — 稳定身份键).

        {workspace}/cells/{normalize(manifest.name)}/. name 而非 address —
        cell 重启后必须找得回自己的记忆. 与实例残迹 (uid 键) 分离,
        永不自动清 (systemd StateDirectory= 同构).

        TODO: CELL.md 声明可覆写归宿, 本期未实施.
        """
        # 用 workspace 的 sub_storage 组合. LocalWorkspace 的 sub_storage 返回
        # 一个 LocalWorkspace 子实例 — 但 Workspace ABC 不承诺 sub 是 Workspace,
        # 而是 Storage. 这里我们要 Workspace, 所以直接构造 LocalWorkspace(path).
        from ghoshell_moss.contracts.workspace import LocalWorkspace
        home_path = (
            self.workspace.root().abspath() / 'cells' / normalize(self._manifest.name)
        )
        home_path.mkdir(parents=True, exist_ok=True)
        return LocalWorkspace(home_path)

    @property
    def container(self) -> IoCContainer:
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

    def is_host_running(self) -> bool:
        """
        §YY blueprint/matrix.py L397-404: cell 侧判断组网状态的 code as prompt.

        本 cell 是 host → 直接返回 self.is_running().
        本 cell 是 worker → 走 mesh() view 查有 is_host=True 的 presence.
          注意: 这会迫使 owner 惰性创建 Watcher — 与 wait_present 同一合理耦合.
        本期实现: worker 侧默认 True (未来补 host liveness 判定).
        """
        if self.this.is_host:
            return self.is_running()
        # TODO: worker 侧走 mesh view host filter. mesh() 是 async, is_host_running
        # 是 sync — 本期先返回 True 兜底, 需求真出现时把签名改成 async 或引入
        # 缓存 (Watcher 已有 view() 是 sync).
        if self._watcher is not None:
            hosts = [p for p in self._watcher.view().values() if p.is_host]
            return len(hosts) > 0
        return True   # mesh 未惰性化 → 兜底

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
            cor: Coroutine,
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        self._check_running()

        async def _wrap():
            try:
                await cor
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._logger.error(
                    "%s inner task %s exception: %r",
                    self._log_prefix, name, e,
                )
                if stop_matrix_on_error:
                    self.close()
            finally:
                self._logger.debug("%s inner task %s done", self._log_prefix, name)

        task = self._event_loop.create_task(_wrap(), name=name)
        self._task_group.add(task)
        task.add_done_callback(self._task_group.discard)
        return task

    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """运行前注册. 运行中调用抛错 (add_lifecycle_object 才是动态添加通道)."""
        if self._closing_event.is_set():
            raise RuntimeError('Matrix already closing')
        if self.is_running():
            raise RuntimeError(
                'Matrix already running; use add_lifecycle_object for dynamic add'
            )
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

    def _prepare_container(self) -> Container:
        """
        sync 阶段: 注册全部 provider, 返回未 bootstrap 的 container.

        §ZZ-5 装配次序: set (5 个已在实例) → matrix_manifests providers →
        matrix default providers → adapter.bind_ioc → adapter.default_providers →
        logger default → (caller 负责 bootstrap).
        """
        container = Container(name=self.this.address)

        # -- 5 个"实例已在"直接 set -- #
        container.set(Matrix, self)
        container.set(MatrixImpl, self)
        container.set(Environment, self._env)
        container.set(Project, self._project)
        container.set(Workspace, self._project.workspace)
        container.set(MatrixNetworkAdapter, self._adapter)

        # -- matrix_manifests().providers 全 register (workspace baseline, §ZZ-2) -- #
        matrix_manifests = self._project.matrix_manifests()
        for provider_manifest in matrix_manifests.providers():
            if provider_manifest.is_error():
                self._logger.warning(
                    "%s skip provider manifest with error: %s (%s)",
                    self._log_prefix, provider_manifest.name(), provider_manifest.error(),
                )
                continue
            container.register(provider_manifest.value())

        # -- matrix default providers — if not bound (§ZZ-2 兜底) -- #
        for provider in self._default_providers():
            if container.bound(provider.contract()):
                continue
            container.register(provider)

        # -- adapter driver-specific 装配 (§ZZ-5) -- #
        # bind_ioc: 通常注册 lazy provider (如 zenoh.Session), 捕捉 adapter 引用,
        # 首次 fetch 时读 adapter._session (那时 adapter.__aenter__ 已完成).
        self._adapter.bind_ioc(container)
        for provider in self._adapter.default_providers():
            if container.bound(provider.contract()):
                continue
            container.register(provider)

        # -- configs (从 matrix_manifests 收集, 走 workspace yaml store) -- #
        # 老代码里在 _default_providers 里注册 WorkspaceYamlConfigStoreProvider,
        # 这里保持一致 — 但 matrix_manifests.configs() 返回 configs 声明,
        # 装配到 provider 里.
        configs = [
            m.value() for m in matrix_manifests.configs()
            if not m.is_error()
        ]
        if not container.bound(ConfigStore):
            container.register(WorkspaceYamlConfigStoreProvider(*configs))

        # -- resources (matrix_manifests.resources → bootstrapper) -- #
        for resource_manifest in matrix_manifests.resources():
            if resource_manifest.is_error():
                continue
            storage_factory = resource_manifest.value()
            bootstrapper = ResourceStorageFactoryBootstrapper(storage_factory)
            container.add_bootstrapper(bootstrapper)

        return container

    def _default_providers(self) -> Iterable[Provider]:
        """
        matrix 层的 default 接线 (§ZZ-2 default 兜底).

        workspace 用户在 MatrixManifest 里显式覆写即可覆盖.
        driver-specific 的 default (topic/session/zenoh.Session) 归 adapter.
        """
        yield MatrixSubprocessesProvider()
        yield MatrixJobSupervisorProvider()
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
        self._container = self._prepare_container()
        self._exit_stack.enter_context(
            self._container_lifecycle_ctx()
        )
        # container.bootstrap 已在 _container_lifecycle_ctx 里完成.

        # pull LoggerItf from IoC 覆写 (§ZZ-6): 有覆写就用, 没有保 self._logger 兜底
        pulled = self._container.get(LoggerItf)
        if pulled is not None:
            # 只在 pulled 不是 root moss logger 时覆写 (避免层级混乱)
            self._logger = pulled if isinstance(pulled, logging.Logger) else self._logger

        # -- async 阶段 -- #
        try:
            await self._async_exit_stack.__aenter__()

            # channel provider tasks 收尾钩子 (承老 _ensure_channel_provider_task_cancelled)
            self._async_exit_stack.push_async_callback(self._cancel_channel_provider_tasks)

            # 1. adapter async 起 (driver 底层: zenoh session + hub)
            await self._async_exit_stack.enter_async_context(self._adapter)

            # 2. new_presence + async ctx (触发首次 announce)
            self._presence = self._adapter.new_presence(
                self._presence_data,
                logger=self._logger,
            )
            await self._async_exit_stack.enter_async_context(self._presence)

            # 3. force_fetch TopicService + Session 触发 provider chain,
            #    进 async ctx 让它们跑到关闭
            topic_service = self._container.force_fetch(TopicService)
            await self._async_exit_stack.enter_async_context(topic_service)

            session = self._container.force_fetch(Session)
            await self._async_exit_stack.enter_async_context(session)

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

    @contextlib.contextmanager
    def _container_lifecycle_ctx(self):
        """container.bootstrap + configs 装载 + shutdown 反卷 (承老代码)."""
        self._container.bootstrap()
        try:
            # configs 装载: matrix_manifests.configs is_override 走 set_config,
            # 否则 get_or_create.
            matrix_manifests = self._project.matrix_manifests()
            config_store = self._container.get(ConfigStore)
            if config_store is not None:
                for cm in matrix_manifests.configs():
                    if cm.is_error():
                        continue
                    cfg = cm.value()
                    # ConfigStore.get_or_create 或 set_config — 本期不引入 is_override
                    # 语义 (老代码里从 config_info.is_override 判断, 那是 mode 侧).
                    try:
                        config_store.get_or_create(cfg)
                    except Exception:
                        self._logger.exception(
                            "%s config get_or_create failed: %s",
                            self._log_prefix, cfg,
                        )
            yield
        finally:
            self._container.shutdown()

    async def _cancel_channel_provider_tasks(self) -> None:
        tasks = list(self._channel_provider_tasks)
        self._channel_provider_tasks.clear()
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def _cancel_task_group(self) -> None:
        tasks = list(self._task_group)
        self._task_group.clear()
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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

        # sync stack 反卷 (container shutdown 在此触发)
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
